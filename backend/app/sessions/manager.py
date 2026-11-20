from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
from threading import RLock
from uuid import UUID, uuid4
from app.core.exceptions import SessionNotFound

FOLLOWUP_TERMS = {
    "ele", "ela", "eles", "elas", "dele", "dela", "deles", "delas",
    "qual", "quais", "evolucao", "evolution", "tipo", "type",
    "onde", "encontro", "encontrar", "capturo", "capturar", "localizacao",
    "localizar", "where", "find", "catch", "get", "obter", "conseguir", "ganhar",
    "como", "evoluo", "evoluir", "evolui", "evolucao",
    "ataque", "defesa", "velocidade", "hp", "attack", "defense", "speed",
}

COMPARISON_ATTRIBUTES = {
    "ataque": "ataque", "attack": "ataque",
    "defesa": "defesa", "defense": "defesa",
    "velocidade": "velocidade", "speed": "velocidade",
    "hp": "HP",
}

@dataclass
class ChatSession:
    session_id: UUID
    engine: object
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_ids: list[int] = field(default_factory=list)
    pending_comparison_ids: list[int] = field(default_factory=list)
    pending_ranking: bool = False
    last_section: str | None = None
    pending_question: str | None = None
    pending_candidates: list[int] = field(default_factory=list)
    pending_kind: str | None = None
    turns: list[dict] = field(default_factory=list)

class SessionManager:
    def __init__(self, engine, ttl_seconds=1800):
        self.engine, self.ttl = engine, timedelta(seconds=ttl_seconds)
        self._sessions, self._lock = {}, RLock()

    def _purge(self):
        now = datetime.now(timezone.utc)
        self._sessions = {k:v for k,v in self._sessions.items() if now-v.last_activity < self.ttl}

    def create(self):
        with self._lock:
            self._purge(); sid = uuid4(); self._sessions[sid] = ChatSession(sid, self.engine); return self._sessions[sid]

    def get(self, sid):
        try: sid = UUID(str(sid))
        except ValueError: raise SessionNotFound(str(sid))
        with self._lock:
            self._purge()
            if sid not in self._sessions: raise SessionNotFound(str(sid))
            session = self._sessions[sid]; session.last_activity = datetime.now(timezone.utc); return session

    def delete(self, sid):
        with self._lock:
            try: del self._sessions[UUID(str(sid))]
            except (KeyError, ValueError): raise SessionNotFound(str(sid))

    @staticmethod
    def _model_clarification(session, question, names, generate):
        if generate and hasattr(session.engine, "generator"):
            context = (
                "INSTRUÇÃO DE CONTROLE: A referência é ambígua. Não responda ao conteúdo factual. "
                "Pergunte ao usuário qual Pokémon quis dizer, citando as opções: " + ", ".join(names)
            )
            try:
                payload = session.engine.generator(question, context)
                answer = str(payload.get("answer", "")).strip()
                if answer:
                    return {"status": "clarification_needed", "answer": answer,
                            "answer_mode": "llm_generated", "final_answer_origin": "model",
                            "source_documents": [], "supporting_excerpts": [],
                            "generation_metadata": {k: v for k, v in payload.items() if k != "answer"}}
            except Exception:
                pass
        return {"status": "clarification_required",
                "answer": f"De qual Pokémon você está falando: {' ou '.join(names)}?",
                "source_documents": [], "supporting_excerpts": []}

    def ask(self, sid, message, generate=False):
        session = self.get(sid)
        previous_active = list(session.active_ids)
        explicit = session.engine.mentioned_ids(message) if hasattr(session.engine, "mentioned_ids") else []
        normalized = str(message).casefold()
        tokens = set(re.findall(r"\w+", normalized))
        followup = not explicit and bool(tokens & FOLLOWUP_TERMS)
        attribute = next((label for token, label in COMPARISON_ATTRIBUTES.items() if token in tokens), None)
        pronoun_comparison = bool(tokens & {"ele", "ela", "dele", "dela", "it", "its"}) and bool(
            tokens & {"forte", "melhor", "stronger", "better"})

        if session.pending_ranking and attribute and not explicit:
            result = session.engine.answer(
                f"Quais são os pokemons com maior {attribute}?", generate=generate)
            session.pending_ranking = False
        elif session.pending_comparison_ids and attribute and not explicit:
            names = [session.engine.retriever.by_id[i].get("name", str(i))
                     for i in session.pending_comparison_ids]
            result = session.engine.answer(
                f"Compare o {attribute} de {' e '.join(names)}.", generate=generate)
            session.pending_comparison_ids = []
        elif explicit and session.active_ids and pronoun_comparison:
            combined_ids = list(dict.fromkeys([*session.active_ids, *explicit]))
            names = [session.engine.retriever.by_id[i].get("name", str(i)) for i in combined_ids]
            result = session.engine.answer(
                f"{message} comparando {' e '.join(names)}", generate=generate)
            session.active_ids = combined_ids
        elif followup and session.active_ids:
            if len(session.active_ids) > 1 and bool(tokens & {"ele", "ela", "dele", "dela"}):
                names = [session.engine.retriever.by_id[i].get("name", str(i)) for i in session.active_ids]
                result = {"status": "clarification_required", "answer": f"De qual Pokémon você está falando: {' ou '.join(names)}?", "source_documents": [], "supporting_excerpts": []}
            else:
                names = [session.engine.retriever.by_id[i].get("name", str(i)) for i in session.active_ids]
                result = session.engine.answer(f"{message} sobre {' e '.join(names)}", generate=generate)
        else:
            if explicit:
                session.pending_comparison_ids = []
            result = session.engine.answer(message, generate=generate)
        if result.get("status") == "clarification_required" and generate and len(session.active_ids) > 1:
            names = [session.engine.retriever.by_id[i].get("name", str(i)) for i in session.active_ids]
            result = self._model_clarification(session, message, names, generate)
        model_answer = result.get("answer") if result.get("final_answer_origin") else None
        if result.get("gate_reason") == "needs_comparison_criterion" and session.active_ids:
            session.pending_comparison_ids = list(session.active_ids)
            names = [session.engine.retriever.by_id[i].get("name", str(i))
                     for i in session.pending_comparison_ids]
            result["answer"] = (
                f"Em qual atributo você quer comparar {' e '.join(names)}: "
                "ataque, defesa, velocidade ou HP?"
            )
        elif result.get("gate_reason") == "needs_comparison_criterion":
            session.pending_ranking = True
            result["answer"] = "Em qual atributo você quer ranquear: ataque, defesa, velocidade ou HP?"
        if model_answer is not None:
            result["answer"] = model_answer
        if explicit:
            if not (explicit and previous_active and pronoun_comparison):
                session.active_ids = list(dict.fromkeys(explicit))
        elif followup and previous_active:
            session.active_ids = previous_active
        else:
            session.active_ids = []
        result.setdefault("original_question", message)
        result.setdefault("resolved_question", message)
        result.setdefault("context_used", bool(session.active_ids))
        result.setdefault("active_pokemon", [
            session.engine.retriever.by_id[i].get("name", str(i))
            for i in session.active_ids if i in session.engine.retriever.by_id
        ])
        session.last_question = message
        session.turns.append({"message": message, "result": result}); session.turns = session.turns[-8:]
        return result

