from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from uuid import UUID, uuid4
from app.core.exceptions import SessionNotFound

FOLLOWUP_TERMS = {
    "ele", "ela", "eles", "elas", "dele", "dela", "deles", "delas",
    "qual", "quais", "evolucao", "evolution", "tipo", "type",
    "onde", "encontro", "encontrar", "capturo", "capturar", "localizacao",
    "localizar", "where", "find", "catch", "get", "obter", "conseguir", "ganhar",
    "como", "evoluo", "evoluir", "evolui", "evolucao",
}

@dataclass
class ChatSession:
    session_id: UUID
    engine: object
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_ids: list[int] = field(default_factory=list)
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

    def ask(self, sid, message, generate=False):
        session = self.get(sid)
        explicit = session.engine.mentioned_ids(message) if hasattr(session.engine, "mentioned_ids") else []
        normalized = str(message).casefold()
        tokens = set(normalized.split())
        followup = not explicit and any(token in normalized.split() for token in ("ele", "ela", "dele", "dela", "qual", "quais", "evolução", "evolucao", "evolution", "tipo", "type"))
        followup = not explicit and bool(tokens & FOLLOWUP_TERMS)
        if followup and session.active_ids:
            if len(session.active_ids) > 1 and any(token in normalized.split() for token in ("ele", "ela", "dele", "dela")):
                names = [session.engine.retriever.by_id[i].get("name", str(i)) for i in session.active_ids]
                result = {"status": "clarification_required", "answer": f"De qual Pokémon você está falando: {' ou '.join(names)}?", "source_documents": [], "supporting_excerpts": []}
            else:
                names = [session.engine.retriever.by_id[i].get("name", str(i)) for i in session.active_ids]
                result = session.engine.answer(f"{message} sobre {' e '.join(names)}", generate=generate)
        else:
            result = session.engine.answer(message, generate=generate)
        ids = {int(item["pokemon_id"]) for item in result.get("source_documents", [])}
        if ids: session.active_ids = sorted(ids)
        session.turns.append({"message": message, "result": result}); session.turns = session.turns[-8:]
        return result

