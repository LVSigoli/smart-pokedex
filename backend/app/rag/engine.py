import difflib
import re
from datetime import datetime, timezone
from .retrieval import normalize_plain
from .generation import FlanT5Generator

REFUSAL = "Não encontrei informações suficientes no corpus de Kanto/FireRed para responder com segurança."
TECHNICAL = re.compile(r"(?i)\b(?:pokeapi|endpoint|schema|jsonl|tf-idf|api)\b")

class RAGEngine:
    def __init__(self, retriever, settings):
        self.retriever = retriever
        self.settings = settings
        self.name_to_id = {normalize_plain(r.get("api_name") or r.get("name")): int(r["id"]) for r in retriever.records}
        self.known_names = sorted(self.name_to_id, key=len, reverse=True)
        self.generator = FlanT5Generator(getattr(settings, "flan_model", "google/flan-t5-small"))

    def canonicalize(self, question):
        result, corrections = [], []
        for token in normalize_plain(question).split():
            if token in self.name_to_id or len(token) < 4:
                result.append(token); continue
            choice = difflib.get_close_matches(token, self.known_names, n=1, cutoff=.84)
            if choice:
                result.append(choice[0]); corrections.append({"from": token, "to": choice[0]})
            else:
                result.append(token)
        return " ".join(result), corrections

    def mentioned_ids(self, question):
        q = normalize_plain(question)
        return list(dict.fromkeys(self.name_to_id[n] for n in self.known_names if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", q)))

    def requested_section(self, question):
        q = normalize_plain(question)
        if re.search(r"\b(evolu\w*|evolv\w*|pre[- ]?evolu\w*)\b", q): return "evolution"
        # Verbos flexionados aparecem em formas como encontrar, encontro, pego e captura.
        if re.search(r"\b(?:onde|where)\b.*\b(?:encontr\w*|localiz\w*|captur\w*|peg\w*|find|catch)\b", q): return "encounters"
        if re.search(r"\b(?:como|how)\b.*\b(?:encontr\w*|localiz\w*|captur\w*|peg\w*|find|catch)\b", q): return "encounters"
        if re.search(r"\b(?:encontr\w*|localiz\w*|captur\w*|peg\w*|find|catch)\b", q): return "encounters"
        if re.search(r"\b(como (?:conseguir|obter|ganhar|receber)|obter|obten\w*|conseguir|obtain|gift|trade|get)\b", q): return "obtainment"
        if re.search(r"\b(fraquez|resist|imun|weak|effectiveness)\b", q): return "type_matchups"
        if re.search(r"\b(habilidad|ability)\b", q): return "abilities"
        if re.search(r"\b(golpe|movimento|move|tm|hm|aprende|learn)\b", q): return "moves"
        if re.search(r"\b(ataque|defesa|velocidade|speed|stat|atribut|hp)\b", q): return "stats"
        return "profile" if self.mentioned_ids(question) else None

    @staticmethod
    def is_same_type_query(question):
        q = normalize_plain(question)
        return bool(re.search(r"\b(?:mesm[oa]s?|igual(?:es)?|same)\b.*\b(?:tipo|types?)\b", q))

    @staticmethod
    def profile_types(row):
        content = str(row.get("content", ""))
        match = re.search(r"Types in Generation III:\s*([^.]*)\.", content, re.IGNORECASE)
        if not match:
            return frozenset()
        return frozenset(normalize_plain(value) for value in match.group(1).split(",") if value.strip())

    def select_same_type_evidence(self, question, targets, top_k):
        if len(targets) != 1:
            return [], "same_type_requires_one_reference"
        target_id = targets[0]
        target_profile = next((row for row in self.retriever.documents
                               if int(row["pokemon_id"]) == target_id and row["section"] == "profile"), None)
        if not target_profile:
            return [], "missing_target_profile"
        target_types = self.profile_types(target_profile)
        if not target_types:
            return [], "missing_target_types"
        candidates = [row for row in self.retriever.documents
                      if row["section"] == "profile"
                      and int(row["pokemon_id"]) != target_id
                      and self.profile_types(row) == target_types]
        candidates.sort(key=lambda row: (int(row["pokemon_id"]), row["document_id"]))
        selected = [target_profile] + candidates[:max(1, top_k - 1)]
        return [dict(row, rank=index + 1, score=float(row.get("score", 0.0)))
                for index, row in enumerate(selected)], "ready"

    def scope_gate(self, question):
        q = normalize_plain(question)
        if re.search(r"\b(ginasio|gym|elite four|detonado|walkthrough|storyline)\b", q): return "out_of_scope"
        match = re.search(r"\b(?:pokemon|pokedex|numero|number|n)\s*(\d{1,4})\b", q)
        if match and int(match.group(1)) not in range(1, 152): return "outside_kanto_dex"
        if re.search(r"\b(mais forte|melhor pokemon|strongest|best pokemon)\b", q) and not re.search(r"\b(ataque|defesa|velocidade|hp|stat|atribut|speed)\b", q): return "needs_comparison_criterion"

    def select_evidence(self, question, top_k=5):
        gate = self.scope_gate(question)
        if gate: return [], gate
        targets, section = self.mentioned_ids(question), self.requested_section(question)
        if self.is_same_type_query(question):
            return self.select_same_type_evidence(question, targets, top_k)
        ranked = self.retriever.search(question, top_k=len(self.retriever.documents))
        if targets: ranked = [r for r in ranked if r["pokemon_id"] in targets]
        if section in {"obtainment", "encounters"}:
            ranked = [r for r in ranked if r["section"] in ("obtainment", "encounters", "evolution")]
        elif section: ranked = [r for r in ranked if r["section"] == section]
        if not ranked: return [], "no_relevant_documents"
        if targets: ranked.sort(key=lambda r: (targets.index(r["pokemon_id"]) if r["pokemon_id"] in targets else len(targets), -r["score"]))
        return [dict(r, rank=i + 1) for i, r in enumerate(ranked[:max(top_k, len(targets))])], "ready"

    def build_context(self, rows, max_chars=13500, max_doc_chars=3500, *, question=None, token_budget=None):
        if question is not None and token_budget is not None:
            parts, used = [], []
            for row in rows:
                header = f"FONTE [{row['document_id']}] | Pokémon: {row['pokemon_name']} | seção: {row['section']}\n"
                candidate = "\n\n".join(parts + [header + str(row["content"])])
                if self.generator.count_prompt_tokens(question, candidate) > token_budget:
                    continue
                parts.append(header + str(row["content"]))
                used.append(dict(row, content=str(row["content"])))
            return "\n\n".join(parts), used
        parts, used, total = [], [], 0
        for row in rows:
            header = f"FONTE [{row['document_id']}] | Pokémon: {row['pokemon_name']} | seção: {row['section']}\n"
            budget = min(max_doc_chars, max_chars - total - len(header) - 2)
            if budget < 100: break
            content = str(row["content"])[:budget]
            parts.append(header + content); used.append(dict(row, content=content)); total += len(header) + len(content) + 2
        return "\n\n".join(parts), used

    @staticmethod
    def verify(payload, used):
        if not isinstance(payload, dict) or payload.get("can_answer") is not True or not isinstance(payload.get("answer"), str): return False
        if TECHNICAL.search(payload["answer"]): return False
        lookup = {r["document_id"]: r["content"] for r in used}
        cited = set(re.findall(r"\[([0-9]{3}_[a-z0-9_]+)\]", payload["answer"]))
        if not cited or not isinstance(payload.get("evidence"), list): return False
        verified = set()
        for item in payload["evidence"]:
            doc, quote = item.get("document_id"), str(item.get("quote", ""))
            if doc not in lookup or len(" ".join(quote.split())) < 8 or " ".join(quote.split()).casefold() not in " ".join(lookup[doc].split()).casefold(): return False
            verified.add(doc)
        return cited.issubset(verified)

    @staticmethod
    def retrieval_answer(rows, question):
        """Fallback factual para permitir inspeção sem o gerador local; usa só o conteúdo recuperado."""
        if not rows:
            return None
        if rows[0].get("section") == "evolution":
            transitions = []
            for line in str(rows[0].get("content", "")).splitlines():
                match = re.search(r"-\s*([a-z0-9-]+)\s*->\s*([a-z0-9-]+)", line, re.I)
                if match:
                    _origin, target = (part.replace("-", " ").title() for part in match.groups())
                    item_match = re.search(r"\bitem:\s*([^;.)]+)", line, re.I)
                    item = item_match.group(1).strip().replace("-", " ").title() if item_match else ""
                    transitions.append(f"{target} usando {item}" if item else target)
            if transitions:
                subject = rows[0].get("pokemon_name", "O Pokémon")
                joined = ", ".join(transitions)
                if re.search(r"\b(como|quais|qual|evolu)\b", normalize_plain(question)):
                    return f"{subject} pode evoluir para {joined}."
                return f"{subject} can evolve into {joined}."
        excerpts = []
        for row in rows[:3]:
            content = str(row.get("content", "")).strip()
            if content:
                excerpts.append(content)
        return "\n\n".join(excerpts)[:9000] if excerpts else None

    def answer(self, question, generate=False):
        if not isinstance(question, str) or not question.strip(): raise ValueError("Pergunta vazia")
        canonical, corrections = self.canonicalize(question)
        rows, gate = self.select_evidence(canonical, self.settings.rag_top_k)
        generation_preflight_error = None
        if generate:
            try:
                _, used = self.build_context(rows, question=canonical, token_budget=self.generator.max_input_tokens)
            except Exception as exc:
                # A falha ao carregar/tokenizar o modelo não invalida a recuperação.
                # Mantemos as fontes e deixamos a resposta transparente como indisponível.
                generation_preflight_error = exc
                _, used = self.build_context(rows)
        else:
            _, used = self.build_context(rows)
        if not used:
            status = "out_of_scope" if gate in {"out_of_scope", "outside_kanto_dex", "needs_comparison_criterion"} else "insufficient_evidence"
            return {"status": status, "answer": REFUSAL, "source_documents": [], "supporting_excerpts": [], "spelling_corrections": corrections}
        base = {"status": "retrieval_only", "answer": None, "source_documents": [{k: r.get(k) for k in ("document_id", "pokemon_id", "pokemon_name", "section", "score")} for r in used], "supporting_excerpts": [], "spelling_corrections": corrections, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        if not generate:
            return {**base, "answer": self.retrieval_answer(used, question)}
        if generation_preflight_error is not None:
            return {**base, "status": "provider_unavailable", "answer": self.retrieval_answer(used, question),
                    "generation_error": type(generation_preflight_error).__name__}
        try:
            context, used = self.build_context(rows, question=canonical, token_budget=self.generator.max_input_tokens)
            if not used:
                return {**base, "status": "insufficient_evidence", "answer": REFUSAL}
            payload = self.generator(question, context)
        except Exception:
            return {**base, "status": "provider_unavailable", "answer": self.retrieval_answer(used, question)}
        answer = str(payload.get("answer", "")).strip()
        if not answer or answer.casefold() == "não sei":
            return {**base, "status": "insufficient_evidence", "answer": REFUSAL}
        excerpts = [{"document_id": row["document_id"], "quote": row["content"]} for row in used]
        return {**base, "status": "answered", "answer": answer,
                "supporting_excerpts": excerpts,
                "generation_metadata": {key: value for key, value in payload.items() if key != "answer"}}

