import difflib
import re
from datetime import datetime, timezone
from .retrieval import normalize_plain
from .generation import GeminiGenerator

REFUSAL = "Não encontrei informações suficientes no corpus de Kanto/FireRed para responder com segurança."
TECHNICAL = re.compile(r"(?i)\b(?:pokeapi|endpoint|schema|jsonl|tf-idf|api)\b")

class RAGEngine:
    def __init__(self, retriever, settings):
        self.retriever = retriever
        self.settings = settings
        self.name_to_id = {normalize_plain(r.get("api_name") or r.get("name")): int(r["id"]) for r in retriever.records}
        self.known_names = sorted(self.name_to_id, key=len, reverse=True)
        self.generator = GeminiGenerator(settings.gemini_api_key, settings.gemini_model) if settings.gemini_api_key else None

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
        ranked = self.retriever.search(question, top_k=len(self.retriever.documents))
        if targets: ranked = [r for r in ranked if r["pokemon_id"] in targets]
        if section in {"obtainment", "encounters"}:
            ranked = [r for r in ranked if r["section"] in ("obtainment", "encounters", "evolution")]
        elif section: ranked = [r for r in ranked if r["section"] == section]
        if not ranked: return [], "no_relevant_documents"
        if targets: ranked.sort(key=lambda r: (targets.index(r["pokemon_id"]) if r["pokemon_id"] in targets else len(targets), -r["score"]))
        return [dict(r, rank=i + 1) for i, r in enumerate(ranked[:max(top_k, len(targets))])], "ready"

    def build_context(self, rows, max_chars=13500, max_doc_chars=3500):
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
        """Fallback factual para permitir uso sem Gemini; só usa o conteúdo recuperado."""
        if not rows:
            return None
        if rows[0].get("section") == "evolution":
            transitions = []
            for line in str(rows[0].get("content", "")).splitlines():
                match = re.search(r"-\s*([a-z0-9-]+)\s*->\s*([a-z0-9-]+).*?item:\s*([^;.)]+)", line, re.I)
                if match:
                    origin, target, item = (part.replace("-", " ").title() for part in match.groups())
                    transitions.append(f"{target} usando {item}")
            if transitions:
                subject = rows[0].get("pokemon_name", "O Pokémon")
                if re.search(r"\b(como|quais|qual|evolu)\b", normalize_plain(question)):
                    return f"{subject} pode evoluir para " + ", ".join(transitions[:-1]) + (f" e {transitions[-1]}." if len(transitions) > 1 else ".")
                return f"{subject} can evolve into " + ", ".join(transitions[:-1]) + (f" and {transitions[-1]}." if len(transitions) > 1 else ".")
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
        _, used = self.build_context(rows)
        if not used:
            status = "out_of_scope" if gate in {"out_of_scope", "outside_kanto_dex", "needs_comparison_criterion"} else "insufficient_evidence"
            return {"status": status, "answer": REFUSAL, "source_documents": [], "supporting_excerpts": [], "spelling_corrections": corrections}
        base = {"status": "retrieval_only", "answer": None, "source_documents": [{k: r.get(k) for k in ("document_id", "pokemon_id", "pokemon_name", "section", "score")} for r in used], "supporting_excerpts": [], "spelling_corrections": corrections, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        if not generate:
            return {**base, "answer": self.retrieval_answer(used, question)}
        if not self.generator:
            return {**base, "status": "provider_unavailable", "answer": self.retrieval_answer(used, question)}
        try:
            payload = self.generator(question, self.build_context(rows)[0])
        except Exception:
            return {**base, "status": "provider_unavailable", "answer": self.retrieval_answer(used, question)}
        if not self.verify(payload, used): return {**base, "status": "insufficient_evidence", "answer": REFUSAL}
        answer = re.sub(r"\s*\[[0-9]{3}_[a-z0-9_]+\]", "", payload["answer"]).strip()
        return {**base, "status": "answered", "answer": answer, "supporting_excerpts": payload["evidence"]}

