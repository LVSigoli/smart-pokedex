import difflib
import re
from datetime import datetime, timezone
from .retrieval import normalize_plain
from .generation import FlanT5Generator

REFUSAL = "Não encontrei informações suficientes no corpus de Kanto/FireRed para responder com segurança."
class RAGEngine:
    def __init__(self, retriever, settings):
        self.retriever = retriever
        self.settings = settings
        self.name_to_id = {normalize_plain(r.get("api_name") or r.get("name")): int(r["id"]) for r in retriever.records}
        self.known_names = sorted(self.name_to_id, key=len, reverse=True)
        self.generator = FlanT5Generator(
            getattr(settings, "flan_model", "google/flan-t5-small"),
            max_input_tokens=getattr(settings, "flan_max_input_tokens", 384),
            max_new_tokens=getattr(settings, "flan_max_new_tokens", 96),
            num_beams=getattr(settings, "flan_num_beams", 1),
        )

    def canonicalize(self, question):
        normalized = normalize_plain(question)
        protected_tokens = set()
        for name in self.known_names:
            if " " in name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", normalized):
                protected_tokens.update(name.split())
        result, corrections = [], []
        for token in normalized.split():
            # Nidoran F/M e outras espécies compostas não podem passar pela
            # correção fuzzy palavra a palavra, que confundiria os irmãos.
            if token in protected_tokens or token in self.name_to_id or len(token) < 4:
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
        if re.search(r"\b(ginasios?|gyms?|elite four|detonado|walkthrough|storyline)\b", q): return "out_of_scope"
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
        if section in {"obtainment", "encounters"}:
            # Evolução não comprova onde o Pokémon é encontrado ou obtido.
            # Mantemos somente evidência de aquisição/localização para evitar
            # respostas como "evolui para..." em perguntas de encontro.
            ranked = [r for r in ranked if r["section"] in ("obtainment", "encounters")]
        elif section:
            ranked = [r for r in ranked if r["section"] == section]
        minimum_score = float(getattr(self.settings, "rag_min_score", 0.15))
        if targets:
            ranked = [r for r in ranked if r["pokemon_id"] in targets]
            found_targets = {
                int(r["pokemon_id"])
                for r in ranked
                if float(r.get("score", 0.0)) >= minimum_score
            }
            # Uma consulta comparativa pode dar pontuação suficiente para um
            # alvo e quase nenhuma para o outro. Reaplicamos o mesmo TF-IDF
            # persistido por alvo apenas para completar a cobertura exigida.
            for target_id in targets:
                if target_id in found_targets:
                    continue
                record = next((item for item in self.retriever.records if int(item["id"]) == target_id), None)
                if not record:
                    continue
                target_name = record.get("api_name") or record.get("name") or str(target_id)
                target_query = f"{target_name} {section or ''}".strip()
                complementary = self.retriever.search(target_query, top_k=len(self.retriever.documents))
                complementary = [r for r in complementary if int(r["pokemon_id"]) == target_id]
                if section in {"obtainment", "encounters"}:
                    complementary = [r for r in complementary if r["section"] in ("obtainment", "encounters")]
                elif section:
                    complementary = [r for r in complementary if r["section"] == section]
                ranked.extend(complementary)
            unique = {}
            for row in ranked:
                current = unique.get(row["document_id"])
                if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                    unique[row["document_id"]] = row
            ranked = list(unique.values())
        if not ranked: return [], "no_relevant_documents"
        ranked = [row for row in ranked if float(row.get("score", 0.0)) >= minimum_score]
        if not ranked: return [], "below_similarity_threshold"
        if targets: ranked.sort(key=lambda r: (targets.index(r["pokemon_id"]) if r["pokemon_id"] in targets else len(targets), -r["score"]))
        return [dict(r, rank=i + 1) for i, r in enumerate(ranked[:max(top_k, len(targets))])], "ready"

    @staticmethod
    def split_passages(content):
        """Divide uma fonte sem cortar frases, linhas, números ou IDs no meio."""
        passages = []
        transition = []

        def flush_transition():
            if transition:
                passages.append("\n".join(transition))
                transition.clear()

        for line in str(content or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^-\s*[a-z0-9-]+\s*->\s*[a-z0-9-]+", line, re.I):
                flush_transition()
                transition.append(line)
                continue
            if transition and re.match(r"^(?:conditions?:|not specifically verified)", line, re.I):
                transition.append(line)
                continue
            flush_transition()
            passages.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
        flush_transition()
        return passages

    @staticmethod
    def balance_rows_by_pokemon(rows):
        """Intercala fontes por Pokémon para consultas comparativas."""
        buckets, order = {}, []
        for row in rows:
            pokemon_id = int(row["pokemon_id"])
            if pokemon_id not in buckets:
                buckets[pokemon_id] = []
                order.append(pokemon_id)
            buckets[pokemon_id].append(row)
        balanced = []
        offset = 0
        while any(offset < len(buckets[pokemon_id]) for pokemon_id in order):
            for pokemon_id in order:
                if offset < len(buckets[pokemon_id]):
                    balanced.append(buckets[pokemon_id][offset])
            offset += 1
        return balanced

    @staticmethod
    def _rank_passages(question, content):
        terms = {token for token in normalize_plain(question).split() if len(token) > 2}
        passages = RAGEngine.split_passages(content)
        evolution_query = bool(re.search(r"\b(?:evolu\w*|evolv\w*)\b", normalize_plain(question)))

        def relevance(item):
            index, passage = item
            normalized = normalize_plain(passage)
            overlap = len(terms & set(normalized.split()))
            evolution = 20 if evolution_query and "->" in passage else 0
            conditions = 5 if evolution_query and re.search(r"\b(?:minimum level|item|trigger)\b", normalized) else 0
            return (-(overlap * 10 + evolution + conditions), index)

        return [passage for _, passage in sorted(
            enumerate(passages),
            key=relevance,
        )]

    def build_context(self, rows, max_chars=13500, max_doc_chars=3500, *, question=None, token_budget=None):
        if question is not None and token_budget is not None:
            ordered = self.balance_rows_by_pokemon(rows)
            entries = []

            def render(candidate_entries):
                return "\n\n".join(
                    f"FONTE [{entry['row']['document_id']}] | Pokémon: {entry['row']['pokemon_name']} | "
                    f"seção: {entry['row']['section']}\n" + "\n".join(entry["passages"])
                    for entry in candidate_entries if entry["passages"]
                )

            def add_passage(row, passage):
                entry = next((item for item in entries if item["row"]["document_id"] == row["document_id"]), None)
                created = entry is None
                if created:
                    entry = {"row": row, "passages": []}
                    entries.append(entry)
                entry["passages"].append(passage)
                if self.generator.count_prompt_tokens(question, render(entries)) > token_budget:
                    entry["passages"].pop()
                    if created:
                        entries.remove(entry)
                    return False
                return True

            # Primeiro reserva uma passagem completa para cada entidade pedida.
            first_by_pokemon = []
            seen_pokemon = set()
            for row in ordered:
                pokemon_id = int(row["pokemon_id"])
                if pokemon_id not in seen_pokemon:
                    seen_pokemon.add(pokemon_id)
                    first_by_pokemon.append(row)
            for row in first_by_pokemon:
                for passage in self._rank_passages(question, row["content"]):
                    if add_passage(row, passage):
                        break

            # Usa o espaço restante para enriquecer as fontes mais relevantes.
            for row in ordered:
                existing = next((item for item in entries if item["row"]["document_id"] == row["document_id"]), None)
                selected = set(existing["passages"]) if existing else set()
                for passage in self._rank_passages(question, row["content"]):
                    if passage not in selected and add_passage(row, passage):
                        selected.add(passage)

            context = render(entries)
            used = [dict(entry["row"], content="\n".join(entry["passages"]))
                    for entry in entries if entry["passages"]]
            return context, used
        parts, used, total = [], [], 0
        for row in rows:
            header = f"FONTE [{row['document_id']}] | Pokémon: {row['pokemon_name']} | seção: {row['section']}\n"
            budget = min(max_doc_chars, max_chars - total - len(header) - 2)
            if budget < 100: break
            content = str(row["content"])[:budget]
            parts.append(header + content); used.append(dict(row, content=content)); total += len(header) + len(content) + 2
        return "\n\n".join(parts), used

    @staticmethod
    def evolution_transitions(rows):
        transitions = []
        for row in rows:
            if row.get("section") != "evolution":
                continue
            content = str(row.get("content", ""))
            pattern = r"-\s*([a-z0-9-]+)\s*->\s*([a-z0-9-]+)\.\s*(?:Conditions:\s*\((.*?)\))?"
            for match in re.finditer(pattern, content, re.I | re.S):
                origin, target, conditions = match.groups()
                conditions = conditions or ""
                item_match = re.search(r"\bitem:\s*([^;.)]+)", conditions, re.I)
                level_match = re.search(r"\bminimum level:\s*(\d+)", conditions, re.I)
                transitions.append({
                    "origin": origin.replace("-", " ").title(),
                    "target": target.replace("-", " ").title(),
                    "item": item_match.group(1).strip().replace("-", " ").title() if item_match else "",
                    "level": level_match.group(1) if level_match else "",
                })
        return transitions

    @classmethod
    def evolution_answer_is_complete(cls, answer, rows):
        transitions = cls.evolution_transitions(rows)
        normalized_answer = normalize_plain(answer)
        if not transitions:
            return True
        return all(
            normalize_plain(transition["target"]) in normalized_answer
            and (not transition["level"] or transition["level"] in normalized_answer)
            and (not transition["item"] or normalize_plain(transition["item"]) in normalized_answer)
            for transition in transitions
        )

    @staticmethod
    def requested_stat(question):
        normalized = normalize_plain(question)
        patterns = (
            ("special attack", r"\b(?:ataque especial|special attack)\b"),
            ("special defense", r"\b(?:defesa especial|special defense)\b"),
            ("speed", r"\b(?:velocidade|speed)\b"),
            ("attack", r"\b(?:ataque|attack)\b"),
            ("defense", r"\b(?:defesa|defense)\b"),
            ("hp", r"\b(?:hp|pontos de vida|vida)\b"),
        )
        return next((name for name, pattern in patterns if re.search(pattern, normalized)), None)

    @staticmethod
    def profile_types_pt(row):
        content = str(row.get("content", ""))
        match = re.search(r"Types PT-BR:\s*([^.]*)\.", content, re.I)
        if not match:
            match = re.search(r"Types in Generation III:\s*([^.]*)\.", content, re.I)
        return [value.strip() for value in match.group(1).split(",") if value.strip()] if match else []

    @staticmethod
    def is_acquisition_question(question):
        normalized = normalize_plain(question)
        return bool(re.search(
            r"\b(?:onde|where|encontr\w*|localiz\w*|captur\w*|peg\w*|find|catch|"
            r"obter|obten\w*|conseguir|receber|ganhar|obtain|gift|get)\b",
            normalized,
        ))

    @staticmethod
    def acquisition_facts(rows):
        """Extrai somente fatos de aquisição explicitamente presentes nos documentos."""
        facts = []
        seen = set()
        pattern = re.compile(
            r"^-\s*([^;\n]+);\s*method:\s*([^;\n]+);\s*levels:\s*(\d+)-(\d+);",
            re.I | re.M,
        )
        for row in rows:
            if row.get("section") not in {"encounters", "obtainment"}:
                continue
            for match in pattern.finditer(str(row.get("content", ""))):
                location, method, minimum, maximum = (value.strip() for value in match.groups())
                fact = {
                    "pokemon": str(row.get("pokemon_name", "Pokémon")).strip(),
                    "location": location.replace("-", " ").title(),
                    "method": normalize_plain(method),
                    "minimum_level": minimum,
                    "maximum_level": maximum,
                }
                key = tuple(fact.values())
                if key not in seen:
                    seen.add(key)
                    facts.append(fact)
        return facts

    @classmethod
    def acquisition_answer(cls, rows, question):
        if not cls.is_acquisition_question(question):
            return None
        facts = cls.acquisition_facts(rows)
        if not facts:
            return None
        sentences = []
        for fact in facts[:6]:
            name, location, method = fact["pokemon"], fact["location"], fact["method"]
            minimum, maximum = fact["minimum_level"], fact["maximum_level"]
            level = f"no nível {minimum}" if minimum == maximum else f"entre os níveis {minimum} e {maximum}"
            if method == "gift":
                sentence = f"{name} é obtido como presente em {location}, {level}"
            else:
                method_text = {
                    "walk": "caminhando",
                    "surf": "usando Surf",
                    "old rod": "pescando com a Old Rod",
                    "good rod": "pescando com a Good Rod",
                    "super rod": "pescando com a Super Rod",
                    "rock smash": "usando Rock Smash",
                }.get(method, f"pelo método {method}")
                sentence = f"{name} pode ser encontrado em {location}, {method_text}, {level}"
            sentences.append(sentence + ".")
        return "No Pokémon FireRed, " + " ".join(sentences)

    @classmethod
    def structured_answer(cls, rows, question):
        acquisition = cls.acquisition_answer(rows, question)
        if acquisition:
            return acquisition
        transitions = cls.evolution_transitions(rows)
        if transitions:
            portuguese = bool(re.search(r"\b(como|quais|qual|evolu)\b", normalize_plain(question)))
            steps = []
            for transition in transitions:
                if portuguese:
                    condition = (f"usando {transition['item']}" if transition["item"]
                                 else f"no nível {transition['level']}" if transition["level"] else "")
                    steps.append(f"{transition['origin']} evolui para {transition['target']} {condition}".strip())
                else:
                    condition = (f"using {transition['item']}" if transition["item"]
                                 else f"at level {transition['level']}" if transition["level"] else "")
                    steps.append(f"{transition['origin']} evolves into {transition['target']} {condition}".strip())
            return "; ".join(steps) + "."

        normalized_question = normalize_plain(question)
        if re.search(r"\b(?:tipos?|types?)\b", normalized_question):
            if re.search(r"\b(?:mesm[oa]s?|igual(?:es)?|same)\b", normalized_question) and len(rows) > 1:
                reference, matches = rows[0], rows[1:]
                names = ", ".join(row["pokemon_name"] for row in matches)
                types = " e ".join(cls.profile_types_pt(reference))
                return f"{names} têm os mesmos tipos ({types}) que {reference['pokemon_name']}."
            answers = []
            for row in rows:
                types = cls.profile_types_pt(row)
                if types:
                    answers.append(f"{row['pokemon_name']} é do tipo {' e '.join(types)}")
            if answers:
                return "; ".join(answers) + "."

        stat = cls.requested_stat(question)
        if stat:
            labels = {"speed": "velocidade base", "attack": "ataque base", "defense": "defesa base",
                      "special attack": "ataque especial base", "special defense": "defesa especial base",
                      "hp": "HP base"}
            values = []
            for row in rows:
                match = re.search(rf"-\s*{re.escape(stat)}:\s*(\d+(?:\.\d+)?)", str(row.get("content", "")), re.I)
                if match:
                    values.append((row["pokemon_name"], match.group(1)))
            if values:
                answer = "; ".join(f"{name} tem {labels[stat]} {value}" for name, value in values)
                if len(values) == 2 and all(value.replace(".", "", 1).isdigit() for _, value in values):
                    first, second = values
                    if float(first[1]) != float(second[1]):
                        greater = first[0] if float(first[1]) > float(second[1]) else second[0]
                        answer += f"; {greater} possui o maior valor"
                return answer + "."
        return None

    @classmethod
    def structured_answer_is_complete(cls, answer, rows, question):
        expected = cls.structured_answer(rows, question)
        if not expected:
            return True
        normalized_answer = normalize_plain(answer)
        acquisition = cls.acquisition_facts(rows) if cls.is_acquisition_question(question) else []
        if acquisition:
            for fact in acquisition:
                location_anchor = next(
                    (token for token in normalize_plain(fact["location"]).split() if len(token) >= 4),
                    "",
                )
                if normalize_plain(fact["pokemon"]) not in normalized_answer:
                    return False
                if location_anchor and location_anchor not in normalized_answer:
                    return False
                if fact["method"] == "gift" and not re.search(
                    r"\b(?:presente|gift|receb\w*|obt\w*)\b", normalized_answer
                ):
                    return False
            return True
        if cls.evolution_transitions(rows):
            return cls.evolution_answer_is_complete(answer, rows)
        stat = cls.requested_stat(question)
        if stat:
            for row in rows:
                match = re.search(rf"-\s*{re.escape(stat)}:\s*(\d+(?:\.\d+)?)", str(row.get("content", "")), re.I)
                if match and (normalize_plain(row["pokemon_name"]) not in normalized_answer or match.group(1) not in normalized_answer):
                    return False
            return True
        for row in rows:
            types = cls.profile_types_pt(row)
            if types and any(normalize_plain(value) not in normalized_answer for value in types):
                return False
        return True

    @classmethod
    def retrieval_answer(cls, rows, question):
        """Resposta extrativa auditável usada quando a geração não é suficiente."""
        if not rows:
            return None
        structured = cls.structured_answer(rows, question)
        if structured:
            return structured
        excerpts = [str(row.get("content", "")).strip() for row in rows[:3] if str(row.get("content", "")).strip()]
        return "\n\n".join(excerpts)[:9000] if excerpts else None

    def answer(self, question, generate=False):
        if not isinstance(question, str) or not question.strip(): raise ValueError("Pergunta vazia")
        canonical, corrections = self.canonicalize(question)
        rows, gate = self.select_evidence(canonical, self.settings.rag_top_k)
        generation_preflight_error = None
        context = ""
        if generate:
            try:
                context, used = self.build_context(rows, question=canonical, token_budget=self.generator.max_input_tokens)
            except Exception as exc:
                # A falha ao carregar/tokenizar o modelo não invalida a recuperação.
                # Mantemos as fontes e deixamos a resposta transparente como indisponível.
                generation_preflight_error = exc
                _, used = self.build_context(rows)
        else:
            _, used = self.build_context(rows)
        if not used:
            status = "out_of_scope" if gate in {"out_of_scope", "outside_kanto_dex", "needs_comparison_criterion"} else "insufficient_evidence"
            return {"status": status, "answer": REFUSAL, "answer_mode": "extractive",
                    "source_documents": [], "supporting_excerpts": [], "spelling_corrections": corrections}
        base = {"status": "retrieval_only", "answer": None, "answer_mode": "extractive", "source_documents": [{k: r.get(k) for k in ("document_id", "pokemon_id", "pokemon_name", "section", "score")} for r in used], "supporting_excerpts": [], "spelling_corrections": corrections, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        if not generate:
            return {**base, "answer": self.retrieval_answer(used, question)}
        if generation_preflight_error is not None:
            return {**base, "status": "provider_unavailable", "answer": self.retrieval_answer(used, question),
                    "generation_error": type(generation_preflight_error).__name__}
        targets = self.mentioned_ids(canonical)
        if len(targets) > 1 and not set(targets).issubset({int(row["pokemon_id"]) for row in used}):
            return {**base, "status": "insufficient_evidence", "answer": REFUSAL}
        try:
            payload = self.generator(question, context)
        except Exception as exc:
            return {**base, "status": "provider_unavailable", "answer": self.retrieval_answer(used, question),
                    "generation_error": type(exc).__name__}
        answer = str(payload.get("answer", "")).strip()
        generation_metadata = {key: value for key, value in payload.items() if key != "answer"}
        generation_metadata["documents_in_prompt"] = [row["document_id"] for row in used]
        structured = self.structured_answer(used, question)
        if structured and not self.structured_answer_is_complete(answer, used, question):
            fallback = self.retrieval_answer(used, question)
            if fallback:
                generation_metadata.update(
                    response_mode="extractive_fallback",
                    fallback_reason=("missing_evolution_conditions" if not self.is_acquisition_question(question)
                                     and self.evolution_transitions(used)
                                     else "missing_structured_facts"),
                )
                excerpts = [{"document_id": row["document_id"], "quote": row["content"]} for row in used]
                return {**base, "status": "answered", "answer": fallback,
                        "answer_mode": "extractive_fallback", "supporting_excerpts": excerpts,
                        "generation_metadata": generation_metadata}
        if not answer or answer.casefold() == "não sei":
            return {**base, "status": "insufficient_evidence", "answer": REFUSAL}
        excerpts = [{"document_id": row["document_id"], "quote": row["content"]} for row in used]
        return {**base, "status": "answered", "answer": answer,
                "answer_mode": "generative",
                "supporting_excerpts": excerpts,
                "generation_metadata": generation_metadata}

