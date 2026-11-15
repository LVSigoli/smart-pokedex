from types import SimpleNamespace
from app.rag.engine import RAGEngine, REFUSAL

class FakeRetriever:
    documents = [{"pokemon_id": 4}, {"pokemon_id": 7}]
    records = [{"id": 4, "api_name": "charmander", "name": "Charmander"}, {"id": 7, "api_name": "squirtle", "name": "Squirtle"}]
    def search(self, question, top_k=5):
        rows = [{"rank":1,"document_id":"004_evolution","pokemon_id":4,"pokemon_name":"Charmander","section":"evolution","score":.9,"content":"Charmander evolves into Charmeleon at level 16."}, {"rank":2,"document_id":"007_evolution","pokemon_id":7,"pokemon_name":"Squirtle","section":"evolution","score":.8,"content":"Squirtle evolves into Wartortle at level 16."}]
        return rows[:top_k]

def engine(): return RAGEngine(FakeRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))


def test_engine_uses_short_generation_limit_by_default():
    assert engine().generator.max_new_tokens == 64


def test_all_pokemon_of_a_type_returns_the_complete_deterministic_list():
    rag = engine()
    rows = [
        {"rank": 1, "document_id": "004_profile", "pokemon_id": 4, "pokemon_name": "Charmander",
         "section": "profile", "score": .9, "content": "Types PT-BR: fogo."},
        {"rank": 2, "document_id": "006_profile", "pokemon_id": 6, "pokemon_name": "Charizard",
         "section": "profile", "score": .8, "content": "Types PT-BR: fogo, voador."},
        {"rank": 3, "document_id": "037_profile", "pokemon_id": 37, "pokemon_name": "Vulpix",
         "section": "profile", "score": .7, "content": "Types PT-BR: fogo."},
    ]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class IncompleteGenerator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context): return 50

        def __call__(self, question, context):
            return {"answer": "Charizard é do tipo fogo e voador."}

    rag.generator = IncompleteGenerator()
    result = rag.answer("Quais são todos os Pokémon do tipo fogo?", generate=True)

    assert result["answer"] == (
        "Charmander é do tipo fogo; Charizard é do tipo fogo e voador; "
        "Vulpix é do tipo fogo."
    )
    assert result["generation_metadata"]["response_mode"] == "grounded_repair"


def test_type_matchup_question_is_rendered_as_natural_weakness_answer():
    rag = engine()
    rows = [{
        "rank": 1, "document_id": "006_type_matchups", "pokemon_id": 6,
        "pokemon_name": "Charizard", "section": "type_matchups", "score": .9,
        "content": (
            "Defensive effectiveness against Charizard (Generation III type chart):\n"
            "- 4x weak: rock.\n"
            "- 2x weak: electric, water.\n"
            "- 0.5x resistant: fighting, fire, steel.\n"
            "- 0.25x resistant: bug, grass.\n"
            "- immune (0x): ground."
        ),
    }]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class Generator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context): return 50

        def __call__(self, question, context):
            return {"answer": "[006_type_matchups] is a Pokémon: Charizard."}

    rag.generator = Generator()
    result = rag.answer("Quais são as fraquezas do Charizard?", generate=True)

    assert result["answer"] == "Charizard é fraco contra pedra (4×), elétrico e água (2×)."
    assert "[006_type_matchups]" not in result["answer"]

def test_multiple_pokemon_reserve_evidence():
    rows, reason = engine().select_evidence("Quais sao as evolucoes de Charmander e Squirtle")
    assert reason == "ready"
    assert {r["pokemon_id"] for r in rows} == {4, 7}


def test_multi_pokemon_retrieval_complements_missing_target_with_targeted_tfidf_query():
    class ComplementaryRetriever:
        documents = [{"pokemon_id": 1}, {"pokemon_id": 2}]
        records = [
            {"id": 1, "api_name": "bulbasaur", "name": "Bulbasaur"},
            {"id": 2, "api_name": "ivysaur", "name": "Ivysaur"},
        ]

        def search(self, question, top_k=5):
            if question.casefold().startswith("quais"):
                return [{
                    "rank": 1, "document_id": "001_profile", "pokemon_id": 1,
                    "pokemon_name": "Bulbasaur", "section": "profile", "score": .9,
                    "content": "Types in Generation III: grass, poison.",
                }]
            return [{
                "rank": 1, "document_id": "002_profile", "pokemon_id": 2,
                "pokemon_name": "Ivysaur", "section": "profile", "score": .8,
                "content": "Types in Generation III: grass, poison.",
            }]

    rag = RAGEngine(ComplementaryRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))

    rows, reason = rag.select_evidence("Quais são os tipos de Bulbasaur e Ivysaur?")

    assert reason == "ready"
    assert {row["pokemon_id"] for row in rows} == {1, 2}

def test_outside_dex_refuses():
    result = engine().answer("Qual é o Pokémon número 152?")
    assert result["status"] == "out_of_scope"
    assert result["answer"] == REFUSAL


def test_plural_gym_question_is_out_of_scope():
    result = engine().answer("Quais ginásios preciso vencer?", generate=True)
    assert result["status"] == "out_of_scope"
    assert result["answer"] == REFUSAL


def test_vague_strength_comparison_asks_for_an_attribute_instead_of_refusing():
    result = engine().answer("Charmander é mais forte que Squirtle?", generate=True)

    assert result["status"] == "clarification_needed"
    assert "Charmander" in result["answer"]
    assert "Squirtle" in result["answer"]
    assert "ataque" in result["answer"]
    assert result["source_documents"] == []

def test_token_budget_keeps_complete_passages_for_each_compared_pokemon():
    rag = engine()

    class WordBudgetGenerator:
        max_input_tokens = 70

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + " " + context).split()) + 18

    rag.generator = WordBudgetGenerator()
    noise = " ".join(["irrelevant"] * 80)
    rows = [
        {"document_id": "004_stats", "pokemon_id": 4, "pokemon_name": "Charmander",
         "section": "stats", "score": .9,
         "content": f"Charmander base speed is 65.\n{noise}"},
        {"document_id": "007_stats", "pokemon_id": 7, "pokemon_name": "Squirtle",
         "section": "stats", "score": .8,
         "content": f"Squirtle base speed is 43.\n{noise}"},
    ]

    context, used = rag.build_context(
        rows, question="Compare a velocidade de Charmander e Squirtle",
        token_budget=rag.generator.max_input_tokens,
    )

    assert {row["pokemon_id"] for row in used} == {4, 7}
    assert "Charmander base speed is 65." in context
    assert "Squirtle base speed is 43." in context
    assert rag.generator.count_prompt_tokens("Compare a velocidade de Charmander e Squirtle", context) <= 70
    assert all(row["content"] in rows[index]["content"] for index, row in enumerate(used))


def test_generation_returns_auditable_selected_passages_without_json_contract():
    rag = engine()

    class GroundedGenerator:
        max_input_tokens = 80

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + " " + context).split()) + 18

        def __call__(self, question, context):
            return {"answer": "Charmander evolui para Charmeleon no nível 16.",
                    "model": "google/flan-t5-small", "input_tokens": 42}

    rag.generator = GroundedGenerator()
    result = rag.answer("Como evoluir Charmander?", generate=True)

    assert result["status"] == "answered"
    assert result["supporting_excerpts"]
    assert result["supporting_excerpts"][0]["document_id"] == "004_evolution"
    assert "can_answer" not in result
    assert "evidence" not in result
    assert result["generation_metadata"]["documents_in_prompt"] == ["004_evolution"]


def test_evolution_passages_keep_transition_and_conditions_together_and_first():
    content = (
        "Evolution family of Charmander: charmander, charmeleon, charizard.\n"
        "Charmander is the first stage and does not evolve from another Pokémon.\n"
        "- charmander -> charmeleon.\n"
        "Conditions: (trigger: level up; minimum level: 16).\n"
        "Not specifically verified for FireRed.\n"
        "- charmeleon -> charizard.\n"
        "Conditions: (trigger: level up; minimum level: 36)."
    )

    passages = RAGEngine._rank_passages("Como evoluir Charmander?", content)

    assert passages[0].startswith("- charmander -> charmeleon.")
    assert "minimum level: 16" in passages[0]


def test_evolution_generation_falls_back_when_model_omits_required_conditions():
    rag = engine()
    evolution = {
        "rank": 1, "document_id": "004_evolution", "pokemon_id": 4,
        "pokemon_name": "Charmander", "section": "evolution", "score": .9,
        "content": (
            "- charmander -> charmeleon. Conditions: (trigger: level up; minimum level: 16).\n"
            "- charmeleon -> charizard. Conditions: (trigger: level up; minimum level: 36)."
        ),
    }
    rag.select_evidence = lambda question, top_k=5: ([evolution], "ready")

    class IncompleteGenerator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + " " + context).split()) + 18

        def __call__(self, question, context):
            return {"answer": "Charmander é o primeiro estágio de sua família.",
                    "model": "google/flan-t5-small", "input_tokens": 50}

    rag.generator = IncompleteGenerator()
    result = rag.answer("Como evoluir Charmander?", generate=True)

    assert result["status"] == "answered"
    assert "Charmander evolui para Charmeleon no nível 16" in result["answer"]
    assert "Charmeleon evolui para Charizard no nível 36" in result["answer"]
    assert result["answer_mode"] == "generative"
    assert result["generation_metadata"]["response_mode"] == "grounded_repair"
    assert result["generation_metadata"]["fallback_reason"] == "missing_evolution_conditions"


def test_branched_evolution_keeps_every_compact_transition_and_repairs_naturally():
    rag = engine()
    evolution = {
        "rank": 1, "document_id": "133_evolution", "pokemon_id": 133,
        "pokemon_name": "Eevee", "section": "evolution", "score": .9,
        "content": (
            "- eevee -> vaporeon. Conditions: (version_group: red blue; item: water stone; "
            "trigger: use item; near_special_rock: False). Not specifically verified for FireRed.\n"
            "- eevee -> jolteon. Conditions: (version_group: red blue; item: thunder stone; "
            "trigger: use item; near_special_rock: False). Not specifically verified for FireRed.\n"
            "- eevee -> flareon. Conditions: (version_group: red blue; item: fire stone; "
            "trigger: use item; near_special_rock: False). Not specifically verified for FireRed."
        ),
    }
    rag.select_evidence = lambda question, top_k=5: ([evolution], "ready")

    class IncompleteGenerator:
        max_input_tokens = 90
        received_context = ""

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + " " + context).split()) + 18

        def __call__(self, question, context):
            self.received_context = context
            return {"answer": "Eevee evolui para Vaporeon usando Water Stone."}

    generator = IncompleteGenerator()
    rag.generator = generator
    result = rag.answer("Como evoluir o Eevee?", generate=True)

    assert all(name in generator.received_context.casefold() for name in ("vaporeon", "jolteon", "flareon"))
    assert all(name in result["answer"] for name in ("Vaporeon", "Jolteon", "Flareon"))
    assert "version_group" not in result["answer"]
    assert "Not specifically verified" not in result["answer"]
    assert result["answer_mode"] == "generative"
    assert result["generation_metadata"]["response_mode"] == "grounded_repair"


def test_low_similarity_named_pokemon_evidence_is_rejected():
    rag = engine()
    rag.retriever.search = lambda question, top_k=5: [{
        "rank": 1, "document_id": "025_profile", "pokemon_id": 4,
        "pokemon_name": "Charmander", "section": "profile", "score": .12,
        "content": "Charmander is a fire Pokémon.",
    }]
    rag.settings.rag_min_score = .15

    rows, reason = rag.select_evidence("Qual é o número do sapato do Charmander?")

    assert rows == []
    assert reason == "below_similarity_threshold"


def test_type_generation_falls_back_when_model_omits_retrieved_types():
    rag = engine()
    profile = {
        "rank": 1, "document_id": "006_profile", "pokemon_id": 4,
        "pokemon_name": "Charmander", "section": "profile", "score": .9,
        "content": "Types in Generation III: fire. Types PT-BR: fogo.",
    }
    rag.select_evidence = lambda question, top_k=5: ([profile], "ready")

    class IncompleteGenerator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context): return 40

        def __call__(self, question, context):
            return {"answer": "Charmander vive em montanhas.", "model": "google/flan-t5-small"}

    rag.generator = IncompleteGenerator()
    result = rag.answer("Qual é o tipo do Charmander?", generate=True)

    assert result["answer"] == "Charmander é do tipo fogo."
    assert result["answer_mode"] == "generative"
    assert result["generation_metadata"]["response_mode"] == "grounded_repair"
    assert result["generation_metadata"]["fallback_reason"] == "missing_structured_facts"


def test_stat_comparison_falls_back_with_values_for_every_pokemon():
    rag = engine()
    rows = [
        {"rank": 1, "document_id": "006_stats", "pokemon_id": 4,
         "pokemon_name": "Charmander", "section": "stats", "score": .9,
         "content": "Base stats for Charmander:\n- speed: 65."},
        {"rank": 2, "document_id": "007_stats", "pokemon_id": 7,
         "pokemon_name": "Squirtle", "section": "stats", "score": .8,
         "content": "Base stats for Squirtle:\n- speed: 43."},
    ]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class IncompleteGenerator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context): return 50

        def __call__(self, question, context):
            return {"answer": "Charmander e Squirtle têm velocidades diferentes.",
                    "model": "google/flan-t5-small"}

    rag.generator = IncompleteGenerator()
    result = rag.answer("Compare a velocidade de Charmander e Squirtle.", generate=True)

    assert "Charmander tem velocidade base 65" in result["answer"]
    assert "Squirtle tem velocidade base 43" in result["answer"]
    assert result["answer_mode"] == "generative"
    assert result["generation_metadata"]["response_mode"] == "grounded_repair"


def test_single_stat_generation_receives_only_the_requested_stat():
    rag = engine()
    rows = [{
        "rank": 1, "document_id": "004_stats", "pokemon_id": 4,
        "pokemon_name": "Charmander", "section": "stats", "score": .9,
        "content": (
            "Base stats for Charmander:\n"
            "- attack: 52.\n"
            "- defense: 43.\n"
            "- hp: 39.\n"
            "- speed: 65."
        ),
    }]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class FocusedGenerator:
        max_input_tokens = 100
        received_context = ""

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + " " + context).split()) + 18

        def __call__(self, question, context):
            self.received_context = context
            return {
                "answer": (
                    "Charmander stats: attack 52, defense 43, hp 39, speed 65."
                ),
                "model": "google/flan-t5-small",
            }

    generator = FocusedGenerator()
    rag.generator = generator

    result = rag.answer("Qual seu ataque base? sobre Charmander", generate=True)

    assert "- attack: 52." in generator.received_context
    assert "- defense: 43." not in generator.received_context
    assert "- hp: 39." not in generator.received_context
    assert "- speed: 65." not in generator.received_context
    assert result["answer"] == "Charmander tem ataque base 52."
    assert result["answer_mode"] == "generative"
    assert result["generation_metadata"]["response_mode"] == "focused_structured"
    assert result["generation_metadata"]["raw_answer"].startswith("Charmander stats:")

def test_location_question_accepts_obtainment_evidence():
    class EeveeRetriever:
        documents = [{"pokemon_id": 133}]
        records = [{"id": 133, "api_name": "eevee", "name": "Eevee"}]
        def search(self, question, top_k=5):
            return [{"rank": 1, "document_id": "133_obtainment", "pokemon_id": 133, "pokemon_name": "Eevee", "section": "obtainment", "score": .9, "content": "Eevee is available as a documented gift."}]

    rag = RAGEngine(EeveeRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))
    rows, reason = rag.select_evidence("Onde encontro o Eevee?")
    assert reason == "ready"
    assert [row["document_id"] for row in rows] == ["133_obtainment"]


def test_location_query_complements_obtainment_with_targeted_encounter_evidence():
    class LocationRetriever:
        records = [{"id": 25, "api_name": "pikachu", "name": "Pikachu"}]
        documents = [
            {"pokemon_id": 25, "document_id": "025_obtainment", "section": "obtainment"},
            {"pokemon_id": 25, "document_id": "025_encounters_01", "section": "encounters"},
        ]

        def search(self, question, top_k=5):
            obtainment = {
                "rank": 1, "document_id": "025_obtainment", "pokemon_id": 25,
                "pokemon_name": "Pikachu", "section": "obtainment", "score": .3,
                "content": "Documented mechanisms: wild_encounter.",
            }
            encounter = {
                "rank": 2, "document_id": "025_encounters_01", "pokemon_id": 25,
                "pokemon_name": "Pikachu", "section": "encounters", "score": .2,
                "content": "- viridian forest area; method: walk; levels: 3-5;",
            }
            return [obtainment, encounter][:top_k] if question == "pikachu encounters" else [obtainment][:top_k]

    rag = RAGEngine(LocationRetriever(), SimpleNamespace(
        flan_model="google/flan-t5-small", rag_top_k=5, rag_min_score=.15,
    ))

    rows, reason = rag.select_evidence("Onde encontro Pikachu?")

    assert reason == "ready"
    assert [row["document_id"] for row in rows] == ["025_obtainment", "025_encounters_01"]


def test_mixed_location_answer_formats_encounter_and_evolution_naturally():
    rows = [
        {
            "document_id": "025_encounters_01", "pokemon_id": 25,
            "pokemon_name": "Pikachu", "section": "encounters", "score": .8,
            "content": "- viridian forest area; method: walk; levels: 3-5;",
        },
        {
            "document_id": "149_obtainment", "pokemon_id": 149,
            "pokemon_name": "Dragonite", "section": "obtainment", "score": .7,
            "content": "Documented mechanisms: evolution.",
        },
    ]

    answer = RAGEngine.retrieval_answer(rows, "Onde encontro Pikachu e Dragonite?")

    assert "Pikachu pode ser encontrado em Viridian Forest Area" in answer
    assert "Dragonite" in answer and "obtido por evolução" in answer
    assert "Acquisition evidence" not in answer
    assert "Documented mechanisms" not in answer


def test_generated_location_answer_is_grounded_and_naturally_postprocessed():
    rag = engine()
    rows = [
        {
            "document_id": "025_encounters_01", "pokemon_id": 25,
            "pokemon_name": "Pikachu", "section": "encounters", "score": .8,
            "content": "- viridian forest area; method: walk; levels: 3-5;",
        },
        {
            "document_id": "149_obtainment", "pokemon_id": 149,
            "pokemon_name": "Dragonite", "section": "obtainment", "score": .7,
            "content": "Documented mechanisms: evolution.",
        },
    ]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class WeakLocationGenerator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context):
            return 50

        def __call__(self, question, context):
            return {
                "answer": "Pikachu e Dragonite.",
                "model": "google/flan-t5-small",
            }

    rag.generator = WeakLocationGenerator()

    result = rag.answer("Onde encontro Pikachu e Dragonite?", generate=True)

    assert "Pikachu pode ser encontrado em Viridian Forest Area" in result["answer"]
    assert "Dragonite" in result["answer"] and "obtido por evolução" in result["answer"]
    assert result["answer_mode"] == "generative"
    assert result["generation_metadata"]["response_mode"] == "grounded_structured"
    assert result["generation_metadata"]["raw_answer"] == "Pikachu e Dragonite."


def test_profile_overview_is_generated_then_presented_as_a_natural_summary():
    rag = engine()
    rows = [{
        "document_id": "149_profile", "pokemon_id": 149,
        "pokemon_name": "Dragonite", "section": "profile", "score": .9,
        "content": (
            "Pokémon Dragonite (National Pokédex #149).\n"
            "Types in Generation III: dragon, flying.\n"
            "Types PT-BR: dragão, voador.\n"
            "Habitat: waters-edge.\n"
            "Height: 2.2 m; weight: 210.0 kg."
        ),
    }]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class EnglishProfileGenerator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context):
            return 50

        def __call__(self, question, context):
            return {
                "answer": "Dragonite is a dragon and flying Pokémon.",
                "model": "google/flan-t5-small",
            }

    rag.generator = EnglishProfileGenerator()

    result = rag.answer("Fale um pouco sobre Dragonite", generate=True)

    assert result["answer"] == (
        "Dragonite é o Pokémon #149, dos tipos dragão e voador. "
        "Mede 2,2 m, pesa 210 kg e vive próximo à água."
    )
    assert result["answer_mode"] == "generative"
    assert result["generation_metadata"]["response_mode"] == "grounded_structured"
    assert result["generation_metadata"]["raw_answer"].startswith("Dragonite is")


def test_profile_summary_with_sabre_typo_repairs_incomplete_generation():
    rag = engine()
    rows = [{
        "rank": 1, "document_id": "006_profile", "pokemon_id": 6,
        "pokemon_name": "Charizard", "section": "profile", "score": .9,
        "content": (
            "Pokémon Charizard (National Pokédex #006).\n"
            "Types in Generation III: fire, flying.\n"
            "Types PT-BR: fogo, voador.\n"
            "Pokédex description (firered): It spits fire that is hot enough to melt boulders.\n"
            "Habitat: mountain.\n"
            "Height: 1.7 m; weight: 90.5 kg."
        ),
    }]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class ShortProfileGenerator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context): return 50

        def __call__(self, question, context):
            return {"answer": "Charizard (National Pokédex #006)."}

    rag.generator = ShortProfileGenerator()
    result = rag.answer("O que sabre sobre o Charizard?", generate=True)

    assert "tipos fogo e voador" in result["answer"]
    assert "Mede 1,7 m" in result["answer"]
    assert "It spits fire" not in result["answer"]
    assert "Descrição da Pokédex" not in result["answer"]
    assert "vive em regiões montanhosas" in result["answer"]
    assert result["generation_metadata"]["response_mode"] == "grounded_structured"

    overview = rag.answer("O que sabe sobre o Charizard?", generate=True)
    assert "tipos fogo e voador" in overview["answer"]
    assert "Mede 1,7 m" in overview["answer"]
    assert "It spits fire" not in overview["answer"]
    assert overview["generation_metadata"]["response_mode"] == "grounded_structured"


def test_profile_overview_uses_portuguese_description_without_technical_labels():
    rag = engine()
    rows = [{
        "rank": 1, "document_id": "006_profile", "pokemon_id": 6,
        "pokemon_name": "Charizard", "section": "profile", "score": .9,
        "content": (
            "Pokémon Charizard (National Pokédex #006).\n"
            "Types PT-BR: fogo, voador.\n"
            "Descrição: Suas asas conseguem levá-lo a grandes altitudes.\n"
            "Ele expele fogo em temperaturas muito altas.\n"
            "Habitat: mountain.\n"
            "Height: 1.7 m; weight: 90.5 kg."
        ),
    }]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class Generator:
        max_input_tokens = 100

        @staticmethod
        def count_prompt_tokens(question, context): return 50

        def __call__(self, question, context):
            return {"answer": "Charizard (National Pokédex #006)."}

    rag.generator = Generator()
    result = rag.answer("O que sabe sobre o Charizard?", generate=True)

    assert result["answer"] == (
        "Charizard é o Pokémon #006, dos tipos fogo e voador. "
        "Suas asas conseguem levá-lo a grandes altitudes. Ele expele fogo em temperaturas muito altas. "
        "Mede 1,7 m, pesa 90,5 kg e vive em regiões montanhosas."
    )
    assert "Geração III" not in result["answer"]
    assert "Descrição:" not in result["answer"]
    assert "Pokédex FireRed" not in result["answer"]


def test_location_fallback_turns_eevee_gift_evidence_into_a_natural_answer():
    rows = [
        {
            "document_id": "133_obtainment", "pokemon_id": 133,
            "pokemon_name": "Eevee", "section": "obtainment", "score": .9,
            "content": (
                "Acquisition evidence for Eevee in FireRed:\n"
                "Documented mechanisms: wild_encounter."
            ),
        },
        {
            "document_id": "133_encounters_01", "pokemon_id": 133,
            "pokemon_name": "Eevee", "section": "encounters", "score": .8,
            "content": (
                "Documented wild encounters for Eevee in Pokémon FireRed:\n"
                "- celadon city celadon mansion; method: gift; levels: 25-25; "
                "API encounter chance weight: 100; conditions: not specified."
            ),
        },
    ]

    answer = RAGEngine.retrieval_answer(rows, "Onde eu encontro o Eevee?")

    assert answer == "No Pokémon FireRed, Eevee é obtido como presente em Celadon City Celadon Mansion, no nível 25."
    assert "Acquisition evidence" not in answer
    assert "Documented wild encounters" not in answer


def test_location_fallback_does_not_answer_with_evolution_data():
    rows = [
        {
            "document_id": "041_evolution", "pokemon_id": 41,
            "pokemon_name": "Zubat", "section": "evolution", "score": .9,
            "content": "- zubat -> golbat. Conditions: (minimum level: 22).",
        },
        {
            "document_id": "041_encounters_01", "pokemon_id": 41,
            "pokemon_name": "Zubat", "section": "encounters", "score": .8,
            "content": (
                "- mt moon 1f; method: walk; levels: 6-8; "
                "API encounter chance weight: 69; conditions: not specified."
            ),
        },
    ]

    answer = RAGEngine.retrieval_answer(rows, "Onde encontrar o Zubat?")

    assert "Mt Moon 1F" in answer
    assert "Golbat" not in answer


def test_location_generation_failure_returns_the_natural_acquisition_fallback():
    rag = engine()
    rows = [{
        "document_id": "133_encounters_01", "pokemon_id": 133,
        "pokemon_name": "Eevee", "section": "encounters", "score": .9,
        "content": (
            "- celadon city celadon mansion; method: gift; levels: 25-25; "
            "API encounter chance weight: 100; conditions: not specified."
        ),
    }]
    rag.select_evidence = lambda question, top_k=5: (rows, "ready")

    class BrokenGenerator:
        max_input_tokens = 384

        @staticmethod
        def count_prompt_tokens(question, context):
            raise RuntimeError("model unavailable")

    rag.generator = BrokenGenerator()

    result = rag.answer("Onde eu encontro o Eevee?", generate=True)

    assert result["status"] == "provider_unavailable"
    assert result["answer"] == "No Pokémon FireRed, Eevee é obtido como presente em Celadon City Celadon Mansion, no nível 25."


def test_location_queries_do_not_use_evolution_documents_as_encounter_evidence():
    class EvolutionOnlyRetriever:
        documents = [{"pokemon_id": 2}]
        records = [{"id": 2, "api_name": "ivysaur", "name": "Ivysaur"}]

        def search(self, question, top_k=5):
            return [{
                "rank": 1, "document_id": "002_evolution", "pokemon_id": 2,
                "pokemon_name": "Ivysaur", "section": "evolution", "score": .9,
                "content": "- bulbasaur -> ivysaur. Conditions: (minimum level: 16).",
            }]

    rag = RAGEngine(EvolutionOnlyRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))

    rows, reason = rag.select_evidence("Onde encontro o Ivysaur?")

    assert rows == []
    assert reason == "no_relevant_documents"


def test_canonicalization_preserves_compound_species_names():
    class CompoundNameRetriever:
        documents = []
        records = [
            {"id": 29, "api_name": "nidoran-f", "name": "Nidoran F"},
            {"id": 32, "api_name": "nidoran-m", "name": "Nidoran M"},
        ]

        def search(self, question, top_k=5):
            return []

    rag = RAGEngine(CompoundNameRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))

    canonical, _ = rag.canonicalize("Fale sobre o Pokémon Nidoran F")

    assert "nidoran f" in canonical
    assert rag.mentioned_ids(canonical) == [29]

def test_find_and_catch_phrasings_select_location_or_acquisition_sections():
    rag = engine()
    assert rag.requested_section("Como encontrar o Eevee?") == "encounters"
    assert rag.requested_section("E como eu encontro ele? sobre Eevee") == "encounters"
    assert rag.requested_section("Como eu pego o Eevee?") in {"obtainment", "encounters"}

def test_question_about_stats_is_restricted_to_named_pokemon():
    class SpeciesRetriever:
        documents = [{"pokemon_id": 133}, {"pokemon_id": 25}]
        records = [{"id": 133, "api_name": "eevee", "name": "Eevee"},
                   {"id": 25, "api_name": "pikachu", "name": "Pikachu"}]
        def search(self, question, top_k=5):
            return [
                {"rank": 1, "document_id": "025_stats", "pokemon_id": 25, "pokemon_name": "Pikachu", "section": "stats", "score": .9, "content": "Pikachu base attack: 55."},
                {"rank": 2, "document_id": "133_stats", "pokemon_id": 133, "pokemon_name": "Eevee", "section": "stats", "score": .8, "content": "Eevee base attack: 55."},
            ][:top_k]
    rag = RAGEngine(SpeciesRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))
    rows, _ = rag.select_evidence("Qual seu ataque? sobre Eevee")
    assert rows and {row["pokemon_id"] for row in rows} == {133}

def test_same_type_query_recovers_matching_profile_documents():
    class TypeRetriever:
        documents = [
            {"pokemon_id": 6, "section": "profile", "document_id": "006_profile", "content": "Types in Generation III: fire, flying."},
            {"pokemon_id": 146, "section": "profile", "document_id": "146_profile", "content": "Types in Generation III: fire, flying."},
            {"pokemon_id": 25, "section": "profile", "document_id": "025_profile", "content": "Types in Generation III: electric."},
        ]
        records = [
            {"id": 6, "api_name": "charizard", "name": "Charizard"},
            {"id": 146, "api_name": "moltres", "name": "Moltres"},
            {"id": 25, "api_name": "pikachu", "name": "Pikachu"},
        ]
        def search(self, question, top_k=5):
            return [
                {"rank": 1, "document_id": "006_profile", "pokemon_id": 6, "pokemon_name": "Charizard", "section": "profile", "score": .9, "content": "Types in Generation III: fire, flying."},
                {"rank": 2, "document_id": "146_profile", "pokemon_id": 146, "pokemon_name": "Moltres", "section": "profile", "score": .8, "content": "Types in Generation III: fire, flying."},
                {"rank": 3, "document_id": "025_profile", "pokemon_id": 25, "pokemon_name": "Pikachu", "section": "profile", "score": .7, "content": "Types in Generation III: electric."},
            ][:top_k]
    rag = RAGEngine(TypeRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))
    rows, reason = rag.select_evidence("Quais os pokemons com o mesmo tipo do charizard?")
    assert reason == "ready"
    assert {row["pokemon_id"] for row in rows} == {6, 146}


def test_type_listing_accepts_fire_as_the_type_name_without_word_tipo():
    class TypeRetriever:
        documents = [
            {"pokemon_id": 6, "section": "profile", "document_id": "006_profile", "pokemon_name": "Charizard", "content": "Types in Generation III: fire, flying."},
            {"pokemon_id": 4, "section": "profile", "document_id": "004_profile", "pokemon_name": "Charmander", "content": "Types in Generation III: fire."},
        ]
        records = []
        def search(self, question, top_k=5):
            return []
    rag = RAGEngine(TypeRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))
    rows, reason = rag.select_evidence("Quais são os pokemons de fogo?")
    assert reason == "ready"
    assert {row["pokemon_name"] for row in rows} == {"Charizard", "Charmander"}
    assert "Charmander" in rag.retrieval_answer(rows, "Quais são os pokemons de fogo?")


def test_same_type_answer_lists_matching_pokemon_naturally():
    rows = [
        {"pokemon_name": "Charizard", "section": "profile", "content": "Types in Generation III: fire, flying."},
        {"pokemon_name": "Moltres", "section": "profile", "content": "Types in Generation III: fire, flying."},
    ]
    answer = RAGEngine.retrieval_answer(rows, "Quais pokemons têm o mesmo tipo do Charizard?")
    assert answer == "Moltres têm os mesmos tipos (fogo e voador) que Charizard."


def test_attributes_of_pokemon_and_evolutions_use_stats_even_with_typo():
    class StatsRetriever:
        documents = [
            {"pokemon_id": 133, "section": "stats", "document_id": "133_stats", "pokemon_name": "Eevee", "content": "- attack: 55."},
            {"pokemon_id": 134, "section": "stats", "document_id": "134_stats", "pokemon_name": "Vaporeon", "content": "- attack: 65."},
            {"pokemon_id": 133, "section": "evolution", "document_id": "133_evolution", "pokemon_name": "Eevee", "content": "- eevee -> vaporeon. Conditions: (item: water stone)."},
        ]
        records = [{"id": 133, "api_name": "eevee", "name": "Eevee"}, {"id": 134, "api_name": "vaporeon", "name": "Vaporeon"}]
        def search(self, question, top_k=5):
            return [dict(row, score=0.9, rank=index + 1) for index, row in enumerate(self.documents) if row["section"] == "stats"]
    rag = RAGEngine(StatsRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))
    rows, reason = rag.select_evidence("Quais são os atronutos do Eevee e das evoluções?")
    assert reason == "ready"
    assert {row["pokemon_id"] for row in rows} == {133, 134}
    answer = rag.retrieval_answer(rows, "Quais são os atronutos do Eevee e das evoluções?")
    assert "Eevee: ataque 55" in answer and "Vaporeon: ataque 65" in answer


def test_type_listing_has_priority_over_active_pokemon_profile():
    rows = [
        {"pokemon_name": "Charmander", "section": "profile", "content": "Types in Generation III: fire."},
        {"pokemon_name": "Charizard", "section": "profile", "content": "Types in Generation III: fire, flying."},
    ]
    answer = RAGEngine.retrieval_answer(rows, "Quais são os pokemon d e tipo fogo? sobre Charizard")
    assert "Charmander" in answer and "Charizard" in answer
    assert "National Pokédex" not in answer


def test_attribute_ranking_returns_all_species_sorted_by_requested_stat():
    class RankingRetriever:
        documents = [
            {"pokemon_id": 4, "section": "stats", "document_id": "004_stats", "pokemon_name": "Charmander", "content": "- attack: 52."},
            {"pokemon_id": 6, "section": "stats", "document_id": "006_stats", "pokemon_name": "Charizard", "content": "- attack: 84."},
            {"pokemon_id": 25, "section": "stats", "document_id": "025_stats", "pokemon_name": "Pikachu", "content": "- attack: 55."},
        ]
        records = []
        def search(self, question, top_k=5):
            return [dict(row, score=.9, rank=i + 1) for i, row in enumerate(self.documents)]
    rag = RAGEngine(RankingRetriever(), SimpleNamespace(flan_model="google/flan-t5-small", rag_top_k=5))
    rows, reason = rag.select_evidence("Quais são os maiores ataques?")
    assert reason == "ready"
    assert [row["pokemon_name"] for row in rows] == ["Charizard", "Pikachu", "Charmander"]

def test_generation_failure_keeps_retrieved_sources():
    rag = engine()
    class BrokenGenerator:
        max_input_tokens = 384
        def count_prompt_tokens(self, question, context):
            raise RuntimeError("model unavailable")
        def __call__(self, question, context):
            raise RuntimeError("model unavailable")
    rag.generator = BrokenGenerator()
    result = rag.answer("Como evoluir Charmander?", generate=True)
    assert result["status"] == "provider_unavailable"
    assert result["source_documents"]
    assert result["source_documents"][0]["pokemon_id"] == 4

def test_evolution_fallback_keeps_targets_without_item():
    rag = engine()
    rag.generator = BrokenGenerator() if 'BrokenGenerator' in globals() else rag.generator
    rows, reason = rag.select_evidence("Como evoluir Charmander?")
    answer = rag.retrieval_answer(rows, "Como evoluir Charmander?")
    assert reason == "ready"
    assert "Charmeleon" in answer
    assert "para ." not in answer

def test_evolution_fallback_explains_method_and_level():
    row = {"pokemon_name": "Charmander", "section": "evolution", "content": (
        "- charmander -> charmeleon. Conditions: (trigger: level up; minimum level: 16).\n"
        "- charmeleon -> charizard. Conditions: (trigger: level up; minimum level: 36)."
    )}
    answer = RAGEngine.retrieval_answer([row], "Como evoluir Charmander?")
    assert "16" in answer and "36" in answer

