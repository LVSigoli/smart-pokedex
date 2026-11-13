from types import SimpleNamespace
from app.rag.engine import RAGEngine, REFUSAL

class FakeRetriever:
    documents = [{"pokemon_id": 4}, {"pokemon_id": 7}]
    records = [{"id": 4, "api_name": "charmander", "name": "Charmander"}, {"id": 7, "api_name": "squirtle", "name": "Squirtle"}]
    def search(self, question, top_k=5):
        rows = [{"rank":1,"document_id":"004_evolution","pokemon_id":4,"pokemon_name":"Charmander","section":"evolution","score":.9,"content":"Charmander evolves into Charmeleon at level 16."}, {"rank":2,"document_id":"007_evolution","pokemon_id":7,"pokemon_name":"Squirtle","section":"evolution","score":.8,"content":"Squirtle evolves into Wartortle at level 16."}]
        return rows[:top_k]

def engine(): return RAGEngine(FakeRetriever(), SimpleNamespace(gemini_api_key=None, gemini_model="test", rag_top_k=5))

def test_multiple_pokemon_reserve_evidence():
    rows, reason = engine().select_evidence("Quais sao as evolucoes de Charmander e Squirtle")
    assert reason == "ready"
    assert {r["pokemon_id"] for r in rows} == {4, 7}

def test_outside_dex_refuses():
    result = engine().answer("Qual é o Pokémon número 152?")
    assert result["status"] == "out_of_scope"
    assert result["answer"] == REFUSAL

def test_grounding_requires_literal_quote():
    used = [{"document_id":"004_evolution", "content":"Charmander evolves into Charmeleon at level 16."}]
    payload = {"can_answer":True,"answer":"Charmander evolui [004_evolution].","evidence":[{"document_id":"004_evolution","quote":"Charmander evolves into Charmeleon at level 16."}]}
    assert RAGEngine.verify(payload, used)

def test_location_question_accepts_obtainment_evidence():
    class EeveeRetriever:
        documents = [{"pokemon_id": 133}]
        records = [{"id": 133, "api_name": "eevee", "name": "Eevee"}]
        def search(self, question, top_k=5):
            return [{"rank": 1, "document_id": "133_obtainment", "pokemon_id": 133, "pokemon_name": "Eevee", "section": "obtainment", "score": .9, "content": "Eevee is available as a documented gift."}]

    rag = RAGEngine(EeveeRetriever(), SimpleNamespace(gemini_api_key=None, gemini_model="test", rag_top_k=5))
    rows, reason = rag.select_evidence("Onde encontro o Eevee?")
    assert reason == "ready"
    assert [row["document_id"] for row in rows] == ["133_obtainment"]

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
    rag = RAGEngine(SpeciesRetriever(), SimpleNamespace(gemini_api_key=None, gemini_model="test", rag_top_k=5))
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
    rag = RAGEngine(TypeRetriever(), SimpleNamespace(gemini_api_key=None, gemini_model="test", rag_top_k=5))
    rows, reason = rag.select_evidence("Quais os pokemons com o mesmo tipo do charizard?")
    assert reason == "ready"
    assert {row["pokemon_id"] for row in rows} == {6, 146}

