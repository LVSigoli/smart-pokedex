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

