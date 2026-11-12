from types import SimpleNamespace
from app.sessions.manager import SessionManager

class Engine:
    def answer(self, message, generate=False): return {"status":"retrieval_only","answer":message}

def test_sessions_are_isolated():
    manager=SessionManager(Engine())
    first, second = manager.create(), manager.create()
    assert first.session_id != second.session_id
    assert manager.ask(first.session_id, "a")["answer"] == "a"
    assert manager.get(second.session_id).turns == []

class ContextualEngine:
    retriever = SimpleNamespace(by_id={133: {"name": "Eevee"}, 25: {"name": "Pikachu"}})
    def mentioned_ids(self, message):
        lowered = message.casefold()
        return [pokemon_id for pokemon_id, name in ((133, "eevee"), (25, "pikachu")) if name in lowered]
    def answer(self, message, generate=False):
        ids = self.mentioned_ids(message)
        return {"status": "retrieval_only", "answer": message, "source_documents": [{"pokemon_id": pokemon_id} for pokemon_id in ids], "supporting_excerpts": []}

def test_location_followup_keeps_active_pokemon():
    manager = SessionManager(ContextualEngine())
    session = manager.create()
    manager.ask(session.session_id, "Fale sobre Eevee e Pikachu")
    result = manager.ask(session.session_id, "E onde eu encontro eles?")
    assert result["answer"].startswith("E onde eu encontro eles? sobre ")
    assert "Eevee" in result["answer"] and "Pikachu" in result["answer"]

def test_location_followup_keeps_single_active_pokemon():
    manager = SessionManager(ContextualEngine())
    session = manager.create()
    manager.ask(session.session_id, "Fale sobre Eevee")
    result = manager.ask(session.session_id, "Onde eu encontro ele?")
    assert result["answer"] == "Onde eu encontro ele? sobre Eevee"

