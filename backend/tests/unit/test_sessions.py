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

