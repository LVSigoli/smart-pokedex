from uuid import uuid4

from app.api.schemas.public import ChatRequest


def test_chat_requests_generate_by_default():
    payload = ChatRequest(session_id=uuid4(), message="Fale sobre Dragonite")

    assert payload.generate is True
