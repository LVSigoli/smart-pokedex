from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field

class SessionResponse(BaseModel):
    session_id: UUID

class ChatRequest(BaseModel):
    session_id: UUID
    message: str = Field(min_length=1, max_length=1000)
    generate: bool = False

class Source(BaseModel):
    document_id: str
    pokemon_id: int
    pokemon_name: str
    section: str
    score: float | None = None

class ChatResponse(BaseModel):
    session_id: UUID
    status: str
    answer: str | None = None
    source_documents: list[Source] = []
    supporting_excerpts: list[dict] = []

