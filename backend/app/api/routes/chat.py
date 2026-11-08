from fastapi import APIRouter, HTTPException, Request
from app.api.schemas.public import ChatRequest, ChatResponse, SessionResponse
from app.core.exceptions import SessionNotFound
router = APIRouter()

@router.post("/sessions", response_model=SessionResponse)
def create_session(request: Request):
    if request.app.state.sessions is None: raise HTTPException(503, "Artefatos RAG não carregados")
    return {"session_id": request.app.state.sessions.create().session_id}

@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, request: Request):
    if request.app.state.sessions is None: raise HTTPException(503, "Artefatos RAG não carregados")
    try: request.app.state.sessions.delete(session_id)
    except SessionNotFound: raise HTTPException(404, "Sessão não encontrada")

@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    if request.app.state.sessions is None: raise HTTPException(503, "Artefatos RAG não carregados")
    try: result = request.app.state.sessions.ask(payload.session_id, payload.message, payload.generate)
    except SessionNotFound: raise HTTPException(404, "Sessão não encontrada")
    return {"session_id": payload.session_id, **result}

