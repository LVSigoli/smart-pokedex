from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.core.exceptions import ArtifactError
from app.rag.retrieval import Retriever
from app.rag.engine import RAGEngine
from app.sessions.manager import SessionManager
from app.api.routes import health, chat, pokemon

def create_app(settings=None):
    settings = settings or get_settings()
    app = FastAPI(title="Pokédex Inteligente API", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=settings.origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    try:
        retriever = Retriever(settings.rag_data_dir); engine = RAGEngine(retriever, settings)
    except ArtifactError:
        retriever = None; engine = None
    app.state.retriever = retriever; app.state.engine = engine
    app.state.sessions = SessionManager(engine, settings.session_ttl_seconds) if engine else None
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1"); app.include_router(pokemon.router, prefix="/api/v1")
    return app

app = create_app()

