from fastapi import APIRouter, Request
router = APIRouter()

@router.get("/health")
def health(): return {"status": "ok"}

@router.get("/ready")
def ready(request: Request): return {"ready": request.app.state.engine is not None}

