from fastapi import APIRouter, HTTPException, Query, Request
router = APIRouter()

@router.get("/pokemon")
def list_pokemon(request: Request, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100), search: str | None = None):
    if request.app.state.retriever is None: raise HTTPException(503, "Artefatos RAG não carregados")
    records = list(request.app.state.retriever.records)
    if search:
        query = search.casefold(); records = [r for r in records if query in str(r.get("name", "")).casefold()]
    start=(page-1)*page_size
    return {"items": records[start:start+page_size], "page": page, "page_size": page_size, "total": len(records)}

@router.get("/pokemon/{pokemon_id}")
def get_pokemon(pokemon_id: int, request: Request):
    if request.app.state.retriever is None: raise HTTPException(503, "Artefatos RAG não carregados")
    record = request.app.state.retriever.by_id.get(pokemon_id)
    if not record: raise HTTPException(404, "Pokémon não encontrado")
    return record

