# Pokédex Inteligente

Migração local do trabalho acadêmico de PLN para um monorepo com FastAPI e cliente React/Vite. O corpus é restrito aos 151 Pokémon de Kanto, com referência FireRed conforme os notebooks.

## Execução

1. Copie `backend/.env.example` para `backend/.env` e ajuste `RAG_DATA_DIR`.
2. Reproduza os dados na ordem `CORPUS_POKEMON.ipynb` → `01_RAG_DATA_PREPARATION.ipynb` → `02_RETRIEVAL_AND_EVALUATION_ADAPTADO.ipynb`, colocando os resultados em `data/processed` e `data/artifacts/retrieval`.
3. Valide com `cd backend; python -m app.cli verify`.
4. Instale o backend com `pip install -e ".[test]"` e execute `uvicorn app.main:app --reload`.
5. Execute o frontend com `cd frontend; npm install; npm run dev`.

Sem os JSON brutos/artefatos, a aplicação inicia apenas o endpoint `/health` e `/ready=false`; nenhum corpus sintético é criado.

## API

`POST /api/v1/sessions`, `POST /api/v1/chat`, `DELETE /api/v1/sessions/{id}`, `GET /api/v1/pokemon`, `GET /api/v1/pokemon/{id}`. A chave Gemini fica exclusivamente no backend em `GEMINI_API_KEY`.

