# Pokédex Inteligente

Migração local do trabalho acadêmico de PLN para um monorepo com FastAPI e cliente React/Vite. O corpus é restrito aos 151 Pokémon de Kanto, com referência FireRed conforme os notebooks.

## Execução

1. Copie `backend/.env.example` para `backend/.env` e ajuste `RAG_DATA_DIR`.
2. Crie um ambiente virtual e instale as dependências:

   ```powershell
   cd "C:\Users\Lucas Sigoli\Documents\Projects\smart-pokedex"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e "backend[test,notebooks]"
   ```

   Para instalar apenas as dependências do backend e dos testes, use `python -m pip install -e "backend[test]"`.
3. Execute `python scripts/run_notebooks.py`. O executor adapta os caminhos do Colab em memória e salva cópias executadas em `data/executed_notebooks`.
3. Valide com `cd backend; python -m app.cli verify`.
4. Instale o backend com `pip install -e ".[test]"` e execute `uvicorn app.main:app --reload`.
5. Execute o frontend com `cd frontend; npm install; npm run dev`.

Sem os JSON brutos/artefatos, a aplicação inicia apenas o endpoint `/health` e `/ready=false`; nenhum corpus sintético é criado.

Para retomar a partir de uma etapa, use `python scripts/run_notebooks.py --from-step 01`, `02` ou `03`. A etapa 00 consulta a PokéAPI e pode demorar; arquivos brutos existentes são reutilizados pelo notebook.

Para executar todas as operações com um único comando na raiz do projeto:

```powershell
.\start.ps1
```

O script verifica os artefatos, executa o pipeline caso estejam ausentes ou inválidos, instala as dependências do frontend quando necessário e inicia a API em `http://127.0.0.1:8000` e o frontend em `http://127.0.0.1:5173`. Encerre com `Ctrl+C`.

O CORS já permite os dois hosts locais (`localhost` e `127.0.0.1`). Se criar `backend/.env`, mantenha ambos em `ALLOWED_ORIGINS`.

Se o comando `python` não for reconhecido, feche e reabra o PowerShell para carregar o PATH atualizado. Como alternativa, use o executável instalado diretamente:

```powershell
& "C:\Users\Lucas Sigoli\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
```

## API

`POST /api/v1/sessions`, `POST /api/v1/chat`, `DELETE /api/v1/sessions/{id}`, `GET /api/v1/pokemon`, `GET /api/v1/pokemon/{id}`. A geração de texto usa localmente `google/flan-t5-small`; não é necessária chave de API. Na primeira geração, os pesos podem ser baixados do Hugging Face.

