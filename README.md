# Pokédex Inteligente

Migração local do trabalho acadêmico de PLN para um monorepo com FastAPI e cliente React/Vite. O corpus é restrito aos 151 Pokémon de Kanto, com referência FireRed conforme os notebooks.

## Variáveis de ambiente

As variáveis do backend ficam no arquivo `backend/.env`. O projeto não lê um `.env` na raiz. Para criar o arquivo local, execute na raiz do projeto:

```powershell
Copy-Item backend/.env.example backend/.env
notepad backend/.env
```

O conteúdo de `backend/.env.example` é:

```dotenv
RAG_DATA_DIR=../application_data
GEMINI_API_KEY=
GEMINI_MODEL_NAME=gemini-3.5-flash-lite
GEMINI_MAX_INPUT_TOKENS=8192
GEMINI_MAX_OUTPUT_TOKENS=768
GEMINI_TEMPERATURE=0.1
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
SESSION_TTL_SECONDS=1800
MAX_MESSAGE_CHARS=1000
RAG_TOP_K=5
RAG_MIN_SCORE=0.15
```

Preencha `GEMINI_API_KEY` com a chave do Gemini para habilitar a geração de respostas. As demais variáveis já possuem valores padrão adequados para o ambiente local; mantenha `RAG_DATA_DIR=../application_data` quando a API for executada a partir de `backend`. Nunca versione `backend/.env` nem envie a chave para o frontend.

## Dados gerados pelos notebooks

Os notebooks gravam os dados na pasta `application_data/`, na raiz do projeto:

| Etapa | Diretório | Conteúdo |
| --- | --- | --- |
| Notebook 00 | `application_data/corpus/raw_data` | JSON brutos extraídos da PokéAPI, um arquivo por Pokémon. |
| Notebook 00 | `application_data/corpus/processed_data/txt` | Textos processados do corpus. |
| Notebook 00 | `application_data/corpus/metadata` | Metadados, CSV/JSON e relatório de qualidade. |
| Notebook 00 | `application_data/corpus/artifacts` | Artefatos de representação, como BoW e sua matriz. |
| Notebook 01 | `application_data/rag/documents` | `pokemon_rag_records.jsonl`, `pokemon_rag_documents.jsonl` e metadados dos documentos. |
| Notebook 01 | `application_data/rag/reference` | Aliases e referências de tradução. |
| Notebook 02 | `application_data/artifacts/retrieval` | Vetorizadores, matrizes TF-IDF, manifesto e regras de normalização. |
| Notebooks 02 e 03 | `application_data/evaluation` | Consultas, métricas, rankings e resultados de avaliação. |
| Notebooks 01, 02 e 03 | `application_data/reports` | Relatórios de qualidade, recuperação e geração. |
| Todos | `application_data/logs` | Logs das execuções. |

Ao executar `python scripts/run_notebooks.py`, as cópias executadas dos arquivos `.ipynb` ficam em `data/executed_notebooks/`. No Google Colab, defina `POKEDEX_PROJECT_DIR` ou use a pasta `SmartPokedex` no Google Drive; nesse caso, a mesma estrutura é criada em `SmartPokedex/application_data/`.

## Execução

1. Crie e configure `backend/.env` conforme a seção [Variáveis de ambiente](#variáveis-de-ambiente).
2. Crie um ambiente virtual e instale as dependências:

   ```powershell
   cd "C:\Users\Lucas Sigoli\Documents\Projects\smart-pokedex"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e "backend[test,notebooks,gemini]"
   ```

   Para instalar apenas as dependências do backend e dos testes, use `python -m pip install -e "backend[test]"`.
3. Execute `python scripts/run_notebooks.py`. O executor adapta os caminhos do Colab em memória e salva cópias executadas em `data/executed_notebooks`.
3. Valide com `cd backend; python -m app.cli verify`.
4. Instale o backend com `pip install -e ".[test,gemini]"`, configure `GEMINI_API_KEY` em `backend/.env` e execute `uvicorn app.main:app --reload`.
5. Execute o frontend com `cd frontend; npm install; npm run dev`.

Na interface web, abra a aba `Teste RAG` para executar uma conversa demonstrativa pela API. O laboratório oferece os cenários `Um Pokémon` e `Comparar dois`, troca de Pokémon ativo e perguntas contextuais como `Como evolui ele` e `Onde encontro eles`. O Gemini é o provedor padrão da aplicação web; as fontes recuperadas continuam disponíveis para auditoria.

Sem os JSON brutos/artefatos produzidos pelos notebooks em `application_data`, a aplicação inicia apenas o endpoint `/health` e `/ready=false`; nenhum corpus sintético é criado.

Para retomar a partir de uma etapa, use `python scripts/run_notebooks.py --from-step 01`, `02` ou `03`. A etapa 00 consulta a PokéAPI e pode demorar; arquivos brutos existentes são reutilizados pelo notebook.

Para executar todas as operações com um único comando na raiz do projeto:

```powershell
.\start.ps1
```

O script verifica os artefatos, executa o pipeline caso estejam ausentes ou inválidos, instala as dependências do frontend quando necessário e inicia a API em `http://127.0.0.1:8000` e o frontend em `http://127.0.0.1:5173`. Encerre com `Ctrl+C`.

Para executar um teste conversacional completo do RAG, use:

```powershell
python scripts/test_rag_conversations.py
```

O teste percorre os 151 Pokémon, cria conversas individuais, mistura pares de Pokémon, troca o Pokémon ativo e testa referências contextuais como `ele` e `eles`. Os turnos e as fontes recuperadas são salvos em `data/evaluation/rag_conversation_test.jsonl`, e o resumo em `data/evaluation/rag_conversation_test_summary.json`. Para uma execução menor:

```powershell
python scripts/test_rag_conversations.py --max-pokemon 10 --pair-limit 3
```

Para também chamar o provedor generativo configurado, acrescente `--generate`; com Gemini isso exige `GEMINI_API_KEY`. Use `--allow-failures` para gerar o relatório mesmo quando o teste encontrar falhas de cobertura ou contexto; sem essa opção, o processo termina com código 1, adequado para CI.

O CORS já permite os dois hosts locais (`localhost` e `127.0.0.1`). Se criar `backend/.env`, mantenha ambos em `ALLOWED_ORIGINS`.

Se o comando `python` não for reconhecido, feche e reabra o PowerShell para carregar o PATH atualizado. Como alternativa, use o executável instalado diretamente:

```powershell
& "C:\Users\Lucas Sigoli\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
```

## API

`POST /api/v1/sessions`, `POST /api/v1/chat`, `DELETE /api/v1/sessions/{id}`, `GET /api/v1/pokemon`, `GET /api/v1/pokemon/{id}`. A geração de texto usa Gemini no backend e exige `GEMINI_API_KEY`; a chave nunca é enviada ao frontend.

## Fluxo RAG alinhado ao material da aula

O projeto segue o mesmo fluxo didático do notebook `S017_LLM_RAG.ipynb`:

1. O Notebook 01 transforma o corpus existente em documentos temáticos em `application_data/rag/documents`.
2. O Notebook 02 ajusta e persiste TF-IDF, matriz, ordem dos documentos e manifesto em `application_data/artifacts/retrieval`.
3. O Notebook 03 reutiliza esses artefatos, recupera por cosseno, seleciona evidências e chama o Gemini para redigir a resposta final.
4. A API usa o mesmo modelo, orçamento e política de evidências. Tipos, estatísticas, evoluções e formas de obtenção são conferidos contra os trechos enviados. Se o modelo omitir valores obrigatórios, a resposta permanece generativa e recebe um reparo fundamentado registrado em `generation_metadata`; o texto público continua natural e não expõe o documento bruto.

O cliente Gemini é inicializado sob demanda. Configure `GEMINI_MODEL_NAME`, `GEMINI_MAX_INPUT_TOKENS`, `GEMINI_MAX_OUTPUT_TOKENS` e `GEMINI_TEMPERATURE` no `backend/.env`. A chave deve permanecer somente no ambiente do backend.

Para executar o Notebook 03 diretamente no VS Code ou Jupyter, abra a pasta `notebooks` e rode as células em ordem. Ele detecta o monorepo e lê `application_data/rag/documents` e `application_data/artifacts/retrieval`; `PLN_AVALIACAO_DIR` pode apontar explicitamente para a raiz.

A avaliação real com e sem RAG é opt-in. No PowerShell:

```powershell
$env:RUN_GENERATION_EVALUATION = "true"
jupyter nbconvert --to notebook --execute notebooks/03_SMART_POKEDEX.ipynb `
  --output 03_SMART_POKEDEX_evaluation.ipynb `
  --output-dir data/executed_notebooks `
  --ExecutePreprocessor.timeout=-1
```

Os resultados ficam em `data/evaluation/rag_generation_results.jsonl`; a planilha `rag_manual_review.csv` preserva notas humanas existentes. Remova a variável ou defina `false` para executar apenas os testes determinísticos e os prechecks.

