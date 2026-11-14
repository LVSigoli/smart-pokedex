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
   python -m pip install -e "backend[test,notebooks,flan]"
   ```

   Para instalar apenas as dependências do backend e dos testes, use `python -m pip install -e "backend[test]"`.
3. Execute `python scripts/run_notebooks.py`. O executor adapta os caminhos do Colab em memória e salva cópias executadas em `data/executed_notebooks`.
3. Valide com `cd backend; python -m app.cli verify`.
4. Instale o backend com `pip install -e ".[test,flan]"` e execute `uvicorn app.main:app --reload`.
5. Execute o frontend com `cd frontend; npm install; npm run dev`.

Na interface web, abra a aba `Teste RAG` para executar uma conversa demonstrativa pela API. O laboratório oferece os cenários `Um Pokémon` e `Comparar dois`, troca de Pokémon ativo e perguntas contextuais como `Como evolui ele` e `Onde encontro eles`. O FLAN-T5 local permanece sempre ativo nas conversas da interface; as fontes recuperadas continuam disponíveis para auditoria.

Sem os JSON brutos/artefatos, a aplicação inicia apenas o endpoint `/health` e `/ready=false`; nenhum corpus sintético é criado.

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

Para também chamar o FLAN-T5 local, acrescente `--generate`; isso pode ser demorado em CPU. Use `--allow-failures` para gerar o relatório mesmo quando o teste encontrar falhas de cobertura ou contexto; sem essa opção, o processo termina com código 1, adequado para CI.

O CORS já permite os dois hosts locais (`localhost` e `127.0.0.1`). Se criar `backend/.env`, mantenha ambos em `ALLOWED_ORIGINS`.

Se o comando `python` não for reconhecido, feche e reabra o PowerShell para carregar o PATH atualizado. Como alternativa, use o executável instalado diretamente:

```powershell
& "C:\Users\Lucas Sigoli\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
```

## API

`POST /api/v1/sessions`, `POST /api/v1/chat`, `DELETE /api/v1/sessions/{id}`, `GET /api/v1/pokemon`, `GET /api/v1/pokemon/{id}`. A geração de texto usa localmente `google/flan-t5-small`; não é necessária chave de API. Na primeira geração, os pesos podem ser baixados do Hugging Face.

## Fluxo RAG alinhado ao material da aula

O projeto segue o mesmo fluxo didático do notebook `S017_LLM_RAG.ipynb`:

1. O Notebook 01 transforma o corpus existente em documentos temáticos.
2. O Notebook 02 ajusta e persiste TF-IDF, matriz, ordem dos documentos e manifesto.
3. O Notebook 03 reutiliza esses artefatos, recupera por cosseno, seleciona passagens completas dentro de 384 tokens e chama o FLAN-T5 com um prompt curto de contexto, pergunta e resposta.
4. A API usa o mesmo modelo, orçamento e política de evidências. Tipos, estatísticas, evoluções e formas de obtenção são conferidos contra os trechos enviados. Se o modelo omitir valores obrigatórios, a resposta permanece generativa e recebe um reparo fundamentado registrado em `generation_metadata`; o texto público continua natural e não expõe o documento bruto.

O modelo é carregado sob demanda. O Hugging Face faz o download no primeiro uso e reutiliza o cache nas execuções seguintes. `FLAN_MODEL=google/flan-t5-base` configura a API e `FLAN_MODEL_NAME=google/flan-t5-base` configura o Notebook 03 para uma comparação opcional; o padrão continua sendo `google/flan-t5-small`.

Para executar o Notebook 03 diretamente no VS Code ou Jupyter, abra a pasta `notebooks` e rode as células em ordem. Ele detecta o monorepo e lê `data/processed` e `data/artifacts/retrieval`; `PLN_AVALIACAO_DIR` pode apontar explicitamente para a raiz.

A avaliação real com e sem RAG é opt-in. No PowerShell:

```powershell
$env:RUN_GENERATION_EVALUATION = "true"
jupyter nbconvert --to notebook --execute notebooks/03_RAG_CHATBOT.ipynb `
  --output 03_RAG_CHATBOT_evaluation.ipynb `
  --output-dir data/executed_notebooks `
  --ExecutePreprocessor.timeout=-1
```

Os resultados ficam em `data/evaluation/rag_generation_results.jsonl`; a planilha `rag_manual_review.csv` preserva notas humanas existentes. Remova a variável ou defina `false` para executar apenas os testes determinísticos e os prechecks.

