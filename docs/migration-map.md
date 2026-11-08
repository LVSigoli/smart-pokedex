# Mapa de migração

| Origem                                       | Responsabilidade                                    | Destino                                          | Risco                                        |
| -------------------------------------------- | --------------------------------------------------- | ------------------------------------------------ | -------------------------------------------- |
| `00_CORPUS_POKEMON.ipynb`                    | Coleta PokéAPI, enriquecimento, FireRed, registros  | `app.pipeline`/dados brutos                      | Rede, limites da PokéAPI e campos históricos |
| `01_RAG_DATA_PREPARATION.ipynb`              | Normalização, chunks, JSONL, aliases                | `app.pipeline` + `data/processed`                | Preservar IDs e escopos factuais             |
| `02_RETRIEVAL_AND_EVALUATION_ADAPTADO.ipynb` | TF-IDF 1–2-gram, BoW, ranking e avaliação           | `app.rag.retrieval` + `data/artifacts/retrieval` | Alinhamento documento/matriz                 |
| `03_RAG_CHATBOT.ipynb`                       | Escopo, seção, contexto, Gemini, grounding e sessão | `app.rag.engine`, `generation`, `sessions`       | Referências ambíguas e citações literais     |

Os notebooks originais foram preservados em `notebooks/`. O backend não coleta nem reconstrói índices durante requisições.
