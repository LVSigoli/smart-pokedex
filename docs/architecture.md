# Arquitetura

O processo de dados produz JSONL separados por documentos e registros, depois o índice TF-IDF é persistido em `data/artifacts/retrieval`. A API carrega esses artefatos uma vez no processo. O motor mantém TF-IDF como recuperador principal; Gemini é opcional e só pode responder quando suas citações e trechos literais passam pela validação.

