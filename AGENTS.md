# Manutenção

- Preserve a ordem dos documentos e o alinhamento com a matriz TF-IDF.
- Não substitua TF-IDF por embeddings sem decisão explícita.
- Não fabrique corpus completo em testes; use fixtures pequenas.
- Rode `python -m app.cli verify` e `pytest` antes de mudanças no motor.
- O servidor não deve executar coleta ou indexação em uma requisição.

