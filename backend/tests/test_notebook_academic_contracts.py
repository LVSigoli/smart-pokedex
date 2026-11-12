"""Contratos estáticos para as correções acadêmicas dos notebooks."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def notebook_source(name: str) -> str:
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_alias_pedra_has_unambiguous_contract() -> None:
    source = notebook_source("01_RAG_DATA_PREPARATION.ipynb")
    assert "'rock':['pedra','rocha']" in source
    assert "'pedra':'stone'" not in source
    assert "'pedra evolutiva':'evolution stone'" in source


def test_notebook_02_fits_each_vectorizer_once() -> None:
    source = notebook_source("02_RETRIEVAL_AND_EVALUATION.ipynb")
    assert source.count("BOW.fit_transform(normalized_texts)") == 1
    assert source.count("TFIDF.fit_transform(normalized_texts)") == 1
    assert "Q25–Q32 já existem ou não possuem documento-alvo" in source


def test_notebook_03_keeps_generation_opt_in_and_compares_retrievers() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "RUN_GENERATION_EVALUATION = False" in source
    assert 'getpass(' not in source
    assert "tfidf_raw" in source and "tfidf_filtered" in source
    assert "não foi confirmada especificamente para FireRed" in source
