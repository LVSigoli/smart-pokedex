"""Contratos dos quatro notebooks oficiais fornecidos para a aplicação."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"


def source(name: str) -> str:
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_repository_contains_the_four_supplied_notebooks():
    assert [path.name for path in sorted(NOTEBOOKS.glob("*.ipynb"))] == [
        "00_CORPUS_POKEMON.ipynb",
        "01_RAG_DATA_PREPARATION.ipynb",
        "02_RAG_RETRIEVAL.ipynb",
        "03_SMART_POKEDEX.ipynb",
    ]


def test_notebook_00_builds_the_enriched_kanto_corpus():
    text = source("00_CORPUS_POKEMON.ipynb")
    assert "range(1, 152)" in text
    assert "APPLICATION_DATA_DIR = PROJECT_ROOT / 'application_data'" in text
    assert "RAW_DIR = root / \"raw_data\"" in text
    assert "enrichment" in text


def test_notebook_01_exports_thematic_documents_without_backend_dependency():
    text = source("01_RAG_DATA_PREPARATION.ipynb")
    assert "google.colab" in text
    assert "application_data/rag/documents" in text
    assert "pokemon_rag_documents.jsonl" in text
    assert "pokemon_rag_records.jsonl" in text
    assert "retrieval_text" in text and "content" in text


def test_notebook_02_persists_tfidf_and_manifest_alignment():
    text = source("02_RAG_RETRIEVAL.ipynb")
    assert "TfidfVectorizer" in text
    assert "cosine_similarity" in text
    assert "TFIDF_PATH" in text and "tfidf_vectorizer.joblib" in text
    assert "TFIDF_MATRIX_PATH" in text and "tfidf_matrix.npz" in text
    assert "retrieval_manifest.json" in text
    assert "loaded_matrix.shape[0]==len(loaded_manifest['document_ids'])==len(documents)" in text


def test_notebook_03_is_gemini_only_and_reuses_the_persisted_tfidf_index():
    text = source("03_SMART_POKEDEX.ipynb")
    normalized = text.casefold()
    assert "google-genai" in normalized
    assert "gemini_model_name" in normalized
    assert "generate_content" in normalized
    assert "model_info['vectorizer']" in text
    assert "model_info['matrix']" in text
    assert "fit(" not in text[text.index("## 5"):]


def test_notebook_03_preserves_evidence_selection_and_scope_rules():
    text = source("03_SMART_POKEDEX.ipynb")
    for marker in (
        "def split_passages(content):",
        "def balance_rows_by_pokemon(rows):",
        "def select_evidence(question",
        "def answer_question(question",
        "out_of_scope",
        "documents_in_prompt",
        "source_documents",
    ):
        assert marker in text


def test_notebook_03_keeps_generation_evaluation_and_colab_ui():
    text = source("03_SMART_POKEDEX.ipynb")
    assert "def validate_generation_questions" in text
    assert "def load_generation_questions" in text
    assert "GEMINI_API_KEY" in text
    assert "CHAT_UI" in text
    assert "GEMINI_MAX_OUTPUT_TOKENS" in text
