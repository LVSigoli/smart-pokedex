"""Testa a validação do CSV de avaliação definida no Notebook 03."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
import pytest


NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "03_RAG_CHATBOT.ipynb"
REQUIRED = {"query_id", "question", "expected_case", "expected_section"}


def load_schema_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    functions = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "validate_generation_questions", "load_generation_questions",
            }:
                functions[node.name] = node
        if len(functions) == 2:
            break
    assert "validate_generation_questions" in functions
    assert "load_generation_questions" in functions
    module = ast.Module(body=[
        functions["validate_generation_questions"],
        functions["load_generation_questions"],
    ], type_ignores=[])
    namespace = {"re": re, "pd": pd, "warnings": __import__("warnings")}
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


def load_validator():
    return load_schema_functions()["validate_generation_questions"]


def test_valid_csv_schema_is_accepted_without_losing_rows():
    validate = load_validator()
    frame = pd.DataFrame([{
        "query_id": "R01", "question": "Tipos do Charizard?",
        "expected_case": "answerable", "expected_section": "profile",
    }])
    before = frame.copy(deep=True)
    validated = validate(frame, "valid.csv")
    assert set(validated.columns) == REQUIRED
    assert validated.to_dict("records") == frame.to_dict("records")
    pd.testing.assert_frame_equal(frame, before)


def test_missing_required_columns_raise_descriptive_error():
    validate = load_validator()
    frame = pd.DataFrame([{"query_id": "R01", "question": "Tipos?"}])
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="expected_case.*expected_section"):
        validate(frame, "incomplete.csv")
    pd.testing.assert_frame_equal(frame, before)


def test_legacy_category_schema_migrates_only_when_values_are_case_labels():
    validate = load_validator()
    frame = pd.DataFrame([
        {"query_id": "R-old-1", "question": "Pergunta respondível?",
         "category": "answerable", "expected_section": "profile"},
        {"query_id": "R-old-2", "question": "Comparação?",
         "category": "comparison", "expected_section": "stats"},
        {"query_id": "R-old-3", "question": "Fora de escopo?",
         "category": "out_of_scope", "expected_section": "none"},
        {"query_id": "R-old-4", "question": "Sem evidência?",
         "category": "missing_evidence", "expected_section": "profile"},
    ])
    validated = validate(frame, "legacy.csv")
    assert validated["expected_case"].tolist() == [
        "answerable", "comparison", "out_of_scope", "missing_evidence",
    ]
    assert validated["category"].tolist() == frame["category"].tolist()
    assert frame.columns.tolist() == ["query_id", "question", "category", "expected_section"]


def test_legacy_thematic_category_is_rejected_without_fabricating_case_labels():
    validate = load_validator()
    frame = pd.DataFrame([{
        "query_id": "R-old", "question": "Pergunta antiga?",
        "category": "perfil", "expected_section": "profile",
    }])
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="category.*expected_case.*perfil"):
        validate(frame, "legacy.csv")
    pd.testing.assert_frame_equal(frame, before)


def test_loader_backs_up_thematic_legacy_csv_and_uses_canonical_seed(tmp_path):
    load = load_schema_functions()["load_generation_questions"]
    path = tmp_path / "rag_generation_questions.csv"
    legacy = pd.DataFrame([{
        "query_id": "old-1", "question": "Pergunta factual?",
        "category": "factual", "expected_section": "profile",
    }])
    legacy.to_csv(path, index=False, encoding="utf-8")
    seed_cases = [
        ("R01", "Tipos do Charizard?", "answerable", "profile"),
        ("R02", "Ginásios?", "out_of_scope", "none"),
    ]

    with pytest.warns(UserWarning, match="backup"):
        questions = load(path, seed_cases)

    backup = tmp_path / "rag_generation_questions_legacy_backup.csv"
    assert backup.exists()
    assert pd.read_csv(backup).to_dict("records") == legacy.to_dict("records")
    assert questions["expected_case"].tolist() == ["answerable", "out_of_scope"]
    assert set(pd.read_csv(path).columns) == REQUIRED
