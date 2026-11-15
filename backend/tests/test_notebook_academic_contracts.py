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


def test_notebook_01_translates_descriptions_offline_and_documents_the_reason() -> None:
    source = notebook_source("01_RAG_DATA_PREPARATION.ipynb")
    assert "facebook/nllb-200-distilled-600M" in source
    assert "eng_Latn" in source and "por_Latn" in source
    assert "description_pt" in source
    assert "description_translations_ptbr.json" in source
    assert "TRANSLATION_PIPELINE_VERSION" in source
    assert "translation_payload.get('pipeline_version')" in source
    assert "description_translations_reviewed_ptbr.json" in source
    assert "human_reviewed" in source
    assert "apply_description_translations" in source
    assert "evita latência no chat" in source
    assert "não depende de tradução durante cada pergunta" in source
    assert "Descrição: {r['description_pt']}" in source


def test_notebook_01_translation_is_self_contained_for_colab_without_backend() -> None:
    source = notebook_source("01_RAG_DATA_PREPARATION.ipynb")
    assert "if BACKEND_DIR.is_dir():" in source
    assert "A pasta backend do projeto é necessária" not in source
    assert "class NllbDescriptionTranslator:" in source
    assert "def apply_description_translations(" in source
    assert "EDITORIAL_TRANSLATIONS_BY_ID" in source
    assert "Suas asas conseguem levá-lo" in source
    assert "DRIVE_AVAILABLE = False" in source
    assert "except Exception as drive_error" in source
    assert "if IN_COLAB and DRIVE_AVAILABLE:" in source


def test_notebook_01_discovers_colab_raw_data_layouts() -> None:
    source = notebook_source("01_RAG_DATA_PREPARATION.ipynb")
    assert "RAW_CANDIDATES" in source
    assert "Path('/content/corpus/raw')" in source
    assert "next((candidate for candidate in RAW_CANDIDATES if candidate.is_dir())" in source


def test_notebook_02_fits_each_vectorizer_once() -> None:
    source = notebook_source("02_RETRIEVAL_AND_EVALUATION.ipynb")
    assert source.count("BOW.fit_transform(normalized_texts)") == 1
    assert source.count("TFIDF.fit_transform(normalized_texts)") == 1
    assert "Q25–Q32 já existem ou não possuem documento-alvo" in source


def test_notebook_02_normalizes_compound_rock_type_aliases() -> None:
    source = notebook_source("02_RETRIEVAL_AND_EVALUATION.ipynb")
    assert "'tipo pedra': 'type rock'" in source
    assert "'tipo rocha': 'type rock'" in source
    assert "normalize_query('Pokémon do tipo pedra').endswith('type rock')" in source
    chatbot = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "('tipo pedra', 'type rock')" in chatbot
    assert "('tipo rocha', 'type rock')" in chatbot

def test_notebook_03_keeps_generation_opt_in_and_compares_retrievers() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "RUN_GENERATION_EVALUATION = os.environ.get(" in source
    assert 'getpass(' not in source
    assert "tfidf_raw" in source and "tfidf_filtered" in source
    assert "FLAN_MODEL_NAME" in source
    assert "google/flan-t5-small" in source
    assert "MAX_INPUT_TOKENS = 384" in source
    assert "truncation=False" in source
    assert "model_without_corpus" in source
    assert "Responda em português de forma breve" in source
    assert "torch.inference_mode" in source
    assert "GEMINI" not in source.upper()

def test_notebook_03_generation_uses_preprovisioned_local_model() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    normalized = source.casefold()
    assert "api_error" not in normalized
    assert "generation_error" in normalized
    assert "consumo de api" not in normalized
    assert "chave de api" not in normalized
    assert "api_key" not in normalized
    assert "remote" not in normalized
    assert source.count("local_files_only=True") >= 2
    assert "cache local previamente provisionado" in normalized

def test_notebook_03_generation_evaluation_normalizes_csv_columns_and_uses_mappings() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "def validate_generation_questions(frame, source_name):" in source
    assert "def load_generation_questions(path, seed_rows):" in source
    assert "rag_generation_questions_legacy_backup.csv" in source
    assert "questions.columns = [re.sub" in source
    assert "questions = load_generation_questions(EVAL_QUESTIONS_PATH, seed_cases)" in source
    assert "Nenhum campo foi convertido automaticamente" in source
    assert "questions.to_dict(orient='records')" in source
    assert "row['expected_case']" in source
    assert "questions.itertuples(index=False)" not in source


def test_notebooks_resolve_the_monorepo_data_layout_before_legacy_application() -> None:
    for notebook in ("01_RAG_DATA_PREPARATION.ipynb", "02_RETRIEVAL_AND_EVALUATION.ipynb", "03_RAG_CHATBOT.ipynb"):
        source = notebook_source(notebook)
        assert "import os" in source or ", os," in source
        monorepo_marker = "candidate / 'backend'"
        legacy_marker = "candidate / 'application' / 'data' / 'rag' / 'documents'"
        assert monorepo_marker in source
        assert legacy_marker in source
        assert source.index(monorepo_marker) < source.index(legacy_marker)
        assert source.index("candidates = (Path.cwd(), *Path.cwd().parents)") < source.index(
            "Path('/content/drive/MyDrive/IFSP-ADS/ultimo_periodo/tc_2/avaliacao')"
        )
        assert "DOC_DIR = DATA_DIR / 'processed'" in source
        if notebook != "01_RAG_DATA_PREPARATION.ipynb":
            assert "ARTIFACTS_DIR = DATA_DIR / 'artifacts' / 'retrieval'" in source


def test_notebook_03_colab_prefers_current_repo_and_provisions_generator() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    candidates = source.index("candidates = (Path.cwd(), *Path.cwd().parents)")
    drive_fallback = source.index("Path('/content/drive/MyDrive/IFSP-ADS/ultimo_periodo/tc_2/avaliacao')")
    assert candidates < drive_fallback
    assert "'1' if IN_COLAB else '0'" in source
    assert "subprocess.check_call" in source
    assert "colab_output.enable_custom_widget_manager()" in source
    assert "CHAT_UI = open_chat_ui() if IN_COLAB else None" in source
    assert "def show_answer(result, question=None):" in source
    assert "f'Falha na geração: {type(error).__name__}: {error}'" in source
    assert "def pokemon_names_by_ids(pokemon_ids):" in source
    assert "def detect_response_language(question):" in source


def test_notebook_03_chat_ui_shows_only_conversation_and_preloads_model() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    ui_source = source[source.index("def show_answer(result") : source.index("## 12")]
    preload = source.index("if PRELOAD_FLAN_MODEL:\n    load_flan_t5()")
    assert preload < source.index("def answer_question(")
    assert "Você" in ui_source
    assert "Smart Pokédex" in ui_source
    assert "Pensando..." in ui_source
    assert "conversation.ask(prompt, generate=True)" in ui_source
    assert "load_flan_t5()" in ui_source
    assert "transcript = widgets.HTML" in ui_source
    assert "widgets.Output" not in ui_source
    assert "overflow-y:auto" in ui_source
    assert "Fontes recuperadas" not in ui_source
    assert "source_documents" not in ui_source
    assert "widgets.Checkbox" not in ui_source


def test_notebook_03_compacts_context_by_complete_passages_and_balances_entities() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "def split_passages(content):" in source
    assert "def compact_evolution_passage(passage):" in source
    assert "def balance_rows_by_pokemon(rows):" in source
    assert "documents_in_prompt" in source
    assert "def evolution_transitions(rows):" in source
    assert "missing_evolution_conditions" in source
    assert "answer_mode" in source
    assert "grounded['returned_citations']" not in source
    assert "rag_input_tokens" in source
    assert "rag_documents_in_prompt" in source
    assert "existing_review = pd.read_csv" in source
    assert "human_review_fields" in source
    assert "stat_passages = [" in source
    assert "requested_stat_name(question)" in source
    assert "response_mode='focused_structured'" in source
    assert "raw_answer=answer" in source
    assert "uma única frase curta" in source
    assert "não liste outros atributos" in source
    assert "'answer_mode': 'extractive', 'generation_error': type(preflight_error).__name__" in source
    assert "if target_id in found_targets and section not in {'obtainment', 'encounters'}:" in source
    assert "response_mode='grounded_structured'" in source
    assert "response_mode='grounded_repair'" in source
    assert "max_new_tokens=64" in source
    assert "obtido por evolução" in source
    assert "conversation.ask(prompt, generate=True)" in source
    assert "def profile_answer(rows, question):" in source
    assert "fale|sabe|sabre|conte" in source
    assert "r'Descri[cç][aã]o:" in source
    assert "connector = ',' if dex_match else ' é'" in source
    assert "Descrição da Pokédex FireRed:" not in source
    assert "vive próximo à água" in source
    assert "EDITORIAL_DESCRIPTION_BY_ID" in source
    assert "editorial_description = EDITORIAL_DESCRIPTION_BY_ID.get" in source
    assert "def is_type_listing_question(question):" in source
    assert "def requested_type(question):" in source
    assert "def select_type_evidence(question):" in source
    assert "def type_matchup_answer(rows, question):" in source


def test_notebook_03_distinguishes_scope_limits_from_missing_evidence() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "SCOPE_LIMIT_REASONS" in source
    assert "'out_of_scope' if gate in SCOPE_LIMIT_REASONS else 'insufficient_evidence'" in source
    assert "('scope_limit' if gate else 'insufficient_evidence')" not in source
    assert "clarification_needed" in source
    assert "Em qual atributo" in source
    assert "pending_kind" in source
    assert "needs_comparison_criterion" in source
    assert "Compare o {attribute_label} de {joined}." in source


def test_notebook_03_formats_acquisition_evidence_instead_of_returning_raw_documents() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "def acquisition_facts(rows):" in source
    assert "def acquisition_answer(rows, question):" in source
    assert "é obtido como presente" in source


def test_notebook_03_protects_compound_species_names_during_normalization() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "protected_tokens = set()" in source
    assert "if ' ' in name and re.search" in source
    assert "token in protected_tokens" in source


def test_notebook_03_prioritizes_attribute_comparisons_over_evolution_intent() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "atronut" in source
    assert "related_ids = set(targets)" in source
    assert "summaries = []" in source


def test_notebook_03_prioritizes_type_listing_over_active_profile() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "if is_type_listing_question(question):" in source
    assert source.index("if is_type_listing_question(question):") < source.index("profile = profile_answer(rows, question)")


def test_notebook_03_never_prefers_raw_generation_for_type_lists_or_evolution() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "fallback_answer = structured_answer(used, user_question) or retrieval_answer" in source
    assert "or is_type_listing_question(user_question) or evolution_transitions(used)" in source


def test_notebook_03_supports_complete_type_lists_and_attribute_rankings() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "def is_ranking_question(question):" in source
    assert "candidates.sort(key=lambda item: (-item[0], int(item[1]['pokemon_id'])))" in source
    assert "collection_answer = structured_answer(retrieved, user_question)" in source
    assert "def ranking_limit(question):" in source
    assert "shown = ranked if limit is None else ranked[:limit]" in source
    assert "mais fortes?" in source


def test_notebook_03_remembers_pending_global_ranking_attribute() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "self.pending_kind == 'ranking' and attribute_label" in source
    assert "Quais são os pokemons com maior {attribute_label}?" in source
    assert "self.pending_kind = 'comparison' if targets else 'ranking'" in source


def test_notebook_03_renders_non_evolving_species_in_portuguese() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "def non_evolving_answer(rows, question):" in source
    assert "não evolui para outro Pokémon" in source
    assert "or non_evolving_answer(used, user_question)" in source


def test_notebook_03_renders_rare_habitat_without_raw_english_label() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert "'rare': 'é uma espécie rara'" in source


def test_notebook_03_runs_retrieval_evaluation_once() -> None:
    source = notebook_source("03_RAG_CHATBOT.ipynb")
    assert source.count("CHATBOT_RETRIEVAL_METRICS = EVAL_DIR / 'chatbot_retrieval_metrics_by_query.csv'") == 1


def test_notebook_02_marks_all_expected_documents_as_reviewed() -> None:
    source = notebook_source("02_RETRIEVAL_AND_EVALUATION.ipynb")
    assert "Documentos esperados revisados" in source
    assert "queries_existing['review_status'] = 'reviewed'" in source
