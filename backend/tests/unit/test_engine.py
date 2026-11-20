from types import SimpleNamespace

from app.rag.engine import RAGEngine


class FakeRetriever:
    documents = [{"pokemon_id": 4}, {"pokemon_id": 7}]
    records = [
        {"id": 4, "api_name": "charmander", "name": "Charmander"},
        {"id": 7, "api_name": "squirtle", "name": "Squirtle"},
    ]

    def search(self, question, top_k=5):
        return [{
            "rank": 1,
            "document_id": "004_evolution",
            "pokemon_id": 4,
            "pokemon_name": "Charmander",
            "section": "evolution",
            "score": 0.9,
            "content": "Charmander evolves into Charmeleon at level 16.",
        }][:top_k]


def engine():
    return RAGEngine(FakeRetriever(), SimpleNamespace(
        gemini_api_key="test-key",
        gemini_model_name="gemini-test",
        gemini_max_input_tokens=8192,
        gemini_max_output_tokens=768,
        gemini_temperature=0.1,
        rag_top_k=5,
    ))


def test_retrieval_only_returns_sources_without_python_authored_answer():
    result = engine().answer("Como evoluir Charmander?", generate=False)

    assert result["status"] == "retrieval_only"
    assert result["answer"] is None
    assert result["source_documents"][0]["document_id"] == "004_evolution"


def test_generated_answer_is_authored_by_gemini_and_keeps_evidence():
    rag = engine()

    class Generator:
        max_input_tokens = 8192

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + context).split())

        def __call__(self, question, context):
            return {"answer": "Charmander evolui para Charmeleon no nível 16.", "model": "gemini-test"}

    rag.generator = Generator()
    result = rag.answer("Como evoluir Charmander?", generate=True)

    assert result["status"] == "needs_manual_validation"
    assert result["answer_mode"] == "llm_generated"
    assert result["final_answer_origin"] == "model"
    assert result["generation_metadata"]["response_mode"] == "llm_only"
    assert result["supporting_excerpts"]


def test_missing_evidence_is_sent_to_gemini_instead_of_using_python_fallback():
    rag = engine()
    rag.select_evidence = lambda question, top_k=5: ([], "no_relevant_documents")
    received = {}

    class Generator:
        max_input_tokens = 8192

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + context).split())

        def __call__(self, question, context):
            received["context"] = context
            return {"answer": "Não encontrei informação verificada suficiente nas fontes."}

    rag.generator = Generator()
    result = rag.answer("Qual é o tamanho do sapato do Charmander?", generate=True)

    assert result["status"] == "needs_manual_validation"
    assert result["answer"] == "Não encontrei informação verificada suficiente nas fontes."
    assert "não há informação verificada" in received["context"]
    assert result["source_documents"] == []


def test_provider_failure_does_not_return_a_deterministic_python_answer():
    rag = engine()

    class BrokenGenerator:
        max_input_tokens = 8192

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + context).split())

        def __call__(self, question, context):
            raise RuntimeError("provider unavailable")

    rag.generator = BrokenGenerator()
    result = rag.answer("Como evoluir Charmander?", generate=True)

    assert result["status"] == "provider_unavailable"
    assert result["answer"] is None


def test_scope_gate_is_preserved_before_generation():
    result = engine().answer("Quais ginásios preciso vencer?", generate=False)
    assert result["gate_reason"] == "out_of_scope"
    assert result["source_documents"] == []
def test_notebook_generation_adds_control_instruction_for_acquisition_questions():
    rag = engine()
    received = {}

    class Generator:
        max_input_tokens = 8192

        @staticmethod
        def count_prompt_tokens(question, context):
            return len((question + context).split())

        def __call__(self, question, context):
            received["context"] = context
            return {"answer": "A resposta foi escrita pelo modelo."}

    rag.generator = Generator()
    result = rag.answer("Como encontro Charizard?", generate=True)

    assert result["final_answer_origin"] == "model"
    assert "Não interprete habitat" in received["context"]
    assert "captura selvagem" in received["context"]
