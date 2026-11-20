from types import SimpleNamespace

import pytest

from app.rag.generation import GeminiGenerator
from app.rag.engine import RAGEngine


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="Pikachu é do tipo elétrico.")


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_gemini_writes_only_from_retrieved_context_and_records_model_call():
    client = FakeClient()
    generator = GeminiGenerator(
        api_key="test-key",
        model_name="gemini-test",
        client_factory=lambda key: client,
    )

    result = generator("Qual é o tipo do Pikachu?", "FONTE [025_profile] | tipo elétrico.")

    assert result["answer"] == "Pikachu é do tipo elétrico."
    assert result["model"] == "gemini-test"
    assert client.models.calls[0]["model"] == "gemini-test"
    assert "FONTE [025_profile]" in client.models.calls[0]["contents"]


def test_gemini_requires_a_key_before_initializing_the_client():
    with pytest.raises(Exception, match="GEMINI_API_KEY"):
        GeminiGenerator(api_key="", client_factory=lambda key: FakeClient())(
            "Pergunta", "FONTE [001_profile] | contexto"
        )


def test_prompt_uses_the_smart_pokedex_conversation_rules():
    prompt = GeminiGenerator.prompt_text("Onde encontro Pikachu?", "contexto")

    assert "Não inicie respostas com \"Com base nos dados disponíveis\"." in prompt
    assert "Para rankings, mostre somente os atributos relevantes." in prompt
    assert "Uma regra específica de FireRed só pode ser afirmada" in prompt
    assert "Não chame estes últimos de captura selvagem." in prompt
    assert "Não mencione o corpus, os documentos, as evidências" in prompt


def test_engine_uses_gemini_with_notebook_model_settings():
    retriever = SimpleNamespace(records=[], documents=[])
    settings = SimpleNamespace(
        gemini_api_key="test-key",
        gemini_model_name="gemini-test",
        gemini_max_input_tokens=8192,
        gemini_max_output_tokens=768,
        gemini_temperature=0.1,
        rag_top_k=5,
    )

    engine = RAGEngine(retriever, settings)

    assert isinstance(engine.generator, GeminiGenerator)
    assert engine.generator.model_name == "gemini-test"
