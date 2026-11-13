from types import SimpleNamespace

from app.rag.generation import FlanT5Generator


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        tokens = text.split()
        return SimpleNamespace(input_ids=[list(range(len(tokens)))])

    def decode(self, ids, skip_special_tokens=True):
        return "Resposta fundamentada no contexto."


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [[1, 2, 3]]


def fake_loader(model_name):
    return FakeTokenizer(), FakeModel()


def test_flan_loads_lazily_and_generates_deterministically():
    generator = FlanT5Generator("google/flan-t5-small", loader=fake_loader, device="cpu")
    assert generator.loaded is False
    result = generator("Qual é o tipo do Charizard?", "Charizard é do tipo fogo e voador.")
    assert generator.loaded is True
    assert result["answer"] == "Resposta fundamentada no contexto."
    assert result["model"] == "google/flan-t5-small"
    assert result["generation_config"]["do_sample"] is False


def test_flan_prompt_reports_input_tokens_without_silent_truncation():
    generator = FlanT5Generator("google/flan-t5-small", loader=fake_loader, device="cpu", max_input_tokens=40)
    prompt, token_count = generator.build_prompt("Qual é o tipo?", "Fonte [006_profile]: fogo e voador.")
    assert token_count <= 40
    assert "Fonte [006_profile]" in prompt
