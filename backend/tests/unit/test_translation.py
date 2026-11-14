import pytest

from app.pipeline.translation import (
    NLLB_MODEL_NAME,
    NllbDescriptionTranslator,
    apply_description_translations,
)


def test_applies_cached_and_new_translations_without_retranslating_known_texts():
    records = [
        {"id": 1, "description": "A seed grows on its back."},
        {"id": 4, "description": "The flame on its tail shows its health."},
    ]
    calls = []

    def translate_batch(texts):
        calls.append(texts)
        return ["A chama em sua cauda indica seu estado de saúde."]

    translated, memory = apply_description_translations(
        records,
        translate_batch,
        translation_memory={"A seed grows on its back.": "Uma semente cresce em suas costas."},
        batch_size=8,
    )

    assert calls == [["The flame on its tail shows its health."]]
    assert translated[0]["description_pt"] == "Uma semente cresce em suas costas."
    assert translated[1]["description_pt"] == "A chama em sua cauda indica seu estado de saúde."
    assert translated[1]["description_translation"]["model"] == NLLB_MODEL_NAME
    assert translated[1]["description_translation"]["source_language"] == "eng_Latn"
    assert translated[1]["description_translation"]["target_language"] == "por_Latn"
    assert memory[records[1]["description"]] == translated[1]["description_pt"]


def test_rejects_empty_translation_instead_of_exporting_an_incomplete_corpus():
    records = [{"id": 25, "description": "It stores electricity in its cheeks."}]

    with pytest.raises(ValueError, match="25"):
        apply_description_translations(records, lambda texts: [""], batch_size=1)


def test_nllb_translates_each_sentence_and_normalizes_pokemon_spelling():
    translator = NllbDescriptionTranslator()
    received = []

    def translate_segments(segments):
        received.extend(segments)
        return [
            "Suas asas conseguem levá-lo a grandes altitudes.",
            "Ele expele fogo em temperaturas muito altas.",
        ]

    translator._translate_texts = translate_segments
    result = translator([
        "Its wings carry this POKéMON to high altitudes. "
        "It blows out fire at very high temperatures."
    ])

    assert len(received) == 2
    assert result == [
        "Suas asas conseguem levá-lo a grandes altitudes. "
        "Ele expele fogo em temperaturas muito altas."
    ]

