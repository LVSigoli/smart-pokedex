"""Tradução offline das descrições antes da indexação do corpus."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any
import re


NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "por_Latn"
TRANSLATION_PIPELINE_VERSION = "2-sentence-preserving"


class NllbDescriptionTranslator:
    """Carrega o NLLB sob demanda e traduz lotes de inglês para português."""

    def __init__(
        self,
        model_name: str = NLLB_MODEL_NAME,
        *,
        source_language: str = SOURCE_LANGUAGE,
        target_language: str = TARGET_LANGUAGE,
        device: str | None = None,
        max_input_tokens: int = 256,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_name = model_name
        self.source_language = source_language
        self.target_language = target_language
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.source_language,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(selected_device)
        self._model.eval()
        self.device = selected_device

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]

    @staticmethod
    def _normalize_translation(text: str) -> str:
        normalized = re.sub(r"\bPOK[ÉE]MON\b", "Pokémon", str(text), flags=re.I)
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
        normalized = re.sub(r"([.!?])\1+", r"\1", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def __call__(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        sentence_groups = [self._sentences(text) for text in texts]
        segments = [sentence for group in sentence_groups for sentence in group]
        translated_segments = self._translate_texts(segments)
        if len(translated_segments) != len(segments):
            raise ValueError("O NLLB não retornou uma tradução para cada frase")
        normalized = [self._normalize_translation(text) for text in translated_segments]
        results = []
        offset = 0
        for group in sentence_groups:
            translated_group = normalized[offset:offset + len(group)]
            if any(not sentence for sentence in translated_group):
                raise ValueError("O NLLB retornou uma frase vazia")
            results.append(" ".join(translated_group))
            offset += len(group)
        return results

    def _translate_texts(self, texts: list[str]) -> list[str]:
        self._load()
        import torch

        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        target_token = self._tokenizer.convert_tokens_to_ids(self.target_language)
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=target_token,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        return [text.strip() for text in self._tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )]


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    if size < 1:
        raise ValueError("batch_size deve ser maior que zero")
    for start in range(0, len(values), size):
        yield values[start:start + size]


def apply_description_translations(
    records: Iterable[Mapping[str, Any]],
    translate_batch: Callable[[list[str]], list[str]],
    *,
    translation_memory: Mapping[str, str] | None = None,
    batch_size: int = 8,
    model_name: str = NLLB_MODEL_NAME,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Adiciona ``description_pt`` sem alterar a descrição original em inglês."""

    translated_records = [dict(record) for record in records]
    memory = {
        str(source): str(target).strip()
        for source, target in (translation_memory or {}).items()
        if str(source).strip() and str(target).strip()
    }
    missing_sources = []
    for record in translated_records:
        source = str(record.get("description") or "").strip()
        existing = str(record.get("description_pt") or "").strip()
        if source and existing:
            memory[source] = existing
        elif source and source not in memory and source not in missing_sources:
            missing_sources.append(source)

    for batch in _batches(missing_sources, batch_size):
        outputs = list(translate_batch(batch))
        if len(outputs) != len(batch):
            raise ValueError("O tradutor retornou uma quantidade inesperada de descrições")
        for source, target in zip(batch, outputs, strict=True):
            cleaned = str(target or "").strip()
            if not cleaned:
                pokemon_ids = [str(record.get("id")) for record in translated_records
                               if str(record.get("description") or "").strip() == source]
                raise ValueError(
                    "Tradução vazia para o Pokémon de ID " + ", ".join(pokemon_ids)
                )
            memory[source] = cleaned

    for record in translated_records:
        source = str(record.get("description") or "").strip()
        if not source:
            raise ValueError(f"Descrição original vazia para o Pokémon de ID {record.get('id')}")
        target = memory.get(source, "").strip()
        if not target:
            raise ValueError(f"Tradução ausente para o Pokémon de ID {record.get('id')}")
        record["description_pt"] = target
        record["description_translation"] = {
            "method": "offline_machine_translation",
            "model": model_name,
            "source_language": SOURCE_LANGUAGE,
            "target_language": TARGET_LANGUAGE,
            "review_status": "machine_translated_pending_review",
        }
    return translated_records, memory
