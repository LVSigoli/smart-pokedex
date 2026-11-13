"""Geração local com FLAN-T5, carregada somente na primeira inferência."""
from __future__ import annotations

import time
from threading import RLock

from app.core.exceptions import ProviderUnavailable


class FlanT5Generator:
    def __init__(self, model_name: str = "google/flan-t5-small", *, loader=None,
                 device: str | None = None, max_input_tokens: int = 384,
                 max_new_tokens: int = 96, num_beams: int = 1):
        self.model_name = model_name
        self._loader = loader or self._load_transformers
        self.device = device
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.tokenizer = None
        self.model = None
        self._lock = RLock()

    @property
    def loaded(self):
        return self.tokenizer is not None and self.model is not None

    @staticmethod
    def _load_transformers(model_name):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderUnavailable(
                "Instale transformers, sentencepiece e torch para usar FLAN-T5."
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # O ambiente pode definir um proxy sem saída (por exemplo, no sandbox).
        # Os pesos são provisionados fora do servidor; inferência não tenta baixar
        # arquivos nem fica dependente da rede em uma requisição.
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True).to(device).eval()
        return tokenizer, model, device

    def _ensure_loaded(self):
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            loaded = self._loader(self.model_name)
            if len(loaded) == 3:
                self.tokenizer, self.model, detected_device = loaded
                self.device = self.device or detected_device
            else:
                self.tokenizer, self.model = loaded
                self.device = self.device or "cpu"
            self.model.eval()

    def _token_count(self, text):
        encoded = self.tokenizer(text, add_special_tokens=True, truncation=False)
        ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        return len(ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids)

    def build_prompt(self, question, context):
        self._ensure_loaded()
        prompt = self.prompt_text(question, context)
        token_count = self._token_count(prompt)
        if token_count > self.max_input_tokens:
            raise ProviderUnavailable(
                f"Contexto excede o orçamento de {self.max_input_tokens} tokens ({token_count})."
            )
        return prompt, token_count

    @staticmethod
    def prompt_text(question, context):
        return (
            "Responda apenas com base no contexto. Se faltar informação, responda "
            "Não sei. Responda no idioma da pergunta, sem inventar fatos.\n"
            f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
        )

    def count_prompt_tokens(self, question, context):
        self._ensure_loaded()
        return self._token_count(self.prompt_text(question, context))

    def __call__(self, question, context):
        self._ensure_loaded()
        prompt, input_tokens = self.build_prompt(question, context)
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=True)
            if hasattr(encoded, "to"):
                encoded = encoded.to(self.device)
            started = time.perf_counter()
            try:
                import torch
                inference = torch.inference_mode()
            except ImportError:
                class _Inference:
                    def __enter__(self): return self
                    def __exit__(self, *args): return False
                inference = _Inference()
            with inference:
                generation_inputs = dict(encoded) if hasattr(encoded, "keys") else encoded.__dict__
                output_ids = self.model.generate(
                    **generation_inputs,
                    max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=self.num_beams,
                )
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if not answer:
                raise ProviderUnavailable("FLAN-T5 não retornou conteúdo.")
            return {
                "answer": answer, "model": self.model_name,
                "input_tokens": input_tokens, "output_tokens": len(output_ids[0]),
                "latency_seconds": time.perf_counter() - started,
                "generation_config": {"max_new_tokens": self.max_new_tokens, "do_sample": False, "num_beams": self.num_beams},
            }
        except ProviderUnavailable:
            raise
        except Exception as exc:
            raise ProviderUnavailable(f"Falha na geração FLAN-T5: {type(exc).__name__}") from exc
