from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerSpec:
    key: str
    label: str
    kind: str
    ref: str


DEFAULT_SPECS: list[TokenizerSpec] = [
    TokenizerSpec("gpt2", "GPT-2", "hf", "gpt2"),
    TokenizerSpec("cl100k", "cl100k (GPT-4)", "tiktoken", "cl100k_base"),
    TokenizerSpec("o200k", "o200k (GPT-4o)", "tiktoken", "o200k_base"),
    TokenizerSpec("llama3", "Llama 3.x", "hf", "NousResearch/Llama-3.2-1B"),
    TokenizerSpec("qwen25", "Qwen 2.5", "hf", "Qwen/Qwen2.5-0.5B"),
    TokenizerSpec("phi3", "Phi-3", "hf", "microsoft/Phi-3-mini-4k-instruct"),
    TokenizerSpec("gemma3", "Gemma 3", "hf", "google/gemma-3-1b-pt"),
    TokenizerSpec("mistral", "Mistral (Nemo)", "hf", "mistralai/Mistral-Nemo-Base-2407"),
]
SPEC_KEYS = [s.key for s in DEFAULT_SPECS]
SPEC_LABELS = {s.key: s.label for s in DEFAULT_SPECS}


class TokenCounter:
    def __init__(self, keys: list[str] | None = None, hf_token: str | None = None) -> None:
        self.specs = [s for s in DEFAULT_SPECS if keys is None or s.key in keys]
        self.hf_token = hf_token or None
        self._encoders: dict[str, object] = {}
        self.failures: dict[str, str] = {}

    def _load(self, spec: TokenizerSpec):
        if spec.key in self._encoders:
            return self._encoders[spec.key]
        if spec.key in self.failures:
            return None
        try:
            if spec.kind == "tiktoken":
                import tiktoken

                enc = tiktoken.get_encoding(spec.ref)
            else:
                from transformers import AutoTokenizer

                enc = AutoTokenizer.from_pretrained(spec.ref, use_fast=True, token=self.hf_token)
            self._encoders[spec.key] = enc
            return enc
        except Exception as e:  # noqa: BLE001
            self.failures[spec.key] = str(e).splitlines()[0][:300]
            return None

    def available_specs(self) -> list[TokenizerSpec]:
        return [s for s in self.specs if self._load(s) is not None]

    def count(self, texts: list[str]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for spec in self.specs:
            enc = self._load(spec)
            if enc is None:
                continue
            if spec.kind == "tiktoken":
                counts = [len(c) for c in enc.encode_ordinary_batch(texts, num_threads=os.cpu_count() or 4)]
            else:
                encoded = enc(texts, add_special_tokens=False, return_length=True, return_attention_mask=False)
                counts = list(encoded["length"])
            out[spec.key] = counts
        return out
