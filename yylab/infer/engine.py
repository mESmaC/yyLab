from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class LoadSpec:
    ref: str
    quant: str
    adapter_path: str | None = None
    adapter_arch: str = ""
    router_path: str | None = None
    settings: dict = field(default_factory=dict)

    def key(self) -> str:
        return "|".join(
            [
                self.ref,
                self.quant,
                self.adapter_path or "",
                self.router_path or "",
                json.dumps(self.settings, sort_keys=True),
            ]
        )


def format_messages(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # noqa: BLE001
        parts = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            parts.append(f"### {role}\n{m.get('content', '')}")
        parts.append("### Assistant\n")
        return "\n\n".join(parts)


def _rope_scaling(settings: dict) -> dict | None:
    kind = settings.get("rope_scaling", "none")
    factor = float(settings.get("rope_factor", 2.0) or 2.0)
    if kind == "linear":
        return {"rope_type": "linear", "factor": factor}
    if kind in {"yarn", "dynamic"}:
        return {"rope_type": "yarn", "factor": factor}
    return None


def _cache_kwargs(settings: dict, cuda: bool) -> tuple[dict, list[str]]:
    notes: list[str] = []
    impl = settings.get("cache_implementation", "dynamic")
    max_ctx = int(settings.get("max_context_tokens", 8192) or 8192)
    chunk = int(settings.get("prefill_chunk_size", 8192) or 8192)
    if impl == "dynamic" or not cuda:
        if impl != "dynamic" and not cuda:
            notes.append("CPU inference: using dynamic cache.")
        return {}, notes
    if impl == "offloaded":
        notes.append("KV cache offloaded to CPU RAM (dynamic).")
        return {"cache_implementation": "offloaded"}, notes
    if impl == "offloaded_static":
        notes.append("KV cache offloaded to CPU RAM (static, chunked prefill).")
        return {
            "cache_implementation": "offloaded_static",
            "max_cache_len": max_ctx,
            "prefill_chunk_size": chunk,
        }, notes
    if impl == "static":
        return {
            "cache_implementation": "static",
            "max_cache_len": max_ctx,
            "prefill_chunk_size": chunk,
        }, notes
    if impl == "quantized":
        kvq = settings.get("kv_quant", "int8")
        if kvq == "none":
            return {}, notes
        if kvq == "fp8":
            notes.append("FP8 KV requested; using INT8 KV quantization.")
            nbits = 8
        else:
            nbits = 8 if kvq == "int8" else 4
        notes.append(f"KV cache quantized to {nbits}-bit.")
        return {
            "cache_implementation": "quantized",
            "cache_config": {"nbits": nbits, "q_group_size": 64, "residual_length": 128},
        }, notes
    if impl == "sliding_window":
        return {
            "cache_implementation": "sliding_window",
            "cache_config": {"window_size": int(settings.get("sliding_window", 4096))},
        }, notes
    return {}, notes


class Engine:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.spec: LoadSpec | None = None
        self.notes: list[str] = []
        self.device: str = "cpu"
        self.cuda = False
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def nll_ids(self, prefix_ids, target_ids) -> float:
        import torch

        if not self.is_loaded():
            raise RuntimeError("no model loaded")
        t = list(target_ids)
        if not t:
            return 0.0
        p = list(prefix_ids)
        dev = torch.device(self.device)
        input_ids = torch.tensor([p + t], dtype=torch.long, device=dev)
        labels = torch.tensor([[-100] * len(p) + t], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, labels=labels)
        return float(out.loss.item())

    def embed_ids(self, ids):
        import torch

        if not self.is_loaded():
            raise RuntimeError("no model loaded")
        ids = list(ids)
        if not ids:
            return None
        dev = torch.device(self.device)
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][0]
        return h.mean(dim=0).float().cpu()

    def is_loaded(self) -> bool:
        return self.model is not None

    def status(self) -> str:
        if not self.is_loaded():
            return "no model loaded"
        s = self.spec
        bits = [s.ref.split("/")[-1].split("\\")[-1]]
        if s.adapter_path:
            bits.append(f'{s.adapter_arch}:{s.adapter_path.split("/")[-1].split(chr(92))[-1]}')
        if s.router_path:
            bits.append("router")
        return " + ".join(bits)

    def unload(self) -> None:
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            self.tokenizer = None
            self.spec = None
            self.notes = []
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def load(self, spec: LoadSpec) -> list[str]:
        from ..models.loading import load_base
        from ..mole.inject import MoLEConfig, inject_mole, set_stage
        from ..mole.store import apply_mole_weights, load_mole_config, load_mole_weights

        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            import torch

            model, tokenizer, cuda, quant, notes = load_base(
                spec.ref,
                spec.quant,
                attn_implementation=spec.settings.get("attn_implementation", "sdpa"),
                rope_scaling=_rope_scaling(spec.settings),
            )
            model.config.use_cache = True
            if spec.adapter_path and spec.adapter_arch == "mole":
                cfg = load_mole_config(spec.adapter_path)
                mole_cfg = MoLEConfig.from_dict(cfg["config"])
                inject_mole(model, mole_cfg, expert_dtype=torch.bfloat16 if cuda else torch.float32)
                src = spec.router_path or spec.adapter_path
                apply_mole_weights(model, load_mole_weights(src))
                set_stage(model, "none")
                notes.append(f"MoLE adapter applied ({len(cfg.get('layers', []))} layers) from {src}")
                if spec.router_path:
                    notes.append("Router snapshot overrides adapter router weights.")
            elif spec.adapter_path and spec.adapter_arch == "lora":
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, spec.adapter_path)
                notes.append("PEFT LoRA adapter applied.")
            model.eval()
            self.model = model
            self.tokenizer = tokenizer
            self.spec = spec
            self.notes = notes
            self.cuda = cuda
            self.device = "cuda" if cuda else "cpu"
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            return list(notes)

    def _generate_kwargs(self, input_ids, attention_mask, temperature: float, top_p: float, max_new_tokens: int) -> dict:
        settings = self.spec.settings if self.spec else {}
        max_ctx = int(settings.get("max_context_tokens", 8192) or 8192)
        prompt_len = input_ids.shape[1]
        budget = max_ctx - prompt_len
        if budget < 2:
            raise ValueError(f"Prompt ({prompt_len} tokens) exceeds max context ({max_ctx}).")
        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": min(int(max_new_tokens), budget),
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs["temperature"] = max(float(temperature), 1e-5)
            kwargs["top_p"] = float(top_p)
        cache_kwargs, cache_notes = _cache_kwargs(settings, self.cuda)
        self.notes.extend(cache_notes)
        kwargs.update(cache_kwargs)

        from transformers import StoppingCriteria, StoppingCriteriaList

        cancel_event = self._cancel

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **_kw) -> bool:
                return cancel_event.is_set()

        kwargs["stopping_criteria"] = StoppingCriteriaList([_Cancel()])
        return kwargs

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> Iterator[str]:
        if not self.is_loaded():
            raise RuntimeError("no model loaded")
        self._cancel.clear()

        prompt = format_messages(self.tokenizer, messages)
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)
        gen_kwargs = self._generate_kwargs(input_ids, attention_mask, temperature, top_p, max_new_tokens)
        had_cache = "cache_implementation" in gen_kwargs
        try:
            yield from self._stream(gen_kwargs)
        except Exception:
            if had_cache:
                gen_kwargs.pop("cache_implementation", None)
                gen_kwargs.pop("cache_config", None)
                self.notes.append("Selected KV cache failed; retried with dynamic cache.")
                yield from self._stream(gen_kwargs)
            else:
                raise

    def _stream(self, gen_kwargs: dict) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        error_box: list[BaseException] = []

        def runner() -> None:
            import torch

            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)
            finally:
                streamer.end()

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        t.join(timeout=5)
        if error_box:
            raise error_box[0]


ENGINE = Engine()
