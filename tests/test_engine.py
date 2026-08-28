from __future__ import annotations

from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM

from yylab.infer.engine import Engine, LoadSpec, format_messages
from yylab.mole.inject import MoLEConfig, inject_mole
from yylab.mole.store import save_mole


class FakeTok:
    def __init__(self) -> None:
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = None

    def __call__(self, texts, return_tensors=None, **_):
        if isinstance(texts, str):
            ids = [[(ord(c) % 100) + 2 for c in texts[:64]]]
        else:
            ids = [[(ord(c) % 100) + 2 for c in t[:64]] for t in texts]
        if return_tensors == "pt":
            import torch as _t

            mask = _t.ones(len(ids), max(len(i) for i in ids), dtype=_t.long)
            padded = [i + [1] * (mask.shape[1] - len(i)) for i in ids]
            return {"input_ids": _t.tensor(padded), "attention_mask": mask}
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        raise NotImplementedError("no template")

    def decode(self, ids, **_):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


def _tiny_llama_dir(tmp_path: Path) -> Path:
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    out = tmp_path / "tiny_llama"
    model.save_pretrained(out)
    return out


def _fake_loader(tmp_path: Path, monkeypatch):
    from transformers import AutoModelForCausalLM

    import yylab.infer.engine as engine_mod

    model_dir = _tiny_llama_dir(tmp_path)

    def fake_load(ref, quant, attn_implementation="sdpa", rope_scaling=None):
        m = AutoModelForCausalLM.from_pretrained(ref)
        m.config.use_cache = True
        return m, FakeTok(), False, quant, []

    monkeypatch.setattr(engine_mod, "load_base", fake_load, raising=False)
    import yylab.models.loading as loading_mod

    monkeypatch.setattr(loading_mod, "load_base", fake_load, raising=False)
    return model_dir


def test_chat_template_fallback():
    out = format_messages(FakeTok(), [{"role": "user", "content": "hi"}])
    assert "### User" in out and "### Assistant" in out


def test_engine_generate_plain(tmp_path, monkeypatch):
    model_dir = _fake_loader(tmp_path, monkeypatch)
    eng = Engine()
    notes = eng.load(LoadSpec(ref=str(model_dir), quant="bf16"))
    assert eng.is_loaded()
    assert isinstance(notes, list)
    chunks = list(eng.generate_stream([{"role": "user", "content": "hello"}], temperature=0.0, max_new_tokens=8))
    text = "".join(chunks)
    assert isinstance(text, str)
    eng.unload()
    assert not eng.is_loaded()


def test_engine_with_mole_adapter(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    model_dir = _fake_loader(tmp_path, monkeypatch)

    base = AutoModelForCausalLM.from_pretrained(model_dir)
    mole_cfg = MoLEConfig(num_experts=2, rank=4, alpha=8.0, top_k=2, target_modules=("q_proj", "v_proj"))
    inject_mole(base, mole_cfg)
    adp_dir = tmp_path / "adp"
    save_mole(adp_dir, base, mole_cfg, base_ref=str(model_dir))

    eng = Engine()
    notes = eng.load(
        LoadSpec(ref=str(model_dir), quant="bf16", adapter_path=str(adp_dir), adapter_arch="mole")
    )
    assert any("MoLE" in n for n in notes)
    chunks = list(eng.generate_stream([{"role": "user", "content": "hi"}], temperature=0.8, top_p=0.9, max_new_tokens=6))
    assert isinstance("".join(chunks), str)


def test_engine_context_budget(tmp_path, monkeypatch):
    model_dir = _fake_loader(tmp_path, monkeypatch)
    eng = Engine()
    eng.load(LoadSpec(ref=str(model_dir), quant="bf16", settings={"max_context_tokens": 8}))
    try:
        list(eng.generate_stream([{"role": "user", "content": "x" * 40}], max_new_tokens=8))
        raise AssertionError("expected context budget error")
    except ValueError as e:
        assert "context" in str(e).lower()
