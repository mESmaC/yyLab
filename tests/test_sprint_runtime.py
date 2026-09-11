from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from yylab.models import SprintConfig, SprintForCausalLM
from yylab.models.families import checkpoint_family, family_from_reference
from yylab.models.loading import load_base, scan_checkpoint


@pytest.fixture
def sprint_checkpoint(tmp_path):
    vocabulary = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocabulary.update({f"token{index}": index + 4 for index in range(60)})
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    config = SprintConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=1,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    torch.manual_seed(72)
    model = SprintForCausalLM(config).eval()
    folder = tmp_path / "sprint"
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder, model


def test_sprint_saved_checkpoint_registers_in_a_fresh_yylab_worker(sprint_checkpoint):
    folder, _ = sprint_checkpoint
    script = (
        "import sys; import yylab.models.loading; "
        "from transformers import AutoConfig, AutoModelForCausalLM; "
        "cfg = AutoConfig.from_pretrained(sys.argv[1]); "
        "model = AutoModelForCausalLM.from_pretrained(sys.argv[1]); "
        "assert cfg.model_type == 'sprint'; "
        "assert type(model).__name__ == 'SprintForCausalLM'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(folder)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_sprint_load_base_roundtrip_and_cached_generation(sprint_checkpoint, monkeypatch):
    folder, original = sprint_checkpoint
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, tokenizer, cuda, precision, notes = load_base(str(folder), "fp32")
    model.eval()
    assert isinstance(model, SprintForCausalLM)
    assert not cuda
    assert precision == "fp32"
    assert notes == []
    assert family_from_reference(folder) == "hf"
    assert checkpoint_family({"ref": str(folder), "meta_json": '{"model_family":"hf","kind":"sprint"}'}) == "hf"
    scanned = scan_checkpoint(folder)
    assert scanned["arch"] == "SprintForCausalLM"
    assert scanned["hidden_size"] == 64
    assert scanned["num_layers"] == 2
    inputs = tokenizer("token4 token7 token10", return_tensors="pt")
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        torch.testing.assert_close(model(**inputs).logits, original(**inputs).logits, rtol=0, atol=0)
        cached = model.generate(**inputs, do_sample=False, max_new_tokens=4, use_cache=True)
        uncached = model.generate(**inputs, do_sample=False, max_new_tokens=4, use_cache=False)
    assert torch.equal(cached, uncached)
    assert cached.shape[1] > inputs["input_ids"].shape[1]
    assert torch.equal(cached[:, : inputs["input_ids"].shape[1]], inputs["input_ids"])
    assert tokenizer.decode(cached[0])


def test_sprint_is_loadable_by_the_existing_export_path(sprint_checkpoint):
    from yylab.export.merge import load_base_full

    folder, original = sprint_checkpoint
    exported, tokenizer = load_base_full(str(folder), dtype=torch.float32)
    assert isinstance(exported, SprintForCausalLM)
    assert exported.get_input_embeddings().weight.device.type == "cpu"
    assert len(tokenizer) == 64
    assert exported.config.use_cache
    original_state = original.state_dict()
    for key, actual in exported.state_dict().items():
        torch.testing.assert_close(actual, original_state[key], rtol=0, atol=0)


@pytest.mark.parametrize("cache_kind", ["dynamic", "static"])
def test_sprint_cached_chunk_continuation_matches_full_prefix(sprint_checkpoint, cache_kind):
    from transformers.cache_utils import DynamicCache, StaticCache

    _, model = sprint_checkpoint
    tokens = torch.tensor([[2, 7, 8, 9, 10, 11], [2, 12, 13, 14, 15, 16]])
    cache = (
        StaticCache(config=model.config, max_cache_len=24)
        if cache_kind == "static"
        else DynamicCache(config=model.config)
    )
    with torch.inference_mode():
        expected = model(tokens, use_cache=False).logits
        for start, stop, masked in [(0, 3, False), (3, 5, True), (5, 6, False)]:
            actual = model(
                tokens[:, start:stop],
                past_key_values=cache,
                attention_mask=torch.ones_like(tokens[:, :stop]) if masked else None,
                use_cache=True,
            )
            torch.testing.assert_close(actual.logits, expected[:, start:stop], atol=1e-6, rtol=1e-5)
    assert cache.get_seq_length() == tokens.shape[1]
    with pytest.raises(ValueError, match="requires use_cache=True"):
        model(tokens[:, :1], past_key_values=cache, use_cache=False)
    assert cache.get_seq_length() == tokens.shape[1]


@pytest.mark.parametrize("cache_kind", ["dynamic", "static"])
def test_sprint_left_padded_generation_matches_individual_prompts(sprint_checkpoint, cache_kind):
    _, model = sprint_checkpoint
    batch = torch.tensor([[0, 0, 2, 4, 5], [2, 6, 7, 8, 9]])
    mask = (batch != 0).long()
    options = {"max_new_tokens": 4, "do_sample": False, "eos_token_id": None, "cache_implementation": cache_kind}
    with torch.inference_mode():
        together = model.generate(batch, attention_mask=mask, **options)
        for row, padding in enumerate([2, 0]):
            prompt = batch[row : row + 1, padding:]
            separate = model.generate(prompt, attention_mask=torch.ones_like(prompt), **options)
            assert torch.equal(together[row, batch.shape[1] :], separate[0, prompt.shape[1] :])


@pytest.mark.parametrize("additive", [False, True])
def test_sprint_four_dimensional_mask_preserves_causality(sprint_checkpoint, additive):
    _, model = sprint_checkpoint
    tokens = torch.tensor([[2, 4, 5, 6]])
    # Even a caller-supplied mask cannot unmask future tokens of a causal model.
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    if additive:
        mask = torch.zeros(1, 1, 4, 4)
    with torch.inference_mode():
        expected = model(tokens, use_cache=False).logits
        actual = model(tokens, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected)


def test_sprint_static_cache_rejects_overflow_without_mutating_it(sprint_checkpoint):
    from transformers.cache_utils import StaticCache

    _, model = sprint_checkpoint
    cache = StaticCache(config=model.config, max_cache_len=4)
    tokens = torch.tensor([[2, 4, 5]])
    with torch.inference_mode():
        model(tokens, past_key_values=cache, use_cache=True)
    with pytest.raises(ValueError, match="cache capacity"):
        model(tokens[:, :2], past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 3


@pytest.mark.parametrize(
    ("precision", "options", "message"),
    [
        ("fp32", {"rope_scaling": {"rope_type": "yarn", "factor": 2}}, "RoPE context extension"),
        ("fp32", {"max_position_embeddings": 64}, "RoPE context extension"),
        ("fp32", {"attn_implementation": "eager"}, "SDPA attention"),
        ("nf4", {}, "dense BF16, FP16 or FP32"),
    ],
)
def test_sprint_loader_rejects_unsupported_execution_options(
    sprint_checkpoint, monkeypatch, precision, options, message
):
    from transformers import AutoTokenizer

    from yylab.models import precision_capabilities

    folder, _ = sprint_checkpoint
    # Test the architecture policy independently of machine precision support.
    monkeypatch.setattr(precision_capabilities, "require_model_precision", lambda *_: None)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *_: pytest.fail("unsupported execution should fail before loading the tokenizer"),
    )
    with pytest.raises(ValueError, match=message):
        load_base(str(folder), precision, **options)
