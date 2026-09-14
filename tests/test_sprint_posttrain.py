"""Saved Sprint PLE/KDA compatibility with CPU SFT, FTPO, adapters and chat."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from yylab.models import SprintConfig, SprintForCausalLM
from yylab.models.loading import load_base, supports_generation_cache
from yylab.train.antidoom import FTPOPair, RepeatHit, ftpo_objective
from yylab.workers.antidoom import (
    TARGET_MODULES,
    _generate_batch,
    _training_batch,
    evaluate_causal_nll,
    resolve_target_modules,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(78)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture(params=["full", "qr"])
def saved_kda_ple(request, tmp_path):
    vocabulary = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocabulary.update({f"token{i}": i + 4 for i in range(28)})
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>",
                                        unk_token="<unk>", bos_token="<bos>", eos_token="<eos>")
    config = SprintConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=4, num_attention_heads=2,
                          max_position_embeddings=64, loss_backend="torch",
                          pad_token_id=0, bos_token_id=2, eos_token_id=3,
                          ple_type=request.param, ple_lanes=4,
                          attention_type="kda", kda_backend="auto", use_cache=False)
    model = SprintForCausalLM(config).eval()
    assert hasattr(model.layers[0].self_attn, "A_log"), "test must exercise actual production KDA"
    path = tmp_path / "checkpoint"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path, model, tokenizer


def test_saved_kda_ple_loads_for_continued_sft_and_resaves(saved_kda_ple, tmp_path):
    path, original, _ = saved_kda_ple
    model, tokenizer, cuda, precision, notes = load_base(str(path), "fp32")
    assert not cuda and precision == "fp32"
    assert model.config.attention_type == "kda" and model.config.ple_type == original.config.ple_type
    assert not model.config.use_cache and not supports_generation_cache(model)
    assert any("full prefix" in note for note in notes)
    for name, parameter in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], parameter, atol=0, rtol=0)
    ids = torch.tensor([[2, 7, 8, 9, 10], [0, 2, 11, 12, 13]])
    mask = ids.ne(0).long()
    labels = ids.masked_fill(~mask.bool(), -100)
    before = model.ple_refresh_layers[0].embedding.weight.detach().clone() if model.config.ple_type == "full" else model.ple_refresh_layers[0].remainder_embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    output = model(ids, attention_mask=mask, labels=labels, use_cache=False)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    current = model.ple_refresh_layers[0].embedding.weight if model.config.ple_type == "full" else model.ple_refresh_layers[0].remainder_embedding.weight
    assert not torch.equal(before, current)
    continued = tmp_path / "continued"
    model.save_pretrained(continued)
    tokenizer.save_pretrained(continued)
    restored = AutoModelForCausalLM.from_pretrained(continued).eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, attention_mask=mask).logits,
                                   model(ids, attention_mask=mask).logits, atol=0, rtol=0)


def pair(prompt, rejected):
    return FTPOPair(batch_index=0, sequence_index=0, return_index=0,
                    prompt_ids=tuple(prompt), chosen_ids=(14, 15), rejected_token_id=rejected,
                    source_prompt_ids=tuple(prompt), generated_ids=(rejected,) * 4,
                    rejected_generation_index=1, repetition=RepeatHit(0, 4, 1, 4, "x"),
                    chosen_surfaces=("token10", "token11"), rejected_surface="token5", generation_text="xxxx")


def test_saved_kda_ple_ftpo_adapter_roundtrip_and_mining_generation(saved_kda_ple, tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model

    path, _, _ = saved_kda_ple
    base, tokenizer, _, _, _ = load_base(str(path), "fp32")
    target_modules = resolve_target_modules(base, TARGET_MODULES)
    assert "qkv_proj" in target_modules
    lora = LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=target_modules, ensure_weight_tying=False)
    model = get_peft_model(base, lora, autocast_adapter_dtype=False)
    batch = _training_batch([pair([2, 7], 9), pair([2, 8, 9], 10)],
                            pad_token_id=0, device=torch.device("cpu"))
    inputs = {key: batch[key] for key in ("input_ids", "attention_mask", "position_ids")}
    model.train()
    policy = model(**inputs, use_cache=False, return_dict=True).logits[:, -1]
    with model.disable_adapter(), torch.no_grad():
        reference = model(**inputs, use_cache=False, return_dict=True).logits[:, -1]
    objective = ftpo_objective(policy, reference, batch["chosen_ids"], batch["rejected_token_ids"], batch["chosen_mask"])
    assert torch.isfinite(objective.loss)
    objective.loss.backward()
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
    assert any("qkv_proj" in name and "lora_B" in name and parameter.grad.abs().max() > 0
               for name, parameter in trainable.items())
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    torch.optim.SGD(trainable.values(), lr=0.1).step()
    model.eval()
    adapter_path = tmp_path / "ftpo-adapter"
    model.save_pretrained(adapter_path, save_embedding_layers=False)
    base_again, _, _, _, _ = load_base(str(path), "fp32")
    restored = PeftModel.from_pretrained(base_again, adapter_path, is_trainable=False,
                                         autocast_adapter_dtype=False).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs, use_cache=False).logits,
                                   restored(**inputs, use_cache=False).logits, atol=0, rtol=0)
    prompts = [[2, 7], [2, 8, 9]]
    generation_inputs, mask, generated = _generate_batch(
        restored, tokenizer, prompts, {"max_new_tokens": 2}, output_scores=True)
    assert generation_inputs.shape == mask.shape == (2, 3)
    assert generated.sequences.shape[0] == 2 and generated.sequences.shape[1] > 3
    assert torch.equal(generated.sequences[:, :3], generation_inputs)
    assert generated.scores and all(torch.isfinite(score).all() for score in generated.scores)
    assert getattr(generated, "past_key_values", None) is None
    nll = evaluate_causal_nll(restored, tokenizer, prompts, {"nll_eval_prompts": 2, "nll_batch_size": 2})
    assert nll["tokens"] == 3 and nll["nll"] > 0


def test_kda_chat_avoids_constructing_an_external_cache(saved_kda_ple, monkeypatch):
    from yylab.infer import engine as inference

    path, model, tokenizer = saved_kda_ple
    engine = inference.Engine()
    engine.model, engine.tokenizer = model, tokenizer
    engine.spec = inference.LoadSpec(ref=str(path), quant="fp32", model_family="hf",
                                     settings={"max_context_tokens": 64, "cache_implementation": "quantized_offloaded",
                                               "kv_quant": "int4"})
    def fail_cache(*args, **kwargs):
        raise AssertionError("KDA must not select a Transformers cache")
    monkeypatch.setattr(inference, "_cache_kwargs", fail_cache)
    ids = torch.tensor([[2, 7, 8]])
    kwargs = engine._generate_kwargs(ids, torch.ones_like(ids), 0.0, 1.0, 2)
    assert kwargs["use_cache"] is False
    assert "past_key_values" not in kwargs and "cache_implementation" not in kwargs
    assert any("full prefix" in note for note in engine.notes)
    with torch.no_grad():
        generated = model.generate(**kwargs)
    assert generated.shape[1] > ids.shape[1]


def test_antidoom_target_mapping_preserves_explicit_packed_and_other_models():
    sprint = SimpleNamespace(config=SimpleNamespace(model_type="sprint"))
    other = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    assert resolve_target_modules(other, TARGET_MODULES) == list(TARGET_MODULES)
    assert resolve_target_modules(sprint, ["qkv_proj", "o_proj", "lm_head"]) == ["qkv_proj", "o_proj", "lm_head"]
    resolved = resolve_target_modules(sprint, TARGET_MODULES)
    assert resolved.count("qkv_proj") == 1 and not set(resolved).intersection({"q_proj", "k_proj", "v_proj"})
    with pytest.raises(ValueError, match="packs Q/K/V"):
        resolve_target_modules(sprint, ["q_proj", "lm_head"])


def test_cache_policy_preserves_ordinary_sprint_and_hf_defaults():
    assert supports_generation_cache(SimpleNamespace(model_type="sprint", attention_type="sdpa"))
    assert supports_generation_cache(SimpleNamespace(model_type="sprint"))
    assert supports_generation_cache(SimpleNamespace(model_type="qwen3", attention_type="kda"))
    assert not supports_generation_cache(SimpleNamespace(config=SimpleNamespace(model_type="sprint", attention_type="kda")))
