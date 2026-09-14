"""CPU contracts for production KDA training, masks and usable checkpoints."""

from __future__ import annotations

import copy
import json
from unittest import mock

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from yylab.models.sprint import (
    SprintAttention,
    SprintConfig,
    SprintForCausalLM,
    estimate_parameter_breakdown,
    estimate_parameter_count,
    plan_sprint_model,
)
from yylab.models.sprint_kda import (
    SprintKDAAttention,
    SprintKDAGatedRMSNorm,
    SprintKDAShortConvolution,
    prepare_kda_layout,
    reference_kda,
)


@pytest.fixture(autouse=True)
def cpu_kernels():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(914)
    # Windows oneDNN does not implement every tiny BF16 depthwise backward.
    with torch.backends.mkldnn.flags(enabled=False):
        yield
    torch.set_num_threads(threads)


def tiny(kind="full", **kwargs):
    values = dict(vocab_size=47, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=2, max_position_embeddings=64, loss_backend="torch",
                  ple_type=kind, attention_type="kda", pad_token_id=0, bos_token_id=1, eos_token_id=2)
    values.update(kwargs)
    return SprintConfig(**values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_reference_recurrence_forward_and_gradients_ignore_outer_autocast(dtype):
    shape = (2, 17, 2, 8)
    values = [torch.randn(shape, dtype=dtype) for _ in range(4)]
    values += [torch.randn(2, 17, 2, dtype=dtype), torch.tensor([.3, .9]), torch.randn(16) - 3]
    probe = torch.randn(shape)

    def evaluate(autocast):
        inputs = [value.detach().clone().requires_grad_(True) for value in values]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            output = reference_kda(*inputs)
        gradients = torch.autograd.grad((output.float() * probe).sum(), inputs)
        return output, gradients

    output, gradients = evaluate(False)
    mixed_output, mixed_gradients = evaluate(True)
    torch.testing.assert_close(mixed_output, output, atol=0, rtol=0)
    for actual, expected in zip(mixed_gradients, gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.isfinite(actual).all()


def test_reference_gated_norm_has_no_autocast_contraction():
    norm = SprintKDAGatedRMSNorm(8, 1e-6)
    value, gate = [torch.randn(2, 7, 2, 8, dtype=torch.bfloat16) for _ in range(2)]
    expected = norm(value, gate, reference=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = norm(value, gate, reference=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_reference_convolution_preserves_precision_until_after_silu(dtype):
    conv = SprintKDAShortConvolution(3)
    inputs = torch.randn(2, 9, 3, dtype=dtype, requires_grad=True)
    probe = torch.randn(2, 9, 3)
    accumulation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    with torch.autocast("cpu", enabled=False):
        padded = torch.nn.functional.pad(inputs.to(accumulation_dtype).transpose(1, 2), (3, 0))
        full_precision = torch.nn.functional.conv1d(padded, conv.weight.to(accumulation_dtype), groups=3)
        expected = torch.nn.functional.silu(full_precision).transpose(1, 2).to(dtype)
    expected_grads = torch.autograd.grad((expected.float() * probe).sum(), (inputs, conv.weight))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = conv(inputs, reference=True)
    actual_grads = torch.autograd.grad((actual.float() * probe).sum(), (inputs, conv.weight))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
        assert torch.isfinite(actual_grad).all()
    if dtype == torch.bfloat16:
        # Ensure this case would catch the old weight/preactivation rounding.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prematurely_rounded = torch.nn.functional.silu(torch.nn.functional.conv1d(
                torch.nn.functional.pad(inputs.transpose(1, 2), (3, 0)), conv.weight, groups=3)).transpose(1, 2)
        assert not torch.equal(prematurely_rounded, expected)


@pytest.mark.parametrize("kind", ["none", "full", "qr"])
@pytest.mark.parametrize("mlp,tied", [("gelu", True), ("swiglu", False)])
def test_parameter_accounting_and_3_to_1_layers(kind, mlp, tied):
    cfg = tiny(kind, mlp_type=mlp, tie_word_embeddings=tied)
    model = SprintForCausalLM(cfg)
    counts = estimate_parameter_breakdown(cfg)
    assert counts["total"] == estimate_parameter_count(cfg) == sum(p.numel() for p in model.parameters())
    assert sum(v for k, v in counts.items() if k not in {"total", "ple"}) == counts["total"]
    assert counts["attention"] == sum(p.numel() for layer in model.layers for p in layer.self_attn.parameters())
    assert [isinstance(layer.self_attn, SprintKDAAttention) for layer in model.layers] == [True, True, True, False]
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied
    assert not cfg.use_cache


@pytest.mark.parametrize("target", [1_000_000, 10_000_000, 150_000_000, 1_000_000_000])
@pytest.mark.parametrize("kind", ["none", "full", "qr"])
@pytest.mark.parametrize("vocab", [257, 8192])
def test_kda_plans_budget_complete_model_and_retain_ffn(target, kind, vocab):
    cfg = plan_sprint_model(target, vocab, attention_type="kda", ple_type=kind)
    assert cfg.num_hidden_layers % 4 == 0
    assert cfg.intermediate_size >= cfg.hidden_size
    assert cfg.intermediate_size % (128 if target >= 10_000_000 else 16) == 0
    with torch.device("meta"):
        model = SprintForCausalLM(cfg)
    assert estimate_parameter_count(cfg) == sum(p.numel() for p in model.parameters()) <= target
    assert cfg.ple_type == kind and cfg.attention_type == "kda"


def test_balanced_and_legacy_plans_are_distinct_and_valid():
    balanced = plan_sprint_model(150_000_000, 8192, attention_type="kda", ple_type="full", profile="balanced")
    assert balanced.num_hidden_layers == 8 and balanced.intermediate_size >= balanced.hidden_size
    for target, shape, count in [(1_000_000, (96, 2, 352), 995808),
                                 (10_000_000, (512, 2, 1792), 9964032),
                                 (150_000_000, (1536, 6, 4352), 149442048),
                                 (1_000_000_000, (2816, 11, 10112), 998505728)]:
        cfg = plan_sprint_model(target, 8192)
        assert cfg.attention_type == "sdpa"
        assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size) == shape
        assert estimate_parameter_count(cfg) == count
    with pytest.raises(ValueError, match="vocabulary|minimum"):
        plan_sprint_model(10_000, 8192, attention_type="kda", ple_type="full")


@pytest.mark.parametrize("key,value", [("attention_type", "linear"), ("attention_type", []),
                                        ("kda_backend", "fake"), ("kda_backend", None),
                                        ("kda_disable_recompute", 1), ("kda_disable_recompute", "true")])
def test_invalid_options(key, value):
    with pytest.raises(ValueError, match=key):
        tiny(**{key: value})
    with pytest.raises(ValueError, match=key):
        plan_sprint_model(150_000_000, 8192, **{key: value})


def test_invalid_hybrid_shape_and_explicit_fla_cpu():
    with pytest.raises(ValueError, match="multiple of four"):
        tiny(num_hidden_layers=3)
    with pytest.raises(ValueError, match="head_dim"):
        tiny(hidden_size=512, num_attention_heads=1)
    with pytest.raises(ValueError, match="requires CUDA"):
        SprintForCausalLM(tiny(kda_backend="fla"))(torch.tensor([[1, 3]]))


@pytest.mark.parametrize("kind", ["none", "full", "qr"])
def test_causal_forward_all_parameter_gradients_and_no_hidden_layout_cache(kind):
    model = SprintForCausalLM(tiny(kind))
    initial_attributes = set(model.__dict__)
    ids = torch.tensor([[1, 3, 3, 8, 8, 9], [1, 5, 5, 7, 4, 4]])
    original = model(ids, labels=ids, return_logits=True)
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([17, 18])
    torch.testing.assert_close(original.logits[:, :4], model(changed).logits[:, :4], atol=1e-6, rtol=1e-5)
    original.loss.backward()
    assert torch.isfinite(original.loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    with mock.patch("yylab.models.sprint.prepare_kda_layout", wraps=prepare_kda_layout) as prepare:
        model(ids, attention_mask=torch.ones_like(ids))
        assert prepare.call_count == 1
        model(ids)
        assert prepare.call_count == 1  # Unmasked training skips all layout preparation.
    assert set(model.__dict__) == initial_attributes


@pytest.mark.parametrize("kind", ["none", "full", "qr"])
@pytest.mark.parametrize("left", [True, False])
def test_padding_matches_each_unpadded_sequence_and_explicit_positions(kind, left):
    model = SprintForCausalLM(tiny(kind)).eval()
    first = torch.tensor([[1, 3, 5, 7, 9, 11]])
    second = torch.tensor([[1, 4, 4, 8]])
    padded = torch.cat([torch.tensor([[0, 0]]), second], 1) if left else torch.cat([second, torch.tensor([[0, 0]])], 1)
    mask = torch.tensor([[1] * 6, [0, 0, 1, 1, 1, 1] if left else [1, 1, 1, 1, 0, 0]])
    ids = torch.cat([first, padded])
    positions = mask.long().cumsum(-1).sub(1).clamp_min(0)
    output = model(ids, attention_mask=mask, position_ids=positions).logits
    torch.testing.assert_close(output[0], model(first).logits[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(output[1, mask[1].bool()], model(second).logits[0], atol=1e-6, rtol=1e-5)
    dirty = ids.clone()
    dirty[~mask.bool()] = 37
    torch.testing.assert_close(model(dirty, attention_mask=mask).logits[mask.bool()], output[mask.bool()], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["full", "qr"])
@pytest.mark.parametrize("mask_kind", ["bool", "additive", "finite_min", "positions", "padding_gap"])
def test_segments_reset_convolution_recurrence_and_sdpa(kind, mask_kind):
    model = SprintForCausalLM(tiny(kind)).eval()
    first, second = torch.tensor([[1, 3, 3]]), torch.tensor([[1, 7, 5]])
    ids = torch.cat([first, second], 1)
    kwargs = {}
    if mask_kind == "positions":
        kwargs["position_ids"] = torch.tensor([[0, 1, 2, 0, 1, 2]])
    elif mask_kind == "padding_gap":
        ids = torch.cat([first, torch.tensor([[0]]), second], 1)
        kwargs["attention_mask"] = torch.tensor([[1, 1, 1, 0, 1, 1, 1]])
    else:
        mask = torch.zeros(6, 6, dtype=torch.bool)
        mask[:3, :3] = mask[3:, 3:] = torch.ones(3, 3, dtype=torch.bool).tril()
        if mask_kind != "bool":
            minimum = float("-inf") if mask_kind == "additive" else torch.finfo(torch.float32).min
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, minimum)
        kwargs["attention_mask"] = mask[None, None]
    output = model(ids, **kwargs).logits
    torch.testing.assert_close(output[:, :3], model(first).logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(output[:, -3:], model(second).logits, atol=1e-6, rtol=1e-5)
    ids[:, :3] = 39
    torch.testing.assert_close(model(ids, **kwargs).logits[:, -3:], output[:, -3:], atol=1e-6, rtol=1e-5)


def test_mask_rejection_does_not_silently_discard_visibility():
    model = SprintForCausalLM(tiny())
    ids = torch.tensor([[1, 2, 3, 4]])
    masks = [torch.tensor([[1., 1., .5, 1.]]), torch.ones(1, 3), torch.zeros(1, 1, 4, 4).fill_(-.2)]
    gap = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    gap[0, 0, 3, 1] = False
    masks.append(gap)
    heads = torch.ones(1, 2, 4, 4, dtype=torch.bool).tril()
    heads[0, 1, 3, 2] = False
    masks.append(heads)
    for mask in masks:
        with pytest.raises(ValueError, match="KDA"):
            model(ids, attention_mask=mask)
    with pytest.raises(ValueError, match="position_ids"):
        model(ids, position_ids=torch.tensor([[0., 1., 2., 3.]]))
    with pytest.raises(ValueError, match="input_ids"):
        model(inputs_embeds=model.embed_tokens(ids))


def test_all_padding_returns_finite_zero_loss_and_zero_gradients():
    model = SprintForCausalLM(tiny())
    ids = torch.zeros(2, 4, dtype=torch.long)
    output = model(ids, attention_mask=ids, labels=torch.full_like(ids, -100), return_logits=True)
    assert output.loss.item() == 0 and torch.isfinite(output.logits).all()
    output.loss.backward()
    assert all(parameter.grad is not None and not parameter.grad.count_nonzero() for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["full", "qr"])
def test_checkpoint_forward_backward_parity_with_segments(kind):
    eager = SprintForCausalLM(tiny(kind)).train()
    checked = copy.deepcopy(eager)
    checked.gradient_checkpointing_enable()
    ids = torch.tensor([[0, 1, 3, 5, 7, 9], [1, 4, 6, 0, 1, 7]])
    mask = (ids != 0).long()
    labels = ids.masked_fill(~mask.bool(), -100)
    a, b = [model(ids, attention_mask=mask, labels=labels).loss for model in (eager, checked)]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.backward()
    b.backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), checked.named_parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5, msg=name)


@pytest.mark.parametrize("kind", ["none", "full", "qr"])
def test_bf16_conversion_preserves_gate_values_and_finite_training(kind):
    model = SprintForCausalLM(tiny(kind))
    gates = {name: (id(p), p.detach().clone()) for name, p in model.named_parameters() if name.endswith(("A_log", "dt_bias"))}
    model.bfloat16()
    for name, p in model.named_parameters():
        if name in gates:
            assert p.dtype == torch.float32 and id(p) == gates[name][0]
            assert torch.equal(p, gates[name][1])
        else:
            assert p.dtype == torch.bfloat16
    ids = torch.tensor([[0, 1, 3, 5], [1, 7, 7, 9]])
    mask = (ids != 0).long()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(ids, attention_mask=mask, labels=ids.masked_fill(~mask.bool(), -100)).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.dtype == p.dtype for p in model.parameters())


@pytest.mark.parametrize("kind", ["none", "full", "qr"])
def test_saved_checkpoint_reload_generation_and_beams(tmp_path, kind):
    model = SprintForCausalLM(tiny(kind)).eval()
    ids = torch.tensor([[0, 1, 3, 5], [1, 7, 7, 9]])
    mask = (ids != 0).long()
    model.save_pretrained(tmp_path)
    restored, info = SprintForCausalLM.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert restored.config.attention_type == "kda" and restored.config.ple_type == kind
    assert restored.embed_tokens.weight is restored.lm_head.weight
    torch.testing.assert_close(model(ids, attention_mask=mask).logits, restored(ids, attention_mask=mask).logits, atol=0, rtol=0)
    generation = restored.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False,
                                   use_cache=True, output_scores=True, return_dict_in_generate=True)
    assert generation.sequences.shape == (2, 7) and len(generation.scores) == 3
    assert generation.past_key_values is None
    assert torch.isfinite(torch.stack(generation.scores)).all()
    uncached = restored.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=False)
    assert torch.equal(generation.sequences, uncached)
    beams = restored.generate(ids, attention_mask=mask, max_new_tokens=2, num_beams=2, use_cache=True)
    assert beams.shape == (2, 6)
    auto = AutoModelForCausalLM.from_pretrained(tmp_path)
    assert isinstance(auto.layers[0].self_attn, SprintKDAAttention)
    with pytest.raises(ValueError, match="external caches"):
        restored.generate(ids, past_key_values=DynamicCache(), max_new_tokens=1)
    with pytest.raises(ValueError, match="use_cache=False"):
        restored(ids, use_cache=True)


def test_bf16_loading_preserves_original_fp32_gate_values(tmp_path):
    model = SprintForCausalLM(tiny()).eval()
    model.save_pretrained(tmp_path)
    loaded = SprintForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16)
    for name, parameter in loaded.named_parameters():
        if name.endswith(("A_log", "dt_bias")):
            assert parameter.dtype == torch.float32
            assert torch.equal(parameter, dict(model.named_parameters())[name])
        else:
            assert parameter.dtype == torch.bfloat16


@pytest.mark.parametrize("corruption", ["missing_gate", "missing_qkv", "mismatched_gate", "changed_to_sdpa"])
def test_incompatible_kda_checkpoint_cannot_load_silently(tmp_path, corruption):
    model = SprintForCausalLM(tiny())
    model.save_pretrained(tmp_path)
    weights_path = tmp_path / "model.safetensors"
    state = load_file(weights_path)
    if corruption == "changed_to_sdpa":
        path = tmp_path / "config.json"
        config = json.loads(path.read_text())
        config["attention_type"] = "sdpa"
        path.write_text(json.dumps(config))
    elif corruption == "missing_gate":
        del state["layers.0.self_attn.A_log"]
    elif corruption == "missing_qkv":
        del state["layers.0.self_attn.qkv_proj.weight"]
    else:
        state["layers.0.self_attn.dt_bias"] = state["layers.0.self_attn.dt_bias"][:1]
    if corruption != "changed_to_sdpa":
        save_file(state, weights_path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="KDA checkpoint"):
        SprintForCausalLM.from_pretrained(tmp_path, ignore_mismatched_sizes=True)


def test_gate_initialization_and_old_sdpa_checkpoint(tmp_path):
    model = SprintForCausalLM(tiny())
    for layer in model.layers[:3]:
        attn = layer.self_attn
        assert (attn.A_log.exp() >= 1).all() and (attn.A_log.exp() <= 16).all()
        dt = torch.nn.functional.softplus(attn.dt_bias)
        assert (dt >= .001).all() and (dt <= .1).all()
        assert torch.equal(attn.g_proj[1].bias, torch.zeros_like(attn.g_proj[1].bias))
        assert attn.qkv_conv.weight.count_nonzero() == attn.qkv_conv.weight.numel()
    legacy = SprintForCausalLM(tiny(attention_type="sdpa", num_hidden_layers=3))
    legacy.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    for key in ("attention_type", "kda_backend", "kda_disable_recompute"):
        config.pop(key)
    config_path.write_text(json.dumps(config))
    restored = SprintForCausalLM.from_pretrained(tmp_path)
    assert restored.config.attention_type == "sdpa" and all(isinstance(layer.self_attn, SprintAttention) for layer in restored.layers)
    assert restored.state_dict().keys() == legacy.state_dict().keys()
