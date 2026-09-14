"""Production Sprint PLE accounting, training, caching and checkpoint contracts."""

from __future__ import annotations

import copy
import json
from unittest import mock

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.cache_utils import DynamicCache, StaticCache

from yylab.models.sprint import (
    SprintConfig,
    SprintForCausalLM,
    estimate_parameter_breakdown,
    estimate_parameter_count,
    plan_sprint_model,
)
from yylab.models.sprint_ple import qr_lane_moduli


@pytest.fixture(autouse=True)
def cpu_settings():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(171)
    yield
    torch.set_num_threads(previous)


def tiny(kind="qr", **kwargs):
    options = dict(vocab_size=47, hidden_size=32, intermediate_size=64,
                   num_hidden_layers=3, num_attention_heads=2,
                   max_position_embeddings=64, loss_backend="torch",
                   ple_type=kind, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    options.update(kwargs)
    return SprintConfig(**options)


@pytest.mark.parametrize("kind", ["none", "full", "qr"])
@pytest.mark.parametrize("gated,tied", [("gelu", True), ("swiglu", False)])
def test_exact_cpu_parameter_breakdown(kind, gated, tied):
    config = tiny(kind, mlp_type=gated, tie_word_embeddings=tied, ple_lanes=7)
    model = SprintForCausalLM(config)
    counts = estimate_parameter_breakdown(config)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert counts["total"] == estimate_parameter_count(config) == actual
    assert sum(value for key, value in counts.items() if key not in {"total", "ple"}) == actual
    assert counts["ple"] == counts["ple_tables"] + counts["ple_controls"]
    assert counts["ple"] == sum(parameter.numel() for layer in model.ple_refresh_layers for parameter in layer.parameters())
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied
    assert len(model.ple_refresh_layers) == (0 if kind == "none" else config.num_hidden_layers - 1)
    if kind == "full":
        assert counts["ple_tables"] == 2 * 47 * 32  # One full table, regardless of QR lane setting.


@pytest.mark.parametrize("kind", ["full", "qr"])
@pytest.mark.parametrize("target", [1_000_000, 10_000_000, 150_000_000, 1_000_000_000])
@pytest.mark.parametrize("vocab", [257, 8192])
def test_continuous_plans_include_requested_ple_without_oversizing(kind, target, vocab):
    config = plan_sprint_model(target, vocab, ple_type=kind)
    with torch.device("meta"):
        model = SprintForCausalLM(config)
    assert estimate_parameter_count(config) == sum(parameter.numel() for parameter in model.parameters())
    assert estimate_parameter_count(config) <= target
    assert config.ple_type == kind and config.ple_lanes == 4
    assert config.vocab_size == vocab
    assert config.num_hidden_layers >= 2 and model.ple_refresh_layers
    assert config.intermediate_size % (128 if target >= 10_000_000 else 16) == 0


@pytest.mark.parametrize("kind,ffn,count", [("full", 896, 148655621), ("qr", 3968, 147985945)])
def test_150m_keeps_measured_backbone_and_pays_for_ple_with_ffn(kind, ffn, count):
    config = plan_sprint_model(150_000_000, 8192, ple_type=kind)
    assert (config.hidden_size, config.num_hidden_layers, config.intermediate_size) == (1536, 6, ffn)
    assert estimate_parameter_count(config) == count


def test_balanced_alignment_and_smaller_shape_fallback_keep_ple():
    baseline = plan_sprint_model(1_000_000, 8192)
    full = plan_sprint_model(1_000_000, 8192, ple_type="full")
    assert full.hidden_size < baseline.hidden_size
    assert full.num_hidden_layers >= 2
    assert full.ple_type == "full"
    assert full.intermediate_size % 16 == 0
    balanced = plan_sprint_model(150_000_000, 8192, ple_type="qr", profile="balanced", ple_lanes=3)
    assert balanced.intermediate_size % 16 == 0 and balanced.ple_lanes == 3
    assert estimate_parameter_count(balanced) <= 150_000_000
    with pytest.raises(ValueError, match="minimum|vocabulary"):
        plan_sprint_model(100_000, 8192, ple_type="full")
    with pytest.raises(ValueError, match="minimum"):
        plan_sprint_model(1_000_000, 32768, ple_type="full")


@pytest.mark.parametrize("target,shape,count", [
    (1_000_000, (96, 2, 352), 995808),
    (10_000_000, (512, 2, 1792), 9964032),
    (150_000_000, (1536, 6, 4352), 149442048),
    (1_000_000_000, (2816, 11, 10112), 998505728),
])
def test_none_preserves_existing_shapes_and_weight_keys(target, shape, count):
    config = plan_sprint_model(target, 8192)
    explicit = plan_sprint_model(target, 8192, ple_type="none")
    assert config.to_dict() == explicit.to_dict()
    assert (config.hidden_size, config.num_hidden_layers, config.intermediate_size) == shape
    assert estimate_parameter_count(config) == count
    with torch.device("meta"):
        model = SprintForCausalLM(config)
    assert not any(name.startswith("ple_") for name in model.state_dict())


@pytest.mark.parametrize("option,value", [
    ("ple_type", "fake_qat"), ("ple_type", "QR"), ("ple_type", None), ("ple_type", []),
    ("ple_lanes", 0), ("ple_lanes", -1), ("ple_lanes", True), ("ple_lanes", 4.0),
])
def test_invalid_ple_options_rejected_by_config_and_planner(option, value):
    with pytest.raises(ValueError, match=option):
        tiny(**{option: value})
    with pytest.raises(ValueError, match=option):
        plan_sprint_model(1_000_000, 8192, **{option: value})


@pytest.mark.parametrize("kind", ["full", "qr"])
def test_custom_initialization_and_lookup_composition_with_repeated_ids(kind):
    model = SprintForCausalLM(tiny(kind, num_hidden_layers=4))
    layer = model.ple_refresh_layers[0]
    assert layer.gate.item() == .5
    ids = torch.tensor([[0, 7, 7, 14], [46, 1, 1, 7]])
    probe = torch.randn(2, 4, 32)
    if kind == "full":
        actual = layer(ids)
        expected = torch.stack([torch.stack([layer.embedding.weight[int(token)] for token in row]) for row in ids]) * layer.gate
    else:
        assert torch.count_nonzero(layer.lane_logits) == 0
        assert abs(layer.quotient_embedding.weight.mean().item() - 1) < .01
        assert .01 < layer.quotient_embedding.weight.std().item() < .03
        indices = model.ple_qr_layout.prepare_indices(ids)
        actual = layer(ids, qr_indices=indices)
        lanes, remainder_offset, quotient_offset = [], 0, 0
        weights = layer.lane_logits.softmax(0)
        for lane, modulus in enumerate(qr_lane_moduli(47)):
            lanes.append(torch.stack([torch.stack([
                layer.remainder_embedding.weight[remainder_offset + int(token) % modulus]
                * layer.quotient_embedding.weight[quotient_offset + int(token) // modulus]
                for token in row]) for row in ids]) * weights[lane])
            remainder_offset += modulus
            quotient_offset += (47 + modulus - 1) // modulus
        expected = torch.stack(lanes).sum(0) * layer.gate
    parameters = list(layer.parameters())
    actual_grads = torch.autograd.grad((actual * probe).sum(), parameters)
    expected_grads = torch.autograd.grad((expected * probe).sum(), parameters)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        assert not left.is_sparse and torch.isfinite(left).all()
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)


def test_qr_addresses_are_prepared_once_and_not_retained():
    model = SprintForCausalLM(tiny("qr", num_hidden_layers=4))
    ids = torch.tensor([[0, 1, 7, 7, 46]])
    original_keys = set(vars(model))
    seen, patches = [], []
    for layer in model.ple_refresh_layers:
        original = layer.forward
        def observe(*args, _original=original, **kwargs):
            seen.append(kwargs["qr_indices"])
            return _original(*args, **kwargs)
        patches.append(mock.patch.object(layer, "forward", side_effect=observe))
    with mock.patch.object(model.ple_qr_layout, "prepare_indices", wraps=model.ple_qr_layout.prepare_indices) as prepare:
        for patch in patches:
            patch.start()
        try:
            model(ids, use_cache=False)
            model(ids.clone(), use_cache=False)
        finally:
            for patch in reversed(patches):
                patch.stop()
        assert prepare.call_count == 2
    assert all(value is seen[0] for value in seen[:3])
    assert all(value is seen[3] for value in seen[3:])
    assert seen[0] is not seen[3]
    assert set(vars(model)) == original_keys


@pytest.mark.parametrize("kind", ["full", "qr"])
def test_zero_gates_match_plain_forward_and_backbone_gradients(kind):
    plain = SprintForCausalLM(tiny("none"))
    injected = SprintForCausalLM(tiny(kind))
    status = injected.load_state_dict(plain.state_dict(), strict=False)
    assert status.missing_keys and all(name.startswith("ple_") for name in status.missing_keys)
    with torch.no_grad():
        for layer in injected.ple_refresh_layers:
            layer.gate.zero_()
    ids = torch.randint(0, 47, (2, 9))
    expected = plain(ids, labels=ids, return_logits=True)
    actual = injected(ids, labels=ids, return_logits=True)
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    actual.loss.backward()
    expected.loss.backward()
    for name, parameter in plain.named_parameters():
        torch.testing.assert_close(dict(injected.named_parameters())[name].grad, parameter.grad, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["full", "qr"])
def test_training_checkpointed_gradients_and_cpu_bf16_residual(kind):
    eager = SprintForCausalLM(tiny(kind)).train()
    checkpointed = copy.deepcopy(eager)
    checkpointed.gradient_checkpointing_enable()
    ids = torch.randint(0, 47, (2, 9))
    left, right = eager(ids, labels=ids).loss, checkpointed(ids, labels=ids).loss
    left.backward()
    right.backward()
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    for a, b in zip(eager.parameters(), checkpointed.parameters(), strict=True):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    eager.zero_grad(set_to_none=True)
    dtypes = []
    handles = [layer.register_forward_pre_hook(lambda _module, args: dtypes.append(args[0].dtype)) for layer in eager.layers]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = eager(ids, labels=ids).loss
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert dtypes == [torch.bfloat16] * eager.config.num_hidden_layers
    assert all(parameter.dtype == parameter.grad.dtype == torch.float32 for parameter in eager.parameters())
    assert torch.isfinite(loss)
    direct = copy.deepcopy(eager).bfloat16()
    assert direct(ids, labels=ids).loss.isfinite()


@pytest.mark.parametrize("kind", ["full", "qr"])
@pytest.mark.parametrize("cache_kind", ["dynamic", "static"])
def test_cached_chunks_and_left_padding_match_full_sequence(kind, cache_kind):
    model = SprintForCausalLM(tiny(kind)).eval()
    ids = torch.tensor([[0, 0, 1, 7, 8, 9], [1, 5, 6, 7, 8, 9]])
    mask = (ids != 0).long()
    cache = StaticCache(config=model.config, max_cache_len=16) if cache_kind == "static" else DynamicCache(config=model.config)
    with torch.inference_mode():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        for start, stop in [(0, 3), (3, 5), (5, 6)]:
            actual = model(ids[:, start:stop], attention_mask=mask[:, :stop], past_key_values=cache, use_cache=True)
            torch.testing.assert_close(actual.logits, full[:, start:stop], atol=1e-6, rtol=1e-5)
        unpadded = model(ids[:1, 2:], attention_mask=torch.ones(1, 4), use_cache=False).logits
    torch.testing.assert_close(full[:1, 2:], unpadded, atol=1e-6, rtol=1e-5)
    assert cache.get_seq_length() == 6


@pytest.mark.parametrize("kind", ["full", "qr"])
def test_beam_search_cached_and_uncached_match(kind):
    model = SprintForCausalLM(tiny(kind)).eval()
    ids = torch.tensor([[0, 0, 1, 7, 8], [1, 5, 6, 7, 8]])
    mask = (ids != 0).long()
    options = dict(attention_mask=mask, num_beams=3, do_sample=False, max_new_tokens=4, eos_token_id=None)
    with torch.inference_mode():
        expected = model.generate(ids, use_cache=False, **options)
        cached = model.generate(ids, use_cache=True, **options)
    assert torch.equal(cached, expected)


@pytest.mark.parametrize("kind", ["full", "qr"])
def test_no_future_leakage_and_inputs_embeds_rejected(kind):
    model = SprintForCausalLM(tiny(kind)).eval()
    ids = torch.randint(0, 47, (2, 9))
    changed = ids.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 47
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        actual = model(changed, use_cache=False).logits
    torch.testing.assert_close(actual[:, :5], expected[:, :5], atol=0, rtol=0)
    with pytest.raises(ValueError, match="requires input_ids"):
        model(inputs_embeds=model.embed_tokens(ids))
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=model.embed_tokens(ids))


@pytest.mark.parametrize("kind", ["none", "full", "qr"])
def test_checkpoint_roundtrip_preserves_config_weights_aliases_and_outputs(tmp_path, kind):
    model = SprintForCausalLM(tiny(kind, ple_lanes=3)).eval()
    model.save_pretrained(tmp_path)
    loaded, info = SprintForCausalLM.from_pretrained(tmp_path, local_files_only=True, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert loaded.config.ple_type == kind and loaded.config.ple_lanes == 3
    assert loaded.embed_tokens.weight is loaded.lm_head.weight
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    ids = torch.randint(0, 47, (2, 9))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, model(ids, use_cache=False).logits, atol=0, rtol=0)
    # HF BF16 loading uses meta initialization; quotient tables must load their
    # saved values rather than being reset to generic embedding initialization.
    bf16 = SprintForCausalLM.from_pretrained(tmp_path, local_files_only=True, dtype=torch.bfloat16)
    for name, value in model.named_parameters():
        torch.testing.assert_close(dict(bf16.named_parameters())[name], value.bfloat16(), atol=0, rtol=0)


def test_legacy_checkpoint_without_ple_config_fields_loads_as_none(tmp_path):
    model = SprintForCausalLM(tiny("none"))
    model.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.pop("ple_type", None)
    config.pop("ple_lanes", None)
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = SprintForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.config.ple_type == "none" and loaded.config.ple_lanes == 4
    assert not loaded.ple_refresh_layers
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["full", "qr"])
@pytest.mark.parametrize("corruption", ["missing", "shape"])
def test_incomplete_ple_checkpoints_fail_instead_of_randomly_initializing(tmp_path, kind, corruption):
    model = SprintForCausalLM(tiny(kind))
    model.save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    key = next(name for name in state if name.startswith("ple_refresh_layers.") and name.endswith("weight"))
    if corruption == "missing":
        state.pop(key)
    else:
        state[key] = state[key][:-1]
    save_file(state, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="PLE checkpoint.*PLE tensors"):
        SprintForCausalLM.from_pretrained(tmp_path, local_files_only=True, ignore_mismatched_sizes=True)


@pytest.mark.parametrize("kind", ["full", "qr"])
def test_loading_ple_checkpoint_as_none_cannot_silently_discard_refreshes(tmp_path, kind):
    model = SprintForCausalLM(tiny(kind))
    model.save_pretrained(tmp_path)
    config = SprintConfig.from_pretrained(tmp_path)
    config.ple_type = "none"
    with pytest.raises(ValueError, match="PLE checkpoint.*PLE tensors"):
        SprintForCausalLM.from_pretrained(tmp_path, config=config, local_files_only=True)
