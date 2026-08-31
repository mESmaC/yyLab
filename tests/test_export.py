from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from yylab.export.merge import expert_usage_weights, merge_lora, merge_mole
from yylab.mole.inject import MoLEConfig, inject_mole, mole_layers, token_routers
from yylab.mole.store import save_mole


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

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<eos>": 0, "<unk>": 1}
    vocab.update({f"t{i}": i + 2 for i in range(100)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<eos>", unk_token="<unk>", pad_token="<eos>"
    )
    fast.save_pretrained(out)
    return out


def test_merge_lora(tmp_path):
    from peft import LoraConfig, get_peft_model

    model_dir = _tiny_llama_dir(tmp_path)
    base = AutoModelForCausalLM.from_pretrained(model_dir)
    base_w = base.model.layers[0].self_attn.q_proj.weight.detach().clone()

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(model, peft_cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.02)
    adp_dir = tmp_path / "lora_adp"
    model.save_pretrained(adp_dir)

    out = merge_lora(str(model_dir), str(adp_dir), tmp_path / "merged_lora")
    assert (out / "config.json").exists()
    assert list(out.glob("*.safetensors"))

    merged = AutoModelForCausalLM.from_pretrained(out)
    merged_w = merged.model.layers[0].self_attn.q_proj.weight.detach()
    assert not torch.allclose(merged_w.float(), base_w.float(), atol=1e-6)
    x = torch.tensor([[2, 3, 4, 5]])
    assert merged(x).logits.shape == (1, 4, 128)


def test_merge_mole_zero_router_exact(tmp_path):
    torch.manual_seed(0)
    model_dir = _tiny_llama_dir(tmp_path)

    mole_model = AutoModelForCausalLM.from_pretrained(model_dir)
    mole_cfg = MoLEConfig(num_experts=2, rank=4, alpha=8.0, top_k=2, target_modules=("q_proj", "v_proj"))
    inject_mole(mole_model, mole_cfg)
    for _, layer in mole_layers(mole_model):
        for expert in layer.experts:
            expert.lora_B.data.normal_(0, 0.02)
    adp_dir = tmp_path / "mole_adp"
    save_mole(adp_dir, mole_model, mole_cfg, base_ref=str(model_dir))

    out = merge_mole(str(model_dir), str(adp_dir), tmp_path / "merged_mole")
    assert (out / "config.json").exists()

    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    assert not mole_layers(merged)

    x = torch.tensor([[2, 3, 4, 5, 6, 7]])
    mole_model.eval()
    with torch.no_grad():
        ref = mole_model(x).logits
        got = merged(x).logits
    assert torch.allclose(ref, got, atol=1e-2)


def test_expert_usage_reflects_sparse_top_k_at_init(tmp_path):
    model_dir = _tiny_llama_dir(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    mole_cfg = MoLEConfig(num_experts=3, rank=4, alpha=8.0, top_k=2, target_modules=("q_proj",))
    inject_mole(model, mole_cfg)
    weights = expert_usage_weights(model, [[2, 3, 4, 5, 6]])
    assert len(weights) == 2
    for w in weights.values():
        assert torch.allclose(w.sum(), torch.tensor(1.0))
        assert torch.count_nonzero(w) == 2
        assert torch.allclose(w[w > 0], torch.tensor([0.5, 0.5]))


def test_merge_mole_with_calibration(tmp_path):
    torch.manual_seed(0)
    model_dir = _tiny_llama_dir(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    mole_cfg = MoLEConfig(num_experts=2, rank=4, alpha=8.0, top_k=1, target_modules=("q_proj",))
    inject_mole(model, mole_cfg)
    for _, layer in mole_layers(model):
        for expert in layer.experts:
            expert.lora_B.data.normal_(0, 0.02)
        layer.router.gate.weight.data.normal_(0, 0.5)
    adp_dir = tmp_path / "mole_adp2"
    save_mole(adp_dir, model, mole_cfg, base_ref=str(model_dir))

    out = merge_mole(
        str(model_dir), str(adp_dir), tmp_path / "merged_mole2", calib_ids=[[2, 3, 4, 5, 6, 7]]
    )
    merged = AutoModelForCausalLM.from_pretrained(out)
    x = torch.tensor([[2, 3, 4]])
    assert merged(x).logits.shape == (1, 3, 128)


def test_token_router_calibration_reuses_each_blocks_sparse_route(tmp_path):
    model_dir = _tiny_llama_dir(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    mole_cfg = MoLEConfig(
        num_experts=3,
        rank=2,
        alpha=4.0,
        top_k=1,
        target_modules=("q_proj", "v_proj"),
        router_type="per_token",
    )
    inject_mole(model, mole_cfg)
    for _, router in token_routers(model):
        router.gate.weight.data.zero_()

    weights = expert_usage_weights(model, [[2, 3, 4, 5, 6]])

    assert len(weights) == 4
    by_block: dict[str, list[torch.Tensor]] = {}
    for name, weight in weights.items():
        block = name.split(".self_attn")[0]
        by_block.setdefault(block, []).append(weight)
        assert torch.allclose(weight.sum(), torch.tensor(1.0))
        assert torch.count_nonzero(weight) == 1
    assert all(len(block_weights) == 2 for block_weights in by_block.values())
    assert all(torch.equal(block_weights[0], block_weights[1]) for block_weights in by_block.values())


def test_merge_token_routed_mole_requires_calibration(tmp_path):
    model_dir = _tiny_llama_dir(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    mole_cfg = MoLEConfig(
        num_experts=2,
        rank=2,
        alpha=4.0,
        top_k=1,
        target_modules=("q_proj",),
        router_type="per_token",
    )
    inject_mole(model, mole_cfg)
    adp_dir = tmp_path / "token_mole_adp"
    save_mole(adp_dir, model, mole_cfg, base_ref=str(model_dir))

    with pytest.raises(ValueError, match="requires calibration data"):
        merge_mole(str(model_dir), str(adp_dir), tmp_path / "uncalibrated")

    out = merge_mole(
        str(model_dir),
        str(adp_dir),
        tmp_path / "calibrated",
        calib_ids=[[2, 3, 4, 5, 6]],
    )
    merged = AutoModelForCausalLM.from_pretrained(out)
    assert not mole_layers(merged)
    assert not token_routers(merged)
    assert merged(torch.tensor([[2, 3, 4]])).logits.shape == (1, 3, 128)
