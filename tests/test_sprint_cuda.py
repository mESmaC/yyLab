"""Bounded native CUDA correctness checks; no emulated precision substitutes."""

import pytest
import torch

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train.sprint import SprintStepEngine, save_sprint_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def model(backend="torch"):
    return SprintForCausalLM(SprintConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=2,
                                        max_position_embeddings=64, loss_backend=backend,
                                        pad_token_id=0, eos_token_id=1))


def test_native_liger_loss_and_gradients_match_torch_bf16():
    torch.manual_seed(27)
    left, right = model().cuda(), model("liger").cuda()
    right.load_state_dict(left.state_dict())
    x = torch.randint(2, 128, (2, 17), device="cuda")
    labels = x.clone()
    labels[:, :5] = -100
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = left(x, labels=labels).loss, right(x, labels=labels).loss
    torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-3)
    a.backward()
    b.backward()
    for p, q in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-4, rtol=0.03)
    right.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        empty = right(x, labels=torch.full_like(x, -100)).loss
    assert empty == 0
    empty.backward()
    assert all(p.grad.isfinite().all() for p in right.parameters())


@pytest.mark.parametrize("backend", ["torch", "liger"])
def test_compile_real_optimizer_updates_and_fp32_state(backend):
    torch.manual_seed(6)
    cfg = {"torch_compile": True, "compile_mode": "default", "lr": 0.001,
           "max_steps": 100, "grad_clip": 1.0, "parameter_dtype": "float32", "optimizer_state_offload": "none"}
    engine = SprintStepEngine(model(backend), cfg, "cuda")
    x = torch.randint(2, 128, (2, 17)).pin_memory()
    first = float(engine.train_step([{"input_ids": x}], step=0).loss)
    final = first
    for step in range(1, 5):
        final = float(engine.train_step([{"input_ids": x}], step=step).loss)
    assert final < first
    assert all(p.dtype == torch.float32 and p.grad.isfinite().all() for p in engine.model.parameters())
    for state in engine.optimizer.state.values():
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32
        assert state["exp_avg"].device.type == "cuda"


def test_cuda_cached_generation_static_dynamic_and_uncached_agree():
    torch.manual_seed(16)
    net = model().cuda().to(torch.bfloat16).eval()
    x = torch.tensor([[0, 2, 5], [2, 6, 7]], device="cuda")
    mask = (x != 0).long()
    options = {"attention_mask": mask, "max_new_tokens": 3, "do_sample": False,
               "eos_token_id": None, "disable_compile": True}
    with torch.inference_mode():
        uncached = net.generate(x, use_cache=False, **options)
        dynamic = net.generate(x, cache_implementation="dynamic", **options)
        static = net.generate(x, cache_implementation="static", **options)
    assert torch.equal(uncached, dynamic)
    assert torch.equal(uncached, static)


def test_bf16_cuda_cpu_offload_checkpoint_roundtrip(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(19)
    cfg = {"lr": 0.001, "max_steps": 20, "grad_clip": 1.0,
           "parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu", "offload_bucket_mb": 1}
    engine = SprintStepEngine(model(), cfg, "cuda")
    # Stress reuse of the pinned buffer across many weight slices.
    engine.optimizer.bucket_elements = 1024
    x = torch.randint(2, 128, (2, 17)).pin_memory()
    engine.train_step([{"input_ids": x}], step=0)
    assert all(p.dtype == torch.bfloat16 and p.grad.dtype == torch.bfloat16 for p in engine.model.parameters())
    for state in engine.optimizer.state.values():
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            assert state[key].device.type == "cpu"
            assert state[key].dtype == torch.float32
    vocabulary = {f"word{i}": i for i in range(128)}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocabulary, unk_token="word0")))
    folder = tmp_path / "offloaded"
    save_sprint_checkpoint(folder, engine, tokenizer, {"step": 1})
    expected = engine.train_step([{"input_ids": x}], step=1)
    loaded = SprintForCausalLM.from_pretrained(folder, dtype="auto")
    assert next(loaded.parameters()).dtype == torch.bfloat16
    resumed = SprintStepEngine(loaded, {**cfg, "_restore_optimizer": True}, "cuda")
    resumed.optimizer.bucket_elements = 1024
    payload = torch.load(folder / "trainer.pt", map_location="cpu", weights_only=True)
    resumed.optimizer.load_state_dict(payload["optimizer"])
    actual = resumed.train_step([{"input_ids": x}], step=1)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    for p, q in zip(engine.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(engine.optimizer.state[p][key], resumed.optimizer.state[q][key], rtol=0, atol=0)
