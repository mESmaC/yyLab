"""CUDA integration of production PLE with compile, offload, checkpoints and generation."""
import copy

import pytest
import torch

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train.sprint import SprintStepEngine, save_sprint_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def model(kind):
    return SprintForCausalLM(SprintConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=3, num_attention_heads=2, max_position_embeddings=256,
        ple_type=kind, loss_backend='torch', pad_token_id=0, eos_token_id=1))


@pytest.mark.parametrize('kind', ['full', 'qr'])
def test_ple_compiled_training_and_cached_generation(kind):
    torch.manual_seed(117)
    engine = SprintStepEngine(model(kind), dict(parameter_dtype='float32', optimizer_state_offload='none',
        torch_compile=True, compile_mode='default', max_steps=100, lr=.001, cpu_threads=4), 'cuda')
    for step in range(4):
        ids = torch.randint(2, 256, (2, 64)).pin_memory()
        result = engine.train_step([dict(input_ids=ids)], step=step)
        assert result.loss.isfinite()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in engine.model.parameters())
    assert all(p.dtype == torch.float32 for p in engine.model.parameters())
    prefix = torch.tensor([[0, 4, 9], [3, 5, 8]], device='cuda')
    options = dict(attention_mask=prefix.ne(0).long(), max_new_tokens=3, do_sample=False,
                   eos_token_id=None, disable_compile=True)
    engine.model.eval()
    with torch.inference_mode():
        uncached = engine.model.generate(prefix, use_cache=False, **options)
        dynamic = engine.model.generate(prefix, cache_implementation='dynamic', **options)
        static = engine.model.generate(prefix, cache_implementation='static', **options)
    assert torch.equal(uncached, dynamic)
    assert torch.equal(uncached, static)


@pytest.mark.parametrize('kind', ['full', 'qr'])
def test_ple_bf16_cpu_optimizer_checkpoint_resumes_exactly(kind, tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(118)
    cfg = dict(parameter_dtype='bfloat16', optimizer_state_offload='cpu', cpu_optimizer_kernel='native',
               offload_bucket_mb=1, lr=.001, max_steps=100, cpu_threads=4)
    engine = SprintStepEngine(model(kind), cfg, 'cuda')
    ids = torch.randint(2, 256, (2, 64)).pin_memory()
    engine.train_step([dict(input_ids=ids)], step=0)
    vocabulary = {f'word{i}': i for i in range(256)}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocabulary, unk_token='word0')))
    destination = tmp_path / kind
    save_sprint_checkpoint(destination, engine, tokenizer, {'step': 1})
    expected = engine.train_step([dict(input_ids=ids)], step=1)
    loaded = SprintForCausalLM.from_pretrained(destination, dtype='auto')
    assert loaded.config.ple_type == kind
    resumed = SprintStepEngine(loaded, {**cfg, '_restore_optimizer': True}, 'cuda')
    state = torch.load(destination/'trainer.pt', map_location='cpu', weights_only=True)
    resumed.optimizer.load_state_dict(state['optimizer'])
    actual = resumed.train_step([dict(input_ids=ids)], step=1)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    for left, right in zip(engine.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['full', 'qr'])
def test_ple_gradient_checkpointing_recomputes_refresh_correctly(kind):
    torch.manual_seed(119)
    ordinary = model(kind).cuda()
    recomputed = copy.deepcopy(ordinary)
    recomputed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.randint(2, 256, (2, 65), device='cuda')
    labels = ids.clone()
    labels[:, :8] = -100
    for net in (ordinary, recomputed):
        net.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            net(ids, labels=labels).loss.backward()
    for (name, left), (_, right) in zip(ordinary.named_parameters(), recomputed.named_parameters(), strict=True):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)
