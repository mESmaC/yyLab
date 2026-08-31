from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from yylab import db
from yylab.train.data import PackedBatches, estimate_steps, plan_expert_data
from yylab.train.loop import LoopOpts, attention_backend_context, make_optimizer, train_loop
from yylab.train.planner import PlanInput, estimate_adapter_trainable_params, plan_training


class WhitespaceTok:
    def __init__(self, vocab: int = 128) -> None:
        self.vocab = vocab
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = None

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False, **_kwargs):
        ids = [[(hash(w) % (self.vocab - 2)) + 2 for w in t.split()] for t in texts]
        return {"input_ids": ids}


def test_packed_batches():
    docs = [{"text": "a b c d e f g"} for _ in range(10)]
    tok = WhitespaceTok()
    pb = PackedBatches(docs, lambda ts: tok(ts)["input_ids"], seq_len=8, eos_token_id=1, micro_batch=2)
    batches = list(pb)
    assert all(len(b["input_ids"]) == 2 for b in batches)
    assert all(len(seq) == 8 for b in batches for seq in b["input_ids"])
    first = batches[0]["input_ids"][0]
    assert first[-1] == 1
    assert pb.docs_seen == 10
    assert pb.tokens_seen == 70


def test_packed_batches_max_batches():
    docs = [{"text": " ".join("w" for _ in range(50))} for _ in range(100)]
    tok = WhitespaceTok()
    pb = PackedBatches(
        docs, lambda ts: tok(ts)["input_ids"], seq_len=10, eos_token_id=1, micro_batch=1, max_batches=3
    )
    batches = list(pb)
    assert len(batches) == 3


def test_packed_batches_bounds_tokenizer_chunk_and_streams_long_docs():
    calls: list[int] = []
    text_sizes: list[int] = []

    def tokenize(texts: list[str]) -> list[list[int]]:
        calls.append(len(texts))
        text_sizes.extend(map(len, texts))
        return [[index % 127 for index, _ in enumerate(text.split())] for text in texts]

    docs = [{"text": " ".join("token" for _ in range(10_000))} for _ in range(9)]
    packed = PackedBatches(
        docs,
        tokenize,
        seq_len=128,
        eos_token_id=1,
        micro_batch=1,
        max_batches=12,
        tokenize_chunk=3,
        max_text_chars=100,
    )
    assert len(list(packed)) == 12
    assert calls and max(calls) <= 3
    assert max(text_sizes) <= 100


def test_packed_batches_adds_eos_once_per_document_not_per_text_chunk():
    calls: list[list[str]] = []

    def tokenize(texts: list[str]) -> list[list[int]]:
        calls.append(list(texts))
        return [[7] * len(text) for text in texts]

    packed = PackedBatches(
        [{"text": "abcdefg"}, {"text": "hijkl"}],
        tokenize,
        seq_len=1,
        eos_token_id=1,
        micro_batch=1,
        tokenize_chunk=2,
        max_text_chars=3,
    )
    tokens = [sequence[0] for batch in packed for sequence in batch["input_ids"]]
    assert tokens == [7] * 7 + [1] + [7] * 5 + [1]
    assert max(map(len, calls)) <= 2
    assert max(len(text) for call in calls for text in call) <= 3


def test_build_batches_replays_each_epoch(monkeypatch):
    from yylab.workers import train as worker

    seeds: list[int] = []

    def fake_docs(_corpora, _domains, *, seed, **_kwargs):
        seeds.append(seed)
        yield {"text": " ".join("word" for _ in range(20))}

    monkeypatch.setattr(worker, "iter_corpus_docs", fake_docs)
    cfg = {
        "corpora": ["corpus"],
        "domains": None,
        "epochs": 3,
        "seq_len": 8,
        "micro_batch": 1,
        "grad_accum": 1,
        "seed": 7,
    }
    batches = worker.build_batches(cfg, WhitespaceTok(), max_steps=100)
    list(batches)
    assert seeds == [7, 8, 9]


def test_build_batches_suppresses_raw_length_warning_without_truncating(monkeypatch):
    from yylab.workers import train as worker

    class RecordingTokenizer:
        eos_token_id = 1

        def __init__(self):
            self.calls: list[dict] = []

        def __call__(self, texts, **kwargs):
            self.calls.append(kwargs)
            # Deliberately much longer than the model-facing sequence length.
            return {"input_ids": [[2] * 40 for _text in texts]}

    monkeypatch.setattr(
        worker,
        "iter_corpus_docs",
        lambda *_args, **_kwargs: iter([{"text": "dense input"}]),
    )
    tokenizer = RecordingTokenizer()
    batches = list(
        worker.build_batches(
            {
                "corpora": ["corpus"],
                "seq_len": 8,
                "micro_batch": 1,
                "grad_accum": 1,
                "epochs": 1,
            },
            tokenizer,
            max_steps=5,
        )
    )

    assert tokenizer.calls == [
        {
            "add_special_tokens": False,
            "return_attention_mask": False,
            "verbose": False,
        }
    ]
    assert len(batches) == 5
    assert all(len(sequence) == 8 for batch in batches for sequence in batch["input_ids"])


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids=None, labels=None, **_):
        logits = self.head(self.emb(input_ids))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return SimpleNamespace(loss=loss)


def test_train_loop(tmp_path, capsys):
    torch.manual_seed(0)
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    docs = [{"text": " ".join(f"w{i % 20}" for i in range(40))} for _ in range(20)]
    tok = WhitespaceTok()
    pb = PackedBatches(
        docs, lambda ts: tok(ts)["input_ids"], seq_len=16, eos_token_id=1, micro_batch=2, max_batches=8
    )
    opts = LoopOpts(micro_batch=2, grad_accum=2, max_steps=3, log_every=1, device="cpu", run_dir=tmp_path)
    result = train_loop(model, iter(pb), opt, sched, opts, aux_fn=lambda: torch.tensor(0.5))
    assert result["steps"] == 3
    assert result["data_exhausted"] is False
    assert result["tokens"] > 0
    lines = (tmp_path / "metrics.jsonl").read_text("utf-8").strip().splitlines()
    assert len(lines) >= 3
    assert (tmp_path / "progress.json").exists()
    terminal = capsys.readouterr().out
    assert "train: step 1/3" in terminal
    assert "tok/s" in terminal


def test_train_loop_reports_data_exhaustion_below_step_cap(tmp_path):
    torch.manual_seed(0)
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    batches = iter([{"input_ids": [[2, 3, 4, 5]]}, {"input_ids": [[6, 7, 8, 9]]}])
    opts = LoopOpts(micro_batch=1, grad_accum=1, max_steps=5, log_every=1, device="cpu", run_dir=tmp_path)
    result = train_loop(model, batches, opt, sched, opts)
    progress = json.loads((tmp_path / "progress.json").read_text("utf-8"))
    assert result["steps"] == 2
    assert result["step_cap"] == 5
    assert result["data_exhausted"] is True
    assert progress["step"] == 2
    assert "data exhausted" in progress["msg"]


def test_attention_context_prioritizes_cudnn_and_retains_fallbacks(monkeypatch):
    from torch.nn.attention import SDPBackend

    from yylab.train import loop

    seen: dict = {}

    @contextmanager
    def fake_sdpa_kernel(backends, set_priority=False):
        seen["backends"] = backends
        seen["priority"] = set_priority
        seen["active"] = True
        yield
        seen["active"] = False

    monkeypatch.setattr(loop, "cudnn_sdpa_eligible", lambda *_args: True)
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", fake_sdpa_kernel)
    with attention_backend_context(TinyLM(), "cuda"):
        assert seen["active"] is True
    assert seen["backends"][0] is SDPBackend.CUDNN_ATTENTION
    assert SDPBackend.MATH in seen["backends"]
    assert seen["priority"] is True
    assert seen["active"] is False


def test_train_loop_attention_context_covers_checkpoint_recompute(tmp_path, monkeypatch):
    from torch.utils.checkpoint import checkpoint

    from yylab.train import loop

    context_active = False
    observations: list[bool] = []

    @contextmanager
    def fake_context(*_args, **_kwargs):
        nonlocal context_active
        context_active = True
        yield
        context_active = False

    class CheckpointLM(TinyLM):
        def forward(self, input_ids=None, labels=None, **_):
            hidden = self.emb(input_ids)

            def block(value):
                observations.append(context_active)
                return self.head(value)

            logits = checkpoint(block, hidden, use_reentrant=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return SimpleNamespace(loss=loss)

    monkeypatch.setattr(loop, "attention_backend_context", fake_context)
    model = CheckpointLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    train_loop(
        model,
        iter([{"input_ids": [[2, 3, 4, 5]]}]),
        opt,
        sched,
        LoopOpts(grad_accum=1, max_steps=1, device="cpu", run_dir=tmp_path),
    )
    assert observations == [True, True]


def test_small_cuda_parameter_set_prefers_fused_adamw(monkeypatch):
    from yylab.train import loop

    sentinel = object()
    calls: list[dict] = []

    class FakeParameter:
        requires_grad = True
        is_cuda = True

        def numel(self):
            return 10_092_544

    class FakeModel:
        def parameters(self):
            return [FakeParameter()]

    def fake_adamw(_params, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(loop.torch.optim, "AdamW", fake_adamw)
    assert make_optimizer(FakeModel(), 3e-4, cuda=True) is sentinel
    assert calls == [{"lr": 3e-4, "weight_decay": 0.0, "fused": True}]


def test_planner_qlora_on_small_gpu():
    plan = plan_training(
        PlanInput(
            params_b=0.5,
            vram_gb=4.0,
            seq_len=1024,
            method="auto",
            adapter_only=True,
            isolated_expert_training=True,
        )
    )
    assert plan.feasible
    assert plan.method in {"lora", "qlora"}
    assert plan.grad_ckpt

    too_big = plan_training(
        PlanInput(params_b=3.0, vram_gb=4.0, seq_len=1024, method="auto", adapter_only=True)
    )
    assert not too_big.feasible
    assert too_big.method == "qlora"

    fits_bigger = plan_training(
        PlanInput(params_b=3.0, vram_gb=8.0, seq_len=1024, method="auto", adapter_only=True)
    )
    assert fits_bigger.feasible


def test_planner_uses_real_qwen_projection_shapes_and_isolates_experts():
    common = dict(
        params_b=0.6,
        vram_gb=12.0,
        seq_len=2048,
        method="auto",
        adapter_only=True,
        lora_rank=16,
        num_experts=4,
        hidden_size=1024,
        intermediate_size=3072,
        num_layers=28,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=151_936,
        model_type="qwen3",
        fused_linear_ce=True,
        full_liger_kernels=True,
        prefer_cudnn_sdpa=True,
        sm_version="12.0",
    )
    isolated = PlanInput(**common, isolated_expert_training=True)
    joint = PlanInput(**common, isolated_expert_training=False)
    assert estimate_adapter_trainable_params(isolated) == 10_092_544
    assert estimate_adapter_trainable_params(joint) == 40_370_176
    assert plan_training(isolated).est_peak_gb < plan_training(joint).est_peak_gb
    assert (plan_training(isolated).micro_batch, plan_training(isolated).grad_accum) == (8, 2)


def test_checkpoint_shape_reads_head_dim_and_only_enables_qwen3_fused_ce(tmp_path):
    from yylab.ui.pages.train_mole import _checkpoint_shape

    qwen_dir = tmp_path / "qwen"
    qwen_dir.mkdir()
    (qwen_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "head_dim": 128, "vocab_size": 151_936}),
        "utf-8",
    )
    llama_dir = tmp_path / "llama"
    llama_dir.mkdir()
    (llama_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "head_dim": 64, "vocab_size": 32_000}),
        "utf-8",
    )

    qwen = _checkpoint_shape({"params_b": 0.6, "ref": str(qwen_dir)})
    llama = _checkpoint_shape({"params_b": 0.6, "ref": str(llama_dir)})
    assert qwen["head_dim"] == 128
    assert qwen["model_type"] == "qwen3"
    assert qwen["fused_linear_ce"] is True
    assert qwen["full_liger_kernels"] is True
    assert qwen["prefer_cudnn_sdpa"] is True
    assert llama["fused_linear_ce"] is False


def test_qwen_liger_enables_full_kernel_set(monkeypatch):
    from liger_kernel.transformers.model.qwen3 import lce_forward

    from yylab.workers import train as worker

    captured: dict = {}
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3"),
        forward=lambda: None,
    )

    def fake_apply(**kwargs):
        captured.update(kwargs)
        kwargs["model"].forward = MethodType(lce_forward, kwargs["model"])

    monkeypatch.setattr("liger_kernel.transformers.apply_liger_kernel_to_qwen3", fake_apply)
    assert worker.enable_fused_linear_ce(model, full_kernels=True)
    assert captured["rope"] is True
    assert captured["rms_norm"] is True
    assert captured["swiglu"] is True
    assert captured["fused_linear_cross_entropy"] is True


def test_make_plan_preserves_base_and_mole_training_semantics(tmp_path, monkeypatch):
    from yylab.ui.pages import train_mole

    gpu = SimpleNamespace(vram_free_gb=64.0, compute_capability="8.9")
    monkeypatch.setattr(train_mole, "best_gpu", lambda: gpu)
    monkeypatch.setattr(train_mole, "ram_total_gb", lambda: 64.0)
    monkeypatch.setattr(train_mole, "has_triton", lambda: False)
    checkpoint = {"params_b": 0.1, "ref": str(tmp_path)}

    base = train_mole._make_plan(
        checkpoint,
        512,
        "auto",
        "nf4",
        16,
        1,
        adapter_only=False,
        isolated_expert_training=False,
    )
    mole = train_mole._make_plan(
        checkpoint,
        512,
        "auto",
        "nf4",
        16,
        4,
        adapter_only=True,
        isolated_expert_training=True,
    )
    assert base.method == "full"
    assert mole.method == "lora"
    assert any("sequentially" in note for note in mole.notes)


def test_blackwell_compute_capability_note_is_numeric():
    plan = plan_training(
        PlanInput(
            params_b=0.6,
            vram_gb=12.0,
            method="qlora",
            quant="nf4",
            sm_version="12.0",
            adapter_only=True,
            isolated_expert_training=True,
        )
    )
    assert any("low-precision" in note for note in plan.notes)


def test_planner_lora_when_model_fits():
    plan = plan_training(PlanInput(params_b=0.5, vram_gb=16.0, seq_len=1024, method="auto"))
    assert plan.feasible
    assert plan.method in {"lora", "full"}


def _seed_corpus(tmp_path: Path) -> str:
    from yylab.ingest.pipeline import IngestConfig, run_ingest

    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "a.txt").write_text(
        "Training data for the tiny model must contain enough characters to pass the filters used by intake. "
        * 6,
        "utf-8",
    )
    (src / "b.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "text": f"Document number {i} padded with enough words to survive minimum length checks easily."
                }
            )
            for i in range(30)
        ),
        "utf-8",
    )
    return run_ingest(
        lambda *a: None,
        IngestConfig(
            name="train-corpus",
            source_dir=str(src),
            tokenizer_keys=[],
            near_dedup=False,
            global_dedup=False,
            language_filter=False,
            quality_filter=False,
        ),
    )


def test_expert_data_plan_falls_back_to_disjoint_document_shards(tmp_path):
    corpus_id = _seed_corpus(tmp_path)
    plans = plan_expert_data([corpus_id], 3)
    assert [plan.shard_index for plan in plans] == [0, 1, 2]
    assert all(plan.num_shards == 3 for plan in plans)
    assert all(plan.domains == ("language",) for plan in plans)


def test_quant_lock_is_committed_only_after_success(tmp_path):
    from yylab.workers.train import commit_quant_lock, enforce_quant_lock

    model_dir = _tiny_llama_dir(tmp_path)
    ckpt_id = db.new_id("ckpt")
    db.execute(
        "INSERT INTO checkpoints(id, name, created_at, kind, source, ref) VALUES(?,?,?,?,?,?)",
        (ckpt_id, "deferred-lock", db.now_iso(), "base", "local", str(model_dir)),
    )
    enforce_quant_lock(ckpt_id, "nf4")
    assert db.q1("SELECT quant_locked FROM checkpoints WHERE id=?", (ckpt_id,))["quant_locked"] == ""
    commit_quant_lock(ckpt_id, "nf4")
    assert db.q1("SELECT quant_locked FROM checkpoints WHERE id=?", (ckpt_id,))["quant_locked"] == "nf4"


def _tiny_llama_dir(tmp_path: Path) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM

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


def test_run_mole_end_to_end(tmp_path, monkeypatch):
    from yylab.workers import train as worker

    corpus_id = _seed_corpus(tmp_path)
    model_dir = _tiny_llama_dir(tmp_path)

    ckpt_id = db.new_id("ckpt")
    db.execute(
        "INSERT INTO checkpoints(id, name, created_at, kind, source, ref, params_b) VALUES(?,?,?,?,?,?,?)",
        (ckpt_id, "tiny", db.now_iso(), "base", "local", str(model_dir), 0.001),
    )
    adapter_id = db.new_id("adp")
    db.execute(
        "INSERT INTO adapters(id, name, created_at, checkpoint_id, arch, quant, status) VALUES(?,?,?,?,?,?,?)",
        (adapter_id, "tiny-mole", db.now_iso(), ckpt_id, "mole", "nf4", "training"),
    )

    def fake_load(ref, quant):
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(ref)
        m.config.use_cache = False
        return m, WhitespaceTok(), False, quant, []

    monkeypatch.setattr(worker, "load_base", fake_load)
    log_messages: list[str] = []
    monkeypatch.setattr(worker, "log", log_messages.append)

    cfg = {
        "checkpoint_id": ckpt_id,
        "checkpoint_ref": str(model_dir),
        "adapter_id": adapter_id,
        "quant": "nf4",
        "num_experts": 2,
        "rank": 4,
        "alpha": None,
        "top_k": 2,
        "target_modules": ["q_proj", "v_proj"],
        "corpora": [corpus_id],
        "domains": None,
        "seq_len": 32,
        "epochs": 1,
        "micro_batch": 1,
        "grad_accum": 2,
        "lr": 1e-3,
        "aux_weight": 0.01,
        "grad_ckpt": False,
        "max_steps": 16,
        "log_every": 1,
        "save_every": None,
    }
    rd = tmp_path / "run"
    summary = worker.run_mole(cfg, rd)
    assert 0 < summary["result"]["steps"] < cfg["max_steps"]
    assert all(expert["data_exhausted"] for expert in summary["experts"])
    assert any("data slice exhausted" in message for message in log_messages)
    progress = json.loads((rd / "progress.json").read_text("utf-8"))
    assert progress["step"] == summary["result"]["steps"]
    metrics = [json.loads(line) for line in (rd / "metrics.jsonl").read_text("utf-8").splitlines()]
    second_expert_first = next(record for record in metrics if record["stage"].startswith("expert 2/"))
    assert second_expert_first["step"] == summary["experts"][0]["steps"] + second_expert_first["stage_step"]

    a = db.q1("SELECT * FROM adapters WHERE id=?", (adapter_id,))
    assert a["status"] == "trained"
    assert (Path(a["path"]) / "mole_weights.safetensors").exists()
    assert (Path(a["path"]) / "mole_config.json").exists()
    ckpt = db.q1("SELECT quant_locked FROM checkpoints WHERE id=?", (ckpt_id,))
    assert ckpt["quant_locked"] == "nf4"

    cfg["quant"] = "int8"
    try:
        worker.run_mole(cfg, rd)
        raise AssertionError("quant lock should have blocked int8")
    except RuntimeError as e:
        assert "locked" in str(e)


def test_run_base_qlora_peft(tmp_path, monkeypatch):
    from yylab.workers import train as worker

    corpus_id = _seed_corpus(tmp_path)
    model_dir = _tiny_llama_dir(tmp_path)

    ckpt_id = db.new_id("ckpt")
    db.execute(
        "INSERT INTO checkpoints(id, name, created_at, kind, source, ref, params_b) VALUES(?,?,?,?,?,?,?)",
        (ckpt_id, "tiny-base", db.now_iso(), "base", "local", str(model_dir), 0.001),
    )
    adapter_id = db.new_id("adp")
    db.execute(
        "INSERT INTO adapters(id, name, created_at, checkpoint_id, arch, quant, status) VALUES(?,?,?,?,?,?,?)",
        (adapter_id, "tiny-lora", db.now_iso(), ckpt_id, "lora", "nf4", "training"),
    )

    def fake_load(ref, quant):
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(ref)
        m.config.use_cache = False
        return m, WhitespaceTok(), False, quant, []

    monkeypatch.setattr(worker, "load_base", fake_load)

    cfg = {
        "checkpoint_id": ckpt_id,
        "checkpoint_ref": str(model_dir),
        "method": "qlora",
        "quant": "nf4",
        "rank": 8,
        "alpha": 16,
        "corpora": [corpus_id],
        "domains": None,
        "seq_len": 32,
        "epochs": 1,
        "micro_batch": 1,
        "grad_accum": 2,
        "lr": 1e-3,
        "grad_ckpt": False,
        "max_steps": 2,
        "log_every": 1,
        "save_every": None,
        "adapter_id": adapter_id,
    }
    summary = worker.run_base(cfg, tmp_path / "run_base")
    assert summary["result"]["steps"] == 2
    a = db.q1("SELECT * FROM adapters WHERE id=?", (adapter_id,))
    assert a["status"] == "trained"
    from yylab.artifacts import adapter_dir

    assert (adapter_dir(adapter_id) / "adapter_config.json").exists()


def test_estimate_steps(tmp_path):
    corpus_id = _seed_corpus(tmp_path)
    steps = estimate_steps([corpus_id], None, 1, 32, 1, 2)
    assert steps >= 1
