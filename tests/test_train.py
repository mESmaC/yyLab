from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from yylab import db
from yylab.train.data import PackedBatches, estimate_steps
from yylab.train.loop import LoopOpts, train_loop
from yylab.train.planner import PlanInput, plan_training


class WhitespaceTok:
    def __init__(self, vocab: int = 128) -> None:
        self.vocab = vocab
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = None

    def __call__(self, texts, add_special_tokens=False, return_attention_mask=False):
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
    pb = PackedBatches(docs, lambda ts: tok(ts)["input_ids"], seq_len=10, eos_token_id=1, micro_batch=1, max_batches=3)
    batches = list(pb)
    assert len(batches) == 3


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids=None, labels=None, **_):
        logits = self.head(self.emb(input_ids))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return SimpleNamespace(loss=loss)


def test_train_loop(tmp_path):
    torch.manual_seed(0)
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    docs = [{"text": " ".join(f"w{i % 20}" for i in range(40))} for _ in range(20)]
    tok = WhitespaceTok()
    pb = PackedBatches(docs, lambda ts: tok(ts)["input_ids"], seq_len=16, eos_token_id=1, micro_batch=2, max_batches=8)
    opts = LoopOpts(micro_batch=2, grad_accum=2, max_steps=3, log_every=1, device="cpu", run_dir=tmp_path)
    result = train_loop(model, iter(pb), opt, sched, opts, aux_fn=lambda: torch.tensor(0.5))
    assert result["steps"] == 3
    assert result["tokens"] > 0
    lines = (tmp_path / "metrics.jsonl").read_text("utf-8").strip().splitlines()
    assert len(lines) >= 3
    assert (tmp_path / "progress.json").exists()


def test_planner_qlora_on_small_gpu():
    plan = plan_training(PlanInput(params_b=3.0, vram_gb=4.0, seq_len=1024, method="auto"))
    assert plan.feasible
    assert plan.method == "qlora"
    assert plan.quant == "nf4"
    assert plan.grad_ckpt

    too_big = plan_training(PlanInput(params_b=7.0, vram_gb=4.0, seq_len=1024, method="auto"))
    assert not too_big.feasible
    assert too_big.method == "qlora"

    fits_bigger = plan_training(PlanInput(params_b=7.0, vram_gb=6.0, seq_len=1024, method="auto"))
    assert fits_bigger.feasible


def test_planner_lora_when_model_fits():
    plan = plan_training(PlanInput(params_b=0.5, vram_gb=16.0, seq_len=1024, method="auto"))
    assert plan.feasible
    assert plan.method in {"lora", "full"}


def _seed_corpus(tmp_path: Path) -> str:
    from yylab.ingest.pipeline import IngestConfig, run_ingest

    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "a.txt").write_text(
        "Training data for the tiny model must contain enough characters to pass the filters used by intake. " * 6,
        "utf-8",
    )
    (src / "b.jsonl").write_text(
        "\n".join(
            json.dumps({"text": f"Document number {i} padded with enough words to survive minimum length checks easily."})
            for i in range(30)
        ),
        "utf-8",
    )
    return run_ingest(
        lambda *a: None,
        IngestConfig(name="train-corpus", source_dir=str(src), tokenizer_keys=[], near_dedup=False, global_dedup=False),
    )


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
        "max_steps": 2,
        "log_every": 1,
        "save_every": None,
    }
    rd = tmp_path / "run"
    summary = worker.run_mole(cfg, rd)
    assert summary["result"]["steps"] == 2

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
