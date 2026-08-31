from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yylab import db
from yylab.artifacts import checkpoint_dir
from yylab.train import data


def _fake_corpus_streams(monkeypatch, documents: dict[str, list[dict]]) -> None:
    monkeypatch.setattr(data, "corpus_shards", lambda corpus_id: [f"{corpus_id}.jsonl.zst"])
    monkeypatch.setattr(data, "corpus_dir", lambda corpus_id: Path(corpus_id))

    def iter_shards(directory: Path, _shards: list[str]):
        yield from documents[directory.parent.name]

    monkeypatch.setattr(data, "_iter_shards", iter_shards)


def test_round_robin_mixes_documents_after_filtering_and_deduplicates(monkeypatch):
    _fake_corpus_streams(
        monkeypatch,
        {
            "alpha": [
                {"text": "filtered alpha", "domain": "math", "origin": "a-filtered"},
                {"text": "alpha zero", "domain": "code", "origin": "a0"},
                {"text": "alpha one", "domain": "code", "origin": "a1"},
            ],
            "beta": [
                {"text": "beta zero", "domain": "code", "origin": "b0"},
                {"text": "beta one", "domain": "code", "origin": "b1"},
            ],
            "empty": [{"text": "filtered empty", "domain": "math", "origin": "x"}],
        },
    )

    docs = list(
        data.iter_corpus_docs(
            ["alpha", "beta", "alpha", "empty"],
            ["code"],
            shuffle_buffer=1,
            seed=19,
            corpus_mix="round_robin",
        )
    )
    assert [doc["origin"] for doc in docs] == ["a0", "b0", "a1", "b1"]


def test_round_robin_is_deterministic_and_preserves_single_corpus_seed(monkeypatch):
    documents = {
        "alpha": [
            {"text": f"alpha {index}", "domain": "code", "origin": f"a{index}"}
            for index in range(12)
        ],
        "beta": [
            {"text": f"beta {index}", "domain": "code", "origin": f"b{index}"}
            for index in range(7)
        ],
    }
    _fake_corpus_streams(monkeypatch, documents)

    def origins(corpora, mix):
        return [
            doc["origin"]
            for doc in data.iter_corpus_docs(
                corpora,
                shuffle_buffer=8,
                shuffle_buffer_bytes=2**20,
                seed=41,
                corpus_mix=mix,
            )
        ]

    first = origins(["alpha", "beta"], "round_robin")
    assert first == origins(["alpha", "beta"], "round_robin")
    assert first[:10:2] == [origin for origin in first[:10:2] if origin.startswith("a")]
    assert first[1:10:2] == [origin for origin in first[1:10:2] if origin.startswith("b")]
    assert sorted(first) == sorted(origins(["beta", "alpha"], "round_robin"))
    assert origins("alpha", "round_robin") == origins(["alpha"], "concatenate")


def test_estimate_steps_accepts_scalar_and_does_not_double_count(monkeypatch):
    stats = json.dumps(
        {
            "domains": {
                "code": {"docs": 4, "chars": 350, "tokens": {"qwen25": 100}},
            }
        }
    )
    monkeypatch.setattr(data.db, "q1", lambda _query, _params: {"stats_json": stats})

    scalar = data.estimate_steps("alpha", None, 1, 10, 1, 1, "qwen25")
    duplicate = data.estimate_steps(["alpha", "alpha"], None, 1, 10, 1, 1, "qwen25")
    combined = data.estimate_steps(["alpha", "beta"], None, 1, 10, 1, 1, "qwen25")
    assert scalar == duplicate == 10
    assert combined == 20


def test_base_data_config_normalizes_legacy_scalar_and_rejects_empty():
    from yylab.workers.train import normalize_base_data_config

    legacy = normalize_base_data_config({"corpora": " alpha ", "domains": "code"})
    assert legacy["corpora"] == ["alpha"]
    assert legacy["domains"] == ["code"]
    assert legacy["corpus_mix"] == "concatenate"

    selected = normalize_base_data_config(
        {
            "corpora": ["beta", "alpha", "beta"],
            "domains": ["math", "code", "math"],
            "corpus_mix": "round_robin",
        }
    )
    assert selected["corpora"] == ["beta", "alpha"]
    assert selected["domains"] == ["math", "code"]
    assert selected["corpus_mix"] == "round_robin"
    with pytest.raises(ValueError, match="at least one corpus"):
        normalize_base_data_config({"corpora": []})


def test_corpus_provenance_snapshots_order_sources_and_filtered_totals(monkeypatch):
    rows = {
        "beta": {
            "id": "beta",
            "name": "Beta approved",
            "source_path": "D:/datasets/beta",
            "status": "ready",
            "stats_json": json.dumps(
                {
                    "domains": {
                        "code": {"docs": 3, "chars": 300},
                        "math": {"docs": 2, "chars": 500},
                    }
                }
            ),
        },
        "alpha": {
            "id": "alpha",
            "name": "Alpha approved",
            "source_path": "D:/datasets/alpha",
            "status": "ready",
            "stats_json": json.dumps(
                {"domains": {"code": {"docs": 7, "chars": 900}}}
            ),
        },
    }
    monkeypatch.setattr(data.db, "q1", lambda _query, params: rows.get(params[0]))

    provenance = data.corpus_data_provenance(
        ["beta", "alpha", "beta"], ["code"], corpus_mix="round_robin"
    )
    assert provenance["corpus_ids"] == ["beta", "alpha"]
    assert provenance["corpus_mix"] == "round_robin"
    assert provenance["domains"] == ["code"]
    assert provenance["estimated_docs"] == 10
    assert provenance["estimated_chars"] == 1200
    assert [entry["source_path"] for entry in provenance["corpora"]] == [
        "D:/datasets/beta",
        "D:/datasets/alpha",
    ]


class _SavedModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            use_cache=False,
            hidden_size=4,
            num_hidden_layers=1,
            architectures=["SavedModel"],
        )
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return [self.weight]

    def save_pretrained(self, destination: Path) -> None:
        Path(destination, "weights.bin").write_bytes(b"weights")


class _SavedTokenizer:
    eos_token_id = 1

    def save_pretrained(self, destination: Path) -> None:
        Path(destination, "tokenizer.json").write_text("{}", "utf-8")


def test_run_base_persists_multi_corpus_provenance(tmp_path, monkeypatch):
    from yylab.workers import train as worker

    corpus_rows = [
        (
            "alpha",
            "Alpha",
            "C:/data/alpha",
            json.dumps({"domains": {"code": {"docs": 2, "chars": 200}}}),
        ),
        (
            "beta",
            "Beta",
            "C:/data/beta",
            json.dumps({"domains": {"code": {"docs": 3, "chars": 300}}}),
        ),
    ]
    now = db.now_iso()
    for corpus_id, name, source_path, stats in corpus_rows:
        db.execute(
            "INSERT INTO corpora(id, name, created_at, updated_at, status, source_path, stats_json) "
            "VALUES(?,?,?,?,?,?,?)",
            (corpus_id, name, now, now, "ready", source_path, stats),
        )
    output_id = db.new_id("ckpt")
    db.execute(
        "INSERT INTO checkpoints(id, name, created_at, kind, source, ref) VALUES(?,?,?,?,?,?)",
        (output_id, "multi", now, "pending", "pending", ""),
    )

    model = _SavedModel()
    tokenizer = _SavedTokenizer()
    monkeypatch.setattr(worker, "load_base", lambda _ref, quant: (model, tokenizer, False, quant, []))
    monkeypatch.setattr(worker, "_make_base_optimizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(worker, "_make_base_scheduler", lambda *_args, **_kwargs: (object(), 0))
    batches = SimpleNamespace(docs_seen=5, tokens_seen=64)
    monkeypatch.setattr(worker, "build_batches", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(
        worker,
        "train_loop",
        lambda *_args, **_kwargs: {"steps": 1, "tokens": 64, "data_exhausted": False},
    )

    summary = worker.run_base(
        {
            "initialization": {
                "mode": "pretrained",
                "checkpoint_id": "source-checkpoint",
                "checkpoint_ref": "source-ref",
            },
            "method": "full",
            "precision": {"parameter_dtype": "fp32", "qat": "none"},
            "corpora": ["alpha", "beta", "alpha"],
            "corpus_mix": "round_robin",
            "domains": ["code"],
            "seq_len": 8,
            "micro_batch": 1,
            "grad_accum": 1,
            "max_steps": 1,
            "grad_ckpt": False,
            "liger_fused_ce": False,
            "torch_compile": False,
            "out_checkpoint_id": output_id,
        },
        tmp_path / "run",
    )

    metadata = json.loads(
        db.q1("SELECT meta_json FROM checkpoints WHERE id=?", (output_id,))["meta_json"]
    )
    assert summary["corpora"] == ["alpha", "beta"]
    assert summary["corpus_mix"] == "round_robin"
    assert summary["domains"] == ["code"]
    assert metadata["data_provenance"] == summary["data_provenance"]
    recipe = json.loads((checkpoint_dir(output_id) / "yylab_recipe.json").read_text("utf-8"))
    assert recipe["corpora"] == ["alpha", "beta"]
    assert recipe["corpus_mix"] == "round_robin"


def test_run_base_rejects_empty_selection_before_loading_model(tmp_path, monkeypatch):
    from yylab.workers import train as worker

    called = False

    def load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("model loading should not happen")

    monkeypatch.setattr(worker, "load_base", load)
    with pytest.raises(ValueError, match="at least one corpus"):
        worker.run_base({"corpora": []}, tmp_path / "run")
    assert called is False
