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
    monkeypatch.setattr(data, "corpus_shards", lambda corpus_id, **_: [f"{corpus_id}.jsonl.zst"])
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


def test_iter_corpus_docs_filters_splits_and_canonicalizes_legacy_lines(monkeypatch):
    _fake_corpus_streams(
        monkeypatch,
        {
            "alpha": [
                {"text": "train row", "domain": "code", "origin": "train", "split": "training"},
                {"text": "test row", "domain": "code", "origin": "test", "split": "test"},
                {
                    "text": "legacy sourced train row",
                    "domain": "code",
                    "origin": "sourced-train",
                    "source": "D:/datasets/archive/train-00001.parquet",
                },
                {"text": "legacy row", "domain": "code", "origin": "legacy"},
            ]
        },
    )

    train = list(data.iter_corpus_docs("alpha", splits="train", shuffle_buffer=1))
    unspecified = list(
        data.iter_corpus_docs("alpha", splits=["unspecified"], shuffle_buffer=1)
    )
    all_docs = list(data.iter_corpus_docs("alpha", shuffle_buffer=1))

    assert [(doc["origin"], doc["split"]) for doc in train] == [
        ("train", "train"),
        ("sourced-train", "train"),
    ]
    assert [(doc["origin"], doc["split"]) for doc in unspecified] == [
        ("legacy", "unspecified")
    ]
    assert [(doc["origin"], doc["split"]) for doc in all_docs] == [
        ("train", "train"),
        ("test", "test"),
        ("sourced-train", "train"),
        ("legacy", "unspecified"),
    ]


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


def test_base_data_config_normalizes_legacy_scalar_and_rejects_empty(monkeypatch):
    from yylab.workers.train import normalize_base_data_config

    monkeypatch.setattr(
        "yylab.workers.train.db.q1",
        lambda _query, params: {
            "name": params[0],
            "status": "ready",
            "kind": "ingested",
            "verdict": "",
        },
    )

    legacy = normalize_base_data_config({"corpora": " alpha ", "domains": "code"})
    assert legacy["corpora"] == ["alpha"]
    assert legacy["domains"] == ["code"]
    assert legacy["splits"] is None
    assert legacy["corpus_mix"] == "concatenate"

    selected = normalize_base_data_config(
        {
            "corpora": ["beta", "alpha", "beta"],
            "domains": ["math", "code", "math"],
            "splits": ["dev", "train", "validation"],
            "corpus_mix": "round_robin",
        }
    )
    assert selected["corpora"] == ["beta", "alpha"]
    assert selected["domains"] == ["math", "code"]
    assert selected["splits"] == ["validation", "train"]
    assert selected["corpus_mix"] == "round_robin"
    with pytest.raises(ValueError, match="at least one corpus"):
        normalize_base_data_config({"corpora": []})
    with pytest.raises(ValueError, match="unknown source split"):
        normalize_base_data_config({"corpora": ["alpha"], "splits": ["holdout"]})


def test_base_worker_rederives_and_persists_explicit_token_budget_fields(monkeypatch):
    from yylab.workers.train import normalize_base_worker_config

    monkeypatch.setattr(
        "yylab.workers.train.db.q1",
        lambda _query, params: {
            "name": params[0],
            "status": "ready",
            "kind": "ingested",
            "verdict": "",
        },
    )

    resolved = normalize_base_worker_config(
        {
            "corpora": "alpha",
            "splits": ["train", "unspecified"],
            "requested_training_tokens": 1_000_001,
            "seq_len": 512,
            "micro_batch": 2,
            "grad_accum": 4,
            "max_steps": 999,
        }
    )

    assert resolved["max_steps"] == 245
    assert resolved["tokens_per_optimizer_step"] == 4096
    assert resolved["scheduled_training_tokens"] == 1_003_520
    assert resolved["token_budget_rounding"] == 3519
    assert resolved["step_cap_source"] == "token_budget"


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
    assert provenance["corpora"][1]["available_splits"] == ["unspecified"]


def test_corpus_provenance_and_preflight_classify_cdas_and_legacy_sources(monkeypatch):
    rows = {
        "catalogued": {
            "id": "catalogued",
            "name": "Catalogued",
            "source_path": "D:/catalogued",
            "status": "ready",
            "kind": "ingested",
            "verdict": "approved",
            "stats_json": json.dumps(
                {
                    "docs_written": 2,
                    "chars_written": 200,
                    "domains": {"language": {"docs": 2, "chars": 200}},
                    "ingest_config": {"selection_profile": "dataset"},
                    "cdas": {"assets": 1, "documents": 2, "issues": 0},
                }
            ),
        },
        "legacy": {
            "id": "legacy",
            "name": "Legacy",
            "source_path": "D:/legacy",
            "status": "ready",
            "kind": "ingested",
            "verdict": "approved",
            "stats_json": json.dumps(
                {
                    "docs_written": 1,
                    "chars_written": 100,
                    "domains": {"language": {"docs": 1, "chars": 100}},
                }
            ),
        },
    }
    monkeypatch.setattr(data.db, "q1", lambda _query, params: rows.get(params[0]))

    report = data.corpus_training_preflight(["catalogued", "legacy"])
    corpora = report["provenance"]["corpora"]

    assert corpora[0]["catalog_status"] == "cdas_catalogued"
    assert corpora[0]["selection_profile"] == "dataset"
    assert corpora[1]["catalog_status"] == "legacy_reviewed"
    assert report["errors"] == []
    assert report["warnings"] == [
        "Corpus 'Legacy' has no CDAS intake catalog (legacy reviewed)"
    ]

    missing = data.corpus_training_preflight(["missing"])
    assert missing["errors"] == ["Missing corpus: missing"]


def test_corpus_provenance_estimates_selected_split_and_legacy_unspecified(monkeypatch):
    rows = {
        "modern": {
            "id": "modern",
            "name": "Modern",
            "source_path": "D:/datasets/modern",
            "status": "ready",
            "stats_json": json.dumps(
                {
                    "docs_written": 4,
                    "chars_written": 400,
                    "domains": {"code": {"docs": 3, "chars": 300}},
                    "splits": {
                        "train": {
                            "docs": 3,
                            "chars": 300,
                            "domains": {"code": {"docs": 2, "chars": 200}},
                        },
                        "test": {
                            "docs": 1,
                            "chars": 100,
                            "domains": {"code": {"docs": 1, "chars": 100}},
                        },
                    },
                }
            ),
        },
        "legacy": {
            "id": "legacy",
            "name": "Legacy",
            "source_path": "D:/datasets/legacy",
            "status": "ready",
            "stats_json": json.dumps(
                {
                    "docs_written": 7,
                    "chars_written": 700,
                    "domains": {"code": {"docs": 7, "chars": 700}},
                }
            ),
        },
    }
    monkeypatch.setattr(data.db, "q1", lambda _query, params: rows.get(params[0]))

    test_only = data.corpus_data_provenance(
        ["modern", "legacy"], ["code"], splits=["test"]
    )
    legacy_only = data.corpus_data_provenance(
        ["modern", "legacy"], ["code"], splits=["unspecified"]
    )

    assert test_only["schema_version"] == 3
    assert test_only["splits"] == ["test"]
    assert (test_only["estimated_docs"], test_only["estimated_chars"]) == (1, 100)
    assert test_only["corpora"][0]["available_splits"] == ["test", "train"]
    assert test_only["corpora"][1]["available_splits"] == ["unspecified"]
    assert (legacy_only["estimated_docs"], legacy_only["estimated_chars"]) == (7, 700)


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
    monkeypatch.setattr(
        worker,
        "load_base",
        lambda _ref, quant, **_options: (model, tokenizer, False, quant, []),
    )
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
