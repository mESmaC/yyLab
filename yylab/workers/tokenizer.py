"""Isolated tokenizer training/import worker with streaming corpus intake."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .. import db
from ..artifacts import run_dir, tokenizer_dir, write_json
from ..config import get_settings
from ..tokenizer_registry import (
    TOKENIZER_ACTION_IMPORT,
    TOKENIZER_ACTION_TRAIN,
    TOKENIZER_RUN_KIND,
    artifact_checksum,
    mark_tokenizer_failed,
    mark_tokenizer_ready,
    parse_tokenizer_source,
)
from ..train.compact_tokenizer import (
    CAUSAL_SPECIAL_TOKENS,
    CHAT_SPECIAL_TOKENS,
    GENERIC_CHAT_TEMPLATE,
    CompactTokenizerSpec,
    train_compact_tokenizer,
)
from ..train.data import (
    CORPUS_MIX_CONCATENATE,
    corpus_data_provenance,
    iter_corpus_docs,
    normalize_corpus_ids,
    normalize_corpus_mix,
    normalize_source_splits,
)


def log(message: str) -> None:
    print(message, flush=True)


def _write_progress(
    rd: Path,
    *,
    stage: str,
    step: int,
    total: int,
    value: float,
    label: str,
    message: str,
) -> None:
    write_json(
        rd / "progress.json",
        {
            "stage": stage,
            "step": int(step),
            "max_steps": max(1, int(total)),
            "value": round(float(value), 6),
            "metric_label": label,
            "msg": message,
        },
    )


def _profile_tokens(profile: str, custom: Iterable[str] = ()) -> tuple[str, ...]:
    normalized = str(profile or "auto").strip().casefold()
    if normalized == "causal":
        return CAUSAL_SPECIAL_TOKENS
    if normalized == "chat":
        return CHAT_SPECIAL_TOKENS
    if normalized == "custom":
        values = tuple(dict.fromkeys(str(value).strip() for value in custom if str(value).strip()))
        if not values:
            raise ValueError("custom import profile requires at least one special token")
        return values
    if normalized != "auto":
        raise ValueError("import special-token profile must be auto, causal, chat, or custom")
    return ()


def _token_exists(tokenizer, token: str) -> bool:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return False
    unk_id = getattr(tokenizer, "unk_token_id", None)
    return token_id != unk_id or token == getattr(tokenizer, "unk_token", None)


def _configure_imported_tokenizer(
    tokenizer,
    *,
    profile: str,
    custom_special_tokens: Iterable[str] = (),
    raw_artifact: bool,
) -> dict[str, Any]:
    """Attach Transformers special-token semantics without touching source files."""

    normalized = str(profile or "auto").strip().casefold()
    tokens = _profile_tokens(normalized, custom_special_tokens)
    if normalized == "auto" and raw_artifact:
        preferred = "<|endoftext|>"
        if not _token_exists(tokenizer, preferred):
            preferred = str(getattr(tokenizer, "eos_token", None) or preferred)
        tokens = (preferred,)
        normalized = "causal"

    added = 0
    if normalized == "chat":
        added = tokenizer.add_special_tokens(
            {
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "additional_special_tokens": list(CHAT_SPECIAL_TOKENS[4:]),
            }
        )
        tokenizer.chat_template = GENERIC_CHAT_TEMPLATE
    elif tokens:
        shared_end = tokens[0]
        added = tokenizer.add_special_tokens(
            {
                "eos_token": shared_end,
                "pad_token": shared_end,
                "additional_special_tokens": list(tokens[1:]),
            }
        )
    else:
        # Preserve a complete HF tokenizer's mappings. Only repair the common
        # decoder-only omission where PAD is unset but EOS already exists.
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token

    if getattr(tokenizer, "eos_token", None) is None:
        fallback = "<|endoftext|>"
        added += tokenizer.add_special_tokens({"eos_token": fallback, "pad_token": fallback})
    elif getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {
        "profile": normalized,
        "special_tokens": list(tokens),
        "tokens_added": int(added),
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
    }


def effective_vocab_size(tokenizer: Any) -> int:
    """Vocabulary size the way SPALMER sizes embeddings.

    The maximum of ``len(tokenizer)``, the declared ``vocab_size``, and the
    highest token id plus one; for contiguous vocabularies all three agree.
    Cataloguing the same number keeps the SPALMER page's "registered vocabulary
    is authoritative" rule exact.
    """

    candidates: list[int] = []
    try:
        candidates.append(int(len(tokenizer)))
    except (TypeError, AttributeError):
        pass
    declared = getattr(tokenizer, "vocab_size", None)
    if declared is not None:
        candidates.append(int(declared))
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:  # noqa: BLE001 - some backends cannot enumerate
            vocab = {}
        if vocab:
            candidates.append(max(int(value) for value in vocab.values()) + 1)
    if not candidates or max(candidates) <= 0:
        raise ValueError("cannot infer a positive vocabulary size from the tokenizer")
    return max(candidates)


def _hub_snapshot_dir(repo_id: str, revision: str | None, token: str | None) -> Path | None:
    """Locate the cached hub snapshot a tokenizer was loaded from (best effort)."""

    try:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id,
                "tokenizer_config.json",
                revision=revision or None,
                token=token or None,
            )
        )
    except Exception:  # noqa: BLE001 - offline or unusual repo layout
        return None
    return path.parent


def import_tokenizer_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    special_token_profile: str = "auto",
    custom_special_tokens: Iterable[str] = (),
    source_kind: str | None = None,
    revision: str | None = None,
    hf_token: str | None = None,
):
    """Load a raw tokenizer.json, an HF directory, or a hub repo and save a clean local copy."""

    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if source_kind is None:
        source_kind, reference = parse_tokenizer_source(source)
    else:
        reference = str(source)
    raw_artifact = False
    if source_kind == "hub":
        tokenizer = AutoTokenizer.from_pretrained(
            reference,
            revision=revision or None,
            use_fast=True,
            trust_remote_code=False,
            token=hf_token or None,
        )
    else:
        source_path = Path(reference).expanduser().resolve()
        raw_artifact = source_path.is_file()
        if raw_artifact:
            if source_path.name.casefold() != "tokenizer.json":
                raise ValueError("raw tokenizer import expects a tokenizer.json file")
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(source_path),
                clean_up_tokenization_spaces=False,
            )
        elif source_path.is_dir():
            tokenizer = AutoTokenizer.from_pretrained(
                source_path,
                local_files_only=True,
                use_fast=True,
                trust_remote_code=False,
            )
        else:
            raise FileNotFoundError(f"tokenizer source does not exist: {source_path}")
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("tokenizer import requires a fast tokenizer convertible to tokenizer.json")
    special = _configure_imported_tokenizer(
        tokenizer,
        profile=special_token_profile,
        custom_special_tokens=custom_special_tokens,
        raw_artifact=raw_artifact,
    )
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    # Transformers 5 can consolidate these mappings into tokenizer_config.json
    # and omit the traditional file. Keep both so older/local HF loaders also
    # understand the wrapper while AddedToken details remain in tokenizer.json.
    write_json(output / "special_tokens_map.json", tokenizer.special_tokens_map)
    # Prove the result is self-contained before publishing it to the registry.
    reloaded = AutoTokenizer.from_pretrained(
        output,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    if not (output / "tokenizer.json").is_file() or not bool(getattr(reloaded, "is_fast", False)):
        raise RuntimeError("imported tokenizer did not produce a self-contained fast artifact")
    if raw_artifact:
        probe = "ASCII ＦＵＬＬＷＩＤＴＨ ① λ 数学 🧪\n"
        probe_ids = reloaded.encode(probe, add_special_tokens=False)
        recovered = reloaded.decode(
            probe_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if recovered != probe:
            raise ValueError(
                "raw tokenizer.json failed the exact mixed-Unicode round-trip validation"
            )
    return reloaded, special


def tokenizer_source_checksum(source: str | Path) -> str:
    """Hash tokenizer payloads without accidentally hashing model weights."""

    path = Path(source).resolve()
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"tokenizer source does not exist: {path}")
    names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "spiece.model",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    }
    candidates = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and (
            candidate.name in names
            or candidate.name.startswith("vocab.")
            or candidate.name.startswith("merges.")
        )
    )
    if not candidates:
        raise FileNotFoundError(f"no tokenizer payload files found in {path}")
    digest = hashlib.sha256()
    for candidate in candidates:
        name = candidate.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _bounded_evaluation_docs(
    docs: Iterable[dict],
    *,
    max_documents: int,
    max_bytes: int,
) -> Iterator[dict]:
    count = 0
    total_bytes = 0
    for doc in docs:
        if count >= max_documents or (max_bytes and total_bytes >= max_bytes):
            return
        text = str(doc.get("text") or "")
        if not text:
            continue
        encoded = text.encode("utf-8")
        if max_bytes and total_bytes + len(encoded) > max_bytes:
            text = encoded[: max_bytes - total_bytes].decode("utf-8", errors="ignore")
            if not text:
                return
            doc = dict(doc)
            doc["text"] = text
            encoded = text.encode("utf-8")
        count += 1
        total_bytes += len(encoded)
        yield doc


def evaluate_tokenizer(
    tokenizer,
    docs: Iterable[dict],
    *,
    max_documents: int = 512,
    max_bytes: int = 8 * 2**20,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Measure compression, vocabulary use, and exact text recovery."""

    doc_count = 0
    byte_count = 0
    token_count = 0
    used_ids: set[int] = set()
    failures = 0
    failure_samples: list[dict[str, str]] = []
    for doc in _bounded_evaluation_docs(
        docs,
        max_documents=max(1, int(max_documents)),
        max_bytes=max(0, int(max_bytes)),
    ):
        text = str(doc["text"])
        ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        doc_count += 1
        byte_count += len(text.encode("utf-8"))
        token_count += len(ids)
        used_ids.update(ids)
        if decoded != text:
            failures += 1
            if len(failure_samples) < 8:
                failure_samples.append(
                    {
                        "source": str(doc.get("source") or "")[-240:],
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                )
        if progress is not None and (doc_count == 1 or doc_count % 25 == 0):
            progress(doc_count, byte_count)
    vocabulary = int(len(tokenizer))
    return {
        "documents": doc_count,
        "utf8_bytes": byte_count,
        "tokens": token_count,
        "bytes_per_token": round(byte_count / token_count, 6) if token_count else None,
        "vocab_size": vocabulary,
        "vocab_ids_used": len(used_ids),
        "vocab_utilization": round(len(used_ids) / vocabulary, 8) if vocabulary else 0.0,
        "round_trip_failures": failures,
        "round_trip_failure_rate": round(failures / doc_count, 8) if doc_count else 0.0,
        "round_trip_failure_samples": failure_samples,
    }


def _document_stream(cfg: dict[str, Any], *, splits: list[str], seed_offset: int = 0) -> Iterator[dict]:
    return iter_corpus_docs(
        cfg["corpora"],
        cfg.get("domains") or None,
        seed=int(cfg.get("seed") or 0) + seed_offset,
        shuffle_buffer=max(1, int(cfg.get("shuffle_buffer") or 128)),
        shuffle_buffer_bytes=max(1, int(cfg.get("shuffle_buffer_bytes") or 64 * 2**20)),
        corpus_mix=cfg.get("corpus_mix") or CORPUS_MIX_CONCATENATE,
        splits=splits or None,
    )


def _canonical_training_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(raw)
    cfg["corpora"] = normalize_corpus_ids(cfg.get("corpora"))
    if not cfg["corpora"]:
        raise ValueError("tokenizer training requires at least one corpus")
    cfg["splits"] = normalize_source_splits(cfg.get("splits")) or ["train", "unspecified"]
    cfg["evaluation_splits"] = normalize_source_splits(
        cfg.get("evaluation_splits") or ["validation", "test"]
    )
    cfg["corpus_mix"] = normalize_corpus_mix(cfg.get("corpus_mix"))
    domains = cfg.get("domains")
    cfg["domains"] = [str(value) for value in domains or () if str(value)] or None
    cfg["spec"] = CompactTokenizerSpec.from_dict(cfg.get("spec")).validate().to_dict()
    return cfg


def _training_progress(rd: Path, spec: CompactTokenizerSpec):
    def update(documents: int, byte_count: int) -> None:
        if documents == 1 or documents % 100 == 0:
            _write_progress(
                rd,
                stage="training tokenizer",
                step=documents,
                total=spec.max_documents,
                value=byte_count,
                label="bytes",
                message=f"{byte_count / 2**20:.1f} MiB streamed; {documents:,} documents",
            )
            log(f"tokenizer intake: docs={documents:,} bytes={byte_count:,}")

    return update


def _evaluate_selected_corpora(
    tokenizer,
    cfg: dict[str, Any],
    rd: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    maximum_documents = max(1, int(cfg.get("evaluation_max_documents") or 512))
    maximum_bytes = max(0, int(cfg.get("evaluation_max_bytes") or 8 * 2**20))

    def evaluate(splits: list[str], mode: str) -> dict[str, Any]:
        return evaluate_tokenizer(
            tokenizer,
            _document_stream(cfg, splits=splits, seed_offset=10_000),
            max_documents=maximum_documents,
            max_bytes=maximum_bytes,
            progress=lambda documents, byte_count: _write_progress(
                rd,
                stage="evaluating tokenizer",
                step=documents,
                total=maximum_documents,
                value=byte_count,
                label="bytes",
                message=f"{mode}; {byte_count / 2**20:.1f} MiB",
            ),
        )

    evaluation_splits = list(cfg.get("evaluation_splits") or ())
    if evaluation_splits:
        evaluation_provenance = corpus_data_provenance(
            cfg["corpora"],
            cfg.get("domains"),
            corpus_mix=cfg["corpus_mix"],
            splits=evaluation_splits,
        )
        # Avoid decompressing a complete corpus just to discover that its
        # manifest contains no validation/test records.
        if int(evaluation_provenance.get("estimated_docs") or 0) > 0:
            metrics = evaluate(evaluation_splits, "held-out " + "+".join(evaluation_splits))
            if metrics["documents"]:
                return metrics, "held_out", evaluation_provenance
    # A deterministic second pass over the training selection is a useful
    # compression smoke test when a corpus has no validation/test split.
    metrics = evaluate(list(cfg["splits"]), "deterministic training-split sample")
    fallback_provenance = corpus_data_provenance(
        cfg["corpora"],
        cfg.get("domains"),
        corpus_mix=cfg["corpus_mix"],
        splits=cfg["splits"],
    )
    return metrics, "deterministic_training_sample", fallback_provenance


def run_train_job(cfg: dict[str, Any], rd: Path) -> dict[str, Any]:
    cfg = _canonical_training_config(cfg)
    tokenizer_id = str(cfg["tokenizer_id"])
    output = tokenizer_dir(tokenizer_id)
    spec = CompactTokenizerSpec.from_dict(cfg["spec"]).validate()
    training_counts = {"documents": 0, "utf8_bytes": 0}
    publish_training_progress = _training_progress(rd, spec)

    def training_progress(documents: int, byte_count: int) -> None:
        training_counts.update(documents=documents, utf8_bytes=byte_count)
        publish_training_progress(documents, byte_count)

    provenance = corpus_data_provenance(
        cfg["corpora"],
        cfg.get("domains"),
        corpus_mix=cfg["corpus_mix"],
        splits=cfg["splits"],
    )
    log(
        f"byte-BPE training: vocab={spec.vocab_size:,} profile={spec.special_token_profile} "
        f"corpora={len(cfg['corpora'])} splits={'+'.join(cfg['splits'])}"
    )
    tokenizer = train_compact_tokenizer(
        (doc.get("text", "") for doc in _document_stream(cfg, splits=cfg["splits"])),
        output,
        spec,
        progress=training_progress,
    )
    if not training_counts["documents"]:
        raise ValueError("selected corpora/splits produced no tokenizer training documents")
    evaluation, evaluation_mode, evaluation_provenance = _evaluate_selected_corpora(
        tokenizer,
        cfg,
        rd,
    )
    checksum = artifact_checksum(output)
    stats = {
        "training": training_counts,
        "evaluation": evaluation,
        "evaluation_mode": evaluation_mode,
        "evaluation_provenance": evaluation_provenance,
        "requested_vocab_size": int(spec.vocab_size),
        "actual_vocab_size": int(len(tokenizer)),
        "tokenizer_payload_sha256": checksum,
        "provenance": provenance,
    }
    artifact_config = {**cfg, "spec": spec.to_dict()}
    write_json(
        output / "yylab_tokenizer.json",
        {
            "format": "yylab_tokenizer_v1",
            "kind": "byte_bpe",
            "config": artifact_config,
            "stats": stats,
        },
    )
    if artifact_checksum(output) != checksum:
        raise RuntimeError("tokenizer payload checksum changed while writing yyLab metadata")
    mark_tokenizer_ready(
        tokenizer_id,
        path=output,
        kind="byte_bpe",
        vocab_size=len(tokenizer),
        config=artifact_config,
        stats=stats,
    )
    return stats


def run_import_job(cfg: dict[str, Any], rd: Path) -> dict[str, Any]:
    tokenizer_id = str(cfg["tokenizer_id"])
    source_kind = str(cfg.get("source_kind") or "").strip()
    if source_kind:
        source_ref = str(cfg.get("source") or "").strip()
    else:
        source_kind, source_ref = parse_tokenizer_source(cfg.get("source"))
    revision = str(cfg.get("hub_revision") or "").strip() or None
    hf_token = (get_settings().get("hf_token") or None) if source_kind == "hub" else None
    _write_progress(
        rd,
        stage="importing tokenizer",
        step=0,
        total=1,
        value=0,
        label="files",
        message=source_ref,
    )
    tokenizer, special = import_tokenizer_artifact(
        source_ref,
        tokenizer_dir(tokenizer_id),
        special_token_profile=str(cfg.get("special_token_profile") or "auto"),
        custom_special_tokens=cfg.get("custom_special_tokens") or (),
        source_kind=source_kind,
        revision=revision,
        hf_token=hf_token,
    )
    hub_commit = ""
    if source_kind == "hub":
        snapshot = _hub_snapshot_dir(source_ref, revision, hf_token)
        source_checksum = tokenizer_source_checksum(snapshot) if snapshot is not None else ""
        hub_commit = snapshot.name if snapshot is not None else ""
    else:
        source_checksum = tokenizer_source_checksum(source_ref)
    output = tokenizer_dir(tokenizer_id)
    payload_checksum = artifact_checksum(output)
    probe_text = "yyLab tokenizer probe: ASCII, λ, 数学, 🧪\n"
    evaluation = evaluate_tokenizer(
        tokenizer,
        [{"text": probe_text, "source": "built-in-import-probe"}],
        max_documents=1,
        max_bytes=len(probe_text.encode("utf-8")),
    )
    vocab_size = effective_vocab_size(tokenizer)
    artifact_config = {
        **cfg,
        "source": source_ref,
        "source_kind": source_kind,
        "hub_revision": revision or "",
        "hub_commit": hub_commit,
        "special_token_configuration": special,
    }
    stats = {
        "actual_vocab_size": int(len(tokenizer)),
        "effective_vocab_size": int(vocab_size),
        "source_sha256": source_checksum,
        "tokenizer_payload_sha256": payload_checksum,
        "evaluation": evaluation,
    }
    write_json(
        output / "yylab_tokenizer.json",
        {
            "format": "yylab_tokenizer_v1",
            "kind": "imported",
            "config": artifact_config,
            "stats": stats,
        },
    )
    if artifact_checksum(output) != payload_checksum:
        raise RuntimeError("tokenizer payload checksum changed while writing yyLab metadata")
    mark_tokenizer_ready(
        tokenizer_id,
        path=output,
        kind="imported",
        vocab_size=vocab_size,
        config=artifact_config,
        stats=stats,
    )
    _write_progress(
        rd,
        stage="importing tokenizer",
        step=1,
        total=1,
        value=1,
        label="files",
        message=f"ready; {vocab_size:,} vocabulary entries",
    )
    return stats


def execute_run(run_id: str) -> dict[str, Any]:
    run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        raise ValueError(f"run {run_id!r} not found")
    if run["kind"] != TOKENIZER_RUN_KIND:
        raise ValueError(f"tokenizer worker requires run kind {TOKENIZER_RUN_KIND!r}")
    cfg = json.loads(run.get("config_json") or "{}")
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "config.json", cfg)
    db.execute(
        "UPDATE runs SET status='running',started_at=?,log_path=? WHERE id=?",
        (db.now_iso(), str(rd / "train.log"), run_id),
    )
    db.execute("UPDATE tokenizers SET status='processing' WHERE id=?", (cfg.get("tokenizer_id"),))
    action = str(cfg.get("action") or TOKENIZER_ACTION_TRAIN).casefold()
    log(f"yylab tokenizer worker: run={run_id} action={action}")
    if action == TOKENIZER_ACTION_TRAIN:
        return run_train_job(cfg, rd)
    if action == TOKENIZER_ACTION_IMPORT:
        return run_import_job(cfg, rd)
    raise ValueError(f"unknown tokenizer action: {action}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    try:
        summary = execute_run(args.run)
        db.execute(
            "UPDATE runs SET status='done',finished_at=?,metrics_json=? WHERE id=?",
            (db.now_iso(), json.dumps(summary), args.run),
        )
        log("done")
    except Exception as exc:  # noqa: BLE001
        error = f"{exc}\n{traceback.format_exc()}"
        log(error)
        db.execute(
            "UPDATE runs SET status='failed',finished_at=?,error=? WHERE id=?",
            (db.now_iso(), error[-4000:], args.run),
        )
        run = db.q1("SELECT config_json FROM runs WHERE id=?", (args.run,))
        try:
            cfg = json.loads((run or {}).get("config_json") or "{}")
        except (TypeError, ValueError):
            cfg = {}
        if cfg.get("tokenizer_id"):
            mark_tokenizer_failed(str(cfg["tokenizer_id"]), error)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
