"""Versioned, finite token quotas for Sprint causal training.

Pack complete source pools once, sample rows without replacement before an
explicitly permitted replay, then shuffle the mixture before its first update.
Every row belongs to one source and records its original pool position. This
makes actual input/target shares and replay measurable for stopped prefixes.
The old character-balanced stream is deliberately unchanged.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .data import CORPUS_ID_FIELD, DEFAULT_TEXT_CHUNK_CHARS, iter_corpus_docs
from .prepared import prepare_for_training
from .sprint_packing import SprintCausalBatches, make_sprint_tokenize
from .sprint_row_filter import QuotaRowFilter, normalize_row_filter

MODE = "fixed_token_quota"
VERSION = "sprint-fixed-token-quota-v1"
POOL_VERSION = "sprint-source-row-pools-v1"
PROVENANCE_DTYPES = {"row_source": "<u4", "row_pool_index": "<u8"}


def normalize_quota_config(cfg: Mapping[str, Any]) -> dict:
    """Validate after shared corpus admission; never reinterpret legacy modes."""
    result = dict(cfg)
    if result.get("corpus_mix") != MODE:
        return result
    if result.get("objective") != "causal_lm":
        raise ValueError("fixed_token_quota currently supports causal_lm only")
    if result.get("fixed_token_mix") is not None:
        raise ValueError("fixed_token_quota cannot replace a materialized fixed_token_mix")
    if result.get("source_max_ratio"):
        raise ValueError("fixed_token_quota uses exact corpus_weights; remove source_max_ratio")
    if result.get("epochs", 0) not in (0, 1):
        raise ValueError("fixed_token_quota uses one finite allocation; control replay with quota_max_source_epochs")
    result.setdefault("quota_version", VERSION)
    if result["quota_version"] != VERSION:
        raise ValueError("unsupported Sprint token quota version")
    row_filter = normalize_row_filter(result.get("quota_row_filter"))
    if row_filter is None:
        result.pop("quota_row_filter", None)
    else:
        result["quota_row_filter"] = row_filter
    corpora = result["corpora"]
    weights = {cid: (result.get("corpus_weights") or {}).get(cid, 1.0) for cid in corpora}
    if any(isinstance(w, bool) or not math.isfinite(float(w)) or float(w) <= 0 for w in weights.values()):
        raise ValueError("quota corpus weights must be finite and positive")
    # Keep original weights: recomputing from already rounded shares on resume
    # can move a largest-remainder tie and change exact row allocation.
    result["corpus_weights"] = {cid: float(w) for cid, w in weights.items()}
    total = sum(Fraction(str(w)) for w in result["corpus_weights"].values())
    result["quota_token_shares"] = {
        cid: float(Fraction(str(w)) / total) for cid, w in result["corpus_weights"].items()
    }
    cap = result.get("quota_max_source_epochs", 1.0)
    if isinstance(cap, Mapping):
        if set(cap) - set(corpora):
            raise ValueError("quota replay cap references an unselected corpus")
        caps = {cid: cap.get(cid, 1.0) for cid in corpora}
    else:
        caps = {cid: cap for cid in corpora}
    for value in caps.values():
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 1:
            raise ValueError("quota_max_source_epochs must be finite and at least one")
    result["quota_max_source_epochs"] = {cid: float(value) for cid, value in caps.items()}
    return result


def allocate_rows(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder quotas; rounding error is less than one full row."""
    if isinstance(total, bool) or int(total) != total or total < 1 or not weights:
        raise ValueError("token quotas require a positive row budget and source weights")
    values = [Fraction(str(value)) for value in weights.values()]
    if any(value <= 0 for value in values):
        raise ValueError("token quota weights must be positive")
    exact = [total * value / sum(values) for value in values]
    rows = [int(value) for value in exact]
    for index in sorted(range(len(rows)), key=lambda i: (-(exact[i] - rows[i]), i))[:total - sum(rows)]:
        rows[index] += 1
    return dict(zip(weights, rows, strict=True))


def quota_identity(cfg: Mapping[str, Any]) -> dict:
    cfg = normalize_quota_config(cfg)
    result = {
        "version": cfg["quota_version"],
        "rows_by_corpus": allocate_rows(int(cfg["max_rows"]), cfg["corpus_weights"]),
        "max_source_epochs": cfg["quota_max_source_epochs"],
        "first_epoch_order": "global_row_permutation_v1",
    }
    if cfg.get("quota_row_filter"):
        result["row_filter"] = cfg["quota_row_filter"]
    return result


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_info(path: Path, dtype: str) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"dtype": dtype, "bytes": path.stat().st_size, "sha256": digest}


def _verify_file(path: Path, info: dict, expected: int, *, verify: str) -> None:
    if path.stat().st_size != expected or info["bytes"] != expected:
        raise ValueError(f"Sprint quota cache size mismatch: {path.name}")
    if verify == "sha256" and _file_info(path, info["dtype"])["sha256"] != info["sha256"]:
        raise ValueError(f"Sprint quota cache checksum mismatch: {path.name}")


def _pool_identity(identity: dict) -> dict:
    selection = identity["selection"]
    result = {
        "schema": POOL_VERSION,
        "selection": {key: selection[key] for key in (
            "corpora", "domains", "splits", "cross_corpus_dedup", "objective", "seq_len")},
        "max_text_chars": identity["max_text_chars"], "tokenizer": identity["tokenizer"],
        "sources": identity["sources"],
        "ordering": "concatenate_unshuffled_before_per_source_packing",
    }
    if selection.get(MODE, {}).get("row_filter"):
        result["row_filter"] = selection[MODE]["row_filter"]
    return result


def _load_pools(path: Path, identity: dict, verify: str) -> dict:
    receipt = json.loads((path / "manifest.json").read_text("utf-8"))
    if receipt.get("identity") != identity or set(receipt["pools"]) != set(identity["selection"]["corpora"]):
        raise ValueError("Sprint source pool identity mismatch")
    seq_len = int(identity["selection"]["seq_len"])
    for index, cid in enumerate(identity["selection"]["corpora"]):
        pool = receipt["pools"][cid]
        dtype = np.dtype(pool["file"]["dtype"])
        if dtype.str not in {"<u2", "<u4"} or int(pool["rows"]) < 0:
            raise ValueError("invalid Sprint source pool dimensions")
        _verify_file(path / f"source-{index}.bin", pool["file"],
                     int(pool["rows"]) * seq_len * dtype.itemsize, verify=verify)
        if identity.get("row_filter"):
            if pool.get("row_filter", {}).get("policy") != identity["row_filter"]:
                raise ValueError("filtered source pool is missing its filter receipt")
            original = pool.get("original_row_file", {})
            if original.get("dtype") != "<u8":
                raise ValueError("filtered source pool is missing original row provenance")
            _verify_file(path / f"source-{index}.original_rows.bin", original,
                         int(pool["rows"]) * 8, verify=verify)
    return receipt


def _prepare_unfiltered_source_pools(cfg: dict, tokenizer: Any, identity: dict, root: Path,
                                    *, progress=print) -> tuple[Path, dict]:
    """Reuse full source rows across mixture weights, budgets and shuffle seeds.

    Deduplication has the same exact-text contract and corpus precedence as the
    existing causal packer. A source selection/sequence/tokenizer change needs
    a new pool group; token-cache records are reused when enabled. No claim of
    held-out or semantic decontamination is made here.
    """
    pool_identity = _pool_identity(identity)
    key = _digest(pool_identity)
    parent = root / "quota-source-pools"
    parent.mkdir(exist_ok=True)
    destination = parent / key
    verify = str(cfg.get("packed_cache_verify", "size"))
    if destination.exists():
        receipt = _load_pools(destination, pool_identity, verify)
        progress("Reusing complete Sprint source pools; only the requested mixture needs materializing")
        return destination, receipt
    lock = parent / f"{key}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Sprint source-pool preparation active or interrupted; inspect {lock}") from exc
    with os.fdopen(fd, "w") as stream:
        stream.write(f"pid={os.getpid()}\n")
    staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=parent))
    docs = reader = None
    try:
        tokenize = make_sprint_tokenize(tokenizer)
        reader = prepare_for_training(cfg, tokenizer, tokenize, progress=progress)
        docs = iter_corpus_docs(cfg["corpora"], cfg.get("domains"), splits=cfg.get("splits"),
                               shuffle_buffer=1, corpus_mix="concatenate",
                               cross_corpus_dedup=cfg.get("cross_corpus_dedup", True),
                               annotate_source=True, corpus_reader=reader)
        vocab = len(tokenizer)
        if vocab > 0x7FFFFFFF:
            raise ValueError("Sprint token vocabulary exceeds the supported int32 label range")
        dtype = "<u2" if vocab <= 65536 else "<u4"
        corpora = cfg["corpora"]
        seq_len = int(cfg["seq_len"])
        pools = {}
        # With unshuffled concatenate, each source is one contiguous group.
        # Packing separately never puts two corpora in the same training row.
        for cid, records in itertools.groupby(docs, key=lambda doc: doc[CORPUS_ID_FIELD]):
            index = corpora.index(cid)
            if cid in pools:
                raise ValueError("source pool stream was unexpectedly interleaved")
            path = staging / f"source-{index}.bin"
            packed = SprintCausalBatches(
                records, tokenize, seq_len=seq_len, eos_token_id=tokenizer.eos_token_id,
                max_rows=None, rows_per_chunk=max(1, min(256, 2**20 // seq_len)),
                tokenize_chunk=int(cfg.get("tokenize_chunk", 8)),
                tokenization_batch_size=int(cfg.get("tokenization_batch_size", 128)),
                max_text_chars=int(cfg.get("max_text_chars", DEFAULT_TEXT_CHUNK_CHARS)))
            rows = 0
            progress(f"Preparing complete source pool: {cid}")
            with path.open("wb") as stream:
                for batch in packed:
                    ids = np.asarray(batch["input_ids"])
                    if ids.ndim != 2 or ids.shape[1] != seq_len or not len(ids):
                        raise ValueError("Sprint source pools require complete fixed-length rows")
                    if ids.min() < 0 or ids.max() >= vocab:
                        raise ValueError("source pool token ID lies outside the tokenizer vocabulary")
                    stream.write(np.asarray(ids, dtype=dtype).tobytes())
                    rows += len(ids)
                stream.flush()
                os.fsync(stream.fileno())
            pools[cid] = {"rows": rows, "file": _file_info(path, dtype), "data_stats": packed.data_stats()}
            progress(f"Source pool ready: {cid}, {rows:,} rows / {rows * seq_len:,} usable tokens")
        for index, cid in enumerate(corpora):
            if cid not in pools:
                path = staging / f"source-{index}.bin"
                path.write_bytes(b"")
                pools[cid] = {"rows": 0, "file": _file_info(path, dtype), "data_stats": {}}
        # Sources may be reimported while the CPU packer runs. Never publish a
        # pool built from mixed generations under an earlier source identity.
        from .sprint import _packed_identity
        if _pool_identity(_packed_identity(cfg, tokenizer)) != pool_identity:
            raise ValueError("corpus or tokenizer changed during Sprint source pooling")
        receipt = {"identity": pool_identity, "pools": pools}
        (staging / "manifest.json").write_text(_json(receipt), encoding="utf-8")
        staging.rename(destination)
        return destination, receipt
    finally:
        if docs is not None:
            docs.close()
        if reader is not None:
            reader.close()
        if staging.exists() and staging.resolve().is_relative_to(parent.resolve()):
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def prepare_source_pools(cfg: dict, tokenizer: Any, identity: dict, root: Path,
                         *, progress=print) -> tuple[Path, dict]:
    """Reuse raw pools, then optionally publish immutable cleaned derivatives.

    The filtering policy is part of both quota and derived-pool identities.
    Enabling it never mutates or relabels old pools and never needs their input
    documents tokenized again. Filtered pool positions have an explicit mapping
    back to original complete rows in the unfiltered pool group.
    """
    policy = normalize_row_filter(cfg.get("quota_row_filter"))
    if policy is None:
        return _prepare_unfiltered_source_pools(cfg, tokenizer, identity, root, progress=progress)
    pool_identity = _pool_identity(identity)
    if pool_identity.get("row_filter") != policy:
        raise ValueError("quota row filter policy is missing from the pool identity")
    key = _digest(pool_identity)
    parent = root / "quota-source-pools"
    parent.mkdir(exist_ok=True)
    destination = parent / key
    verify = str(cfg.get("packed_cache_verify", "size"))
    if destination.exists():
        receipt = _load_pools(destination, pool_identity, verify)
        progress("Reusing cleaned Sprint source pools")
        return destination, receipt
    raw_cfg = dict(cfg)
    raw_cfg.pop("quota_row_filter", None)
    raw_identity = {**identity, "selection": {**identity["selection"]}}
    raw_identity["selection"][MODE] = {**identity["selection"][MODE]}
    raw_identity["selection"][MODE].pop("row_filter", None)
    raw_root, raw_receipt = _prepare_unfiltered_source_pools(
        raw_cfg, tokenizer, raw_identity, root, progress=progress)
    lock = parent / f"{key}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Sprint filtered-pool preparation active or interrupted; inspect {lock}") from exc
    with os.fdopen(fd, "w") as stream:
        stream.write(f"pid={os.getpid()}\n")
    staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=parent))
    try:
        pools = {}
        seq_len = int(cfg["seq_len"])
        state_path = staging / "filter-state.sqlite3"
        with QuotaRowFilter(state_path, policy) as row_filter:
            for index, cid in enumerate(cfg["corpora"]):
                raw_pool = raw_receipt["pools"][cid]
                dtype = raw_pool["file"]["dtype"]
                output_path = staging / f"source-{index}.bin"
                original_path = staging / f"source-{index}.original_rows.bin"
                progress(f"Removing exact duplicate and strongly repetitive rows: {cid}")
                with (raw_root / f"source-{index}.bin").open("rb") as input_stream, \
                        output_path.open("wb") as output_stream, original_path.open("wb") as original_stream:
                    for start in range(0, int(raw_pool["rows"]), 256):
                        count = min(256, int(raw_pool["rows"]) - start)
                        ids = np.fromfile(input_stream, dtype=dtype, count=count * seq_len).reshape(count, seq_len)
                        keep = row_filter.filter_batch(ids, source=cid, row_offset=start)
                        ids[keep].tofile(output_stream)
                        np.arange(start, start + count, dtype="<u8")[keep].tofile(original_stream)
                        if (start + count) % 16384 == 0:
                            progress(f"Checking source pool {cid}: {start + count:,} / {raw_pool['rows']:,} rows")
                    for stream in (output_stream, original_stream):
                        stream.flush()
                        os.fsync(stream.fileno())
                filtered = row_filter.source_receipt(cid)
                rows = filtered["retained_rows"]
                slots = rows * seq_len
                targets = rows * (seq_len - 1)
                pools[cid] = {
                    "rows": rows, "file": _file_info(output_path, dtype),
                    "original_row_file": _file_info(original_path, "<u8"),
                    "row_filter": filtered,
                    "raw_data_stats": raw_pool["data_stats"],
                    "data_stats": {
                        "scope": "retained_complete_source_pool_rows_after_filter",
                        "sequences": rows,
                        "tokens": {"forwarded_model_slots": slots, "forwarded_real_inputs": slots,
                                   "covered_rendered_positions": slots, "packing_replay_inputs": 0,
                                   "padding_inputs": 0, "model_target_slots": targets,
                                   "optimizer_supervised_targets": targets, "ignored_target_slots": 0},
                        "accounting_note": "Retained row/input/target counts only. Original document/rendered-content/EOS counters remain separately under raw_data_stats; selected document boundaries are not inferred.",
                    },
                }
                progress(f"Cleaned source pool {cid}: kept {rows:,} / {filtered['input_rows']:,} rows")
        state_path.unlink()
        from .sprint import _packed_identity
        if _pool_identity(_packed_identity(cfg, tokenizer)) != pool_identity:
            raise ValueError("corpus or tokenizer changed during Sprint source pool filtering")
        receipt = {
            "identity": pool_identity, "pools": pools,
            "unfiltered_pool_key": raw_root.name,
            "unfiltered_pool_manifest": _file_info(raw_root / "manifest.json", "json"),
            "row_position_scope": "row_pool_index addresses retained rows; source-N.original_rows.bin maps each retained row to its original unfiltered source row offset",
        }
        (staging / "manifest.json").write_text(_json(receipt), encoding="utf-8")
        staging.rename(destination)
        return destination, receipt
    finally:
        if staging.exists() and staging.resolve().is_relative_to(parent.resolve()):
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def sample_pool_rows(pool_rows: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Every complete cycle covers the full pool before any position repeats."""
    if count < 0 or pool_rows < 1:
        raise ValueError("cannot sample an empty source pool")
    output = np.empty(count, dtype="<u8")
    for start in range(0, count, pool_rows):
        size = min(pool_rows, count - start)
        output[start:start + size] = rng.choice(pool_rows, size=size, replace=False)
    return output


def quota_prefix_stats(provenance: Mapping[str, np.ndarray], corpora: list[str],
                       pools: Mapping[str, Any], seq_len: int, rows: int) -> dict:
    if rows < 0 or rows > len(provenance["row_source"]):
        raise ValueError("invalid token quota prefix length")
    sources, positions = provenance["row_source"][:rows], provenance["row_pool_index"][:rows]
    per_corpus, covered = {}, 0
    for index, cid in enumerate(corpora):
        chosen = positions[sources == index]
        count, unique = len(chosen), len(np.unique(chosen))
        covered += unique * seq_len
        per_corpus[cid] = {
            "sequences": count, "forwarded_real_inputs": count * seq_len,
            "optimizer_supervised_targets": count * (seq_len - 1),
            "covered_rendered_positions": unique * seq_len,
            "packing_replay_inputs": (count - unique) * seq_len,
            "input_token_share": count / rows if rows else 0.0,
            "available_pool_rows": int(pools[cid]["rows"]),
            "pool_exposures": count / int(pools[cid]["rows"]) if pools[cid]["rows"] else 0.0,
        }
    slots, targets = rows * seq_len, rows * (seq_len - 1)
    result = {
        "scope": "fixed_token_quota_prefix", "sequences": rows,
        "source_attribution": "exact_row_source_and_pool_position_v1",
        "unique_position_scope": ("retained_source_pool_positions_after_configured_row_filter"
                                  if any(pool.get("row_filter") for pool in pools.values())
                                  else "source_pool_positions_after_configured_exact_dedup"),
        "tokens": {"forwarded_real_inputs": slots, "forwarded_model_slots": slots,
                   "covered_rendered_positions": covered, "packing_replay_inputs": slots - covered,
                   "padding_inputs": 0, "model_target_slots": targets,
                   "optimizer_supervised_targets": targets, "ignored_target_slots": 0},
        "per_corpus": per_corpus,
    }
    filtered = {cid: pool["row_filter"] for cid, pool in pools.items() if pool.get("row_filter")}
    if filtered:
        result["row_filter"] = {
            "scope": "complete_source_pool_preparation_not_consumed_prefix",
            "per_corpus": filtered,
        }
    return result


def materialize_quota(cfg: dict, tokenizer: Any, identity: dict, root: Path,
                       staging: Path, *, progress=print) -> dict:
    """Write model arrays and exact row provenance into the caller's staging dir."""
    pool_root, receipt = prepare_source_pools(cfg, tokenizer, identity, root, progress=progress)
    contract = identity["selection"][MODE]
    corpora = cfg["corpora"]
    allocations = contract["rows_by_corpus"]
    pools = receipt["pools"]
    for cid in corpora:
        count, available = allocations[cid], int(pools[cid]["rows"])
        cap = Fraction(str(contract["max_source_epochs"][cid]))
        allowed = int(available * cap)
        if count and (not available or count > allowed):
            raise ValueError(
                f"fixed_token_quota source {cid!r} needs {count:,} rows / {count * cfg['seq_len']:,} tokens; "
                f"its available pool has {available:,} rows and quota_max_source_epochs={float(cap):g} "
                f"permits {allowed:,}. Reduce its weight/budget or explicitly permit more replay; "
                "other sources will not silently replace it.")
    rows, seq_len = int(cfg["max_rows"]), int(cfg["seq_len"])
    seed = int(cfg.get("seed", 42))
    sources = np.repeat(np.arange(len(corpora), dtype="<u4"), [allocations[cid] for cid in corpora])
    positions = np.empty(rows, dtype="<u8")
    offset = 0
    for index, cid in enumerate(corpora):
        count = allocations[cid]
        if count:
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
            positions[offset:offset + count] = sample_pool_rows(int(pools[cid]["rows"]), count, rng)
        offset += count
    order = np.random.default_rng(np.random.SeedSequence([seed, 2])).permutation(rows)
    sources, positions = sources[order], positions[order]
    del order
    provenance = {"row_source": sources, "row_pool_index": positions}
    provenance_files = {}
    for name, values in provenance.items():
        path = staging / f"{name}.bin"
        with path.open("wb") as stream:
            values.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        provenance_files[name] = _file_info(path, PROVENANCE_DTYPES[name])
    dtype = next(iter(pools.values()))["file"]["dtype"]
    maps = {}
    output_path = staging / "input_ids.bin"
    try:
        for index, cid in enumerate(corpora):
            if allocations[cid]:
                maps[index] = np.memmap(pool_root / f"source-{index}.bin", dtype=dtype, mode="r",
                                        shape=(int(pools[cid]["rows"]), seq_len))
        with output_path.open("wb") as stream:
            for start in range(0, rows, 256):
                end = min(rows, start + 256)
                chunk = np.empty((end - start, seq_len), dtype=dtype)
                for index in np.unique(sources[start:end]):
                    selected = sources[start:end] == index
                    chunk[selected] = maps[int(index)][positions[start:end][selected]]
                chunk.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        for array in maps.values():
            array._mmap.close()
    stats = quota_prefix_stats(provenance, corpora, pools, seq_len, rows)
    stats["scope"] = "fixed_token_quota_allocation"
    stats["requested_token_shares"] = normalize_quota_config(cfg)["quota_token_shares"]
    if receipt.get("unfiltered_pool_key"):
        stats["unique_position_scope"] = "retained_source_pool_positions_after_configured_row_filter"
    for cid, source in stats["per_corpus"].items():
        progress(f"Token quota {cid}: {source['forwarded_real_inputs']:,} tokens "
                 f"({source['input_token_share']:.2%}); {source['packing_replay_inputs']:,} replayed positions")
    return {
        "rows": rows, "seq_len": seq_len, "data_stats": stats,
        "files": {"input_ids": _file_info(output_path, dtype)},
        "row_provenance": {"version": VERSION, "files": provenance_files,
                           "corpora": corpora, "pools": pools, "pool_key": pool_root.name,
                           **({"unfiltered_pool_key": receipt["unfiltered_pool_key"],
                               "row_position_scope": receipt["row_position_scope"]}
                              if receipt.get("unfiltered_pool_key") else {})},
    }
