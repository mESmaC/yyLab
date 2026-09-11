"""Sprint's single-device, throughput-oriented pretraining and SFT runner.

The reusable packed cache is the data contract: CPU workers copy fixed rows
from memory maps, and checkpoints record the exact next row. CUDA defaults to
BF16 model storage and bucketed FP32 CPU optimizer state; FP32 GPU-resident
AdamW remains available when the model fits and throughput takes precedence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import db
from ..artifacts import checkpoint_dir
from ..config import PATHS
from ..tokenizer_registry import get_tokenizer, resolve_tokenizer_artifact
from .data import (
    DEFAULT_TEXT_CHUNK_CHARS,
    OBJECTIVE_CHAT_SFT,
    ChatSFTBatches,
    PackedBatches,
    iter_corpus_docs,
    normalize_corpus_ids,
    normalize_source_splits,
)
from .loop import CPUOffloadAdamW
from .prefetch import BufferedBatches, batch_stream
from .prepared import prepare_for_training, source_snapshot, tokenizer_fingerprint

PACKED_SCHEMA = "sprint-packed-v1"
TRAINER_SCHEMA = "sprint-trainer-v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json(value), encoding="utf-8")
    os.replace(temporary, path)


def _positive_int(value: Any, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or int(value) != float(value) or int(value) < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return int(value)


def normalize_sprint_config(config: Mapping[str, Any]) -> dict:
    """Accept UI aliases and validate the complete training request."""
    # Shared corpus admission and mixing rules remain authoritative in yyLab.
    from ..workers.train import normalize_base_data_config

    cfg = dict(config)
    if cfg.get("objective") == "pretrain":
        cfg["objective"] = "causal_lm"
    cfg = normalize_base_data_config(cfg)
    defaults = {
        "seq_len": 1024, "micro_batch": 8, "grad_accum": 1, "seed": 42,
        "max_steps": 0, "epochs": 0, "warmup_steps": 0, "eval_steps": 20,
        "data_prefetch_batches": 2, "shuffle_buffer": 128,
        "shuffle_buffer_bytes": 64 * 2**20, "shuffle_block_rows": 1024,
    }
    for key, default in defaults.items():
        cfg[key] = _positive_int(cfg.get(key, default), key, zero=key in {
            "seed", "max_steps", "epochs", "warmup_steps", "data_prefetch_batches", "shuffle_block_rows",
        })
    if cfg["seq_len"] < 2:
        raise ValueError("Sprint seq_len must be at least two")
    for canonical, alias, default in (
        ("log_interval", "log_every", 10), ("save_interval", "save_every", 0),
        ("eval_interval", "eval_every", 0),
    ):
        cfg[canonical] = _positive_int(cfg.get(canonical, cfg.get(alias)) or default, canonical,
                                       zero=canonical != "log_interval")
    budget = cfg.get("token_budget", cfg.get("requested_training_tokens", cfg.get("training_tokens", 0)))
    cfg["token_budget"] = _positive_int(budget or 0, "token_budget", zero=True)
    if cfg["token_budget"] and cfg["token_budget"] < cfg["seq_len"]:
        raise ValueError("token_budget must cover at least one complete sequence")
    budget_rows = cfg["token_budget"] // cfg["seq_len"]
    rows_per_step = cfg["micro_batch"] * cfg["grad_accum"]
    if budget_rows:
        steps = math.ceil(budget_rows / rows_per_step)
        cfg["max_steps"] = min(cfg["max_steps"], steps) if cfg["max_steps"] else steps
    if not cfg["max_steps"]:
        raise ValueError("Sprint requires max_steps or token_budget")
    cfg["max_rows"] = min(budget_rows or cfg["max_steps"] * rows_per_step,
                          cfg["max_steps"] * rows_per_step)
    for key, default, low, high in (
        ("lr", 3e-4, 0.0, None), ("weight_decay", 0.1, 0.0, None),
        ("min_lr_ratio", 0.1, 0.0, 1.0), ("grad_clip", 1.0, 0.0, None),
        ("eps", 1e-8, 0.0, None),
    ):
        value = float(cfg.get(key, default))
        if not math.isfinite(value) or value < low or (high is not None and value > high):
            raise ValueError(f"invalid {key}: {value}")
        if key in {"lr", "eps"} and value == 0:
            raise ValueError(f"{key} must be positive")
        cfg[key] = value
    cfg["betas"] = [float(v) for v in cfg.get("betas", (0.9, 0.95))]
    if len(cfg["betas"]) != 2 or any(not 0 <= v < 1 for v in cfg["betas"]):
        raise ValueError("betas must contain two values in [0, 1)")
    cfg["torch_compile"] = cfg.get("torch_compile", cfg.get("compile", False))
    cfg["gradient_checkpointing"] = cfg.get("gradient_checkpointing", cfg.get("activation_checkpointing", False))
    cfg.setdefault("compile_mode", "default")
    if cfg["compile_mode"] not in {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
        raise ValueError("invalid torch.compile mode")
    for key, default in (("torch_compile", False), ("gradient_checkpointing", False),
                         ("token_cache", False), ("pin_memory", True)):
        cfg.setdefault(key, default)
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be Boolean")
    requested_device = str(cfg.get("device", "auto"))
    cpu = requested_device == "cpu" or requested_device == "auto" and not torch.cuda.is_available()
    cfg.setdefault("parameter_dtype", "float32" if cpu else "bfloat16")
    cfg["parameter_dtype"] = {"bf16": "bfloat16", "fp32": "float32"}.get(cfg["parameter_dtype"], cfg["parameter_dtype"])
    if cfg["parameter_dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("Sprint parameter_dtype must be bfloat16 or float32")
    cfg.setdefault("optimizer_state_offload", "cpu")
    if cfg["optimizer_state_offload"] not in {"cpu", "none"}:
        raise ValueError("Sprint optimizer_state_offload must be cpu or none")
    cfg.setdefault("cpu_optimizer_kernel", "auto")
    if cfg["cpu_optimizer_kernel"] not in {"auto", "native", "compiled"}:
        raise ValueError("Sprint cpu_optimizer_kernel must be auto, native or compiled")
    if cfg["cpu_optimizer_kernel"] == "compiled" and (
        cfg["optimizer_state_offload"] != "cpu" or cfg["parameter_dtype"] != "bfloat16"
    ):
        raise ValueError("the compiled CPU optimizer requires BF16 weights and CPU offload")
    if cfg["parameter_dtype"] == "bfloat16" and cfg["optimizer_state_offload"] != "cpu":
        raise ValueError("Sprint BF16 weights require CPU offload to retain FP32 masters and moments")
    if cfg.get("optimizer_state_dtype", "float32") not in {"float32", "fp32"}:
        raise ValueError("Sprint optimizer_state_dtype must be float32")
    cfg["offload_bucket_mb"] = _positive_int(cfg.get("offload_bucket_mb", 32), "offload_bucket_mb")
    if cfg.get("cpu_threads") is not None:
        cfg["cpu_threads"] = _positive_int(cfg["cpu_threads"], "cpu_threads")
    cfg.setdefault("attention_backend", "auto")
    if cfg["attention_backend"] not in {"auto", "cudnn", "efficient", "flash", "math"}:
        raise ValueError("invalid Sprint attention_backend")
    cfg["eval_splits"] = normalize_source_splits(cfg.get("eval_splits")) or None
    cfg["eval_corpora"] = normalize_corpus_ids(cfg.get("eval_corpora")) or cfg["corpora"]
    if cfg["eval_interval"]:
        if not cfg["eval_splits"]:
            raise ValueError("evaluation requires explicit eval_splits")
        if set(cfg["eval_corpora"]) & set(cfg["corpora"]):
            if not cfg["splits"] or set(cfg["eval_splits"]) & set(cfg["splits"]):
                raise ValueError("training and evaluation splits must be disjoint")
        normalize_base_data_config({**cfg, "corpora": cfg["eval_corpora"], "splits": cfg["eval_splits"],
                                    "corpus_mix": "concatenate", "corpus_weights": {}, "source_max_ratio": None})
    cfg["model_config"] = dict(cfg.get("model_config", cfg.get("sprint", {})))
    cfg.setdefault("shape_profile", "throughput")
    if cfg["shape_profile"] not in {"throughput", "balanced"}:
        raise ValueError("Sprint shape_profile must be throughput or balanced")
    cfg["initialization"] = dict(cfg.get("initialization") or {"mode": "scratch"})
    if cfg["initialization"].get("mode", "scratch") not in {"scratch", "checkpoint", "resume"}:
        raise ValueError("initialization mode must be scratch, checkpoint, or resume")
    cfg["model_family"] = "hf"
    cfg["architecture_type"] = "sprint"
    return cfg


def _packed_identity(cfg: Mapping[str, Any], tokenizer: Any) -> dict:
    keys = ("corpora", "domains", "splits", "corpus_mix", "cross_corpus_dedup", "corpus_weights",
            "source_max_ratio", "objective", "seq_len", "seed", "shuffle_buffer", "shuffle_buffer_bytes")
    return {
        "schema": PACKED_SCHEMA,
        "selection": {key: cfg.get(key) for key in keys},
        "max_rows": int(cfg["max_rows"]),
        "max_text_chars": int(cfg.get("max_text_chars", DEFAULT_TEXT_CHUNK_CHARS)),
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "sources": [source_snapshot(cid, verify=str(cfg.get("token_cache_verify_sources", "stat")))
                    for cid in cfg["corpora"]],
    }


def prepare_sprint_cache(cfg: Mapping[str, Any], tokenizer: Any, *, progress=print) -> SprintPackedCache:
    """Pack a bounded corpus pass once; subsequent runs only map fixed rows."""
    identity = _packed_identity(cfg, tokenizer)
    key = hashlib.sha256(_json(identity).encode()).hexdigest()
    root = Path(cfg.get("packed_cache_dir") or PATHS.cache / "sprint-packed")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / key
    if destination.exists():
        cache = SprintPackedCache(destination, verify=str(cfg.get("packed_cache_verify", "size")))
        if cache.manifest["identity"] != identity:
            cache.close()
            raise ValueError("Sprint packed-cache identity mismatch")
        progress(f"Reusing Sprint packed data: {cache.rows:,} rows")
        return cache
    lock = root / f"{key}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Sprint packing already active or interrupted; inspect {lock}") from exc
    with os.fdopen(fd, "w") as stream:
        stream.write(f"pid={os.getpid()}\n")
    staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
    reader = None
    files = {}
    docs = None
    try:
        def tokenize(texts):
            return tokenizer(texts, add_special_tokens=False, return_attention_mask=False, verbose=False)["input_ids"]

        reader = prepare_for_training(cfg, tokenizer, tokenize, progress=progress)
        docs = iter_corpus_docs(
            cfg["corpora"], cfg.get("domains"), splits=cfg.get("splits"),
            shuffle_buffer=int(cfg.get("shuffle_buffer", 128)),
            shuffle_buffer_bytes=int(cfg.get("shuffle_buffer_bytes", 64 * 2**20)),
            seed=int(cfg.get("seed", 42)), corpus_mix=cfg.get("corpus_mix", "concatenate"),
            cross_corpus_dedup=bool(cfg.get("cross_corpus_dedup", True)),
            corpus_weights=cfg.get("corpus_weights") or None, source_max_ratio=cfg.get("source_max_ratio"),
            annotate_source=True, corpus_reader=reader,
        )
        chat = cfg.get("objective") == OBJECTIVE_CHAT_SFT
        pack_opts = dict(seq_len=int(cfg["seq_len"]), micro_batch=1, max_batches=int(cfg["max_rows"]))
        packed = (ChatSFTBatches(docs, tokenizer, **pack_opts) if chat else
                  PackedBatches(docs, tokenize, eos_token_id=tokenizer.eos_token_id,
                                tokenize_chunk=int(cfg.get("tokenize_chunk", 8)),
                                max_text_chars=int(cfg.get("max_text_chars", DEFAULT_TEXT_CHUNK_CHARS)),
                                **pack_opts))
        vocab = len(tokenizer)
        if vocab > 0x7FFFFFFF:
            raise ValueError("Sprint token vocabulary exceeds the supported int32 label range")
        specs = {"input_ids": "<u2" if vocab <= 65536 else "<u4"}
        if chat:
            specs.update(labels="<i4", attention_mask="u1")
        files = {name: (staging / f"{name}.bin").open("wb") for name in specs}
        hashes = {name: hashlib.sha256() for name in specs}
        rows = 0
        progress("Packing Sprint fixed-length training rows")
        for batch in packed:
            ids = np.asarray(batch["input_ids"])
            if ids.min() < 0 or ids.max() >= vocab:
                raise ValueError("packed token ID lies outside the tokenizer vocabulary")
            for name, dtype in specs.items():
                values = np.asarray(batch[name], dtype=dtype)
                if values.shape != (1, int(cfg["seq_len"])):
                    raise ValueError("Sprint packing requires complete fixed-length rows")
                payload = values.tobytes()
                files[name].write(payload)
                hashes[name].update(payload)
            rows += 1
            if rows % 10000 == 0:
                progress(f"Packing Sprint data: {rows:,} rows, {rows * cfg['seq_len']:,} token slots")
        for stream in files.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        if not rows:
            raise ValueError("selected Sprint data contains no complete supervised sequence")
        if identity != _packed_identity(cfg, tokenizer):
            raise ValueError("corpus or tokenizer changed during Sprint packing")
        manifest = {
            "schema": PACKED_SCHEMA, "key": key, "identity": identity, "rows": rows,
            "seq_len": int(cfg["seq_len"]), "data_stats": packed.data_stats(),
            "files": {name: {"dtype": dtype, "bytes": (staging / f"{name}.bin").stat().st_size,
                             "sha256": hashes[name].hexdigest()} for name, dtype in specs.items()},
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
        return SprintPackedCache(destination)
    finally:
        for stream in files.values():
            stream.close()
        if docs is not None:
            docs.close()
        if reader is not None:
            reader.close()
        if staging.exists() and staging.resolve().is_relative_to(root.resolve()):
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


class SprintPackedCache:
    """Read-only packed arrays with deterministic epoch ordering and O(1) seek."""

    def __init__(self, root: str | Path, *, verify: str = "size") -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
        if self.manifest.get("schema") != PACKED_SCHEMA or verify not in {"size", "sha256"}:
            raise ValueError("unsupported Sprint packed cache or verification mode")
        self.rows, self.seq_len = int(self.manifest["rows"]), int(self.manifest["seq_len"])
        if self.rows < 1 or self.seq_len < 2:
            raise ValueError("invalid Sprint packed-cache dimensions")
        self.arrays = {}
        self._orders = {}
        try:
            for name, info in self.manifest["files"].items():
                if name not in {"input_ids", "labels", "attention_mask"}:
                    raise ValueError("invalid Sprint packed-cache field")
                path = self.root / f"{name}.bin"
                dtype = np.dtype(info["dtype"])
                expected = self.rows * self.seq_len * dtype.itemsize
                if path.stat().st_size != expected or expected != info["bytes"]:
                    raise ValueError(f"Sprint packed-cache size mismatch: {name}")
                if verify == "sha256":
                    with path.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != info["sha256"]:
                            raise ValueError(f"Sprint packed-cache checksum mismatch: {name}")
                self.arrays[name] = np.memmap(path, dtype=dtype, mode="r", shape=(self.rows, self.seq_len))
            if "input_ids" not in self.arrays or ("labels" in self.arrays) != ("attention_mask" in self.arrays):
                raise ValueError("incomplete Sprint packed cache")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for array in getattr(self, "arrays", {}).values():
            array._mmap.close()
        self.arrays = {}

    def _indices(self, cursor: int, count: int, seed: int, block_rows: int) -> np.ndarray:
        epoch, offset = divmod(cursor, self.rows)
        if count < 1 or count > self.rows - offset:
            raise ValueError("cache reads must stay inside one epoch")
        positions = np.arange(offset, offset + count, dtype=np.int64)
        if block_rows <= 0 or epoch == 0:
            return positions
        # Only full blocks shuffle; the final short block remains last. This
        # keeps every read a permutation while preserving mostly linear I/O.
        blocks = self.rows // block_rows
        order_key = (epoch, seed, block_rows)
        if order_key not in self._orders:
            self._orders = {order_key: np.random.default_rng(seed + epoch).permutation(blocks)}
        order = self._orders[order_key]
        active = positions < blocks * block_rows
        positions[active] = order[positions[active] // block_rows] * block_rows + positions[active] % block_rows
        return positions

    def read_batch(self, cursor: int, count: int, *, seed: int = 42, block_rows: int = 1024,
                   pin_memory: bool = False) -> dict:
        indices = self._indices(cursor, count, seed, block_rows)
        # Advanced indexing copies away from mmap ownership; no training tensor
        # outlives a mapped buffer or writes into the shared on-disk cache.
        batch = {name: torch.from_numpy(np.asarray(array[indices], dtype=np.int64))
                 for name, array in self.arrays.items()}
        labels = batch.get("labels", batch["input_ids"])
        batch["num_targets"] = int((labels[:, 1:] != -100).sum())
        batch["num_tokens"] = count * self.seq_len
        batch["real_tokens"] = int(batch["attention_mask"].sum()) if "attention_mask" in batch else batch["num_tokens"]
        if pin_memory:
            batch = {key: value.pin_memory() if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
        return batch


class _CacheBatches:
    def __init__(self, cache: SprintPackedCache, cfg: dict, cursor: int, limit: int, *, pin_memory: bool):
        self.cache, self.cfg, self.cursor, self.limit = cache, cfg, cursor, limit
        self.pin_memory = pin_memory

    def data_stats(self):
        return {"next_row": self.cursor, "packed_key": self.cache.manifest["key"]}

    def __iter__(self):
        while self.cursor < self.limit:
            remaining = min(self.cfg["micro_batch"], self.limit - self.cursor)
            parts = []
            while remaining:
                count = min(remaining, self.cache.rows - self.cursor % self.cache.rows)
                parts.append(self.cache.read_batch(self.cursor, count, seed=self.cfg["seed"],
                                                  block_rows=self.cfg["shuffle_block_rows"]))
                self.cursor += count
                remaining -= count
            batch = parts[0] if len(parts) == 1 else {
                key: torch.cat([part[key] for part in parts]) if isinstance(value, torch.Tensor)
                     else sum(part[key] for part in parts)
                for key, value in parts[0].items()
            }
            if self.pin_memory:
                batch = {key: value.pin_memory() if isinstance(value, torch.Tensor) else value
                         for key, value in batch.items()}
            yield batch


def sprint_device(value: str = "auto") -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Sprint supports CPU and CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def sprint_learning_rate(cfg: Mapping[str, Any], step: int) -> float:
    """Zero-based update index; warmup followed by cosine decay."""
    lr = float(cfg.get("lr", 3e-4))
    warmup = int(cfg.get("warmup_steps", 0))
    total = max(1, int(cfg.get("max_steps", 1)))
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    minimum = float(cfg.get("min_lr_ratio", 0.1))
    return lr * (minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress)))


@dataclass
class SprintStepResult:
    loss: torch.Tensor
    tokens: int
    supervised_tokens: int
    real_tokens: int


class SprintCPUOffloadAdamW(CPUOffloadAdamW):
    """FP32 CPU masters/moments with pipelined BF16 transfers and fused CPU Adam.

    Large tensors are split into bounded views, so a vocabulary matrix never
    requires a full-size gradient or Adam denominator temporary in system RAM.
    CUDA uses two pinned buffers to overlap the next gradient transfer and the
    previous weight writeback with the current CPU update. Each buffer's reuse
    waits for its writeback event. FP32 gradient scratch is shared and bounded.
    The saved per-parameter state remains compatible with yyLab's CPU offloader.
    """

    def __init__(self, params, *, bucket_mb=32, cpu_kernel="auto", **kwargs):
        if cpu_kernel not in {"auto", "native", "compiled"}:
            raise ValueError("Sprint CPU optimizer kernel must be auto, native, or compiled")
        super().__init__(params, **kwargs)
        parameters = [p for group in self.param_groups for p in group["params"]]
        self.device = parameters[0].device
        self.parameter_dtype = parameters[0].dtype
        if any(p.device != self.device or p.dtype != self.parameter_dtype for p in parameters):
            raise ValueError("Sprint CPU offload requires one parameter device and dtype")
        self.bucket_elements = max(1, int(bucket_mb) * 2**20 // parameters[0].element_size())
        self._staging = self._gradient = None
        self._staging_buffers = []
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._writeback_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._read_events = [torch.cuda.Event() for _ in range(2)] if self._stream is not None else []
        self._writeback_events = [None, None]
        self.cpu_fused = torch._C._dispatch_has_kernel_for_dispatch_key("aten::_fused_adamw_", "CPU")
        from yylab.train.sprint_cpu_adam import MIN_COMPILED_ELEMENTS, CompiledCPUAdam

        self._compiled_cpu_adam = None
        self._min_compiled_elements = MIN_COMPILED_ELEMENTS
        self.resolved_cpu_kernel = "native"
        self.cpu_kernel_reason = "native CPU Adam explicitly selected"
        if cpu_kernel != "native":
            representative = min(self.bucket_elements, max(p.numel() for p in parameters))
            eligible = self.parameter_dtype == torch.bfloat16 and representative >= MIN_COMPILED_ELEMENTS
            if not eligible:
                self.cpu_kernel_reason = "compiled CPU Adam requires BF16 weights and a slice of at least 1,048,576 elements"
                if cpu_kernel == "compiled":
                    raise ValueError(self.cpu_kernel_reason)
            else:
                try:
                    self._compiled_cpu_adam = CompiledCPUAdam(
                        cpu_threads=torch.get_num_threads(), warmup_elements=representative)
                except Exception as error:
                    # Preparation only touched disposable buffers. Once live
                    # updates begin, exceptions must propagate without retry.
                    self.cpu_kernel_reason = f"compiled CPU Adam preparation failed: {type(error).__name__}: {error}"
                    if cpu_kernel == "compiled":
                        raise RuntimeError(self.cpu_kernel_reason) from error
                    print(f"Sprint: using native CPU Adam; {self.cpu_kernel_reason}", flush=True)
                else:
                    self.resolved_cpu_kernel = "compiled"
                    self.cpu_kernel_reason = "compiled CPU Adam for large BF16 slices; native CPU Adam for small slices"

    def synchronize(self) -> None:
        """Make CPU-side checkpoint/state operations observe completed writebacks."""
        if self._stream is not None:
            self._stream.synchronize()
            self._writeback_stream.synchronize()

    def state_dict(self) -> dict:
        self.synchronize()
        return super().state_dict()

    def initialize_master_weights(self, initial_weights: list[torch.Tensor]) -> None:
        self.synchronize()
        parameters = [p for group in self.param_groups for p in group["params"]]
        if len(parameters) != len(initial_weights):
            raise ValueError("offload master weights do not match the model")
        for parameter, master in zip(parameters, initial_weights, strict=True):
            if master.shape != parameter.shape or master.device.type != "cpu" or master.dtype != torch.float32:
                raise ValueError("offload master weights must be matching FP32 CPU tensors")
            self.state[parameter] = {"step": 0, "master_param": master}

    def load_state_dict(self, state_dict: dict) -> None:
        # Stock Optimizer.load_state_dict would cast these tensors to BF16/CUDA.
        # The existing yyLab CPU offloader avoids that, but clones every state;
        # adopting already-CPU tensors here also avoids a second 12 GB copy for
        # a 1B-parameter resume. Saved state is owned by this optimizer thereafter.
        self.synchronize()
        saved_groups = state_dict.get("param_groups", [])
        saved_states = state_dict.get("state", {})
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("loaded offload optimizer has a different number of parameter groups")
        for saved, live in zip(saved_groups, self.param_groups, strict=True):
            if len(saved["params"]) != len(live["params"]):
                raise ValueError("loaded offload optimizer parameter group has a different size")
            if saved.get("master_weight_storage", "fp32") != "fp32":
                raise ValueError("Sprint offload requires FP32 CPU masters")
        self.state.clear()
        for saved, live in zip(saved_groups, self.param_groups, strict=True):
            parameters = live["params"]
            live.update({key: value for key, value in saved.items() if key != "params"})
            for saved_id, parameter in zip(saved["params"], parameters, strict=True):
                value = saved_states.get(saved_id, {})
                if not value:
                    continue
                if "master_fp8" in value or "master_param" not in value:
                    raise ValueError("Sprint checkpoint lacks a supported FP32 CPU master")
                restored = {"step": int(value.get("step", 0))}
                for key in ("master_param", "exp_avg", "exp_avg_sq"):
                    if key not in value:
                        if restored["step"]:
                            raise ValueError(f"Sprint offload checkpoint is missing {key}")
                        continue
                    tensor = value[key]
                    if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or tensor.dtype != torch.float32:
                        raise ValueError(f"Sprint offload checkpoint {key} must match the parameter in FP32")
                    restored[key] = tensor.detach().to(device="cpu").contiguous()
                self.state[parameter] = restored

    def _buckets(self, parameters):
        entries, used = [], 0
        for parameter in parameters:
            offset = 0
            while offset < parameter.numel():
                count = min(parameter.numel() - offset, self.bucket_elements - used)
                entries.append((parameter, offset, count, used))
                offset += count
                used += count
                if used == self.bucket_elements:
                    yield entries, used
                    entries, used = [], 0
        if entries:
            yield entries, used

    def _ensure_staging(self, elements: int) -> None:
        if self._staging is not None and self._staging.numel() >= elements:
            return
        self.synchronize()
        self._staging_buffers = [
            torch.empty(elements, dtype=self.parameter_dtype, device="cpu", pin_memory=self._stream is not None)
            for _ in range(2 if self._stream is not None else 1)
        ]
        self._staging = self._staging_buffers[0]
        self._gradient = (self._staging if self.parameter_dtype == torch.float32 else
                          torch.empty(elements, dtype=torch.float32, device="cpu"))

    def _read_gradients(self, entries, slot: int) -> None:
        staging = self._staging_buffers[slot]
        with torch.cuda.stream(self._stream) if self._stream is not None else nullcontext():
            if self._stream is not None and self._writeback_events[slot] is not None:
                # GPU must finish reading this pinned buffer before D2H reuses it.
                self._stream.wait_event(self._writeback_events[slot])
            for parameter, offset, count, destination in entries:
                staging[destination:destination + count].copy_(
                    parameter.grad.detach().view(-1)[offset:offset + count], non_blocking=self._stream is not None)
            if self._stream is not None:
                self._read_events[slot].record(self._stream)

    def _write_parameters(self, entries, slot: int) -> None:
        staging = self._staging_buffers[slot]
        with torch.cuda.stream(self._writeback_stream) if self._stream is not None else nullcontext():
            for parameter, offset, count, source in entries:
                parameter.view(-1)[offset:offset + count].copy_(
                    staging[source:source + count], non_blocking=self._stream is not None)
            if self._stream is not None:
                if self._writeback_events[slot] is None:
                    self._writeback_events[slot] = torch.cuda.Event()
                self._writeback_events[slot].record(self._writeback_stream)

    def _update_bucket(self, group, entries, used: int, slot: int) -> None:
        from torch.optim import _functional

        staging = self._staging_buffers[slot]
        gradient = staging if self.parameter_dtype == torch.float32 else self._gradient
        compiled_entries = ([(parameter, offset, count, source) for parameter, offset, count, source in entries
                             if count >= self._min_compiled_elements]
                            if self._compiled_cpu_adam is not None else [])
        native_entries = ([entry for entry in entries if entry[2] < self._min_compiled_elements]
                          if compiled_entries else entries)
        if self.parameter_dtype != torch.float32 and not compiled_entries:
            gradient[:used].copy_(staging[:used])
        for parameter, offset, count, source in compiled_entries:
            state = self.state[parameter]
            self._compiled_cpu_adam.update(
                state["master_param"].view(-1)[offset:offset + count],
                state["exp_avg"].view(-1)[offset:offset + count],
                state["exp_avg_sq"].view(-1)[offset:offset + count],
                staging[source:source + count], step=state["step"], lr=group["lr"],
                beta1=group["betas"][0], beta2=group["betas"][1],
                eps=group["eps"], weight_decay=group["weight_decay"],
            )
        if not native_entries:
            return
        masters, gradients, moments, variances, steps = [], [], [], [], []
        for parameter, offset, count, source in native_entries:
            state = self.state[parameter]
            if compiled_entries:
                # The compiled portions already contain updated weights. Only
                # cast the untouched native gradient slices into FP32 scratch.
                gradient[source:source + count].copy_(staging[source:source + count])
            masters.append(state["master_param"].view(-1)[offset:offset + count])
            moments.append(state["exp_avg"].view(-1)[offset:offset + count])
            variances.append(state["exp_avg_sq"].view(-1)[offset:offset + count])
            gradients.append(gradient[source:source + count])
            # Splits of one parameter share the same logical update index;
            # functional Adam increments each private scalar exactly once.
            steps.append(torch.tensor(float(state["step"] - 1), dtype=torch.float32))
        _functional.adamw(
            masters, gradients, moments, variances, [], steps,
            fused=self.cpu_fused, foreach=not self.cpu_fused, capturable=False, differentiable=False,
            amsgrad=False, beta1=group["betas"][0], beta2=group["betas"][1], lr=group["lr"],
            weight_decay=group["weight_decay"], eps=group["eps"], maximize=False,
        )
        for master, (_, _, count, destination) in zip(masters, native_entries, strict=True):
            staging[destination:destination + count].copy_(master)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._stream is not None:
            caller = torch.cuda.current_stream(self.device)
            self._stream.wait_stream(caller)
            self._writeback_stream.wait_stream(caller)
        work = []
        for group in self.param_groups:
            parameters = [p for p in group["params"] if p.grad is not None]
            for parameter in parameters:
                if parameter.grad.is_sparse or not parameter.is_contiguous() or not parameter.grad.is_contiguous():
                    raise ValueError("Sprint CPU offload requires dense contiguous parameters and gradients")
                state = self.state[parameter]
                if "master_param" not in state:
                    state.update(step=0, master_param=parameter.detach().to(device="cpu", dtype=torch.float32, copy=True))
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(state["master_param"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master_param"])
                state["step"] = int(state["step"]) + 1
            work.extend((group, entries, used) for entries, used in self._buckets(parameters))
        if not work:
            return loss
        self._ensure_staging(max(used for _, _, used in work))
        if self._stream is None:
            for group, entries, used in work:
                self._read_gradients(entries, 0)
                self._update_bucket(group, entries, used, 0)
                self._write_parameters(entries, 0)
            return loss

        # At most two buckets are resident in pinned host memory. Prefetching
        # bucket N+1 overlaps its D2H with CPU Adam for bucket N. Reusing slot N
        # for N+2 is a GPU event dependency, not a host-wide stream barrier.
        for index in range(min(2, len(work))):
            self._read_gradients(work[index][1], index)
        for index, (group, entries, used) in enumerate(work):
            slot = index % 2
            self._read_events[slot].synchronize()
            self._update_bucket(group, entries, used, slot)
            self._write_parameters(entries, slot)
            if index + 2 < len(work):
                self._read_gradients(work[index + 2][1], slot)
        if self._stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._writeback_stream)
        return loss


class SprintStepEngine:
    """Reusable real forward/backward/optimizer path used by training benchmarks."""

    def __init__(self, model, cfg: Mapping[str, Any], device: str | torch.device = "cuda"):
        self.cfg, self.device = dict(cfg), sprint_device(str(device))
        if cfg.get("cpu_threads") is not None:
            # Worker processes own their CPU pool. More threads can slow the
            # memory-bound offloaded optimizer; benchmark this on the host.
            torch.set_num_threads(_positive_int(cfg["cpu_threads"], "cpu_threads"))
        self.cpu_threads = torch.get_num_threads()
        self.attention_backend = str(cfg.get("attention_backend", "auto"))
        if self.attention_backend not in {"auto", "cudnn", "efficient", "flash", "math"}:
            raise ValueError("invalid Sprint attention_backend")
        if self.device.type != "cuda" and self.attention_backend not in {"auto", "math"}:
            raise ValueError("the selected Sprint attention backend requires CUDA")
        dtype_name = str(cfg.get("parameter_dtype", "bfloat16" if self.device.type == "cuda" else "float32"))
        self.parameter_dtype = {"bf16": "bfloat16", "fp32": "float32"}.get(dtype_name, dtype_name)
        if self.parameter_dtype not in {"float32", "bfloat16"}:
            raise ValueError("Sprint parameter_dtype must be bfloat16 or float32")
        self.optimizer_state_offload = str(cfg.get("optimizer_state_offload", "cpu"))
        if self.optimizer_state_offload not in {"cpu", "none"}:
            raise ValueError("Sprint optimizer_state_offload must be cpu or none")
        cpu_kernel = str(cfg.get("cpu_optimizer_kernel", "auto"))
        if cpu_kernel not in {"auto", "native", "compiled"}:
            raise ValueError("Sprint cpu_optimizer_kernel must be auto, native or compiled")
        if cpu_kernel == "compiled" and (
            self.optimizer_state_offload != "cpu" or self.parameter_dtype != "bfloat16"
        ):
            raise ValueError("the compiled CPU optimizer requires BF16 weights and CPU offload")
        if self.parameter_dtype == "bfloat16" and self.optimizer_state_offload != "cpu":
            raise ValueError("Sprint BF16 weights require CPU offload to retain FP32 masters and moments")
        with torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext():
            supported = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        if self.device.type == "cuda" and self.parameter_dtype == "bfloat16" and not supported:
            raise RuntimeError("Sprint BF16 model weights require CUDA hardware with BF16 support")
        initial_masters = {}
        if self.optimizer_state_offload == "cpu" and not cfg.get("_restore_optimizer", False):
            # Preserve initialization precision before rounding accelerator weights.
            initial_masters = {name: p.detach().to(device="cpu", dtype=torch.float32, copy=True)
                               for name, p in model.named_parameters() if p.requires_grad}
        self.model = model.to(device=self.device, dtype=getattr(torch, self.parameter_dtype))
        if cfg.get("gradient_checkpointing", cfg.get("activation_checkpointing", False)):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        compute = str(cfg.get("compute_dtype", "auto"))
        if compute not in {"auto", "bfloat16", "bf16", "float32", "fp32"}:
            raise ValueError("Sprint compute_dtype must be auto, bfloat16, or float32")
        if compute in {"bfloat16", "bf16"} and not supported and self.parameter_dtype != "bfloat16":
            raise RuntimeError("Sprint BF16 training requires CUDA hardware with BF16 support")
        if self.parameter_dtype == "bfloat16" and compute in {"float32", "fp32"}:
            raise ValueError("BF16 model weights cannot execute FP32 matrix multiplications; select float32 parameters")
        self.autocast_enabled = supported and compute not in {"float32", "fp32"}
        self.compute_dtype = "bfloat16" if self.autocast_enabled else self.parameter_dtype
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        if not decay and not no_decay:
            raise ValueError("Sprint training requires trainable parameters")
        groups = [{"params": decay, "weight_decay": float(cfg.get("weight_decay", 0.1))},
                  {"params": no_decay, "weight_decay": 0.0}]
        optimizer_options = dict(lr=float(cfg.get("lr", 3e-4)), betas=tuple(cfg.get("betas", (0.9, 0.95))),
                                 eps=float(cfg.get("eps", 1e-8)))
        if self.optimizer_state_offload == "cpu":
            self.optimizer = SprintCPUOffloadAdamW(groups, bucket_mb=int(cfg.get("offload_bucket_mb", 32)),
                                                   cpu_kernel=cpu_kernel,
                                                   **optimizer_options)
            if initial_masters:
                by_parameter = {id(p): initial_masters[name] for name, p in self.model.named_parameters() if p.requires_grad}
                self.optimizer.initialize_master_weights([by_parameter[id(p)] for group in groups for p in group["params"]])
        else:
            self.optimizer = torch.optim.AdamW(
                groups, **optimizer_options, **({"fused": True} if self.device.type == "cuda" else {"foreach": True}))
        self.cpu_optimizer_kernel = (self.optimizer.resolved_cpu_kernel
                                     if self.optimizer_state_offload == "cpu" else None)
        self.execution_model = self.model
        if cfg.get("torch_compile", cfg.get("compile", False)):
            self.execution_model = torch.compile(self.model, mode=str(cfg.get("compile_mode", "default")))
        self.transfer_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

    def _autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast_enabled else nullcontext())

    def _attention_context(self):
        if self.attention_backend == "auto":
            return nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = {"cudnn": SDPBackend.CUDNN_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                   "flash": SDPBackend.FLASH_ATTENTION, "math": SDPBackend.MATH}[self.attention_backend]
        # Explicit choices fail when unsupported, so a benchmark cannot silently
        # recommend a different kernel. The context restores process defaults.
        return sdpa_kernel(backend)

    @staticmethod
    def _cpu_batch(batch: Mapping[str, Any]) -> dict:
        result = {name: torch.as_tensor(batch[name], dtype=torch.long)
                  for name in ("input_ids", "labels", "attention_mask") if name in batch}
        if result["input_ids"].ndim != 2 or result["input_ids"].shape[1] < 2:
            raise ValueError("Sprint batches must contain [batch, sequence] input IDs, sequence >= 2")
        if any(t.device.type != "cpu" for t in result.values()):
            raise ValueError("SprintStepEngine expects CPU batches so token accounting cannot synchronize CUDA")
        result.setdefault("labels", result["input_ids"])
        if any(t.shape != result["input_ids"].shape for t in result.values()):
            raise ValueError("Sprint batch labels and masks must match input IDs")
        result["num_targets"] = int(batch.get("num_targets", -1))
        if result["num_targets"] < 0:
            result["num_targets"] = int((result["labels"][:, 1:] != -100).sum())
        result["num_tokens"] = result["input_ids"].numel()
        result["real_tokens"] = int(batch.get("real_tokens", result["num_tokens"]))
        return result

    def _device_batches(self, batches: Iterable[dict]):
        def transfer(batch):
            converted = {key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                         for key, value in batch.items()
                         if key not in {"num_targets", "num_tokens", "real_tokens", "labels"}}
            converted["labels"] = (converted["input_ids"] if batch["labels"] is batch["input_ids"]
                                    else batch["labels"].to(self.device, non_blocking=True))
            return converted

        if self.transfer_stream is None:
            for batch in batches:
                yield transfer(batch)
            return
        iterator = iter(batches)
        pending = None
        for batch in iterator:
            with torch.cuda.stream(self.transfer_stream):
                next_batch = transfer(batch)
                ready = self.transfer_stream.record_event()
            if pending is not None:
                current, event = pending
                torch.cuda.current_stream(self.device).wait_event(event)
                for tensor in current.values():
                    tensor.record_stream(torch.cuda.current_stream(self.device))
                yield current
            pending = next_batch, ready
        if pending is not None:
            current, event = pending
            torch.cuda.current_stream(self.device).wait_event(event)
            for tensor in current.values():
                tensor.record_stream(torch.cuda.current_stream(self.device))
            yield current

    def train_step(self, cpu_batches: Iterable[Mapping[str, Any]], *, step: int = 0) -> SprintStepResult:
        batches = [self._cpu_batch(batch) for batch in cpu_batches]
        total = sum(batch["num_targets"] for batch in batches)
        if total < 1:
            raise ValueError("Sprint optimizer step has no supervised shifted targets")
        self.execution_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group["lr"] = sprint_learning_rate(self.cfg, step)
        single_microbatch = len(batches) == 1
        loss_sum = None if single_microbatch else torch.zeros((), device=self.device)
        with self._attention_context():
            for original, batch in zip(batches, self._device_batches(batches), strict=True):
                with self._autocast():
                    loss = self.execution_model(**batch).loss
                    weighted = loss if single_microbatch else loss * (original["num_targets"] / total)
                weighted.backward()
                if single_microbatch:
                    # Keep the independent result storage and accumulation dtype
                    # of zero-plus-loss, including when compiled outputs are reused.
                    loss_sum = weighted.detach().to(dtype=torch.get_default_dtype(), copy=True)
                else:
                    loss_sum.add_(weighted.detach())
        clip = float(self.cfg.get("grad_clip", 1.0))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip, foreach=True)
        self.optimizer.step()
        return SprintStepResult(loss_sum, sum(batch["num_tokens"] for batch in batches), total,
                                sum(batch["real_tokens"] for batch in batches))

    @torch.no_grad()
    def evaluate(self, cpu_batches: Iterable[Mapping[str, Any]], max_batches: int | None = None) -> dict:
        was_training = self.execution_model.training
        self.execution_model.eval()
        loss_sum = torch.zeros((), device=self.device)
        targets, count = 0, 0
        try:
            for raw in cpu_batches:
                original = self._cpu_batch(raw)
                for batch in self._device_batches([original]):
                    with self._attention_context(), self._autocast():
                        loss = self.execution_model(**batch).loss
                loss_sum.add_(loss * original["num_targets"])
                targets += original["num_targets"]
                count += 1
                if max_batches is not None and count >= max_batches:
                    break
        finally:
            self.execution_model.train(was_training)
        if not targets:
            raise ValueError("Sprint evaluation has no supervised targets")
        loss = float((loss_sum / targets).item())
        if not math.isfinite(loss):
            raise FloatingPointError("Sprint evaluation loss is not finite")
        return {"eval_loss": loss, "eval_perplexity": math.exp(min(80, loss)),
                "eval_supervised_tokens": targets, "eval_batches": count}


def _rng_state() -> dict:
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _model_contract(config) -> dict:
    # HF records load-path/dtype/library-version metadata when saving/loading.
    # Only execution-relevant configuration participates in resume identity.
    keys = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "max_position_embeddings", "tie_word_embeddings", "mlp_type", "rope_theta", "rms_norm_eps",
            "initializer_range", "loss_backend", "loss_chunk_size", "bos_token_id", "eos_token_id", "pad_token_id")
    return {key: getattr(config, key, None) for key in keys}


def _resume_contract(cfg: Mapping[str, Any], cache: SprintPackedCache, engine: SprintStepEngine) -> dict:
    keys = ("micro_batch", "grad_accum", "seed", "shuffle_block_rows", "lr", "betas", "weight_decay",
            "eps", "warmup_steps", "max_steps", "min_lr_ratio", "grad_clip", "max_rows", "epochs",
            "torch_compile", "compile_mode", "gradient_checkpointing", "parameter_dtype", "optimizer_state_offload")
    return {"training": {**{key: cfg[key] for key in keys}, "attention_backend": engine.attention_backend,
                         "cpu_optimizer_kernel": engine.cpu_optimizer_kernel,
                         "cpu_threads": engine.cpu_threads,
                         "offload_bucket_mb": cfg.get("offload_bucket_mb", 32)
                                              if engine.optimizer_state_offload == "cpu" else None},
            "packed_key": cache.manifest["key"],
            "model_config": _model_contract(engine.model.config), "compute_dtype": engine.compute_dtype,
            "parameter_dtype": engine.parameter_dtype, "optimizer_state_offload": engine.optimizer_state_offload}


@torch.no_grad()
def _validate_checkpoint_state(engine: SprintStepEngine) -> None:
    tensors = [parameter.detach() for parameter in engine.model.parameters()]
    tensors.extend(state[key] for state in engine.optimizer.state.values()
                   for key in ("master_param", "exp_avg", "exp_avg_sq") if key in state)
    # Infinity norms reduce without allocating parameter-sized Boolean masks.
    # One aggregate host read validates both updated weights and Adam moments,
    # outside the training hot path and before publishing any checkpoint files.
    groups = {}
    for tensor in tensors:
        groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)
    checks = []
    for values in groups.values():
        maxima = torch.stack(torch._foreach_norm(values, float("inf")))
        checks.append(torch.isfinite(maxima).all().to(device=engine.device))
    if not bool(torch.stack(checks).all().item()):
        raise FloatingPointError("Sprint checkpoint contains nonfinite model weights, FP32 masters, or optimizer moments")


def save_sprint_checkpoint(path: str | Path, engine: SprintStepEngine, tokenizer: Any, state: Mapping[str, Any]) -> Path:
    """Publish complete model/tokenizer/optimizer/cursor bundles by directory rename."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing Sprint checkpoint: {path}")
    _validate_checkpoint_state(engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        engine.model.save_pretrained(temporary)
        tokenizer.save_pretrained(temporary)
        payload = dict(state)
        payload.update(schema=TRAINER_SCHEMA, optimizer=engine.optimizer.state_dict(), rng=_rng_state())
        torch.save(payload, temporary / "trainer.pt")
        _write_json(temporary / "sprint_training.json", {key: value for key, value in state.items()
                                                        if key not in {"optimizer", "rng"}})
        temporary.rename(path)
        return path
    finally:
        if temporary.exists() and temporary.resolve().is_relative_to(path.parent.resolve()):
            shutil.rmtree(temporary)


def _checkpoint_reference(initialization: Mapping[str, Any]) -> Path:
    value = initialization.get("checkpoint_ref") or initialization.get("path")
    if not value and initialization.get("checkpoint_id"):
        row = db.q1("SELECT path,ref FROM checkpoints WHERE id=?", (initialization["checkpoint_id"],))
        if row:
            value = row.get("path") or row.get("ref")
    if not value:
        raise ValueError("Sprint checkpoint initialization requires a local checkpoint reference")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Sprint checkpoint directory not found: {path}")
    return path


def _load_training_inputs(cfg: dict, resume_path: Path | None):
    from transformers import AutoTokenizer

    from ..models.sprint import SprintConfig, SprintForCausalLM, plan_sprint_model

    initialization = cfg["initialization"]
    parent = resume_path or (_checkpoint_reference(initialization)
                             if initialization.get("mode") in {"checkpoint", "resume"} else None)
    if parent:
        tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True)
        model = SprintForCausalLM.from_pretrained(parent, local_files_only=True, dtype="auto")
        tokenizer_ref = cfg.get("tokenizer_id") or cfg.get("tokenizer_ref") or initialization.get("tokenizer_ref")
        if tokenizer_ref:
            selected = _load_tokenizer(tokenizer_ref)
            if tokenizer_fingerprint(selected) != tokenizer_fingerprint(tokenizer):
                raise ValueError("continued training tokenizer differs from the checkpoint tokenizer")
        if cfg["model_config"]:
            incoming = SprintConfig.from_dict({**model.config.to_dict(), **cfg["model_config"]})
            # Scratch recipes resolve special-token IDs from the tokenizer;
            # UI shape previews may still contain null IDs in the saved request.
            for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
                if cfg["model_config"].get(key) is None:
                    setattr(incoming, key, getattr(model.config, key, None))
            if _model_contract(incoming) != _model_contract(model.config):
                raise ValueError("continued training cannot change the Sprint model architecture")
    else:
        tokenizer_ref = cfg.get("tokenizer_id") or cfg.get("tokenizer_ref") or initialization.get("tokenizer_ref")
        if not tokenizer_ref:
            raise ValueError("Sprint scratch training requires a tokenizer_id or tokenizer_ref")
        tokenizer = _load_tokenizer(tokenizer_ref)
        values = dict(cfg["model_config"])
        target = values.pop("target_parameters", cfg.get("target_parameters"))
        if target and not values.get("hidden_size"):
            if {"intermediate_size", "num_hidden_layers", "num_attention_heads"} & values.keys():
                raise ValueError("target planning cannot override individual shape dimensions; specify hidden_size for an explicit shape")
            model_config = plan_sprint_model(
                int(target), len(tokenizer),
                max_position_embeddings=int(values.get("max_position_embeddings", cfg["seq_len"])),
                mlp_type=str(values.get("mlp_type", "gelu")),
                tie_word_embeddings=bool(values.get("tie_word_embeddings", True)),
                profile=cfg.get("shape_profile", "throughput"),
            )
            values = {**model_config.to_dict(), **values}
        values.setdefault("max_position_embeddings", cfg["seq_len"])
        if values.get("vocab_size", len(tokenizer)) != len(tokenizer):
            raise ValueError("model vocabulary must match the selected tokenizer")
        values["vocab_size"] = len(tokenizer)
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            values[key] = getattr(tokenizer, key, None)
        model_config = SprintConfig.from_dict(values)
        model_config.validate()
        model = SprintForCausalLM(model_config)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("checkpoint tokenizer and model vocabularies differ")
    if cfg["seq_len"] > model.config.max_position_embeddings:
        raise ValueError("sequence length exceeds Sprint model context length")
    return model, tokenizer, str(parent) if parent else None


def _load_tokenizer(reference: str):
    from transformers import AutoTokenizer

    resolved = resolve_tokenizer_artifact(reference) if get_tokenizer(str(reference)) else str(reference)
    return AutoTokenizer.from_pretrained(str(resolved))


def _register_checkpoint(cfg: dict, path: Path, engine: SprintStepEngine, metadata: dict) -> None:
    identifier = cfg.get("out_checkpoint_id")
    if not identifier:
        return
    parameters = sum(parameter.numel() for parameter in engine.model.parameters())
    db.execute(
        "INSERT INTO checkpoints(id,name,created_at,kind,source,ref,arch,params_b,hidden_size,num_layers,path,meta_json) "
        "VALUES(?,?,?,'sprint','local',?,'SprintForCausalLM',?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET kind='sprint',source='local',ref=excluded.ref,arch=excluded.arch,"
        "params_b=excluded.params_b,hidden_size=excluded.hidden_size,num_layers=excluded.num_layers,"
        "path=excluded.path,meta_json=excluded.meta_json",
        (identifier, str(cfg.get("name") or cfg.get("checkpoint_name") or f"Sprint {identifier}"), db.now_iso(), str(path), parameters / 1e9,
         engine.model.config.hidden_size, engine.model.config.num_hidden_layers, str(path), _json(metadata)),
    )


def run_sprint_training(config: dict, rd: Path) -> dict:
    """Worker entry point. Resume retains the original schedule and exact data cursor."""
    rd = Path(rd)
    rd.mkdir(parents=True, exist_ok=True)
    request = dict(config)
    # Canonicalize explicit updates before overlaying a resumed recipe: saved
    # canonical fields must not silently shadow a user's alias in this request.
    for alias, canonical in (
        ("compile", "torch_compile"), ("activation_checkpointing", "gradient_checkpointing"),
        ("log_every", "log_interval"), ("save_every", "save_interval"), ("eval_every", "eval_interval"),
        ("source_splits", "splits"), ("corpus", "corpora"), ("sprint", "model_config"),
        ("requested_training_tokens", "token_budget"), ("training_tokens", "token_budget"),
    ):
        if alias in request and canonical not in request:
            request[canonical] = request[alias]
    resume_ref = request.get("resume_from")
    if not resume_ref and (request.get("initialization") or {}).get("mode") == "resume":
        resume_ref = str(_checkpoint_reference(request["initialization"]))
    resume_path, saved = None, None
    if resume_ref:
        resume_path = Path(resume_ref).expanduser().resolve()
        if resume_path.name == "trainer.pt":
            resume_path = resume_path.parent
        saved = torch.load(resume_path / "trainer.pt", map_location="cpu", weights_only=True, mmap=True)
        if saved.get("schema") != TRAINER_SCHEMA:
            raise ValueError("unsupported Sprint trainer state")
        # Older checkpoints used native CPU Adam. An implicit new auto setting
        # must not change their numerical update implementation on exact resume.
        if "cpu_optimizer_kernel" not in saved["config"]:
            request.setdefault("cpu_optimizer_kernel", "native")
        request = {**saved["config"], **request}
    cfg = normalize_sprint_config(request)
    output = checkpoint_dir(cfg["out_checkpoint_id"]) if cfg.get("out_checkpoint_id") else rd / "final"
    if output.exists():
        raise FileExistsError(f"choose a new output checkpoint; Sprint output already exists: {output}")
    device = sprint_device(str(cfg.get("device", "auto")))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"] % 2**32)
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    started = time.perf_counter()

    def progress(message):
        print(message, flush=True)
        _write_json(rd / "progress.json", {"stage": "prepare", "step": 0, "max_steps": cfg["max_steps"],
                                            "loss": 0, "msg": message})

    progress("Preparing Sprint model and packed data")
    model, tokenizer, parent = _load_training_inputs(cfg, resume_path)
    engine = SprintStepEngine(model, {**cfg, "_restore_optimizer": saved is not None}, device)
    cfg["cpu_threads"] = engine.cpu_threads
    if engine.cpu_optimizer_kernel is not None:
        cfg["cpu_optimizer_kernel"] = engine.cpu_optimizer_kernel
    cache, eval_cache = None, None
    try:
        cache = prepare_sprint_cache(cfg, tokenizer, progress=progress)
        contract = _resume_contract(cfg, cache, engine)
        step, cursor, tokens, supervised, real_tokens = 0, 0, 0, 0, 0
        if saved:
            # Checkpoints predating attention tuning used PyTorch's automatic
            # dispatch. Preserve their identity while rejecting a new override.
            saved_contract = {**saved["contract"], "training": {
                "attention_backend": "auto",
                "cpu_optimizer_kernel": "native" if engine.optimizer_state_offload == "cpu" else None,
                "cpu_threads": saved["config"].get("cpu_threads", engine.cpu_threads),
                "offload_bucket_mb": saved["config"].get("offload_bucket_mb", 32)
                                     if engine.optimizer_state_offload == "cpu" else None,
                **saved["contract"]["training"]}}
            if saved_contract != contract:
                raise ValueError("resume configuration or data differs; use checkpoint initialization for a new schedule")
            engine.optimizer.load_state_dict(saved["optimizer"])
            _restore_rng(saved["rng"])
            step, cursor = int(saved["step"]), int(saved["next_row"])
            tokens, supervised = int(saved["tokens"]), int(saved["supervised_tokens"])
            real_tokens = int(saved.get("real_tokens", tokens))
        limit = min(cfg["max_rows"], cache.rows * cfg["epochs"]) if cfg["epochs"] else cfg["max_rows"]
        if cursor > limit or step > cfg["max_steps"]:
            raise ValueError("resume cursor exceeds the requested training budget")
        if cfg["eval_interval"]:
            eval_cfg = {**cfg, "corpora": cfg["eval_corpora"], "splits": cfg["eval_splits"],
                        "max_rows": cfg["eval_steps"] * cfg["micro_batch"], "shuffle_buffer": 1,
                        "corpus_mix": "concatenate", "corpus_weights": {}, "source_max_ratio": None}
            eval_cache = prepare_sprint_cache(eval_cfg, tokenizer, progress=progress)
        preparation_seconds = time.perf_counter() - started
        runtime = {"device": str(device), "compute_dtype": engine.compute_dtype, "parameter_dtype": engine.parameter_dtype,
                   "master_weight_dtype": "float32", "optimizer_state_dtype": "float32",
                   "optimizer_state_offload": engine.optimizer_state_offload,
                   "cpu_threads": engine.cpu_threads,
                   "attention_backend": engine.attention_backend,
                   "cpu_optimizer_kernel": engine.cpu_optimizer_kernel,
                   "cpu_optimizer_kernel_reason": getattr(engine.optimizer, "cpu_kernel_reason", None),
                   "offload_bucket_mb": cfg["offload_bucket_mb"] if engine.optimizer_state_offload == "cpu" else None,
                   "optimizer": "sprint_cpu_offload_adamw" if engine.optimizer_state_offload == "cpu" else
                                "fused_adamw" if device.type == "cuda" else "foreach_adamw",
                   "cpu_optimizer_update": ("fused" if engine.optimizer.cpu_fused else "foreach")
                                           if engine.optimizer_state_offload == "cpu" else None,
                   "torch_compile": bool(cfg["torch_compile"]), "compile_mode": cfg["compile_mode"],
                   "gradient_checkpointing": cfg["gradient_checkpointing"],
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "packed_cache": str(cache.root), "packed_key": cache.manifest["key"],
                   "packed_rows": cache.rows, "token_budget_unit": "forwarded_model_slots",
                   "requested_token_budget": cfg["token_budget"], "planned_token_slots": limit * cfg["seq_len"],
                   "preparation_seconds": preparation_seconds}
        _write_json(rd / "sprint_runtime.json", runtime)
        _write_json(rd / "resolved_config.json", cfg)
        pin = device.type == "cuda" and cfg["pin_memory"]
        source = _CacheBatches(cache, cfg, cursor, limit, pin_memory=pin)
        batches = BufferedBatches(source, cfg["data_prefetch_batches"])
        initial_tokens, initial_step, initial_supervised = tokens, step, supervised
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        training_start = time.perf_counter()
        train_seconds = 0.0
        overhead_seconds = 0.0
        window_loss = torch.zeros((), device=device)
        window_targets, window_tokens = 0, 0
        last_loss = float(saved.get("loss", 0.0)) if saved else 0.0
        eval_metrics = {}
        last_checkpoint = None

        def state():
            return {"config": cfg, "contract": contract, "step": step, "next_row": cursor,
                    "tokens": tokens, "supervised_tokens": supervised, "real_tokens": real_tokens,
                    "loss": last_loss, "parent_checkpoint": parent}

        def publish(stage="train"):
            nonlocal window_loss, window_targets, window_tokens, last_loss
            if window_targets:
                last_loss = float((window_loss / window_targets).item())
                if not math.isfinite(last_loss):
                    raise FloatingPointError("Sprint loss is not finite; reduce learning rate or inspect the data")
            elapsed = time.perf_counter() - training_start
            train_elapsed = max(1e-9, elapsed - overhead_seconds)
            result = {"stage": stage, "step": step, "max_steps": cfg["max_steps"], "loss": last_loss,
                      "lr": engine.optimizer.param_groups[0]["lr"], "tokens": tokens,
                      "forwarded_model_tokens": tokens, "supervised_tokens": supervised,
                      "tokens_per_second": (tokens - initial_tokens) / train_elapsed,
                      "supervised_tokens_per_second": (supervised - initial_supervised) / train_elapsed,
                      "end_to_end_tokens_per_second": (tokens - initial_tokens) / max(elapsed + preparation_seconds, 1e-9),
                      "training_seconds": train_elapsed, "elapsed_seconds": elapsed,
                      "next_row": cursor, "epoch": cursor // cache.rows,
                      "msg": f"{tokens:,} token slots; {step:,} optimizer steps", **eval_metrics}
            _write_json(rd / "progress.json", result)
            with (rd / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(_json(result) + "\n")
            window_loss = torch.zeros((), device=device)
            window_targets, window_tokens = 0, 0
            return result

        with batch_stream(batches) as iterator:
            while step < cfg["max_steps"] and cursor < limit:
                window = []
                for _ in range(cfg["grad_accum"]):
                    batch = next(iterator, None)
                    if batch is None:
                        break
                    window.append(batch)
                if not window:
                    break
                result = engine.train_step(window, step=step)
                step += 1
                tokens += result.tokens
                supervised += result.supervised_tokens
                real_tokens += result.real_tokens
                cursor += sum(len(batch["input_ids"]) for batch in window)
                window_loss.add_(result.loss * result.supervised_tokens)
                window_targets += result.supervised_tokens
                window_tokens += result.tokens
                evaluate = eval_cache is not None and step % cfg["eval_interval"] == 0
                save = cfg["save_interval"] and step % cfg["save_interval"] == 0
                if step % cfg["log_interval"] == 0 or evaluate or save:
                    publish()
                if evaluate or save:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    overhead_start = time.perf_counter()
                    if evaluate:
                        eval_source = _CacheBatches(eval_cache, cfg, 0, eval_cache.rows, pin_memory=pin)
                        eval_metrics = engine.evaluate(eval_source, max_batches=cfg["eval_steps"])
                    if save:
                        last_checkpoint = save_sprint_checkpoint(rd / "checkpoints" / f"step-{step:08d}",
                                                                 engine, tokenizer, state())
                        _write_json(rd / "latest_checkpoint.json", {"path": str(last_checkpoint), "step": step})
                    overhead_seconds += time.perf_counter() - overhead_start
                if (rd / "STOP").exists():
                    break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - training_start - overhead_seconds
        if step == 0:
            raise ValueError("Sprint consumed no trainable data")
        summary = publish("complete")
        save_start = time.perf_counter()
        save_sprint_checkpoint(output, engine, tokenizer, state())
        summary.update({"checkpoint_path": str(output), "checkpoint_id": cfg.get("out_checkpoint_id"),
                        "model_family": "hf", "architecture_type": "sprint", "params_b": runtime["parameters"] / 1e9,
                        "arch": "SprintForCausalLM", "steps": step, "new_steps": step - initial_step,
                        "forwarded_real_tokens": real_tokens, "padding_tokens": tokens - real_tokens,
                        "sequence_length": cfg["seq_len"], "training_seconds": train_seconds,
                        "preparation_seconds": preparation_seconds,
                        "checkpoint_seconds": time.perf_counter() - save_start,
                        "runtime": runtime, "packed_data_stats": cache.manifest["data_stats"],
                        "stop_reason": "stop_file" if (rd / "STOP").exists() else
                                       "data_exhausted" if cursor >= limit and limit < cfg["max_rows"] else "budget_complete",
                        "last_periodic_checkpoint": str(last_checkpoint) if last_checkpoint else None})
        summary["wall_seconds"] = time.perf_counter() - started
        summary["end_to_end_tokens_per_second"] = (tokens - initial_tokens) / max(summary["wall_seconds"], 1e-9)
        if device.type == "cuda":
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        _write_json(output / "sprint_summary.json", summary)
        _register_checkpoint(cfg, output, engine, {"model_family": "hf", "training_architecture": "sprint",
                                                  "sprint": summary, "parent_checkpoint": parent})
        _write_json(rd / "summary.json", summary)
        _write_json(rd / "progress.json", summary)
        return summary
    finally:
        if cache is not None:
            cache.close()
        if eval_cache is not None:
            eval_cache.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a Sprint language model using a yyLab recipe")
    parser.add_argument("--config", type=Path, required=True, help="JSON training recipe with registered corpus IDs")
    parser.add_argument("--output", type=Path, required=True, help="run directory for progress, metrics, and checkpoints")
    args = parser.parse_args()
    config = json.loads(args.config.read_text("utf-8"))
    if not isinstance(config, dict):
        parser.error("the training recipe must be a JSON object")
    db.init_db()
    result = run_sprint_training(config, args.output.expanduser().resolve())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
