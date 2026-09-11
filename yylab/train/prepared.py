"""Reusable token records beneath yyLab's existing mixing and packing contracts.

Only derived tokens, masks, and record metadata are written. Source text stays
in the registered corpus (including GCCF/text pointers to original datasets).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import corpus_dir
from ..config import PATHS
from .data import (
    CORPUS_ID_FIELD,
    DEFAULT_TEXT_CHUNK_CHARS,
    OBJECTIVE_CHAT_SFT,
    PREPARED_FIELD,
    _shuffle_bytes,
    corpus_data_dir,
    corpus_shards,
    iter_corpus_docs,
    normalize_corpus_ids,
    normalize_training_objective,
    render_chat_record,
)

SCHEMA = "yylab-token-records-v1"
DEFAULT_SHARD_BYTES = 64 * 2**20


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Fingerprint actual tokenizer machinery, rather than a model/path name."""
    raw = getattr(tokenizer, "tokenizer", tokenizer)
    backend = getattr(raw, "backend_tokenizer", None)
    identity = getattr(tokenizer, "identity", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        artifact = json.loads(backend.to_str())
        if (artifact.get("model") or {}).get("dropout"):
            raise ValueError("prepared tokens require deterministic tokenization; disable BPE dropout")
    elif identity is not None and getattr(identity, "fingerprint", None):
        artifact = {"native_identity": identity.fingerprint}
    elif callable(getattr(raw, "save_pretrained", None)):
        # Slow HF tokenizers may have merges/models outside get_vocab().
        with tempfile.TemporaryDirectory(prefix="yylab-tokenizer-identity-") as temporary:
            root = Path(temporary)
            raw.save_pretrained(root)
            artifact = {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    else:
        raise ValueError(
            "prepared tokens require a serializable tokenizer or native identity; "
            "set token_cache=false for a custom streaming tokenizer"
        )
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")}
    payload = {
        "class": f"{type(raw).__module__}.{type(raw).__qualname__}",
        "artifact": artifact,
        "chat_template": getattr(raw, "chat_template", None),
        "special_tokens": str(getattr(raw, "special_tokens_map", None)),
        "eos": getattr(raw, "eos_token_id", None),
        "pad": getattr(raw, "pad_token_id", None),
        "versions": versions,
        "split_special_tokens": getattr(raw, "split_special_tokens", False),
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def source_snapshot(corpus_id: str, *, verify: str = "stat") -> dict:
    """Track index contents and every referenced source, without copying text.

    Default reuse trusts file size/mtime/ctime. `sha256` additionally reads and
    hashes source contents on every check, for datasets whose timestamps cannot
    be trusted. Index/pointer contents are always hashed.
    """
    if verify not in {"stat", "sha256"}:
        raise ValueError("token_cache_verify_sources must be 'stat' or 'sha256'")
    root = corpus_dir(corpus_id)
    controls: set[Path] = set()
    sources: set[Path] = set()
    text_pointer = root / "text-source.json"
    gccf_pointer = root / "gccf-source.json"
    if text_pointer.exists():
        controls.add(text_pointer)
        index = Path(json.loads(text_pointer.read_text("utf-8"))["index_directory"])
        manifest = index / "text-index.json"
        controls.add(manifest)
        sources.update(Path(row["path"]) for row in json.loads(manifest.read_text("utf-8"))["files"])
    elif gccf_pointer.exists():
        controls.add(gccf_pointer)
        pointer = json.loads(gccf_pointer.read_text("utf-8"))
        index = Path(pointer["index_directory"])
        controls.update((index / "gccf-index.json", index / "index.sqlite3"))
        connection = sqlite3.connect((index / "index.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for (entry,) in connection.execute(
                "SELECT entry FROM files WHERE source=? ORDER BY id", (pointer["source"],)
            ):
                sources.add(Path(json.loads(entry)["abs_path"]))
        finally:
            connection.close()
    else:
        data_dir = corpus_data_dir(corpus_id)
        sources.update(data_dir / name for name in corpus_shards(corpus_id, data_dir))
        if (root / "current.json").exists():
            controls.add(root / "current.json")

    def describe(path: Path, *, digest: bool) -> dict:
        path = path.resolve(strict=True)
        stat = path.stat()
        row = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        if digest:
            row["sha256"] = _sha(path)
        return row

    return {
        "corpus_id": corpus_id,
        "controls": [describe(p, digest=True) for p in sorted(controls)],
        "sources": [describe(p, digest=verify == "sha256") for p in sorted(sources)],
        "verification": verify,
    }


class _ShardWriter:
    def __init__(self, root: Path, target: int, chat: bool) -> None:
        self.root, self.chat = root, chat
        self.capacity = target // (5 if chat else 4)
        if self.capacity < 1:
            raise ValueError("token shard payload limit is too small")
        self.index, self.count = -1, 0
        self.ids = self.mask = None
        self.files: list[dict] = []
        self.tokens = 0

    def close(self) -> None:
        for stream in (self.ids, self.mask):
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
                path = Path(stream.name)
                stream.close()
                self.files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)})
        self.ids = self.mask = None

    def write(self, ids: list[int], labels: list[int] | None = None) -> list[list[int]]:
        if any(token < 0 or token > 0xFFFFFFFF for token in ids):
            raise ValueError("prepared token IDs must fit uint32")
        if self.chat and (
            labels is None
            or len(labels) != len(ids)
            or any(label not in (-100, token) for token, label in zip(ids, labels, strict=True))
        ):
            raise ValueError("prepared chat labels must equal their token ID or -100")
        spans = []
        position = 0
        while position < len(ids):
            if self.ids is None or self.count == self.capacity:
                self.close()
                self.index += 1
                self.count = 0
                self.ids = (self.root / f"tokens-{self.index:06d}.bin").open("wb")
                if self.chat:
                    self.mask = (self.root / f"mask-{self.index:06d}.bin").open("wb")
            take = min(len(ids) - position, self.capacity - self.count)
            self.ids.write(np.asarray(ids[position : position + take], dtype="<u4").tobytes())
            if self.mask is not None:
                self.mask.write(bytes(label != -100 for label in labels[position : position + take]))
            spans.append([self.index, self.count, take])
            position += take
            self.count += take
            self.tokens += take
        return spans


@dataclass(frozen=True)
class TokenUnit:
    store: TokenStore
    spans: list[list[int]]

    def read_ids(self) -> list[int]:
        return self.store.read(self.spans, "tokens", 4)

    def read_chat(self) -> tuple[list[int], list[int]]:
        ids = self.read_ids()
        mask = self.store.read(self.spans, "mask", 1)
        return ids, [token if supervised else -100 for token, supervised in zip(ids, mask, strict=True)]


class TokenStore:
    """Lazy reads with at most eight open shard files per corpus.

    Each shard is checksummed once before its first read in this store. The
    metadata index is checked eagerly. Handles are owned by the reading thread.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported prepared-token schema")
        self.files = {entry["name"]: entry for entry in self.manifest["files"]}
        self._handles: OrderedDict[str, Any] = OrderedDict()
        self._verified: set[str] = set()
        self._verify("records.sqlite3")

    def _verify(self, name: str) -> None:
        if Path(name).name != name:
            raise ValueError("invalid token shard filename")
        path, entry = self.root / name, self.files[name]
        if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
            raise ValueError(f"prepared-token checksum mismatch: {path}")
        self._verified.add(name)

    def read(self, spans: list[list[int]], kind: str, width: int) -> list[int]:
        result: list[int] = []
        for shard, offset, count in spans:
            if min(shard, offset, count) < 0:
                raise ValueError("negative prepared-token span")
            name = f"{kind}-{shard:06d}.bin"
            if name not in self._verified:
                self._verify(name)
            stream = self._handles.pop(name, None)
            if stream is None:
                stream = (self.root / name).open("rb")
            self._handles[name] = stream
            while len(self._handles) > 8:
                self._handles.popitem(last=False)[1].close()
            stream.seek(offset * width)
            payload = stream.read(count * width)
            if len(payload) != count * width:
                raise ValueError("truncated prepared-token span")
            result.extend(np.frombuffer(payload, dtype="<u4" if width == 4 else "u1").tolist())
        return result

    def records(self) -> Iterator[dict]:
        connection = sqlite3.connect(
            (self.root / "records.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        )
        try:
            for (payload,) in connection.execute("SELECT metadata FROM records ORDER BY ordinal"):
                record = json.loads(payload)
                prepared = record[PREPARED_FIELD]
                prepared["units"] = [TokenUnit(self, spans) for spans in prepared["units"]]
                yield record
        finally:
            connection.close()

    def close(self) -> None:
        while getattr(self, "_handles", None):
            self._handles.popitem()[1].close()

    def __del__(self) -> None:
        self.close()


def prepare_corpus(
    corpus_id: str,
    tokenizer: Any,
    tokenize: Callable[[list[str]], list[list[int]]],
    *,
    objective: str = "causal_lm",
    max_text_chars: int = DEFAULT_TEXT_CHUNK_CHARS,
    tokenize_chunk: int = 8,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    cache_dir: Path | None = None,
    verify_sources: str = "stat",
    progress: Callable[[str], None] = print,
) -> TokenStore:
    objective = normalize_training_objective(objective)
    if min(max_text_chars, tokenize_chunk) < 1 or shard_bytes < (5 if objective == OBJECTIVE_CHAT_SFT else 4):
        raise ValueError("preparation chunk and shard sizes must be positive")
    snapshot = source_snapshot(corpus_id, verify=verify_sources)
    identity = {
        "schema": SCHEMA,
        "source": snapshot,
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "objective": objective,
        "max_text_chars": max_text_chars,
        "shard_bytes": shard_bytes,
        # Preserve the existing byte-bounded shuffle decisions on this runtime.
        "python": list(sys.version_info[:2]),
        "pointer_bytes": np.dtype(np.intp).itemsize,
    }
    key = hashlib.sha256(_json(identity).encode()).hexdigest()
    cache = Path(cache_dir) if cache_dir is not None else PATHS.cache / "prepared-tokens"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    if destination.exists():
        store = TokenStore(destination)
        if store.manifest["identity"] != identity:
            raise ValueError("prepared-token identity mismatch")
        progress(f"Reusing prepared tokens for {corpus_id}: {store.manifest['tokens']:,} tokens")
        return store
    lock = cache / f"{key}.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"token preparation is already active (or interrupted); inspect {lock}") from exc
    with os.fdopen(lock_fd, "w") as stream:
        stream.write(f"pid={os.getpid()}\n")
    staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache))
    connection = None
    writer = _ShardWriter(staging, shard_bytes, objective == OBJECTIVE_CHAT_SFT)
    try:
        progress(f"Preparing tokens for {corpus_id}; payload shards up to {shard_bytes / 2**20:g} MiB")
        connection = sqlite3.connect(staging / "records.sqlite3")
        connection.execute("CREATE TABLE records(ordinal INTEGER PRIMARY KEY, metadata TEXT NOT NULL)")
        documents = 0
        records = iter_corpus_docs(
            [corpus_id], shuffle_buffer=1, cross_corpus_dedup=False, annotate_source=True
        )
        try:
            for doc in records:
                text = doc.get("text") or ""
                prepared = {
                    "chars": len(text),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "shuffle_bytes": _shuffle_bytes(doc),
                    "units": [],
                }
                if objective == OBJECTIVE_CHAT_SFT:
                    prepared["chat"], examples = render_chat_record(tokenizer, doc)
                    prepared["units"] = [writer.write(ids, labels) for ids, labels in examples]
                else:
                    starts = range(0, max(len(text), 1), max_text_chars)
                    for start_index in range(0, len(starts), tokenize_chunk):
                        texts = [
                            text[start : start + max_text_chars]
                            for start in starts[start_index : start_index + tokenize_chunk]
                        ]
                        encodings = tokenize(texts)
                        if len(encodings) != len(texts):
                            raise ValueError("tokenizer returned a different number of records")
                        prepared["units"].extend(writer.write(ids) for ids in encodings)
                metadata = {
                    key: doc[key]
                    for key in (
                        CORPUS_ID_FIELD,
                        "id",
                        "sample_id",
                        "source",
                        "domain",
                        "split",
                        "lang",
                        "provenance",
                    )
                    if key in doc
                }
                metadata[PREPARED_FIELD] = prepared
                connection.execute("INSERT INTO records VALUES(?, ?)", (documents, _json(metadata)))
                documents += 1
                if documents % 1000 == 0:
                    connection.commit()
                    progress(f"Preparing {corpus_id}: {documents:,} records, {writer.tokens:,} tokens")
        finally:
            records.close()
        connection.commit()
        connection.close()
        connection = None
        writer.close()
        if source_snapshot(corpus_id, verify=verify_sources) != snapshot:
            raise ValueError("corpus changed during token preparation; cache was not published")
        index = staging / "records.sqlite3"
        writer.files.append({"name": index.name, "bytes": index.stat().st_size, "sha256": _sha(index)})
        manifest = {
            "schema": SCHEMA,
            "identity": identity,
            "documents": documents,
            "tokens": writer.tokens,
            "files": writer.files,
        }
        with (staging / "manifest.json").open("w", encoding="utf-8") as stream:
            stream.write(_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        staging.rename(destination)
        progress(f"Prepared {corpus_id}: {documents:,} records, {writer.tokens:,} tokens at {destination}")
        return TokenStore(destination)
    finally:
        if connection is not None:
            connection.close()
        writer.close()
        if staging.exists() and staging.resolve().is_relative_to(cache.resolve()):
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


class PreparedCorpora:
    def __init__(self, stores: dict[str, TokenStore]) -> None:
        self.stores = stores

    def __call__(self, corpus_id: str) -> Iterator[dict]:
        return self.stores[corpus_id].records()

    def close(self) -> None:
        for store in self.stores.values():
            store.close()


def prepare_for_training(
    cfg: Mapping[str, Any], tokenizer: Any, tokenize: Callable, *, progress=print
) -> PreparedCorpora | None:
    if int(cfg.get("data_prefetch_batches", 2)) < 0:
        raise ValueError("data_prefetch_batches must be non-negative")
    if not isinstance(cfg.get("token_cache", True), bool):
        raise ValueError("token_cache must be a boolean")
    if not cfg.get("token_cache", True):
        return None
    stores: dict[str, TokenStore] = {}
    try:
        for corpus_id in normalize_corpus_ids(cfg["corpora"]):
            stores[corpus_id] = prepare_corpus(
                corpus_id,
                tokenizer,
                tokenize,
                objective=cfg.get("objective", "causal_lm"),
                max_text_chars=int(cfg.get("max_text_chars", DEFAULT_TEXT_CHUNK_CHARS)),
                tokenize_chunk=int(cfg.get("tokenize_chunk", 8)),
                shard_bytes=int(cfg.get("token_shard_bytes", DEFAULT_SHARD_BYTES)),
                cache_dir=Path(cfg["token_cache_dir"]) if cfg.get("token_cache_dir") else None,
                verify_sources=str(cfg.get("token_cache_verify_sources", "stat")),
                progress=progress,
            )
        return PreparedCorpora(stores)
    except BaseException:
        PreparedCorpora(stores).close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare reusable yyLab token records without loading model weights"
    )
    parser.add_argument("--corpora", nargs="+", required=True, help="registered yyLab corpus IDs")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path or registry ID")
    parser.add_argument("--objective", choices=("causal_lm", "chat_sft"), default="causal_lm")
    parser.add_argument("--shard-mib", type=int, default=64)
    parser.add_argument("--max-text-chars", type=int, default=DEFAULT_TEXT_CHUNK_CHARS)
    parser.add_argument("--verify-sources", choices=("stat", "sha256"), default="stat")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from ..tokenizer_registry import get_tokenizer, resolve_tokenizer_artifact

    ref = str(resolve_tokenizer_artifact(args.tokenizer)) if get_tokenizer(args.tokenizer) else args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ref)

    def tokenize(texts):
        return tokenizer(texts, add_special_tokens=False, return_attention_mask=False, verbose=False)[
            "input_ids"
        ]

    reader = prepare_for_training(
        {
            "corpora": args.corpora,
            "objective": args.objective,
            "token_shard_bytes": args.shard_mib * 2**20,
            "max_text_chars": args.max_text_chars,
            "token_cache_verify_sources": args.verify_sources,
            "token_cache_dir": args.cache_dir,
        },
        tokenizer,
        tokenize,
    )
    reader.close()


if __name__ == "__main__":
    main()
