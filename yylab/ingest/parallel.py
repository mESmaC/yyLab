"""Process-pool execution of document preparation with ordered results.

The coordinator in :mod:`yylab.ingest.pipeline` submits :class:`PrepareTask`
objects in asset/chunk order and consumes their results strictly in that
order, so every policy decision (exact and near duplicates, split leakage,
sample ids, locators) sees documents in the same sequence as the historical
single-threaded pipeline.  Worker processes never touch yyLab's SQLite
registry or the UI; they only import :mod:`yylab.ingest.parallel`,
:mod:`yylab.ingest.prepare`, and the adapters.
"""

from __future__ import annotations

import multiprocessing
import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.base import Adapter, ChunkSpec
from .adapters.loader import discover_adapters
from .prepare import AnalysisParams, ChunkResult, PreparedDoc, augment_documents, prepare_chunk

WORKERS_ENV = "YYLAB_INGEST_WORKERS"
DEFAULT_MAX_AUTO_WORKERS = 8
#: Tasks submitted but not yet consumed, per worker.  Running tasks cost the
#: coordinator nothing, so this only bounds queue depth, not memory.
WINDOW_PER_WORKER = 4
#: Finished-but-unconsumed results are what occupy coordinator memory; stop
#: submitting once their estimated resident size exceeds this many bytes.
#: Residency is bounded by this budget plus up to ``workers`` results that
#: finish concurrently.
RESULT_BUDGET_BYTES = 512 * 2**20


def auto_worker_count() -> int:
    cpus = os.cpu_count() or 2
    return max(1, min(cpus - 2, DEFAULT_MAX_AUTO_WORKERS))


def resolve_worker_count(requested: int | None, settings_default: object = None) -> int:
    """Explicit request, then ``YYLAB_INGEST_WORKERS``, then settings, then auto.

    Zero at any level means "auto"; negative and unparsable values are ignored.
    """

    for candidate in (requested, os.environ.get(WORKERS_ENV), settings_default):
        if candidate is None or candidate == "":
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
        if value == 0:
            return auto_worker_count()
    return auto_worker_count()


@dataclass(frozen=True, slots=True)
class PrepareTask:
    path: str
    adapter_name: str
    chunk: ChunkSpec | None
    asset_logical_path: str
    asset_format_name: str
    params: AnalysisParams
    hash_file: bool = False
    build_payload: bool = False


_WORKER_ADAPTERS: dict[str, Adapter] = {}


def _worker_init() -> None:
    """Per-process setup: bounded native thread pools and adapter instances."""

    import sys
    import warnings

    heavy = [
        name
        for name in ("nicegui", "torch", "transformers", "yylab.db", "yylab.ingest.pipeline")
        if name in sys.modules
    ]
    if heavy:
        # Spawned workers must only import the ingest package: the UI stack
        # costs seconds and hundreds of MB per worker, and yylab.db runs
        # registry migrations at import time.
        warnings.warn(
            f"ingest worker imported {', '.join(heavy)}; check the launcher's __main__ guard"
            " and yylab/ingest/__init__.py",
            RuntimeWarning,
            stacklevel=1,
        )
    try:
        import pyarrow as pa

        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
    except Exception:  # noqa: BLE001 - pyarrow is optional for non-Parquet corpora
        pass
    _WORKER_ADAPTERS.clear()
    _WORKER_ADAPTERS.update({adapter.name: adapter for adapter in discover_adapters()})


def _worker_adapter(name: str) -> Adapter:
    if not _WORKER_ADAPTERS:
        _worker_init()
    try:
        return _WORKER_ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown ingest adapter {name!r}") from exc


def run_prepare_task(task: PrepareTask) -> ChunkResult:
    adapter = _worker_adapter(task.adapter_name) if task.adapter_name else None
    return prepare_chunk(
        Path(task.path),
        adapter,
        task.chunk,
        asset_logical_path=task.asset_logical_path,
        asset_format_name=task.asset_format_name,
        params=task.params,
        hash_file=task.hash_file,
        build_payload=task.build_payload,
    )


def run_augment_task(docs: list[PreparedDoc], params: AnalysisParams) -> list[PreparedDoc]:
    return augment_documents(docs, params)


class _Deferred:
    """A lazily evaluated, memoized stand-in for a Future.

    ``done()`` reports whether the value is already materialized (or, when the
    deferred wraps an ``upstream`` future, whether that future finished), so
    :func:`ordered_results` can count its size against the result budget.
    """

    __slots__ = ("_args", "_evaluated", "_fn", "_result", "_upstream")

    def __init__(self, fn: Callable[..., Any], *args: Any, upstream: Any = None) -> None:
        self._fn = fn
        self._args = args
        self._upstream = upstream
        self._evaluated = False
        self._result: Any = None

    @classmethod
    def completed(cls, value: Any) -> _Deferred:
        deferred = cls(lambda: value)
        deferred._result = value
        deferred._evaluated = True
        return deferred

    def done(self) -> bool:
        if self._evaluated:
            return True
        upstream_done = getattr(self._upstream, "done", None)
        return bool(callable(upstream_done) and upstream_done())

    def result(self) -> Any:
        if not self._evaluated:
            self._result = self._fn(*self._args)
            self._evaluated = True
            self._fn = None
            self._args = ()
        return self._result

    def cancel(self) -> bool:
        cancel = getattr(self._upstream, "cancel", None)
        if callable(cancel):
            cancel()
        return True


class SerialExecutor:
    """In-process execution; honours whatever adapters the caller supplies."""

    workers = 1

    def __init__(self, adapters: Iterable[Adapter]) -> None:
        self._adapters = {adapter.name: adapter for adapter in adapters}

    def submit_prepare(self, task: PrepareTask) -> _Deferred:
        adapter = None
        if task.adapter_name:
            adapter = self._adapters.get(task.adapter_name)
            if adapter is None:
                raise ValueError(f"unknown ingest adapter {task.adapter_name!r}")
        return _Deferred(self._run, task, adapter)

    @staticmethod
    def _run(task: PrepareTask, adapter: Adapter | None) -> ChunkResult:
        return prepare_chunk(
            Path(task.path),
            adapter,
            task.chunk,
            asset_logical_path=task.asset_logical_path,
            asset_format_name=task.asset_format_name,
            params=task.params,
            hash_file=task.hash_file,
            build_payload=task.build_payload,
        )

    def submit_augment(self, docs: list[PreparedDoc], params: AnalysisParams) -> _Deferred:
        return _Deferred(augment_documents, docs, params)

    def close(self, *, abort: bool = False) -> None:
        return None


class ProcessExecutor:
    """Spawned worker processes (never fork: the UI process holds threads)."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, int(workers))
        context = multiprocessing.get_context("spawn")
        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_worker_init,
        )

    def submit_prepare(self, task: PrepareTask):
        return self._pool.submit(run_prepare_task, task)

    def submit_augment(self, docs: list[PreparedDoc], params: AnalysisParams):
        return self._pool.submit(run_augment_task, docs, params)

    def close(self, *, abort: bool = False) -> None:
        """Shut the pool down; ``abort`` terminates running workers immediately.

        After an ingest error there is no point waiting for a worker that may
        be minutes into a large PDF or row group, and the caller must be able
        to mark the corpus failed promptly.
        """

        if not abort:
            self._pool.shutdown(wait=True, cancel_futures=True)
            return
        processes = list((self._pool._processes or {}).values())  # noqa: SLF001
        self._pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            try:
                process.terminate()
            except Exception:  # noqa: BLE001 - best effort during failure handling
                pass


def make_executor(workers: int, adapters: Iterable[Adapter]) -> SerialExecutor | ProcessExecutor:
    if workers <= 1:
        return SerialExecutor(adapters)
    return ProcessExecutor(workers)


def _finished_bytes(future: Any) -> int | None:
    """Return the approximate size of a finished result, ``None`` if still running."""

    done = getattr(future, "done", None)
    if not callable(done) or not done():
        return None
    try:
        result = future.result()
    except BaseException:  # noqa: BLE001 - surfaced when the result is consumed
        return 0
    return int(getattr(result, "approx_bytes", 0) or 0)


def ordered_results(
    submissions: Iterable[tuple[Any, Any]],
    window: int,
    *,
    result_budget_bytes: int = RESULT_BUDGET_BYTES,
) -> Iterator[tuple[Any, Any]]:
    """Yield ``(key, result)`` in submission order.

    ``submissions`` is a lazy iterable of ``(key, future_like)`` pairs.  It is
    advanced while fewer than ``window`` tasks are outstanding *and* the
    finished-but-unconsumed results stay under ``result_budget_bytes``; a slow
    task at the head therefore never idles the pool, while memory remains
    bounded by the budget plus one result.
    """

    window = max(1, int(window))
    pending: deque[tuple[Any, Any]] = deque()
    iterator = iter(submissions)
    exhausted = False
    try:
        while True:
            while not exhausted and len(pending) < window:
                finished = sum(size for _key, future in pending if (size := _finished_bytes(future)))
                if pending and finished >= result_budget_bytes:
                    break
                try:
                    key, future = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append((key, future))
            if not pending:
                return
            ready_key, ready = pending.popleft()
            yield ready_key, ready.result()
    finally:
        for _key, future in pending:
            cancel = getattr(future, "cancel", None)
            if callable(cancel):
                cancel()
