"""Bounded CPU read-ahead with consumer-visible accounting and cleanup."""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager


@contextmanager
def batch_stream(batches):
    iterator = iter(batches)
    try:
        yield iterator
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


class BufferedBatches:
    """Prefetch at most `depth` queued batches plus one producer batch.

    The producer only reads/decodes/packs on CPU; CUDA remains on the training
    thread. Statistics are published when a batch is consumed, not produced.
    """

    def __init__(self, batches, depth: int = 2, *, resources=None) -> None:
        if depth < 0:
            raise ValueError("data_prefetch_batches must be non-negative")
        self.batches, self.depth, self.resources = batches, depth, resources
        self._stats = batches.data_stats()
        self._counts = self._snapshot_counts()
        self._active = False

    def _snapshot_counts(self):
        return {key: value for key, value in vars(self.batches).items() if isinstance(value, (int, float))}

    def __getattr__(self, name):
        counts = self.__dict__.get("_counts", {})
        if name in counts:
            return counts[name]
        raise AttributeError(name)

    def data_stats(self):
        stats = dict(self._stats)
        if self.resources is not None:
            stats["prepared_tokens"] = {
                corpus: {
                    "directory": str(store.root),
                    "documents": store.manifest["documents"],
                    "tokens": store.manifest["tokens"],
                }
                for corpus, store in self.resources.stores.items()
            }
        stats["data_prefetch_batches"] = self.depth
        return stats

    def __iter__(self):
        if self._active:
            raise RuntimeError("batch stream already has an active consumer")
        self._active = True
        stop = threading.Event()
        pending = queue.Queue(maxsize=max(1, self.depth))
        thread = None

        def entries():
            iterator = iter(self.batches)
            try:
                for batch in iterator:
                    if stop.is_set():
                        return
                    yield "batch", batch, self.batches.data_stats(), self._snapshot_counts()
                yield "end", None, self.batches.data_stats(), self._snapshot_counts()
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
                docs = getattr(self.batches, "docs", None)
                close = getattr(docs, "close", None)
                if callable(close):
                    close()
                if self.resources is not None:
                    self.resources.close()

        def put(entry):
            while not stop.is_set():
                try:
                    pending.put(entry, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def produce():
            try:
                with batch_stream(entries()) as iterator:
                    for entry in iterator:
                        put(entry)
                        if stop.is_set():
                            break
            except BaseException as exc:
                put(("error", exc, None, None))

        direct = None
        try:
            if self.depth:
                thread = threading.Thread(target=produce, name="yylab-data-prefetch", daemon=True)
                thread.start()
            else:
                direct = entries()
            while True:
                kind, value, stats, counts = pending.get() if self.depth else next(direct)
                if kind == "error":
                    raise value
                self._stats, self._counts = stats, counts
                if kind == "end":
                    break
                yield value
        finally:
            stop.set()
            if direct is not None:
                direct.close()
            if thread is not None:
                thread.join()
            self._active = False
