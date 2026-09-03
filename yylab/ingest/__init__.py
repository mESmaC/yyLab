"""Corpus ingestion package.

The public entry points live in :mod:`yylab.ingest.pipeline`, which imports
the SQLite registry.  They are re-exported lazily so that spawned ingest
workers, which only import :mod:`yylab.ingest.parallel` and
:mod:`yylab.ingest.prepare`, never touch the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["IngestConfig", "reimport_corpus", "run_ingest"]

if TYPE_CHECKING:
    from .pipeline import IngestConfig, reimport_corpus, run_ingest


def __getattr__(name: str):
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
