# Import GCCF classified corpora

A GCCF receipt is a manifest of original source locations and processing
decisions. **yyLab imports receipts by reference. It does not create an extracted
text export or a second set of compressed corpus shards.**

In **Corpora**, select **Import GCCF receipt** and enter the path to a
`gccf.receipt/1` JSON receipt. yyLab builds a metadata index over the
original datasets. It applies approved-file and bad-record decisions, deduplicates,
preserves held-out splits, and creates one selectable corpus per approved source.
Corpus text stays in the datasets directory. Names begin with `GCCF ·`.

The reviewed source files must still exist and match their recorded sizes and
modification times. File approval reflects sampled records, not a classifier
verdict on every document. The import retains that distinction in provenance.

From the yyLab checkout:

```powershell
uv sync
.venv\Scripts\python.exe -m yylab.ingest.gccf C:\path\gccf-receipt.json
```

The shared reader lives in the sibling `../spalmer` checkout, now used as yyLab's
editable uv dependency. Both checkouts are required for this local development
configuration. For a bounded ingestion smoke test, add `--limit-per-file 32`;
this is explicitly recorded as a partial prefix sample, not a training corpus
quality assessment. `--output` chooses the metadata-index directory. Partial and
full scans have separate identities.

The index stores record locations, content identities, deduplication/split
decisions, and counts. Training and previews extract canonical text and messages
from the original files on demand. No source-text spool or processed corpus is
created during indexing. Source size/mtime and record identities are checked on
read; changed sources require a new receipt/index. Repeating the same import
reuses its index and ready registrations. The shared index is also a valid input
to SPALMER `prepare-data`, which writes token shards directly from bounded memory
buffers instead of making a temporary text corpus.

Training can separately create reusable token records and packed batches, as
described in [prepared training data](PREPARED_TRAINING_DATA.md). These are
tokenizer- and training-specific caches; the GCCF importer does not make duplicate
text corpora to feed them.

## Earlier copied imports

Earlier yyLab versions offered **Create a processed copy**, `--materialize`, and
`--export-dir`. Those copying paths have been disabled. Legacy API calls that
request copying fail before creating output or changing registry entries.

Already registered native corpora remain readable. Their saved training caches
and checkpoints are preserved; this change does not silently migrate an existing
corpus or reinterpret a saved training schedule. To refresh GCCF data, import the
original receipt through the source-index path. Migrating existing corpus IDs
requires an explicit, separately verified migration of their saved data identity.

The September 13 artifact cleanup removed an unused 62.76 GB uncompressed import
export while preserving its receipt, original sources, native corpus shards, and
saved-run dependencies. Those old copied corpora may still record the former
export folder as their historical re-import source. Generic Re-import of a
missing source fails without replacing the existing corpus. Use the original
receipt for future ingestion; do not recreate the copied export as the normal
workflow.

## Training

Select the **train** split for training and preserve original-source holdouts.
Choose measured token quotas and source/task-family exposure limits. A corpus's
approval label or document count does not establish topic diversity or factual
correctness. The [duplication and diversity audit](SPRINT_DATA_DUPLICATION_2026-09-13.md)
documents the current library's limitations.

The English prose collection also reads its original files. See
[ENGLISH_PROSE.md](ENGLISH_PROSE.md) for its preparation and holdout receipts.

The full schema and extraction/split guarantees are documented in SPALMER's
`docs/gccf-ingestion.md`.
