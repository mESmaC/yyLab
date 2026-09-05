# Import GCCF classified corpora

In **Corpora**, select **Import GCCF receipt** and enter the path to a
`gccf.receipt/1` JSON receipt. By default, yyLab builds a metadata index over the
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

# Optional: explicitly create a processed copy.
.venv\Scripts\python.exe -m yylab.ingest.gccf C:\path\gccf-receipt.json --materialize

# Reuse a SPALMER export rather than extracting the original corpus again.
.venv\Scripts\python.exe -m yylab.ingest.gccf --export-dir C:\data\gccf-approved
```

The shared reader lives in the sibling `../spalmer` checkout, now used as yyLab's
editable uv dependency. Both checkouts are required for this local development
configuration. For a bounded ingestion smoke test, add `--limit-per-file 32`;
this is explicitly recorded as a partial prefix sample, not a training corpus
quality assessment. The cache separates metadata indexes, materialized exports,
and partial/full scans.

The index stores record locations, content identities, deduplication/split
decisions, and counts. Training and previews extract canonical text and messages
from the original files on demand. No source-text spool or processed corpus is
created during indexing. Source size/mtime and record identities are checked on
read; changed sources require a new receipt/index. Repeating the same import
reuses its index and ready registrations. The shared index is also a valid input
to SPALMER `prepare-data`, which writes token shards directly from bounded memory
buffers instead of making a temporary text corpus.

## Optional processed copy

Enable **Create a processed copy** in the dialog, or pass `--materialize`, to
request compressed yyLab shards. The already completed September 5 import remains
available in this form. It was not deleted or recreated when the default changed.

Registration verifies each record's normalized text against its export ID.
Changed records are excluded and listed, without their contents, in each corpus's
`gccf-integrity-rejections.jsonl`; `stats.json` records the count. Failed imports
can be retried with the same command. Ready corpora are reused.
Missing records are separately counted in `gccf-integrity-summary.json` and the
verified manifest; their IDs cannot be recovered from the remaining export.
An export with more records than its manifest is rejected.

The importer also publishes `verified/data/` beneath the export directory, using
the normal native `.jsonl.zst` shards. `verified/verified-export.json` records the
usable counts and receipt lineage. Point standalone SPALMER `prepare-data` at
that **verified/data** directory. Its reader understands the native fields and
checks GCCF identities too. This verifies normalized canonical text, not every
metadata field or the exact original byte representation.

On this machine, Windows Defender remediated embedded scripts in part of the
security-code export. Changed records are excluded; security settings and
quarantined content are left alone. Use the verified view when using that existing copy.

## Training

Select the **train** split for training. Use token-balanced corpus mixing and
source weights to prevent the largest code collections from dominating. Start
with classified-only data; raw general prose can later supply a controlled
10–20% of training tokens. Global ingest deduplication does not let previously
ingested raw data erase approved copies; training-time cross-corpus deduplication
is still available.

The supplied 16-book collection is now available as **Raw prose · english-prose**,
backed directly by the originals. See [ENGLISH_PROSE.md](ENGLISH_PROSE.md) for
its book-level holdouts, unclassified provenance, and a 90/10 starting target.

The full schema and extraction/split guarantees are documented in SPALMER's
`docs/gccf-ingestion.md`. GCCF export artifacts are reusable by SPALMER's
`prepare-data` command and are not tied to yyLab's database.
