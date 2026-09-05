# Import GCCF classified corpora

In **Corpora**, select **Import GCCF receipt** and enter the path to a
`gccf.receipt/1` JSON receipt. yyLab extracts only included files, applies the
receipt's bad-record locators, deduplicates, preserves held-out splits, and creates
one selectable corpus per approved source. Corpus names begin with `GCCF ·`.

The reviewed source files must still exist and match their recorded sizes and
modification times. File approval reflects sampled records, not a classifier
verdict on every document. The import retains that distinction in provenance.

From the yyLab checkout:

```powershell
uv sync
.venv\Scripts\python.exe -m yylab.ingest.gccf C:\path\gccf-receipt.json

# Reuse a SPALMER export rather than extracting the original corpus again.
.venv\Scripts\python.exe -m yylab.ingest.gccf --export-dir C:\data\gccf-approved
```

The shared reader lives in the sibling `../spalmer` checkout, now used as yyLab's
editable uv dependency. Both checkouts are required for this local development
configuration. For a bounded ingestion smoke test, add `--limit-per-file 32`;
this is explicitly recorded as a partial prefix sample, not a training corpus
quality assessment. The cache separates partial and full exports.

GCCF imports compress already approved records directly into yyLab shards.
They preserve canonical causal text and structured messages for chat SFT without
repeating raw-corpus quality analysis. Native split membership and receipt hashes
flow into yyLab/SPALMER training provenance. Importing the same export again
reuses its ready corpus registrations. A later explicit Re-import can run the
ordinary yyLab analysis pipeline against the approved export.

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
quarantined content are left alone. Use the verified view for experiments.

Select the **train** split for training. Use token-balanced corpus mixing and
source weights to prevent the largest code collections from dominating. Start
with classified-only data; raw general prose can later supply a controlled
10–20% of training tokens. Global ingest deduplication does not let previously
ingested raw data erase approved copies; training-time cross-corpus deduplication
is still available.

The full schema and extraction/split guarantees are documented in SPALMER's
`docs/gccf-ingestion.md`. GCCF export artifacts are reusable by SPALMER's
`prepare-data` command and are not tied to yyLab's database.
