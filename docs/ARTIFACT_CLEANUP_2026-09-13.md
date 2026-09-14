# Artifact cleanup — 13 September 2026

Removed **391 reviewed files totaling 110,311,787,483 bytes**: **110.3 GB / 102.7 GiB**.
The artifact inventory decreased from **222.1 GB to 111.8 GB** of logical file
sizes. The measured increase in free drive space was 110.31 GB. There were no
deletion failures.

## Removed

| Category | Bytes removed | Reason |
| --- | ---: | --- |
| Uncompressed GCCF import export | 62,755,900,190 | Redundant staging created by the earlier copying importer; native training shards and original receipt/source files remain |
| Superseded periodic optimizer states | 43,326,032,514 | 56 intermediate `trainer.pt` files outside registered, latest, and best checkpoints |
| Rejected repetition diagnostic weights | 3,002,059,820 | Five unregistered intervention/control models; original model, scripts, metrics, and generated samples remain |
| Packing benchmark payload copies | 679,477,248 | Completed before/after and tokenizer benchmarks |
| Completed canonicalization hash index | 184,741,888 | Temporary deduplication index; canonical source records remain |
| Verified redundant release archives | 207,761,273 | Spalmer archive members match retained local files; Sprint 10M archive matches its published release asset |
| Optimizer test vectors | 71,395,990 | Completed diagnostic tensors |
| Abandoned packing staging | 60,588,032 | Incomplete hidden cache with no manifest, references, lock, or active writer |
| Audit scratch hash index | 23,830,528 | Completed duplicate-filter dry scan; audit receipts and original payload remain |

The [file-level plan](../artifacts/cleanup-20260913/deletion-plan.json) and
[deletion receipt](../artifacts/cleanup-20260913/deletion-receipt.jsonl) record every
absolute target, size, and reason. Deletion used individual literal paths checked
against the resolved workspace artifact root, protected paths, file sizes, and
modification times. No reparse points were traversed. Only reviewed empty staging
directories were removed afterward, without recursive deletion.

## Preserved and verified

- **Every registered model and every periodic model-weight save** remains
  available for sampling. SHA-256 hashes of all 156 weight files under registered
  checkpoints, runs, and adapters matched before/after (26.90 GB checked).
- **17 complete trainer states** remain, including registered final models,
  each run's latest checkpoint, and best saved evaluation checkpoints. The
  user's registered 100M **16k** checkpoint and its **14k** best save remain;
  the recent 150M **29k** best and **30k** latest saves also remain.
- All **782 registered native corpus shards** remain. For the 13 GCCF corpora
  associated with the removed export, all 290 source-file identities matched
  exactly, and the real training reader returned a document from every corpus.
- All **18 complete packed caches**, **108 prepared-token files**, and the
  recent 150M source pools/evaluation inputs remain. Expensive usable preparation
  was retained.
- Corpus originals, canonical agent-trace sources, tokenizer files, adapters,
  annotations, registry records, reports, generated samples, profiler traces,
  and report-rendering dependencies remain.

The before/after inventory found exactly the planned missing files and no
unexpected payload changes. SQLite WAL/shared-memory timestamps changed during
registry access; their sizes stayed fixed. Registry counts, run statuses, and
protected paths were checked separately and remained unchanged.

Verification receipts:
[model hashes](../artifacts/cleanup-20260913/model-verification-after.json),
[registry and checkpoints](../artifacts/cleanup-20260913/registry-verification-after.json),
[training data and caches](../artifacts/cleanup-20260913/data-verification-after.json),
[inventory comparison](../artifacts/cleanup-20260913/inventory-verification-after.json).

The earlier GCCF export's `verified/data` tree contains hardlinks to retained
native corpus shards. It was preserved; counting both names inflates logical
directory totals without consuming a second copy of those disk blocks. Every
deleted file had a single hardlink, and the observed drive-space recovery agrees
with the removal total.

## GCCF receipt-only correction

The user clarified that a GCCF receipt represents source locations and processing
instructions and must not cause corpus text to be copied. yyLab previously
offered an explicit copying UI checkbox, a `materialize=True` backend branch,
and `--materialize` / `--export-dir` CLI options. That implementation produced
the old materialized export; copying is not part of the intended receipt contract.

The copying implementation and UI option have now been removed. Future receipt
imports build metadata indexes and read original source files. Compatibility API
calls that request copying fail before receipt inspection, output creation, or
registry changes; retired CLI flags fail before database initialization.
Existing native corpus readers remain supported. **No existing corpus ID was
migrated, no original source was changed, and no real dataset was re-imported.**

The [GCCF ingestion guide](GCCF_INGESTION.md) now specifies this contract. The
change passed **60 focused/adjacent tests**, including actual source-index
ingestion, selection/splits/provenance, idempotent registration, source-file
dependence, failure before copying, and continued readability of existing native
corpora. Ruff and scoped whitespace checks passed. See the
[test log](../artifacts/cleanup-20260913/gccf-source-index-tests.log).

## Practical limits

Old periodic saves whose optimizer state was pruned still support sampling and
fresh continued training from their weights. They no longer support exact
optimizer resume at that step. Registered, best, and latest resume states remain.
Rejected diagnostic models must be rerun to generate new samples from those
particular experimental weights.

Some legacy GCCF corpus records still name the removed export as their old
generic re-import source. A missing source fails before replacing the current
corpus. **Use the original receipt to register source references for future
ingestion.** The pre-cleanup audit's export-rematerialization recovery instructions
describe the previous implementation and are superseded by the user's
receipt-only requirement. Migration of existing IDs and saved data identities is
separate work; preserving current native shards avoids invalidating saved runs.

This was a one-time cleanup. Checkpoint save intervals and automatic retention
settings were not changed.
