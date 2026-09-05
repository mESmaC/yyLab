# Original English prose supplement

**Raw prose · english-prose** is registered as `cor_20260904_230255_cb24da`.
It reads the 16 original files from
`C:\Users\engin\Documents\datasets\english-prose`; no corpus text was copied.
The 742,835-byte metadata index resides at
`artifacts/cache/text-sources/283724bfb2def47f0de5/text-index.json`.

The read view contains **13.62 MB / 2.27 million words** after wrapper and tail
exclusions. Twelve books supply 11.58 MB of training text. Two whole books each
are reserved for validation and test. This is sufficient for the next short
controlled pilots; broader contemporary English can be added if evaluation
shows a specific gap. The historical books and Pessoa OCR are not a comprehensive
or uniformly cleaned general-language corpus.

The existing GCCF-selected corpora remain the majority of the proposed mix.
Start with a target near 90% classified / 10% raw prose tokens, keeping identical
data weights across architecture arms. yyLab's token-balanced mode schedules
characters as a token proxy; measure actual token shares and repetition. It does
not automatically replay an exhausted source. Plain concatenation into the much
larger classified corpus would give these books negligible weight.

## Reproduce registration

From the yyLab checkout, with its sibling SPALMER editable dependency installed:

```powershell
.venv\Scripts\python.exe -m yylab.ingest.text_source `
  C:\Users\engin\Documents\datasets\english-prose `
  --selection docs\english-prose-selection.json
```

The selection file holds whole-book validation/test choices and the exact
trailing Pessoa boundary. Gutenberg wrappers are excluded automatically.
Repeated registration reuses the ready corpus. A changed inventory or selection
creates a new index; attempting to reuse an explicit output for different inputs
fails. The original files must remain available and unchanged for training.

The corpus appears in the ordinary corpus selector. Select `train` for training,
`validation` for tuning, and reserve `test` for the final comparison. The generic
Re-import action is disabled for indexes; use the command above to index a new
selection. Corpus previews and training both read from the originals.

Provenance uses `catalog_status=raw_source_index`,
`selection_origin=user_supplied_raw`, and `classified=false`, with source and
index hashes. No GCCF receipt or classifier verdict is assigned to the books.
Token preparation in standalone SPALMER accepts the same index directory and
writes token shards directly without a temporary text corpus.

See [english-prose-audit.json](english-prose-audit.json) for the complete inventory,
hashes, counts, split membership, and verification results. The implementation
and experimental limitations are described in the sibling SPALMER checkout's
`docs/english-prose.md`. All source hashes were verified unchanged; both project
readers produced identical text. SPALMER's full suite passed 463 tests (4 skipped),
and the yyLab integration selection passed 192. No model-quality result is
implied by these ingestion checks.

The first real-corpus B/BR/P1/P2 comparison has since completed using a measured
90.2% GCCF / 9.8% prose training mix. It confirmed the optimizer-state reduction
but showed a modest validation-loss penalty for rotating plasticity. See the
[SPALMER experiment report](../../spalmer/docs/real-corpus-plasticity-results-2026-09-04.md)
for measurements and limitations; this was a single-seed architecture pilot,
not a classified-versus-raw data efficacy comparison.

The follow-up [plastic expert budget study](../../spalmer/docs/plasticity-budget-results.md)
completed five 1,600-update runs with the same source pool and a realized
89.734% GCCF / 10.266% raw prose mix. Increasing the trainable expert subset
reduced the validation-loss penalty, but no rotating arm passed its preset
quality/resource screen. This study also read existing sources without creating
another corpus text copy. The current prose supply remains sufficient for the
next controlled architecture diagnostics.
