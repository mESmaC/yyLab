# Original English prose supplement

## Latest prepared corpus: September 11, 2026

Select **Prepared prose · english-prose — 2026-09-11 (51 files)** in yyLab,
corpus `cor_20260911_200936_0c330d`. This refresh includes all 51 source files,
up from 35, using the existing `prose_v1` cleanup and book-level split policy.
It contains **4,344 retained records**: **4,100 train**, **103 validation** and
**141 test**. The 16 added files contribute 956 additional training records.
The prepared view contains 36,924,275 UTF-8 bytes and 6,273,275 words.

Both existing tokenizers have verified reusable 64 MiB token-record caches:

| Tokenizer | Train content tokens | Validation | Test | All splits |
| --- | ---: | ---: | ---: | ---: |
| Sprint English Math Byte BPE 8K | 10,045,354 | 246,489 | 326,770 | 10,618,613 |
| SPALMER Mixed Byte BPE 16K | 10,412,803 | 247,873 | 331,064 | 10,991,740 |

Counts exclude document-boundary EOS tokens and training-specific packing.
All cached token IDs matched fresh tokenization; payload hashes, cache reuse,
source hashes, and every held-out record's identity and rendered hash passed
verification. All 35 previous source files remain unchanged. Historical corpus
IDs and caches remain available; select this refreshed corpus instead of adding
both old and new versions to the same training mixture. No training was started.

The existing filter excluded 336 duplicate windows in total, including 11 from
the newly added duplicate `pg3711134.txt` (*The Elements of Style*). A filename
claiming *The Pragmatic Programmer* actually contains *The Craft Journey* by
J. R. Cleland; its supplied content and filename were preserved. The audit found
no added normalized windows matching existing holdouts; it is not a paragraph-
level or near-duplicate isolation guarantee.

See the [refresh receipt and source audit](english-prose-reprocess-2026-09-11.json)
for exact cache locations, tokenizer IDs, source inventory, and verification.

## Historical September 4 registration

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
  --selection docs\english-prose-selection.json --preprocessing none
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

## September 7 preprocessing update

The command above explicitly preserves the historical raw view. New imports
default to `prose_v1`: LF line endings, capped excess blank lines and conservative
paragraph reflow before tokenization. Existing corpus IDs and caches remain
unchanged. The profile is part of the new index/import/cache identity.

The expanded collection is now also available as **Prepared prose · english-prose**
(`cor_20260907_194430_e2959b`), with 7,942,957 training-content tokens. See the
[preprocessing results](../../spalmer/docs/prose-preprocessing-2026-09-07.md)
and before/after examples there. No training was launched for this update.

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
