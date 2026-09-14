# Sprint packing and cache reuse â€” 2026-09-11

Sprint now reuses compatible packed data across token budgets and packs new
causal data faster. No setting change is required for new workers.

## Measured result

Latest selected ten-corpus mix from `run_20260911_233414_7e34c9`, using its
selected tokenizer, sequence length 1,024, and 16,777,216 token slots. Three
fresh-output builds per implementation, alternating execution order; medians:

| Operation | Before | After |
| --- | ---: | ---: |
| Build 16.8M packed tokens | 11.952 s | 4.597 s |
| Packing throughput | 1.404M tokens/s | 3.650M tokens/s |
| Reopen the completed cache | â€” | 0.033 s |
| Reuse its 8.4M-token prefix | Repacked | 0.034 s |

Cold cache construction is **2.64Ã— faster** in this measurement. All three
comparisons produced identical token-file hashes and identical document/source
accounting. These are CPU data-preparation measurements on the Core Ultra 7
265F, not model-training throughput. The OS page cache was not flushed. Timing
includes source identity checks, reading, tokenization, packing, hashing, writing
and syncing the new files; it excludes imports, loading the tokenizer and model
setup. The sample does not establish a whole-10B-corpus runtime.

The [measurement receipt](sprint-packing-2026-09-11.json) includes every timing,
the selected corpora/tokenizer, output checksums and baseline revision.

## Reuse rules

Finished caches live under `artifacts/cache/sprint-packed`. Sprint first tries
the exact request, then searches compatible completed caches. It can map a
prefix of a larger cache, or repeat an already complete source pass for a larger
budget. It never treats a shorter budget-truncated cache as a complete corpus.
No packed files are copied or modified when reusing a prefix.

Model size, learning rate and optimizer settings do not invalidate packed data.
Changes to tokenizer contents, source snapshots, objective, sequence length,
seed, corpus order/mixing or filters still require different rows. Resumed
training retains a logical data identity and exact cursor even if a different
physical cache provides the same rows. Prefix statistics report only known
prefix totals; full backing-cache statistics are labelled separately.

Interrupted packing directories are not reusable checkpoints. The abandoned
lock from the cancelled 10B-token run was archived after verifying its worker
had exited; its partial data was left intact. No training run was restarted.

## First-build changes

Causal packing now fills bounded NumPy row chunks and validates/casts/writes
many rows together. Tokenization uses up to 128 text chunks per physical call,
while retaining the original groups of eight for exposure accounting. A private
clone of the generic fast tokenizer uses its IDs-only encoder; custom tokenizer
classes keep their original preprocessing path. The source tokenizer and its
fingerprint are unchanged. Chat SFT keeps its assistant-label rendering and
benefits from compatible packed-cache reuse.

The optional per-document token cache remains available for experiments that
change mixtures or sequence length. It prepares entire corpora before packing,
so it remains opt-in. The packed cache is always enabled.

Validation covers raw/prepared token parity, EOS placement, source attribution,
SFT masks/padding, tokenizer special tokens and Unicode, cache invalidation,
truncated-cache refusal, and exact resumed training across physical caches.

All 336 focused tests passed, and Ruff reported no issues. Custom tokenizer
call boundaries and required versus unused read-ahead failures are also covered.
