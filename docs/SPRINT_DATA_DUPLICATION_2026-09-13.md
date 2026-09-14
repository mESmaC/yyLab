# Sprint training data: duplication and diversity — 13 September 2026

The actual 150M / 1B training payload contains substantial deliberate replay and highly repetitive task scaffolding. Whole-record exact deduplication was working, but it does not make the material diverse. **These measurements identify data problems; they do not establish that those problems alone caused the checkpoint's generation loops or that repetition is now fixed.**

The audit targets `ckpt_20260913_014748_0f6e04`: 488,281 saved rows of 2,048 input tokens, totaling 999,999,488 input slots and 999,511,207 shifted targets. The payload and provenance hashes match the saved manifest. The sampler's entire schedule was independently reconstructed and matched. Original corpora, training caches, checkpoint weights, and optimizer state were not changed. No new model-training run or data download was started; validation included a tiny disposable CPU training fixture.

## How much repeated data reached training?

| Measurement | Result | Interpretation |
| --- | ---: | --- |
| Explicit replay of the same source-pool position | **100,450,304 slots / 10.0450%** | Already-seen positions deliberately reallocated to meet source quotas |
| Exact duplicate whole 2,048-token rows, including replay | 100,689,920 slots / 10.0690% | Only 117 additional redundant rows at different pool positions; all in FABLE |
| Exact duplicate aligned 256-token blocks, including replay | 104,947,456 slots / 10.4948% | Additional repeated passages beyond whole-row replay |
| Exact duplicate aligned 128-token blocks, including replay | **113,563,008 slots / 11.3563%** | Includes 13,112,704 additional redundant slots beyond replay |

The block-length rows are alternative measurements and must not be added together. At 128 tokens, the nonoverlapping decomposition is 100,450,304 replay slots + 13,101,568 same-source content matches at other positions + 11,136 cross-source matches remaining after within-source deduplication. This census checks aligned offsets only; shifted passages, modified templates, and paraphrases can escape it.

Concrete non-replay matches include eight distinct FABLE rows made entirely of `A`, six rows of repeated DNA-like text, and many repeated box-drawing separators. Cross-source duplication was comparatively small at the measured alignments. See the [complete packed census](../artifacts/corpus-duplication-audit-20260913/PACKED_DUPLICATION_FINDINGS.md) and [machine-readable receipt](../artifacts/corpus-duplication-audit-20260913/packed-duplicate-census.json).

An independent [EOS-bounded passage audit](../artifacts/corpus-duplication-audit-20260913/complete-segment-audit.json) inspected all rows and found 1,580,514 complete segments of at least 32 tokens, covering 470,866,487 tokens (47.09% of the payload). All 21,044,613 redundant segment tokens were replay of the same source position; no exact matches at different positions were found in this coverage. Every hash match was also verified token for token. These are complete segments between EOS markers, **not necessarily whole source documents**; long documents and row-boundary fragments are underrepresented. This result cannot establish that the remainder is duplicate-free.

## Templates are the larger gap in exact deduplication

The [template-family audit](../artifacts/corpus-duplication-audit-20260913/template-family-audit.json) samples up to 1,024 unique consumed pool positions per source, removing sampler replay first. It measures overlapping 16-word/punctuation sequences occurring in earlier sampled rows.

| Source | Raw cross-row overlap | Overlap with numeric word tokens masked |
| --- | ---: | ---: |
| Elementary math | **75.05%** | **99.66%** |
| Kimi | 24.58% | 45.16% |
| Qwen | 12.82% | 13.62% |
| FABLE | 10.09% | 10.33% |
| Reasoning | 8.61% | 10.34% |
| Quotes | 0.271% | 0.271% |
| Prose | **0.177%** | 0.373% |

These are sampled, overlapping text-window statistics, **not percentages of training tokens proved wasted**. Nevertheless, the math sample has exceptionally repeated scaffolding. Its examples can remain different as whole records: 1,380 sampled complete math segments yielded 1,378 distinct whole-segment identities after digit/decimal masking. Removing identical records alone will therefore leave most template repetition intact. Representative task-family sampling and exposure caps are needed before treating this material as broad coverage. Automatically deleting every number-masked match would also discard some valid mathematical variation.

## The current prose allocation mostly replayed a small collection

| Source | Actual exposure | Distinct selected pool positions | Replay within that exposure |
| --- | ---: | ---: | ---: |
| Qwen | 100.00M slots | 43.50M slots | 56.50% |
| Prose | 50.00M slots | **10.05M slots** | **79.91%** |
| Kimi | 9.80M slots | 5.86M slots | 40.23% |
| Quotes | 0.201M slots | 0.141M slots | 29.59% |

Reasoning, math, and FABLE did not replay pool positions. Positional uniqueness still does not establish unique meaning. The globally shuffled training schedule interleaved first and repeated source cycles; these are provenance counts, not consecutive chronological epochs.

The [library diversity audit](../artifacts/corpus-duplication-audit-20260913/corpus-diversity-report.md) found **10,045,354 prepared prose content tokens across 44 training books/files**. Packing and EOS account for the small difference from 10,047,488 pool input slots. The collection is concentrated in historical reference, philosophy, education, and scientific material. Coarse book-level classification assigns approximately **1.05M tokens to narrative fiction**. One biographical compendium contributes 15.03% of prose; the five largest books contribute 31.77%. These are book-level categories, not a token-level topic classifier.

Increasing this prose corpus to 45% of a billion-token run would require roughly **44.8 passes** before further cleaning. That would increase its weight without creating new diversity. The library's 44 ready, nonempty, non-rejected corpus entries also include overlapping imports and aggregates; they are not 44 independent sources. Samples found synthetic distractor filler in Kimi, quote encoding problems and a dataset card in the quote pool, and repeated answer boilerplate in another teacher-labeled source. Source names and approval labels do not establish quality or authentic provenance.

## Safeguards implemented

New fixed-quota recipes in Sprint's training page enable **Remove duplicate and extremely repetitive sequences**. The optional, versioned policy:

- Removes later exact complete-token-row matches across selected sources.
- Removes rows of at least 256 tokens when at least 90% of their overlapping token 8-gram occurrences repeat an earlier occurrence within the row.
- Builds separate filtered pools from reusable raw pools, preserving original-row provenance. It does not retokenize unchanged sources or alter existing checkpoints/caches.
- Applies quota and replay admission to retained capacity, and records removals separately from consumed training tokens. Existing recipes with the option absent or disabled preserve their old data identity.

The [dry scan of unique consumed positions](../artifacts/corpus-duplication-audit-20260913/row-filter-dry-audit.json) would reject **129 rows / 264,192 slots**, only **0.02642%** of this run: 29 extreme-repetition rows and 100 other exact copies. Reasons are exclusive; low-information rejection takes precedence, explaining the difference from the 117-row exact-duplicate census. It retained every math row. This deliberately narrow filter catches concrete defects, but does not solve the larger template or replay problem. The dry scan covers selected positions, not every available pool row, and its old-run counts are not a prediction for a new mixture. It took 96.43 seconds on CPU, outside the optimizer path.

The ingest UI and documentation now accurately call MinHash **near-duplicate detection**: the existing implementation flags candidates and retains them. It does not remove near duplicates. See [policy and cache details](SPRINT_TRAINING.md#optional-cleanup-of-packed-rows).

## Diversity change warranted before another large run

Make varied natural and explanatory English a major component, with explicit coverage of everyday topics, narrative, history/culture, science, and ordinary dialogue. Acquire and count genuinely new material instead of increasing replay of the current books or adding more aliases of agent traces. Broad web text in [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), its educational selection [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), and full articles in [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) are source candidates to assess. They are not imported or locally quality-approved by this audit, and they do not by themselves cover all these needs.

The local inventory supports a **conditional 10M-token pilot proposal**, recorded in the [diversity report](../artifacts/corpus-duplication-audit-20260913/corpus-diversity-report.md#concrete-proposed-mixture-using-existing-data) and [proposal receipt](../artifacts/corpus-duplication-audit-20260913/diversity-pilot-proposal.json): 45% selected prose, 30% cleaned general-language reasoning, 15% broad general direct answers, 5% science direct answers, 3% selected Qwen general answers, and 2% substantive dictionary definitions. It excludes the named math pool and the problematic/specialized sources from the first general-English study.

**Those cleaned selections do not yet exist as ready corpora. Applying these percentages to the unfiltered corpus IDs would not implement the proposal.** Their retained capacities, correctness, family overlap, and rendering must be measured first. Cap individual books and task families, preserve original/book/family holdouts, and require the selected token budget to fit without replay. No pilot was queued. A billion-token recipe needs a much larger verified English base than is currently demonstrated.

## Remaining gaps and validation

The [code audit](../artifacts/corpus-duplication-audit-20260913/code-audit.md) also proved two separate unresolved issues: common SFT deduplication uses flat text instead of rendered inputs plus supervision masks, and ordinary training/evaluation iterators do not share an exact held-out exclusion set. **Neither is established as an error in this causal 150M run**: its custom preparation preloaded held-out hashes. They need separate versioned fixes before relying on those ordinary paths. General near-duplicate removal and template-family caps are not implemented by the new row filter.

The final [regression run](../artifacts/corpus-duplication-audit-20260913/regression-tests.log) passed **472 tests**, covering filtering, quota capacity/replay, cache compatibility, prepared tokens, UI contracts, and actual CPU training/evaluation/saving. Ruff passed for the touched production and new test modules. Final SHA-256 checks of the original checkpoint's weights and trainer state still match their protected original hashes. These checks verify tooling and preservation, not learned language quality. Any future data intervention still needs independent generation and capability evaluation alongside held-out loss; the [previous repair study](SPRINT_REPETITION_REPAIR_2026-09-13.md) showed why a lower exact repetition metric alone is insufficient.
