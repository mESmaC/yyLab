# Prepared training data

Implemented 2026-09-05 for HF base training, HF MoLE training, and the native
SPALMER worker. This is a data/runtime improvement before further architecture
experiments. It does not change the intended SPALMER-base → MoLE pipeline.

Training now prepares reusable token records before the first step. Later runs
with the same tokenizer and corpus reuse them. GCCF and English-prose source
pointers still read the original datasets during preparation: there is **no
new corpus text export**. Native scratch preparation runs before model allocation;
the HF worker obtains its tokenizer while loading the model. The standalone
preparation command below avoids loading any model weights.

## Configuration and use

The existing training entry points enable this automatically. Recipes accept:

| Field | Default | Meaning |
| --- | --- | --- |
| `token_cache` | `true` | Prepare/reuse tokens; `false` retains live tokenization. |
| `token_shard_bytes` | `67108864` | Maximum combined token/mask payload per physical shard; 64 MiB. |
| `data_prefetch_batches` | `2` | CPU batches queued ahead; `0` reads synchronously. One additional batch can be in production. |
| `token_cache_dir` | `artifacts/cache/prepared-tokens` | Shared cache location, independent of the run directory. |
| `token_cache_verify_sources` | `stat` | Validate original-file size, mtime and ctime at cache lookup. `sha256` also hashes original contents. |
| `max_text_chars` | `65536` | Existing causal tokenization chunk boundary; part of the cache identity. |

SPALMER's advanced training controls expose shard size, read-ahead depth, and
optimizer moment placement. `training.optimizer_state_offload` accepts `cpu`
(default) or `none` (moments on the parameter device). Both retain BF16 parameters,
FP32 moments, and stochastic parameter rounding. GPU moments require roughly
8 additional bytes per dense trainable parameter. The worker verifies physical
placement after initialization and records it in the run receipt.

From the yyLab directory, prepare the existing prose supplement with the saved
16K tokenizer without constructing a model:

```powershell
.venv/Scripts/python.exe -m yylab.train.prepared --corpora cor_20260904_230255_cb24da --tokenizer tok_20260904_124041_6ec536 --shard-mib 64
```

The command accepts multiple corpus IDs, `--objective chat_sft`,
`--max-text-chars`, `--verify-sources sha256`, and `--cache-dir`.
HF tokenizers and their native SPALMER HF wrappers share the tokenization identity.
Custom tokenizers must expose a stable native identity or serializable tokenizer
artifact; otherwise use `token_cache=false`. BPE dropout is rejected for caching.

## What is preserved

The cache sits below the existing corpus mixer and packers. Causal documents keep
their existing text-chunk tokenization and one EOS per original document. Physical
file boundaries add no tokens. HF rows keep their internal label shift; native
SPALMER still receives N+1 input tokens for N targets. SFT caches store rendered
assistant-turn examples and supervision masks; existing continuation overlap,
padding, and assistant-only labels remain in the packer.

Source order, domain/split filters, exact-text deduplication, expert document
sharding, corpus weights/caps, and seeded shuffling remain in the existing mixer.
Original character counts and byte-budget estimates are retained so switching to
compact records does not change mixing/shuffling decisions. The preparation pass
includes all records in each selected corpus; held-out records remain excluded
from training by the same split filter. Preparation does not train a tokenizer.

Read-ahead publishes statistics when its consumer receives a batch. It stops and
closes readers on normal completion, early termination, or an exception. Native
resume reproduces the consumed prefix from token records without source parsing
or tokenization. It still replays that prefix; direct seeking into a persisted
packing/shuffle state is not implemented.

Run `tokenized_view_manifest.json` files record which prepared cache directories
were used, their token/document counts, and the read-ahead setting.

## Storage, validation, and bounds

Tokens are little-endian uint32: 4 bytes/token for causal preparation. SFT adds
one supervision-mask byte per rendered token. SQLite holds document/unit offsets,
source metadata and hashes, but no `text` or `messages` fields. A token unit can
span files, so even an oversized document respects the physical payload limit.
Index metadata is separate from that limit. This format differs deliberately from
SPALMER's existing random document-window shard format; substituting that sampler
would change yyLab's training stream.

Preparation buffers bounded causal text chunks and writes shards incrementally.
The existing whole-record SFT rendering limit remains. Reads keep at most eight
shard handles per corpus, plus the bounded batch queue and existing packing/shuffle
buffers. Cross-corpus deduplication still keeps a hash set proportional to corpus
size. No claim is made that the complete pipeline uses constant memory.

Cache identity covers tokenizer contents/settings/version, objective, chunk
policy, source generation/index, original file inventory, and shard size. Seed,
mixture, sequence length, batch size, and epoch count do not cause retokenization.
For versioned prose indexes, cleaning happens before this cache layer. The
profile and rendered hashes are included in the index, giving cleaned prose a
distinct cache identity. Existing raw corpus IDs retain their original data.
Source pointers/indexes and the prepared metadata index are checksummed; token
files are checksummed before their first read in a reader. Builds publish an
immutable directory only after a final source-snapshot check. Failed builds remove
their staging output. A process-killing interruption can leave a `.lock` and
staging directory; inspect the recorded PID before removing an abandoned build.

The default original-source stat check assumes dataset contents are not changed
while preserving all recorded file metadata. Use `sha256` for a content audit;
that intentionally rereads source bytes at lookup. Treat published token caches
as immutable during use. Changing a corpus, tokenizer, objective or shard size
creates a new artifact. Unused variants are not automatically deleted. Large
corpus preparation has a real disk cost: a billion causal tokens require about
3.73 GiB plus metadata. Confirm the intended tokenizer/view and available disk
capacity before a full-sized conversion.

CPU read-ahead overlaps reading/packing with the consumer. It does not implement
a separate CUDA copy stream or claim DMA/compute overlap. HF target/padding counts
now use host tensors, native causal labels reuse the transferred input tensor,
and native loss metric host transfers occur at the accumulation boundary.
Other native routing/telemetry synchronization and the custom BF16 optimizer's
update kernels still need profiling. No fused-loss or compilation change is included.

## Measured delivery results

RTX 5070 workstation; **CPU-only data benchmark**, no model constructed or trained.
Input: existing 16-book source index, saved `SPALMER 50M Mixed Byte BPE 16K v1`
tokenizer, train split, seed 42, 512-token rows, microbatch 1. Every comparison
delivered 2,048 batches / 1,048,576 identical input tokens. Three repetitions per
case; the OS page cache was not flushed. Timing `next()` excludes digest work and
any simulated consumer delay. These numbers do not measure NVMe bandwidth or
end-to-end training throughput.

| Delivery mode | Median accumulated input wait |
| --- | ---: |
| Live tokenization, synchronous | 272.8 ms |
| Prepared 1 MiB shards, synchronous | 49.6 ms |
| Prepared 4 MiB shards, synchronous | 49.0 ms |
| Prepared 16 MiB shards, synchronous | 53.1 ms |
| Prepared 64 MiB shards, synchronous | 51.8 ms |

The default prepared path is about 5.3× faster on this delivery-only measure.
Initial preparation took 3.8–4.1 seconds per layout; reuse setup took about
24–29 ms. The complete supplement produced 4,136,621 tokens, 16,546,484 payload
bytes, and about 17.3 MiB including metadata. Only the default 64 MiB cache is
retained; the three benchmark layout variants were removed after measurement.
No full GCCF corpus conversion was started.

Read-ahead adds queue overhead when a consumer drains as fast as possible. With
a **simulated** 1 ms pause after each batch, the 64 MiB prepared path's median
input wait fell from 138.7 ms to 44.6 ms with two-batch read-ahead; total drain time
including the sleeps and digest work changed from 3.260 to 3.195 seconds. This
supports keeping read-ahead configurable, not claiming a model speedup.

The token payload is only 15.8 MiB, so 16 and 64 MiB targets both produced one
file. This test cannot select a shard-size optimum for a much larger corpus or
establish cold-storage bandwidth gains. A 64 MiB default is provisional; 16/64/256
MiB comparisons remain useful when a representative larger stream is ready.

Raw measurements: [delivery](training-data-benchmark-2026-09-05.json) and
[simulated consumer](training-data-read-ahead-2026-09-05.json).
Reproduce with `python -m yylab.train.benchmark_data --help`.

Validation: 239 yyLab checks passed across cache/packing, training, source
ingestion, native integration and UI recipes; 32 SPALMER engine/optimizer/resume
checks passed. GPU checks exercised both physical FP32 moment placements. Cache
tests cover exact tokens/masks/counters, reuse/invalidation, corruption rejection,
shard boundaries, read-ahead cleanup/errors, and native resume. New architecture
experiments remained paused pending a representative model-throughput baseline.

## Subsequent native model measurements

The [real 50M training benchmark](../../spalmer/docs/training-benchmark-50m-results-2026-09-05.md)
completed 1,000 updates / 40.96M targets through the prepared-data path. The early
diagnostic sample spent 0.17% of step time waiting for input. This does not
establish a raw-versus-prepared end-to-end speedup.

The following [forward/backward optimization](../../spalmer/docs/training-compute-optimization-2026-09-05.md)
improved grouped expert dispatch in native SPALMER. A short order-balanced
comparison measured 14,961 → 16,992 supervised targets/s (+13.6%) with identical
paired losses and final learned parameters/optimizer state. It is picked up by
yyLab's native path directly; no recipe or corpus changes are required.
