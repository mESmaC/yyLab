# Sprint: measured BF16 / CPU-offload training

These are the original measurements. See the
[new 50M–1B comparison](SPRINT_SPEED_2026-09-11.md) for the faster sizing and CPU
optimizer implementation.

The new default uses BF16 model weights and gradients on the RTX 5070, with
FP32 master weights and both Adam moments in CPU RAM. The roughly 1B model
completed full forward/backward/clipping/optimizer updates on this 12GB card.

Hardware: NVIDIA RTX 5070, Intel Core Ultra 7 265F (20 cores), 32GB RAM;
PyTorch 2.11.0+cu128. Every case below uses sequence length 512, effective batch
32 sequences, no activation checkpointing, and the native Sprint training-step
engine. The 1M and 50M shapes use different vocabularies; their token rates are
not directly comparable measures of text throughput or model quality.

| Exact parameters | Vocabulary | Microbatch × accumulation | Compilation | Supervised targets/s | Peak GPU allocation | Measured updates | Timed duration |
|---:|---:|---:|---|---:|---:|---:|---:|
| 994,080 | 2,048 | 32 × 1 | default | 1,813,934 | 0.35 GiB | 400 | 3.61 s |
| 49,902,720 | 8,192 | 32 × 1 | default | 92,739 | 3.60 GiB | 20 | 3.53 s |
| 999,625,536 | 8,192 | 4 × 8 | eager | 5,458 | 7.22 GiB | 4 | 11.98 s |

The 1B case retained 11,995,506,432 bytes (11.17 GiB) of FP32 CPU optimizer
state and peaked at 7.22 GiB of PyTorch GPU allocation. Smaller microbatches
leave room for accumulated gradients. The batch8/accumulation4 trial approached
the card's memory limit and slowed to about 2,506 targets/s; the batch4/accumulation8
case processed the same number of tokens per update at about 5,458 targets/s.
The near-capacity compilation trial was stopped before producing a usable
measurement. Those diagnostic artifacts remain under artifacts/sprint-validation.

These are short, single-run synthetic measurements on repeating token batches.
They include input transfer, forward, backward, clipping, CPU Adam updates and
weight transfers; they exclude compilation/warmup, corpus preparation, validation
and checkpoint IO. They are not convergence experiments and do not establish
equal quality to SPALMER or a universal fastest architecture. Full-corpus runs
record preparation and end-to-end throughput separately.

The benchmark intentionally keeps precision, optimizer location, sequence length
and effective batch visible. The page can apply completed settings only when
those execution settings, architecture, context and device match. Use the table
as a starting point and repeat the benchmark for the intended tokenizer/context.

All candidate results, warmup timings, model configurations, memory figures and
final source hashes are in [the receipt](sprint-benchmarks-2026-09-10.json).
See [the training guide](SPRINT_TRAINING.md) for usage and memory requirements.

Final validation passed 227 focused model, trainer, UI, loading and regression
tests, including five native CUDA tests. These check causal masking, loss and
gradient agreement, compilation, static/dynamic generation, BF16 CPU-offload
state placement and exact resumed updates. Ruff passed on the changed Python
implementation and tests. Only dependency deprecation warnings remained.
