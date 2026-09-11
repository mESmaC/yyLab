# Sprint: faster training from 50M to 1B parameters

Sprint's new throughput sizing and CPU optimizer update increased measured
complete-step throughput by **23–31%** on the RTX 5070. BF16 GPU weights and
gradients still use FP32 master weights and Adam moments in CPU RAM.

| Target | Before, targets/s | After, targets/s | Throughput gain | Before / after peak GPU allocation |
|---|---:|---:|---:|---:|
| 50M | 92,843 | 121,352 | 30.7% | 3.59 / 2.32 GiB |
| 500M | 11,045 | 13,589 | 23.0% | 6.21 / 4.65 GiB |
| 1B | 5,599 | 6,969 | 24.5% | 7.22 / 8.13 GiB |

These improvements reduce time per update by 23.5%, 18.7%, and 19.7%,
respectively. The 1B result uses a larger microbatch made possible by lower
activation memory. Effective batch remains fixed. The earlier 1B measurement
was eager; this comparison re-measured the old implementation with compilation
enabled, giving it the stronger 5,599 targets/s baseline.

## What changed

- In this earlier experiment phase, new targets of at least 50M parameters
  defaulted to a wider decoder with fewer
  layers, 256-aligned width, 128-dimensional attention heads and 64-aligned MLP
  width. Select **Original proportions** (`shape_profile="balanced"`) for the
  old planner. Smaller targets and explicit saved architectures kept their
  previous sizing behavior. The later [10M profile update](SPRINT_TRAINING.md#architecture)
  extends throughput sizing to 10M; this report preserves the earlier measurements.
- Two bounded pinned buffers overlap gradient transfers and weight writebacks
  with CPU updates. Events protect buffer reuse and checkpoint synchronization.
- Large BF16 slices use a compiled CPU Adam update that combines the conversions
  with FP32 master/moment updates. Small slices use native Adam. The default auto
  mode validates compilation on disposable buffers and uses native Adam if no
  working local compiler is available. It does not install a compiler.
- The benchmark and training page now retain CPU threads, transfer bucket size,
  attention implementation and the resolved optimizer kernel along with batch
  and compilation settings. Exact resume checks these numerical execution choices.

The measured profiles are:

| Target | Exact parameters before → after | New width / layers / heads / MLP | Microbatch × accumulation before → after |
|---|---|---|---|
| 50M | 49,902,720 → 49,816,576 | 1,024 / 4 / 8 / 3,008 | 32 × 1 → 32 × 1 |
| 500M | 499,821,696 → 499,329,792 | 2,304 / 9 / 18 / 6,976 | 8 × 4 → 8 × 4 |
| 1B | 999,625,536 → 998,505,728 | 2,816 / 11 / 22 / 10,112 | 4 × 8 → 8 × 4 |

The final measured settings use `torch_compile=true`, `compile_mode="default"`,
`attention_backend="cudnn"`, `cpu_threads=20`, `offload_bucket_mb=32`,
`cpu_optimizer_kernel="compiled"`, and no activation checkpointing. Auto attention
remains the portable default; explicit backends fail if unsupported. Re-run the
benchmark for a different tokenizer, context length or machine.

## Measurement and validation

Hardware: NVIDIA RTX 5070 12GB, Intel Core Ultra 7 265F, 32GB system RAM,
PyTorch 2.11.0+cu128. All six final cases use vocabulary 8,192, sequence length
512, effective batch 32, identical seeded synthetic inputs and the same optimizer
schedule. Each has five warmup updates. The 50M cases measure 60 updates; the
500M and 1B cases measure eight. Before uses saved original model/trainer source.

Timing includes forward, backward, clipping, CPU updates and transfers, with
CUDA synchronization around the measured interval. It excludes initialization,
compilation, corpus preparation, validation and checkpoint IO. Runs are short,
repeat the same synthetic batch and can be affected by desktop hardware noise.
These are throughput comparisons across different model shapes: **equal
convergence or final model quality has not been established**.

All **335 focused tests passed**, including native CUDA training, compiled CPU
update parity, shared-buffer reuse, mixed native/compiled slices, and bitwise
checkpoint continuation. The suite also covers model loading/generation, data
packing, budgets, UI recipes and existing training integration. Dependency
deprecation warnings remained. The compiler-cache isolation added after the
timing runs changes startup caching only; fused math and step dispatch are unchanged.

See the [full receipts and source hashes](sprint-speed-2026-09-11.json) and
[training guide](SPRINT_TRAINING.md).
