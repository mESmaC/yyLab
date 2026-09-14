# Disposable BF16-moment optimizer experiment

These are byte-identical archives of `bf16_moments_candidate.py` and its six CPU oracle tests from `artifacts/sprint-1b-speed-20260911/`. They preserve the experiment outside the ignored artifacts directory. Restore them to that original location before executing: their repository-root discovery assumes that path.

The candidate changes AdamW numerics. Large compiled slices store first and second moments in BF16; small native slices retain FP32 moments. Masters and update arithmetic stay FP32, with explicit moment conversion to FP32 inside the fused update. Each update uses the newly computed, unrounded FP32 moments, then stores rounded BF16 moments for the next step. BF16 accelerator weights/gradients, clipping, learning-rate scheduling, weight decay, and the existing CPU-offload transfer pipeline remain in use. Fixed slice layouts keep missing gradients from changing moment storage midway through a run.

All six small CPU oracle tests passed on September 11, 2026. They cover persistent rounding, native tails, missing gradients, learning rates/decay, state bytes, and failure handling. These tests mock compilation; they do not establish real compiled-kernel performance or GPU correctness.

A subsequent RTX 5070 run using the [confirmed recipe](../confirm-bf16-moments-micro8-b32.json) completed five warmup and 64 measured steps: **13,180.519 supervised targets/s**, versus 12,095.279 for the 24-step FP32-moment baseline (+8.97%). Both used the same 999,346,176-parameter shape, `nvfp4_mxfp8_fast`, microbatch 8, accumulation 4, and sequence 512; the measurement lengths differ. The confirmation processed 1,046,528 supervised targets in 79.3996 seconds and retained 8,002,322,432 bytes of CPU optimizer state, including masters, versus 11,992,154,112 baseline bytes. See the [full receipts](../../../sprint-speed-round3-2026-09-11.json). This is repeated synthetic-batch throughput with finite loss and sampled decoder updates, not held-out evaluation. There is no convergence or model-quality claim, and no production optimizer integration or checkpoint/resume support.

The final effective-batch-128 confirmation used microbatch 8 and accumulation 16 for 32 measured steps: **17,454.029 supervised targets/s**, from 2,093,056 targets in 119.9182 seconds. Its matching effective-batch-128 FP32-moment baseline measured 17,026.519 targets/s over 16 steps, a +2.51% difference. State memory remained 8,002,322,432 bytes, and all 24 sampled decoder matrices changed with finite loss and gradient norm. This comparison also has different measurement lengths; changing effective batch from 32 to 128 changes optimizer update frequency, so results across those batch settings are separate comparisons.

From the repository root, restore the archived files into a clean experiment directory:

```powershell
New-Item -ItemType Directory -Path artifacts/sprint-1b-speed-20260911 -Force | Out-Null
Copy-Item -LiteralPath docs/recipes/sprint-speed-round3/experiments/bf16_moments_candidate.py -Destination artifacts/sprint-1b-speed-20260911/bf16_moments_candidate.py
Copy-Item -LiteralPath docs/recipes/sprint-speed-round3/experiments/test_bf16_moments_candidate.py -Destination artifacts/sprint-1b-speed-20260911/test_bf16_moments_candidate.py
.venv/Scripts/python.exe -m pytest artifacts/sprint-1b-speed-20260911/test_bf16_moments_candidate.py -q
```

Use the same saved official precision-benchmark recipe for the baseline and candidate. The example below assumes it is saved as `recipe.json`; selected recipes and measured results are recorded separately. The recipe must request `cpu_optimizer_kernel="compiled"`, CPU optimizer offload, and BF16 parameter storage. It cannot contain `experimental_cpu_adam_min_elements`: this candidate retains the baseline large-slice cutoff. A complete benchmark receipt containing a `recipe` object is also accepted.

Run each command in a fresh process, serially, using new output directories:

```powershell
.venv/Scripts/python.exe -m yylab.train.benchmark_sprint_precision --config recipe.json --output artifacts/reproduce-round3-fp32-moments
.venv/Scripts/python.exe artifacts/sprint-1b-speed-20260911/bf16_moments_candidate.py --config recipe.json --output artifacts/reproduce-round3-bf16-moments
```

The candidate uses the official benchmark's model, seed, batches, warmup, measured steps, and validation checks. Its CPU kernel is prepared and checked on disposable buffers before model steps. Preparation errors fail the case; live update errors propagate without retry. Use the dependency and source versions from the resulting receipt when reproducing a measurement.

`benchmark.json` records `optimizer_candidate`, the candidate source hash, the normalized recipe, runtime settings, and actual memory inventory. The candidate verifies CPU optimizer state storage equals `8 × BF16-moment parameter elements + 12 × FP32-moment parameter elements` bytes, including FP32 masters. Baseline FP32 masters and moments need `12 × parameter elements` bytes. Accelerator tensors, staging buffers, and execution workspaces are separate. Compare speed only with matching model, matrix precision, effective batch, and timing settings, and always identify the changed optimizer numerics.

Archive SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `bf16_moments_candidate.py` | `a3797f6fb7fba385e3f1f9d246b96365d85ee6ad0eac59db4c3b7324fc6794f0` |
| `test_bf16_moments_candidate.py` | `c9302e9b3b93eb8e348859f9848a54b7216133f1e2c6df5212bcfcbc22ffab20` |
