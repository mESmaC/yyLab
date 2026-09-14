# Sprint 1B speed experiments, round 3 — 11 September 2026

The confirmed improvement in this round is **13,181 supervised targets/s, +8.97% over the fresh 12,095-target/s baseline**, for the same 999,346,176-parameter Sprint architecture at effective batch 32. At effective batch 128, it reaches **17,454 targets/s, +2.51% over its own 17,027-target/s baseline**. Both gains come from storing large-slice Adam moments in BF16 while keeping FP32 masters and update arithmetic. **This changes optimizer numerics**; these are experimental speed/memory results without a model-quality or convergence comparison.

The 64-update confirmation reproduces the earlier 24-update BF16-moment result. Short microbatch-16 screens also looked faster, but the longer confirmation was interrupted without a completed timing; that speedup remains **unconfirmed for sustained execution**. A targeted diagnostic observed high memory pressure during its first warmup step, with no evidence of step-count-dependent growth.

## Selected completed results

All rows use microbatch 8, auto/Liger loss, the same model and the fast NVFP4-forward/MXFP8-backward path. Accumulation is 4 for B32 and 16 for B128. Gains compare only within the same effective batch; the optimizer-state change is explicit.

| Recipe | CPU moment storage | Targets/s | Same-batch change | Seconds/update | Measured updates | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-b32` | FP32 | 12,095 | — | 1.352 | 24 | 8.605 |
| `confirm-bf16-moments-micro8-b32` | BF16 large slices / FP32 tails | **13,181** | **+8.97%** | **1.241** | **64** | **8.605** |
| `baseline-b128` | FP32 | 17,027 | — | 3.842 | 16 | 8.605 |
| `confirm-bf16-moments-micro8-b128` | BF16 large slices / FP32 tails | **17,454** | **+2.51%** | **3.747** | **32** | **8.605** |

The B32 confirmation processes **1,046,528 supervised targets in 79.3996 seconds**. Its exact rate is 13,180.5193695 targets/s, close to the earlier screen's 13,182.9509222. The B128 confirmation processes **2,093,056 targets in 119.9182 seconds**, or 17,454.0288795 targets/s. Compared windows differ: candidate versus baseline uses 64 versus 24 measured updates at B32, and 32 versus 16 at B128. Retained CPU master/moment state falls from **11,992,154,112 to 8,002,322,432 bytes**, saving **3.716 GiB** in both cases. Peak reserved CUDA memory remains 10,626,269,184 bytes. These are short benchmarks, not completed billion-token training runs.

## Final trial status

The final evidence snapshot contains **16 accepted measurements, ten interrupted runs and two pretraining wrapper errors**. All failed/incomplete cases have no accepted rate. All benchmark processes have finished or were stopped; no further trials are scheduled for this round.

| Additional trial | Recorded status | Accepted targets/s |
| --- | --- | ---: |
| Fresh FP32-moment B128 baseline, microbatch 8 | Completed, 16 measured updates | 17,027 |
| First-warmup release, microbatch 16 / B32 retry | Interrupted without completed timing | — |
| First-warmup release, microbatch 16 / B128 retry | Interrupted without completed timing | — |
| BF16-moment microbatch 8 / B128 confirmation | Completed, 32 measured updates | 17,454 |
| Combined allocator/BF16-moment microbatch 16 cases | Prepared; unrun | — |

The native allocator GC/fraction and end-of-warmup-release recipes are also prepared but unrun. Source availability and admission checks are not execution results.

## Measurement controls

Hardware/runtime: NVIDIA GeForce RTX 5070 with 12 GiB VRAM on Windows; Intel Core Ultra 7 265F, 20 logical CPU cores and 32 GB RAM. Receipts record PyTorch 2.11.0+cu128, TorchAO 0.18.0, Transformers 5.16.1 and Triton Windows 3.7.1.post27. Cases run in separate processes with seed 42, vocabulary 8,192 and sequence length 512.

The baseline is a tied-embedding GELU Sprint decoder with **999,346,176 parameters**, width 4,096, six layers, FFN width 11,456 and 32 attention heads of dimension 128. Configured context capacity is 2,048; the timed workload uses length 512. Settings are default `torch.compile`, cuDNN attention, no activation checkpointing, auto/Liger loss, 20 CPU threads and a 32 MiB offload bucket. The CPU optimizer uses its compiled kernel for large slices and native Adam for small slices, with the default 1,048,576-element cutoff.

`nvfp4_mxfp8_fast` means native NVFP4 decoder-projection forward GEMMs and native MXFP8 backward GEMMs, with fused transpose/column-cast preparation. Trainable weights and gradients remain BF16. Embeddings, the tied output head, attention and loss retain BF16/FP32 operations. The ordinary optimizer retains FP32 CPU master weights and both FP32 Adam moments: **11,992,154,112 bytes**, exactly 12 bytes per baseline parameter, plus transfer/scratch buffers. This is not a model whose entire training state is stored in FP4.

The main comparison fixes effective batch 32. Each update forwards 16,384 token slots and supervises **16,352 shifted targets**. Reported throughput divides supervised targets by synchronized elapsed time; 24 measured updates therefore count 392,448 targets. Timers include forward, backward, gradient clipping, CPU optimizer work and transfers. Initialization, compilation/warmup, synthetic-data preparation, correctness checks, evaluation and checkpoint writes are excluded. Five warmup steps precede every screen. Peak allocated memory covers initialization, warmup and measurement; it is not total process memory or only retained model state.

Receipts require finite loss and gradient-norm metrics, finite sampled decoder parameters and observed updates in sampled decoder matrices. The assembler verifies successful exit/status, token counts, rate denominators, step durations, model/recipe controls, runtime state precision and source/log hashes. Its acceptance does not assert that every tensor element was independently checked or that training converges. All runs reuse seeded synthetic tokens, so final losses are not evidence of language quality.

## Same-model B32 screening

Every row below keeps the baseline architecture, precision recipe and FP32 optimizer state. Each uses 24 measured updates. Gains use the fresh `baseline-b32` receipt; unreplicated changes around one percent remain weak evidence. The microbatch-16 rows are screening results, not confirmed sustained improvements.

| Case / changed setting | Micro x accumulation | Targets/s | Change | Seconds/update | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline-b32` | 8 x 4 | 12,095 | — | 1.352 | 8.605 |
| `micro16-b32` | 16 x 2 | **12,975** | **+7.28%** | 1.260 | 9.468 |
| `torch-micro16-b32`: Torch loss | 16 x 2 | 12,914 | +6.77% | 1.266 | 9.530 |
| `torch-loss-b32`: Torch loss | 8 x 4 | 12,252 | +1.30% | 1.335 | 6.655 |
| `cpu12-b32`: 12 CPU threads | 8 x 4 | 12,098 | +0.03% | 1.352 | 8.605 |
| `cpu8-b32`: 8 CPU threads | 8 x 4 | 12,032 | −0.52% | 1.359 | 8.605 |
| `bucket128-b32`: 128 MiB bucket | 8 x 4 | 12,046 | −0.41% | 1.357 | 8.605 |

Larger microbatches reduce the number of accumulation passes while keeping one optimizer update per 32 examples. They can alter floating-point reduction order and quantization grouping, so this is the same model/objective and update frequency, without a claim of bitwise equality. The CPU-thread and bucket screens do not establish a useful speed improvement.

The completed microbatch-16 Torch-loss combination reaches 12,914 targets/s, close to auto/Liger's 12,975, with 9.530 GiB peak allocation; its microbatch-8 memory advantage does not carry over to a lower measured peak at microbatch 16. A separate **BF16 GEMM control** at the same shape, microbatch 16 / accumulation 2, auto/Liger loss and 24 measured updates reaches **6,857 targets/s**, 2.385 s/update and 10.215 GiB peak allocation. That precision change is separate from the unchanged-precision table and does not establish equal training quality.

The fresh baseline is within 0.1% of the [previous round's 12,107 targets/s confirmation](SPRINT_SPEED_ROUND2_2026-09-11.md). That agreement supports using this round's fresh receipt as the comparison baseline.

## Architecture screening at B32

These candidates change the model, and each has a slightly different parameter count. All use microbatch 8, accumulation 4, auto/Liger loss, the same precision/optimizer settings and **16 measured updates**. Width and FFN width are multiples of 128; heads retain dimension 128.

| Case | Width / layers / FFN | Actual parameters | Targets/s | Versus baseline | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| `shape-aligned6` | 4096 / 6 / 11392 | 996,200,448 | 12,419 | +2.68% | 6.646 |
| `shape-wide4` | 5120 / 4 / 13056 | 996,193,280 | 12,543 | +3.70% | 6.417 |
| `shape-attention4` | 6144 / 4 / 6912 | 994,105,344 | 12,436 | +2.82% | 6.455 |
| `shape-wide2` | 6144 / 2 / 26240 | 997,226,496 | 12,551 | +3.77% | 5.865 |

The aligned six-layer candidate removes 64 FFN channels and about 0.315% of baseline parameters. Its measured memory reduction is larger than that parameter difference; allocator peaks include temporary execution workspaces and should not be interpreted as weights alone. The wider candidates redistribute compute between attention and dense FFNs and reduce depth. These results establish measured throughput only; they do not establish equal representational capacity, convergence or quality. The `shape-ffn2` candidate is unrun; the combined aligned and two-layer microbatch-32 attempts were interrupted.

## What the diagnostic profile says

Six instrumented baseline steps average **1.344 s/update**: **0.832 s** for computation/setup/clipping and **0.512 s** for optimizer/transfers. Bucket CPU work averages **0.500 s**, processing all 999,346,176 parameter elements across 61 buckets. Optimizer/transfers account for approximately **38.1%** of the synchronized diagnostic step.

These measurements identify CPU optimizer work as a substantial remaining cost. The bucket measurement includes staging conversion and bookkeeping and can overlap transfers; it must not be added to optimizer time or treated as isolated arithmetic time. The profiler inserts synchronization that changes overlap. Its run is explicitly excluded from throughput rankings, and these timings are diagnostic rather than a separate speed result.

## Isolated experiments and rejected attempts

**Per-update forward-weight cache — first attempt rejected:** the artifact candidate packs each decoder weight into NVFP4 once per optimizer update and reuses it across accumulation microbatches. It retains the original BF16 parameters and the existing fresh MXFP8 backward calculation. Refresh and validity checks are inside the measured training step; compilation occurs during warmup. Retained packed buffers total **543,326,304 bytes (0.506 GiB)** for the baseline shape, excluding packing workspaces. Cache buffers are not serialized.

The combined candidate validation finished with **9 passed and 2 failed**: six BF16-moment CPU oracle tests and three cache payload/validity tests passed, while both eager and compiled full-model exact-parity tests failed. The first recorded eager loss comparison was 5.5551805496 versus 5.5542011261. Passing the packed-buffer tests therefore did not establish full training equivalence. This cache attempt is rejected for a speed comparison, and there is **no accepted cache throughput**. The [rejected-cache archive](recipes/sprint-speed-round3/experiments/cache-rejected/README.md) preserves the validation and source/test snapshots; its validation summary was transcribed from output and is not a complete pytest log.

The subsequent tiny first-difference diagnostic started from identical BF16 parameters and found that compiled versus eager packing differed in **45 packed-code byte entries in QKV and five in the FFN up projection**, while blocked scales and global scales matched. Eager cache refresh matched eager packing and both tested microbatch losses; compiled refresh matched compiled packing but produced differing losses against that eager reference. This identifies a packing-execution difference that must be addressed before an equivalence claim. It does not validate optimizer updates, prove the complete cause of both failed tests or establish a speed result. The successful diagnostic process and its source hashes are preserved separately from benchmark rows.

**BF16-stored Adam moments:** a separate artifact keeps FP32 CPU master weights and FP32 update arithmetic, but stores the two moments in BF16 for large compiled slices. Smaller slices retain FP32 moments. The current update uses newly computed FP32 moments; their BF16-rounded values are persisted for subsequent updates. This deliberately changes optimizer numerics. State storage is `8 × large-slice parameters + 12 × remaining parameters` bytes, with transfer/scratch buffers separate. Any accepted result must report the actual dtype inventory and explicit candidate metadata. It cannot be presented as equivalent to the baseline FP32-moment optimizer without a separate training-quality study.

The confirmed B32 result above uses BF16 moments for 997,457,920 parameters and FP32 moments for 1,888,256 parameters. Its receipt passed finite-metric, update-sample and actual-state-inventory checks. Six small CPU oracle tests cover the candidate's own rounding/update rule; they mock compilation and do not assert equivalence to FP32-moment AdamW. The [archived implementation](recipes/sprint-speed-round3/experiments/README.md) remains a disposable experiment without production checkpoint/resume support.

Both candidates are isolated benchmark artifacts, with no production-default or checkpoint-contract change implied. A failed compile, parity check, process or receipt remains a failed/incomplete result and receives no accepted throughput value.

**Interrupted execution:** each case below was stopped before a completed timing and retained an unfinished benchmark receipt. The assembler preserves its nonzero process exit and intervention metadata with no accepted rate.

| Interrupted case | Requested measured updates | Seconds before intervention | Recorded observation |
| --- | ---: | ---: | --- |
| `torch-micro32-b32` | 24 | 313.4 | 11,860 of 12,227 MiB device use |
| `bf16-torch-micro32-b32` | 8 | 209.4 | Near-capacity device memory |
| `micro32-b32` | 8 | 147.9 | 2,099,249,152 bytes of shared GPU memory |
| `aligned6-torch-micro32-b32` | 8 | 142.9 | 2,122,317,824 bytes of shared GPU memory |
| `aligned6-auto-micro32-b32` | 16 | 113.0 | 2,457,862,144 shared bytes; 11,797 MiB dedicated use |
| `confirm-micro16-b32` | 64 | 323.1 | 2,063,597,568 shared bytes; short speedup unconfirmed |
| `micro16-b128` | 16 | 366.6 | No completed larger-batch timing |
| `wide2-torch-micro32-b32` | 24 | 137.0 | 2,837,446,656 shared bytes |
| `release-first-micro16-b32-r2` | 64 | 412.3 | 1,933,574,144 shared bytes despite attempted warmup release |
| `release-first-micro16-b128-r2` | 24 | 231.6 | 1,839,202,304 shared bytes; no completed timing |

The elapsed values are intervention timestamps; process completion can occur slightly later. These observations describe memory pressure and stopped work, not measured slowdowns or confirmed out-of-memory exceptions. The three eight-update recipes were shortened before launch; other requested lengths are retained. No rate is inferred from time spent compiling or waiting before a completed timing.

The [startup-memory diagnostic](recipes/sprint-speed-round3/experiments/startup-memory/README.md) completed only warmup **step 0**, then stopped at its reservation guard. That step took 50.1156 seconds including compilation, peaked at **12,541,987,840 allocated bytes**, and ended with **6,402,138,624 allocated**, **13,392,412,672 reserved** and **1,042,685,440 inactive-split bytes**. It recorded zero allocator retries and zero allocator OOMs. No measured-training step ran. This demonstrates startup/warmup pressure in that instrumented run; it neither proves a memory leak nor isolates compiler workspace from first-step execution.

The allocator wrapper attempts one cache release after warmup step zero, before all remaining warmup/measured work; release time belongs to warmup. Its first two runs failed before training because an unindexed CUDA device was passed to the fraction query. Those source versions and errors remain archived, and the explicit-index fix received new retry names/hashes. Both retries were interrupted. The [allocator archive](recipes/sprint-speed-round3/experiments/allocator/README.md) records exact settings, versions and outcomes. No allocator throughput improvement is accepted.

## B32 versus B128

A B128 update supervises **65,408 targets**, four times B32. For a fixed token budget, it performs one quarter as many optimizer updates. Amortizing CPU Adam can increase targets/s, but changes update frequency, gradient averaging/clipping and a step-based learning-rate schedule. B128 is therefore reported separately, with its own fresh B128 baseline. A B128 result must not be described as a same-training-recipe speedup over B32.

The fresh B128 FP32-moment baseline reaches **17,026.5186521 targets/s**, using microbatch 8 and accumulation 16 over 16 measured updates. It processes 1,046,528 targets in 61.4646 seconds. The BF16-moment confirmation retains that batch structure and reaches **17,454.0288795 targets/s over 32 updates**, a **2.51%** gain against this B128 baseline. Its smaller relative gain is consistent with amortizing optimizer work over more targets, without isolating a causal timing breakdown. The microbatch-16 B128 attempt was interrupted; no larger-microbatch improvement is established.

## Reproduction and evidence

The [versioned consolidated evidence](sprint-speed-round3-2026-09-11.json) preserves available original recipe/receipt contents, process exit/log hashes, validation and diagnostic evidence, source snapshots, failures and incomplete runs. Raw local files live under `artifacts/sprint-1b-speed-20260911`. Comparisons use named same-batch baselines; architecture, precision, loss, allocator execution and optimizer-numerics changes remain separate.

The [selected exact recipes](recipes/sprint-speed-round3/README.md) and byte-hash manifests archive both fresh baselines and both confirmed BF16-moment recipes. The CPU-only audit suite passed **17 tests**, and allocator-hook tests passed **seven**; these validate receipt/admission/hook behavior, not GPU speed or convergence. No tolerance was relaxed to accept the failed cache parity tests.

Run an ordinary recipe through the official benchmark in a fresh process, with an unused output directory:

```powershell
.venv/Scripts/python.exe -m yylab.train.benchmark_sprint_precision --config docs/recipes/sprint-speed-round3/baseline-b32.json --output artifacts/reproduce-round3-baseline-b32
```

Restore the archived BF16-moment implementation to its original artifact path, as shown in the recipe index, then execute the confirmed recipe through that wrapper. Passing an experimental flag to the ordinary benchmark does not activate the optimizer or allocator implementation. Acceptance requires matching runtime metadata/source hashes, actual CPU state inventory and the requested release timing/counts where applicable.

To refresh the local audit after completed processes have recorded their exit status and log checksums:

```powershell
.venv/Scripts/python.exe artifacts/sprint-1b-speed-20260911/assemble.py
```

This round measures steady-state training-step speed on one machine. It excludes corpus preparation, checkpoint/evaluation overhead and startup costs from throughput, and does not estimate final language-model quality or claim a global speed optimum.
