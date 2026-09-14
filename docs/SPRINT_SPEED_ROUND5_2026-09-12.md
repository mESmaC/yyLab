# Sprint speed round 5 — 2026-09-12

**Status: final experiment report.**

## Findings

GPU8 optimizer state plus full NVFP4+SR4 produced **1B, batch 32: 21.48K shifted targets/s** over 80 timed updates (+72.3% versus the fresh same-batch CPU FP32 hybrid baseline); **1B, batch 128: 22.60K shifted targets/s** over 24 timed updates (+28.1% versus the fresh same-batch CPU FP32 hybrid baseline); **500M, batch 32: 40.25K shifted targets/s** over 40 timed updates (+67.9% versus the fresh same-batch CPU FP32 hybrid baseline). The batch-32 1B result is the 80-update confirmation; the batch-128 and 500M windows contain 24 and 40 updates, respectively. These are combined configuration gains on the RTX 5070.

In completed matching 4.19M-token pilots, final token-weighted heldout-loss changes versus CPU FP32 are CPU BF16 moments +0.02034; GPU4 +0.00211; GPU8 -0.02897. Negative means lower loss. These short convergence/stability results do not establish long-run or final-task quality equivalence.

With GPU8 held fixed, the matching SR4 precision pilot changes final heldout loss by -0.00892 versus hybrid compute. This compares complete compute recipes: full NVFP4 also changes forward weight scaling. Initial loss is 9.870815 versus hybrid 9.894193 despite matching weight samples, so it is not an isolated backward-precision quality test.

The delayed_cpu quality pilot failed after 16 recorded completed updates and does not supply a final quality comparison. [Failure receipt](../artifacts/sprint-1b-optimizer-20260912/quality-delayed_cpu-256/pilot.json).

1. **128-wide FFN alignment:** the shared throughput planner now aligns FFN widths to 128; its balanced fallback is preserved. The measured 1B configuration is a manual width-4096, 6-layer, FFN-11392 model with **996,200,448 parameters** and an 8K tied vocabulary. FFN-11520 would be **1,002,491,904**, above the requested 1B budget. Automatic 1B planning instead selects width 2816, 11 layers and FFN 10112 (998,505,728 parameters); it is a distinct model, not the benchmarked architecture. The historical unaligned run is not a fresh alignment-only control, and microbatch 16 did not provide a blanket speed gain.
2. **Remove CPU optimizer work:** GPU TorchAO 8-bit/4-bit moments with BF16 stochastic rounding and no FP32 masters were implemented as experiment adapters. GPU8 hybrid microbatch 8 fits and reaches 19.06K shifted targets/s; the optimizer profile records zero CPU optimizer calls. Delayed FP32 CPU variants were also measured, including failures and single/preallocated host-buffer variants. The quality study retains the original delayed implementation and includes the prior BF16 large-moment CPU candidate.
3. **Quantization and weight cache:** profiling identified conversion cost, especially the original stochastic FP4 backward cast, as substantial. SR4 improves that path. The same-mode compiled forward-weight cache failed untouched numerical parity and was closed with no accepted speed. A further per-update weight-amax hoist and additional activation-quantization kernel work have not been deployed.
4. **Smaller changes:** MXFP8 output-head math measured +2.12% at microbatch 8, but changes numerics and remains an artifact experiment. The first reduce-overhead run failed a graph-lifetime check; the explicit update-boundary wrapper produced valid follow-up measurements, with every completed slow result retained. Folding the clipping coefficient into CPU Adam remains unimplemented.
5. **Full NVFP4 backward recipe:** the repository now exposes `nvfp4_all_portable_sr4`, using native FP4 GEMMs, 2D weight scaling, Hadamard transforms and portable stochastic rounding. It consumes all four Philox outputs with 24-bit uniforms and changes the deterministic RNG layout. Holding GPU8 and microbatch 8 fixed gives **+12.69%** over the hybrid recipe; this avoids the earlier microbatch-4 comparison. GPU4's uncapped short screen did not sustain its rate; the separate cap80 allocator run completed 80 updates at 20.17K, with its own memory observations.

The planner change is used by normal Sprint sizing. SR4 is available through the repository's explicit experimental precision API. The GPU/delayed optimizers, graph-boundary and head wrappers, and quality runner remain artifact tooling; they were not added to the ordinary Sprint UI or production resume path. Normal Sprint BF16 training still requires CPU FP32 masters/moments. These experiments do not produce a production optimizer-resume checkpoint; an optional pilot weight export is a separate artifact.

The measurements separate GPU matrix arithmetic, conversion kernels, CPU optimizer work and memory residency. Every accepted speed row is a completed real forward/backward/clip/update benchmark; synthetic losses are validation checks, not language-quality evidence.

Tokens/sec below means supervised shifted targets per second, matching prior rounds. Input token-slot rates are secondary fields in the JSON; with sequence length 512, shifted targets equal 511/512 of input slots. Setup and warmup/compilation are excluded from accepted benchmark timers. Profilers, aborted runs and small validation gates never supply accepted rates.

## Completed speed measurements

| Experiment | Actual parameters | Precision / optimizer | Micro × accumulation | Steps | Shifted targets/s | Peak allocated / reserved GiB |
|---|---:|---|---:|---:|---:|---:|
| [bench-full-fp4](../artifacts/sprint-500m-fp4-profile-20260912/bench-full-fp4/benchmark.json) | 499,159,040 | nvfp4_all_portable / CPU FP32 masters/moments | 16 × 2 | 40 | 22,438 | 5.45 / 5.81 |
| [bench-hybrid](../artifacts/sprint-500m-fp4-profile-20260912/bench-hybrid/benchmark.json) | 499,159,040 | nvfp4_mxfp8_fast / CPU FP32 masters/moments | 16 × 2 | 40 | 23,981 | 6.58 / 7.57 |
| [bench-sr4](../artifacts/sprint-500m-fp4-profile-20260912/bench-sr4/benchmark.json) | 499,159,040 | nvfp4_all_portable + SR4 / CPU FP32 masters/moments | 16 × 2 | 40 | 25,548 | 6.74 / 7.20 |
| [confirm-sr4](../artifacts/sprint-500m-fp4-profile-20260912/confirm-sr4/benchmark.json) | 499,159,040 | nvfp4_all_portable + SR4 / CPU FP32 masters/moments | 16 × 2 | 80 | 25,772 | 6.74 / 7.13 |
| [aligned-micro16-b128-reduce-marked](../artifacts/sprint-1b-optimizer-20260912/aligned-micro16-b128-reduce-marked/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast + graph update boundary / CPU FP32 masters/moments | 16 × 8 | 24 | 14,150 | 9.44 / 11.31 |
| [aligned-micro16-b32](../artifacts/sprint-1b-optimizer-20260912/aligned-micro16-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / CPU FP32 masters/moments | 16 × 2 | 32 | 10,751 | 9.45 / 10.76 |
| [aligned-micro16-b32-reduce-marked](../artifacts/sprint-1b-optimizer-20260912/aligned-micro16-b32-reduce-marked/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast + graph update boundary / CPU FP32 masters/moments | 16 × 2 | 32 | 13,157 | 9.44 / 11.31 |
| [aligned-micro8-b128](../artifacts/sprint-1b-optimizer-20260912/aligned-micro8-b128/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / CPU FP32 masters/moments | 8 × 16 | 24 | 17,640 | 8.59 / 9.69 |
| [aligned-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/aligned-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / CPU FP32 masters/moments | 8 × 4 | 32 | 12,467 | 8.59 / 9.69 |
| [aligned-micro8-b32-reduce-boundary](../artifacts/sprint-1b-optimizer-20260912/aligned-micro8-b32-reduce-boundary/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast + graph update boundary / CPU FP32 masters/moments | 8 × 4 | 32 | 12,543 | 6.64 / 8.03 |
| [confirm80-gpu4-hybrid-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu4-hybrid-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / GPU AdamW4bit + BF16 SR | 8 × 4 | 80 | 17,981 | 8.59 / 10.70 |
| [confirm80-gpu4-sr4-cap80](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu4-sr4-cap80/benchmark.json) | 996,200,448 | nvfp4_all_portable_sr4 / GPU AdamW4bit + BF16 SR | 8 × 4 | 80 | 20,165 | 9.05 / 9.27 |
| [confirm80-gpu4-sr4-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu4-sr4-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_all_portable_sr4 / GPU AdamW4bit + BF16 SR | 8 × 4 | 80 | 9,690 | 9.05 / 10.69 |
| [confirm80-gpu8-sr4-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu8-sr4-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_all_portable_sr4 / GPU AdamW8bit + BF16 SR | 8 × 4 | 80 | 21,480 | 9.04 / 9.44 |
| [delayed-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/delayed-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / delayed CPU FP32 | 8 × 4 | 32 | 15,045 | 8.59 / 9.69 |
| [delayed-preallocated-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/delayed-preallocated-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / delayed CPU FP32 / preallocated host buffer | 8 × 4 | 32 | 13,068 | 8.59 / 9.69 |
| [delayed-single-micro8-b128](../artifacts/sprint-1b-optimizer-20260912/delayed-single-micro8-b128/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / delayed CPU FP32 / one host buffer | 8 × 16 | 24 | 17,805 | 8.59 / 9.69 |
| [gpu4-micro8-b128](../artifacts/sprint-1b-optimizer-20260912/gpu4-micro8-b128/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / GPU AdamW4bit + BF16 SR | 8 × 16 | 24 | 19,608 | 8.59 / 10.70 |
| [gpu4-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/gpu4-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / GPU AdamW4bit + BF16 SR | 8 × 4 | 32 | 17,932 | 8.59 / 10.70 |
| [gpu4-sr4-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/gpu4-sr4-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_all_portable_sr4 / GPU AdamW4bit + BF16 SR | 8 × 4 | 32 | 19,987 | 9.04 / 10.42 |
| [gpu8-hybrid-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/gpu8-hybrid-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / GPU AdamW8bit + BF16 SR | 8 × 4 | 32 | 19,061 | 8.59 / 9.91 |
| [gpu8-micro4-b128](../artifacts/sprint-1b-optimizer-20260912/gpu8-micro4-b128/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / GPU AdamW8bit + BF16 SR | 4 × 32 | 24 | 17,214 | 7.81 / 9.29 |
| [gpu8-micro4-b32](../artifacts/sprint-1b-optimizer-20260912/gpu8-micro4-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast / GPU AdamW8bit + BF16 SR | 4 × 8 | 32 | 16,513 | 7.74 / 8.85 |
| [gpu8-sr4-500m-micro16-b32](../artifacts/sprint-1b-optimizer-20260912/gpu8-sr4-500m-micro16-b32/benchmark.json) | 499,159,040 | nvfp4_all_portable_sr4 / GPU AdamW8bit + BF16 SR | 16 × 2 | 40 | 40,254 | 6.74 / 7.22 |
| [gpu8-sr4-micro8-b128](../artifacts/sprint-1b-optimizer-20260912/gpu8-sr4-micro8-b128/benchmark.json) | 996,200,448 | nvfp4_all_portable_sr4 / GPU AdamW8bit + BF16 SR | 8 × 16 | 24 | 22,595 | 9.04 / 9.44 |
| [gpu8-sr4-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/gpu8-sr4-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_all_portable_sr4 / GPU AdamW8bit + BF16 SR | 8 × 4 | 32 | 21,508 | 9.04 / 9.44 |
| [head-mxfp8-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/head-mxfp8-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_mxfp8_fast + MXFP8 head / CPU FP32 masters/moments | 8 × 4 | 32 | 12,731 | 8.64 / 9.70 |
| [sr4-micro8-b32](../artifacts/sprint-1b-optimizer-20260912/sr4-micro8-b32/benchmark.json) | 996,200,448 | nvfp4_all_portable_sr4 / CPU FP32 masters/moments | 8 × 4 | 32 | 13,428 | 7.00 / 7.63 |

500M rows use 499,159,040 parameters; the aligned 1B shape uses 996,200,448 parameters. Comparisons across different microbatches, effective batches, optimizer numerics, output-head precision or allocator policies are configuration comparisons, not an isolated precision speedup. Full controls and numerical recipes are retained per row in the JSON.

SR4 consumes all four outputs of ten-round Philox with 24-bit uniforms. It retains adjacent-level stochastic rounding probabilities but changes the deterministic RNG layout; it is not bitwise-equivalent to the earlier portable recipe. GPU low-bit optimizers also change moment storage and remove FP32 master weights. These numerical changes need the separate matched quality study.

### Longer confirmations

Short screening rates remain provisional until their longer repetitions finish. All completed measurements are retained, including slow results; observed memory spill is context, not a reason to remove a completed regression.

With GPU8 and microbatch 8 held fixed, hybrid compute measured 19,061 shifted targets/s and full FP4+SR4 measured 21,480 (**+12.69%**). Their windows contain 32 and 80 updates, respectively; the model, source hashes, seed, batch and schedule match. [Hybrid receipt](../artifacts/sprint-1b-optimizer-20260912/gpu8-hybrid-micro8-b32/benchmark.json) · [FP4 confirmation](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu8-sr4-micro8-b32/benchmark.json). The earlier GPU8 hybrid microbatch-4 measurement is a different batch configuration.

- `gpu4-hybrid-micro8-b32`: 17,932 shifted targets/s over 32 updates, then 17,981 over 80 updates (+0.3%). [Confirmation receipt](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu4-hybrid-micro8-b32/benchmark.json).
- `gpu4-sr4-micro8-b32`: 19,987 shifted targets/s over 32 updates, then 9,690 over 80 updates (-51.5%). **The short rate does not support a sustained speed recommendation.** [Confirmation receipt](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu4-sr4-micro8-b32/benchmark.json).
- `gpu8-sr4-micro8-b32`: 21,508 shifted targets/s over 32 updates, then 21,480 over 80 updates (-0.1%). [Confirmation receipt](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu8-sr4-micro8-b32/benchmark.json).

The GPU4+SR4 regression used the same runtime/source hashes and settings, with a longer measured window. Its allocator reservation rose from 11.186 GB to 11.474 GB. Memory spill is a possible explanation, but no residency counter was captured for that completed run, so the cause remains unconfirmed.

A separate `cap80` allocator configuration completed 80 updates at 20,165 shifted targets/s with 9.957 GB peak reservation. The attested process fraction is 80%. Boundary snapshots show 85,983,232 shared bytes before measurement and 85,983,232 shared bytes after measurement. This changes allocation policy; it does not retrospectively prove the cause of another run's slowdown. [Allocator receipt](../artifacts/sprint-1b-optimizer-20260912/confirm80-gpu4-sr4-cap80/benchmark.json).

## Retained optimizer memory

| Actual parameters | Optimizer | CPU state GiB | GPU state GiB | Additional optimizer buffers GiB |
|---:|---|---:|---:|---:|
| 499,159,040 | CPU FP32 masters/moments | 5.5785 | 0.0000 | 0.1250 |
| 996,200,448 | CPU FP32 masters/moments | 11.1334 | 0.0000 | 0.1250 |
| 996,200,448 | GPU AdamW4bit + BF16 SR | 0.0000 | 0.9858 | 0.0000 |
| 996,200,448 | GPU AdamW8bit + BF16 SR | 0.0000 | 1.8846 | 0.0000 |
| 996,200,448 | delayed CPU FP32 | 11.1334 | 0.0000 | 3.8361 |
| 996,200,448 | delayed CPU FP32 / preallocated host buffer | 11.1334 | 0.0000 | 1.9806 |
| 996,200,448 | delayed CPU FP32 / one host buffer | 11.1334 | 0.0000 | 1.9806 |
| 499,159,040 | GPU AdamW8bit + BF16 SR | 0.0000 | 0.9443 | 0.0000 |

Physical low-bit state includes packed codes, scales and lookup maps; logical wrapper dimensions are not used as storage estimates. GPU optimizers retain tiny CPU step/LR control scalars but no FP32 masters or CPU moments. Delayed CPU Adam retains FP32 masters/moments and adds full pinned gradient/writeback buffers. These retained allocations differ from temporary workspaces and CUDA allocator peaks.

## Diagnostic bottlenecks

- `nvfp4_all_portable`: synchronized step 693.662 ms, compute/setup/clip 437.427 ms, optimizer/transfers 256.127 ms. [Diagnostic receipt](../artifacts/sprint-500m-fp4-profile-20260912/full-fp4-diagnostic/profiling.json).
  Backward fused RHT/quantization costs 132.681 ms per profiled step; backward FP4 GEMMs cost 68.733 ms. Kernel attribution uses native CUDA events correlated to CPU autograd scopes, without adding wrapper and kernel durations.
- `nvfp4_mxfp8_fast`: synchronized step 678.649 ms, compute/setup/clip 415.099 ms, optimizer/transfers 263.443 ms. [Diagnostic receipt](../artifacts/sprint-500m-fp4-profile-20260912/hybrid-diagnostic/profiling.json).
- 1B `gpu8bit`: average owner-stream elapsed intervals over 8 synchronized diagnostic updates: forward 343.011 ms, backward 570.783 ms, clip 11.494 ms, optimizer 56.437 ms. Actual CPU optimizer calls: 0; CPU bucket calls: 0. [Diagnostic receipt](../artifacts/sprint-1b-optimizer-20260912/profile-gpu8-micro4-b32/profiling.json).

Diagnostic synchronizations disturb overlap. CUDA-event stage intervals can include launch gaps and are not sums of busy kernel time. CPU optimizer work overlaps copies; kernel activity sums and CPU timings must not be added as wall time. Delayed-update diagnostics flush each measured update and destroy their usual cross-step overlap. CUDA allocator peaks are not WDDM residency measurements.

## Memory residency and excluded measurements

- `aligned-micro16-b128`: aborted; no accepted throughput. Sustained WDDM spill: process shared GPU memory measured at 2264924160 bytes; no completed throughput accepted Observed shared GPU memory: 2,264,924,160 bytes (2.11 GiB). [Abort receipt](../artifacts/sprint-1b-optimizer-20260912/aligned-micro16-b128-aborted.json). [Residency receipt](../artifacts/sprint-1b-optimizer-20260912/aligned-micro16-b128-memory.json).
- `aligned-micro16-b32-release`: aborted; no accepted throughput.  Observed shared GPU memory: 1,788,870,656 bytes (1.67 GiB). [Abort receipt](../artifacts/sprint-1b-optimizer-20260912/aligned-micro16-b32-release-aborted.json). [Residency receipt](../artifacts/sprint-1b-optimizer-20260912/aligned-micro16-b32-release-memory.json).
- `sr4-micro16-b32`: aborted; no accepted throughput. Sustained shared GPU memory spill while testing micro16; no throughput accepted [Abort receipt](../artifacts/sprint-1b-optimizer-20260912/sr4-micro16-b32-aborted.json).
- `aligned-micro8-b32-reduce`: failed (reported stage: warmup); no accepted throughput. RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "C:\Users\engin\Desktop\yy\yyLab\yylab\models\sprint.py", line 339, in forward. [Failure receipt](../artifacts/sprint-1b-optimizer-20260912/aligned-micro8-b32-reduce/benchmark.json).
- `delayed-direct-micro8-b32`: failed (reported stage: measured_training); no accepted throughput. AcceleratorError: CUDA error: out of memory. [Failure receipt](../artifacts/sprint-1b-optimizer-20260912/delayed-direct-micro8-b32/benchmark.json).
- `delayed-micro8-b128`: failed (reported stage: measured_training); no accepted throughput. AcceleratorError: CUDA error: out of memory. [Failure receipt](../artifacts/sprint-1b-optimizer-20260912/delayed-micro8-b128/benchmark.json).
- `delayed-single-micro8-b32`: failed (reported stage: measured_training); no accepted throughput. AcceleratorError: CUDA error: out of memory. [Failure receipt](../artifacts/sprint-1b-optimizer-20260912/delayed-single-micro8-b32/benchmark.json).

The delayed-optimizer failure metadata can reflect the last synchronization snapshot, rather than state at the failing allocation. It does not establish the exact failing update or allocation; CUDA errors may be reported asynchronously.

External read-only snapshots preserve live process arguments, current config-file hashes, host memory and WDDM counters. They are isolated observations, not a full-run residency history:

- [2026-09-12T20:21:40.700931+00:00](../artifacts/sprint-1b-optimizer-20260912/environment-evidence/20260912T202140Z.json) — counters ok; 6 matching Python processes, including launchers/observers where present.

The compiled forward-weight cache failed repeated untouched numerical-parity checks and the bounded investigation was closed. No cache throughput is accepted. Instrumented traces changed the graph and therefore did not establish the cause. [Cache investigation conclusion](../artifacts/sprint-1b-optimizer-20260912/weight-cache-final-conclusion.md).

## Matched real-data quality pilot

Pilot arms are configured to use the same packed real tokens, initialization seed, tokenizer, heldout rows, total learning-rate schedule and effective batch. Initial weight samples must match both before and after precision/optimizer conversion. The checksum samples at most 128 values per parameter tensor; it is not a full-weight hash. Frozen source files are rehashed against their receipts, and no in-run source changes are accepted. Existing validation splits cover six sources; Cursor and Reasoning have no supplied heldout split. Optimizer candidates are compared only when precision, configuration, source files, input and heldout hashes, initialization samples, completed updates and token budgets match the CPU FP32 baseline.

| Candidate / precision | Status | Updates | Input tokens | Latest heldout loss | Delta vs matched CPU FP32 |
|---|---|---:|---:|---:|---:|
| [cpu_bf16_moments](../artifacts/sprint-1b-optimizer-20260912/quality-cpu_bf16_moments-256/pilot.json) / nvfp4_mxfp8_fast | complete | 256 | 4,194,304 | 4.017634 (step 256) | +0.020341 |
| [cpu_fp32](../artifacts/sprint-1b-optimizer-20260912/quality-cpu_fp32-256/pilot.json) / nvfp4_mxfp8_fast | complete | 256 | 4,194,304 | 3.997292 (step 256) | — |
| [delayed_cpu](../artifacts/sprint-1b-optimizer-20260912/quality-delayed_cpu-256/pilot.json) / nvfp4_mxfp8_fast | failed | 16 | 262,144 | 9.894193 (step 0) | — |
| [gpu4bit](../artifacts/sprint-1b-optimizer-20260912/quality-gpu4bit-256/pilot.json) / nvfp4_mxfp8_fast | complete | 256 | 4,194,304 | 3.999399 (step 256) | +0.002107 |
| [gpu8bit](../artifacts/sprint-1b-optimizer-20260912/quality-gpu8bit-256/pilot.json) / nvfp4_mxfp8_fast | complete | 256 | 4,194,304 | 3.968323 (step 256) | -0.028969 |
| [gpu8bit](../artifacts/sprint-1b-optimizer-20260912/quality-gpu8bit-sr4-256/pilot.json) / nvfp4_all_portable_sr4 | complete | 256 | 4,194,304 | 3.959406 (step 256) | — |

`delayed_cpu` failed with AcceleratorError: CUDA error: out of memory. [Full traceback](../artifacts/sprint-1b-optimizer-20260912/quality-delayed_cpu-256.log). No final or warmup-only quality comparison is accepted.
The traceback reports the error at pinned CPU gradient-snapshot allocation in `_allocate_snapshots`, after 16 synchronous updates were recorded. CUDA's asynchronous error reporting prevents this stack location from establishing the underlying allocation cause. No evaluation of stale-update training completed.

The separate precision ablation holds the optimizer fixed and compares complete compute recipes, including full NVFP4 forward 2D weight scaling and backward quantization. It is not an isolated backward or optimizer-only test. Its configuration/preparation receipt hashes intentionally differ, while the configuration with precision removed, source code, training/heldout tokens, initialization samples, schedule and completed budgets must match. The changed forward recipe can change initial loss despite identical weight samples. SR4 changes model-rounding RNG consumption, so matching seeds do not imply the same random trajectory.

| Fixed optimizer | Updates | Hybrid heldout loss | SR4 heldout loss | SR4 loss delta |
|---|---:|---:|---:|---:|
| [gpu8bit](../artifacts/sprint-1b-optimizer-20260912/quality-gpu8bit-sr4-256/pilot.json) | 256 | 3.968323 | 3.959406 | -0.008917 |

Final per-source complete-recipe comparison at update 256 (4,194,304 training input tokens), with GPU8 held fixed:

| Heldout source | Targets | Hybrid loss | Full NVFP4 + SR4 loss | Loss delta |
|---|---:|---:|---:|---:|
| complete-fable | 16,352 | 5.229650 | 5.223386 | -0.006265 |
| claude-mythos | 16,352 | 0.610307 | 0.611307 | +0.001001 |
| fable-max | 16,352 | 5.595341 | 5.581115 | -0.014225 |
| fable-premium | 16,352 | 4.758595 | 4.755126 | -0.003469 |
| qwen-distillation | 16,352 | 4.536402 | 4.524009 | -0.012393 |
| kimi-distillation | 16,352 | 3.079645 | 3.061495 | -0.018150 |
| **Token-weighted aggregate** | **98,112** | **3.968323** | **3.959406** | **-0.008917** |

![Preliminary 4.19M-token optimizer and precision loss comparisons](../artifacts/sprint-1b-optimizer-20260912/quality-loss-4m.png)

[Standalone loss plot](../artifacts/sprint-1b-optimizer-20260912/quality-loss-4m.png) · [Exact plotted data and receipt hashes](../artifacts/sprint-1b-optimizer-20260912/quality-loss-4m.json).

A gate is a short stability check. Even the full 256-update pilot is preliminary convergence evidence and cannot establish final agent quality or long-run equivalence. Evaluation and final snapshots flush pending delayed updates; that boundary resets the lag. Pilot wall times include setup/evaluation and must not be compared as throughput.

## Provenance and limits

The older 12,095 targets/s result in [round 3](SPRINT_SPEED_ROUND3_2026-09-11.md) used a historical unaligned shape; it is not a fresh measurement from this run. Today's aligned measurements are listed above with exact parameter counts.

The assembler validates token denominators, parameter/storage formulas, finite/update evidence, requested precision, runtime controls, GPU optimizer state payloads and delayed final-flush counters. It stores source receipt hashes. Quality runs additionally retain frozen source copies and exact packed-token hashes.

Reproduce this report with `.venv/Scripts/python.exe artifacts/sprint-1b-optimizer-20260912/assemble_round5.py`. Finalization requires `--final` after all experiments are explicitly finished. Generated 2026-09-12T21:20:48.437642+00:00.
