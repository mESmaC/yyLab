# Sprint 1B step profile and GEMM microbenchmarks (round 4 diagnostics)

Two disposable scripts that measure where a Sprint optimizer update spends its time on this RTX 5070, before any round-4 optimization is attempted. Neither one produces a throughput result: the profiler synchronizes after every update and CUPTI instrumentation changes overlap, and the microbenchmarks run kernels in isolation. Their numbers belong in a diagnostic section, never in a speed table. Neither script modifies `yylab/`, checkpoints or the resume contract; both only read a recipe and write to `--output`.

## Why

The round-3 receipts fix the budget of the confirmed B32 recipe at 1.241 s per update: 0.832 s of GPU work for four microbatches (the diagnostic profile's compute number) and about 0.41 s of serialized CPU Adam and transfers. The GPU part is 50.8 µs per token, or about 120 TFLOP/s against the model's 6.07 GFLOP per token. At the whitepaper dense rates (FP4 493.9, FP8 246.9 or 123.5 with FP32 accumulation, BF16 61.7 TFLOP/s) the GEMM-only floor is either ~24 or ~40 µs per token depending on which FP8 rate the SM120 block-scaled kernels actually reach. That single unknown decides whether the remaining non-GEMM work is ~45% or ~8% of the compute step, so it is measured first.

## Run

From the repository root, in a fresh process, with nothing else using the GPU or the CPU:

```powershell
.venv/Scripts/python.exe docs/recipes/sprint-speed-round4/experiments/profile/profile_sprint_step.py --config docs/recipes/sprint-speed-round3/baseline-b32.json --output artifacts/profile-1b-b32
.venv/Scripts/python.exe docs/recipes/sprint-speed-round4/experiments/profile/bench_sprint_gemms.py --config docs/recipes/sprint-speed-round3/baseline-b32.json --tokens 2048,4096,8192 --output artifacts/gemm-bench-1b
```

The profiler builds the recipe's case exactly as `benchmark_sprint_precision._measure_precision_case` does (seeded model, `apply_sprint_precision`, `SprintStepEngine` with the compiled CPU optimizer), runs three untimed warmup updates (compilation happens here; about a minute at 1B), then records one update with `torch.profiler`. Expect roughly two minutes and a 50–150 MB gzipped trace. `--precision`, `--batch` and `--compile-mode eager` override the recipe's first case; `--profile-steps 2` records two updates; `--analyze-only artifacts/profile-1b-b32/trace.json.gz` re-runs only the analysis. Setting `$env:TORCH_LOGS="graph_breaks,recompiles"` before launching captures Dynamo's own log lines in addition to the counters the script records.

The GEMM benchmark takes the decoder shapes (packed QKV, attention output, MLP up, MLP down) and the tied head from the recipe and times, per projection and token count: BF16 cuBLAS forward/dgrad/wgrad; plain tensorwise FP8 `_scaled_mm`; MXFP8 forward as `mxfp8_mm_nt_fast` runs it and as a bare GEMM on pre-cast operands; the complete `mxfp8_linear_backward_fast` (pad + four casts + two GEMMs) and its two bare GEMMs; each Triton cast alone; the `F.pad` copies when a dimension is not a multiple of 128; NVFP4 forward exactly as `_FastLowPrecisionFunction.forward` executes it (two amax reductions, two torch-op quantizations, the GEMM) and as a bare GEMM; the torch-op NVFP4 quantization of X and W alone; and, if the `mslk` package is installed, TorchAO's Triton NVFP4 quantizer. Fifteen timed calls after five warmup calls, with three rotating operand copies so the 48 MB L2 does not flatter the smaller operands. A few minutes per token count.

The profiler uses the ordinary FP32-moment `SprintCPUOffloadAdamW`. The BF16-moment candidate only changes CPU work; the GPU-side attribution is the same.

## Outputs

`artifacts/profile-1b-b32/`: `receipt.json` (recipe hash, source hashes, runtime versions, hardware, case, warmup/compile seconds, CUDA memory after warmup, Dynamo counters), `dynamo_counters.json` and `dynamo.log` (graph breaks and recompiles during warmup), `trace.json.gz` (Chrome trace; open in Perfetto or `chrome://tracing`), `profile_summary.json` and `profile_summary.md`.

The summary attributes every device event to a phase by walking the CPU launch stack: `sprint.forward[mb=k]`, `sprint.clip_grad_norm` and `sprint.optimizer_step` markers, anything launched under the autograd engine as backward, the rest as other. It reports, per profiled update: wall, device busy and idle time; the tail between the last backward kernel and the next forward (the optimizer critical path); kernel time and device-timeline span per phase and per microbatch; kernel time by family (FP4/FP8/BF16 GEMM from the launching ATen op's dtypes, torchao MXFP8 casts, NVFP4 quantization kernels, inductor fused kernels, attention, Liger, eager ATen elementwise and reductions, copies, memcpy/memset); forward/backward time launched inside versus outside compiled regions; achieved TFLOP/s per GEMM precision from recorded shapes; the forty most expensive kernels with their launch sites; and the largest device idle gaps with the kernels around them.

`artifacts/gemm-bench-1b/`: `gemm_bench.json` and `gemm_bench.md` with one row per (token count, projection, role, precision, path) and a per-token-count roll-up: the decoder GEMM+cast floor per microbatch as Sprint runs it, the bare-GEMM floor, their difference (cast and padding overhead), the BF16 equivalent and the tied head's three BF16 GEMMs.

## What to read off, decided before running

1. MXFP8 rate. If the bare MXFP8 dgrad/wgrad rows exceed roughly 150 TFLOP/s at 4096 tokens, SM120 block-scaled GEMMs are not subject to the FP32-accumulate halving and the GEMM floor is ~24 µs per token; if they sit near 100 TFLOP/s it is ~40 µs and most of the remaining time is GEMM. The tensorwise FP8 row separates kernel selection from the accumulate question.
2. Cast and padding overhead. `decoder_cast_and_padding_overhead_ms` divided by `decoder_sprint_cast_plus_gemm_ms`. Above ~15%, a cast round (Triton NVFP4 quantization, per-update weight amax and weight cache, 128-aligned FFN width) is worth its own experiments; below ~5% it is not.
3. Compiled or eager. `compiled_region_ms` for forward/backward and the graph-break counters. If more than ~10% of forward/backward kernel time is launched outside compiled regions, or the counters show a break inside `_FastLowPrecisionFunction`, graph breaks are fixed before any kernel work, because a fused inductor graph changes every other number.
4. Non-GEMM share in situ. In the profile, forward+backward kernel time minus the GEMM families, divided by forward+backward kernel time. This is the ceiling for what fusion and cast work can recover; compare it with the roll-up from the microbenchmark, which excludes attention, norms, GELU, RoPE, Liger and gradient accumulation.
5. Serialization. `tail_after_last_backward_ms` should be close to the optimizer marker's wall; the device busy fraction over the update should be close to 0.832 / 1.24. Both confirm that the CPU optimizer is fully exposed, which is the premise of the delayed-update and GPU-resident optimizer candidates.
6. Tokens per microbatch. The 2048 / 4096 / 8192 rows show how much GEMM efficiency microbatch 4 gives up and microbatch 16 gains, independent of memory.
7. Idle gaps. Gaps of a millisecond or more inside forward or backward point at host synchronization (a `.item()`, a target-count sync, allocator work) or launch-bound stretches of many tiny kernels; the mean kernel duration and event count per update are the second indicator.

Nothing here establishes training quality, and none of these timings is a candidate for the speed tables.
