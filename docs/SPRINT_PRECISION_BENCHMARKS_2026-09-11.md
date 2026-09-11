# Sprint precision training benchmarks — 2026-09-11

The [follow-up speed round](SPRINT_SPEED_ROUND2_2026-09-11.md) adds 10M measurements, FFN-budget and shape experiments, and longer confirmations through 1B. This report preserves the original precision comparisons.

This experiment compares complete Sprint training updates with BF16, native
NVFP4, native MXFP8, and a BitNet-style W1.58A8 training recipe. It measures
throughput on synthetic tokens; it does not establish convergence, language
quality, or time to reach a useful model. These adapters are separate from the
normal Sprint training recipes.

The [full benchmark receipt](sprint-precision-2026-09-11.json) is the reference
for results, complete recipes, exact execution settings, failures, and source hashes. The
[benchmark CLI](../yylab/train/benchmark_sprint_precision.py) runs the real
`SprintStepEngine`, including the CPU optimizer and transfers.

## Results

The confirmed choices use NVFP4 forward projection math and MXFP8 backward
projection math. Fast column casts help at 50M and 500M; the original cast path
is the reliable choice in these 1B measurements.

### Longer confirmation runs

| Target | BF16 targets/sec | Selected hybrid targets/sec | Throughput gain |
|---|---:|---:|---:|
| 50M | 121,341 | **162,433** | **33.9%** |
| 500M | 13,868 | **22,970** | **65.6%** |
| 1B | 6,956 | **11,582** | **66.5%** |

The selected mode is `nvfp4_mxfp8_fast` at 50M and 500M, and `nvfp4_mxfp8`
at 1B. Both sides use identical microbatch/accumulation settings: **32 × 1**,
**16 × 2**, and **8 × 4** respectively. Each confirmation uses a fresh process,
five warmup updates, and respectively **200, 40, or 24 measured updates**.
The receipt verifies matching shapes, seeds, effective batches, optimizer
schedules, and execution settings before calculating these ratios.

Selected peak GPU allocations were 2.15, 7.70, and 6.89 GiB, versus 2.32, 7.29,
and 8.13 GiB for the corresponding BF16 confirmations. The faster 500M path
therefore uses slightly more peak memory than BF16 at the same batch size.

These are single longer confirmations for the final choices, rather than
statistical confidence intervals. Some larger-batch and fast-cast 1B probes did
not hold their initial throughput; those results remain in the receipt and are
discussed below. The final 1B original-hybrid result closely matches its initial
11,602-target/sec measurement at the same batch size.

### Matched microbatch comparison across all core modes

The first matched-microbatch comparison completed all fifteen cases. Values are
supervised targets per second, rounded to the nearest whole target.

| Target | BF16 | NVFP4 forward / BF16 backward | NVFP4 forward / MXFP8 backward | MXFP8 throughout projections | BitNet-style BF16 QAT |
|---|---:|---:|---:|---:|---:|
| 50M | 121,274 | 133,410 | **157,893** | 156,757 | 116,160 |
| 500M | 13,579 | 15,624 | **21,338** | 21,040 | 13,098 |
| 1B | 6,966 | 8,114 | **11,602** | 11,381 | 6,492 |

NVFP4 forward with MXFP8 backward improved complete-step throughput by
**30.2%, 57.1%, and 66.6%** respectively. BitNet-style BF16 QAT was 4.2%, 3.5%,
and 6.8% slower than the corresponding BF16 baseline. These are measured
implementation comparisons, not statements about the formats' maximum possible
performance or model quality.

| Target | BF16 peak GPU allocation | NVFP4/MXFP8 peak GPU allocation | BitNet-style BF16 QAT peak GPU allocation |
|---|---:|---:|---:|
| 50M | 2.32 GiB | 2.06 GiB | 2.67 GiB |
| 500M | 4.65 GiB | 3.97 GiB | 5.70 GiB |
| 1B | 8.13 GiB | 6.89 GiB | 10.20 GiB |

## What each precision label means

The converted operations are the decoder's packed QKV, attention output, MLP
up, and MLP down projections. Embeddings, the tied output head, attention, and
the loss implementation retain their usual BF16/FP32 execution.

| CLI mode | Decoder forward matrix multiplication | Decoder input-gradient / weight-gradient matrix multiplications |
|---|---|---|
| `bf16` | Dense BF16 baseline | Dense BF16 / BF16 |
| `nvfp4_bf16` | Native NVFP4 | Dense BF16 / BF16 |
| `nvfp4_mxfp8` | Native NVFP4 | Native MXFP8 / MXFP8 |
| `mxfp8` | Native MXFP8 | Native MXFP8 / MXFP8 |
| `bitnet_b158` | W1.58A8 quantized values represented in BF16, using dense BF16 multiplication | Dense BF16 / BF16 with straight-through estimators |
| `mxfp8_fast` | Native MXFP8 | Native MXFP8 / MXFP8, with fused column conversion |
| `nvfp4_mxfp8_fast` | Native NVFP4 | Native MXFP8 / MXFP8, with fused column conversion |
| `bitnet_b158_int8` | Native INT8, INT32 accumulation, FP32 rescaling | Dense BF16 / BF16 with straight-through estimators |
| `nvfp4_all_portable` | Native NVFP4 with TorchAO's training transforms | Native NVFP4 / NVFP4 with portable stochastic rounding |

For the two mixed NVFP4 modes, backward uses the saved BF16 shadow tensors,
converted to the requested backward format when needed. It does not reconstruct
backward inputs from the packed FP4 forward values. The MXFP8 mode independently
quantizes the operands of each of its three matrix multiplications. The receipt
records the provider and actual forward/backward formats. Unsupported native
execution fails explicitly; it is never silently counted as a BF16 result.

TorchAO distinguishes quantized matrix multiplication from emulated execution
and documents its MXFP8/NVFP4 workflows as prototypes. Its published throughput
numbers use other hardware and workloads, so they are not Sprint speed claims.
See the official [quantized training documentation](https://docs.pytorch.org/ao/stable/workflows/training.html#mxfp8).

### BitNet scope

The local adapter follows the BitNet b1.58 training pattern: per-matrix absmean
ternary weights, per-token absmax INT8 activations, and straight-through
gradients. Scale reductions use FP32. Every converted projection has an internal
RMSNorm. The original block norms move inside the QKV/up projections; new norms
inside the output/down projections add parameters. Sprint's dimensions, packed
QKV, GELU, attention, and embedding/head tie stay intact. The packed QKV matrix
shares one ternary scale.

This is a Sprint-shaped BitNet-style QAT comparison, rather than a reproduction
of a published BitNet model or a packed ternary inference kernel. Microsoft's
[training tips, code and FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)
provides the training pattern and explicitly describes floating-point training
matrix multiplications. The [BitNet b1.58 paper](https://arxiv.org/abs/2402.17764)
introduces the ternary representation; its quality results do not validate this
Sprint variant. This benchmark also keeps one common optimizer schedule instead
of tuning a BitNet-specific convergence recipe.

## Matched setup

The core comparison uses the original Sprint throughput shapes saved for this
experiment, with vocabulary 8,192, sequence length 512, maximum context 2,048,
head dimension 128, and effective batch 32 sequences in every case.

| Target | Base parameters | Width / layers / heads / MLP width | Microbatch × accumulation | Warmup / measured updates | Extra BitNet norm parameters |
|---|---:|---|---|---|---:|
| 50M | 49,816,576 | 1,024 / 4 / 8 / 3,008 | 32 × 1 | 5 / 60 | 16,128 |
| 500M | 499,329,792 | 2,304 / 9 / 18 / 6,976 | 8 × 4 | 5 / 20 | 83,520 |
| 1B | 998,505,728 | 2,816 / 11 / 22 / 10,112 | 8 × 4 | 5 / 12 | 142,208 |

Each case starts from seed 42 and identical BF16-rounded initial weights for its
base shape. Conversion occurs before optimizer creation. Consequently the FP32
CPU masters initially contain copies of those BF16-rounded values. BitNet's
additional norms start at their normal initialization; its exact post-conversion
parameter count is recorded separately.

The shared execution settings are `torch_compile=true`, `compile_mode="default"`,
`attention_backend="cudnn"`, `cpu_threads=20`, `offload_bucket_mb=32`,
`cpu_optimizer_kernel="compiled"`, and no activation checkpointing. The compiled
CPU Adam kernel updates large BF16 slices; small slices use native CPU Adam.
All cases use the same Sprint optimizer schedule with initial learning rate
`3e-4`, Adam betas `(0.9, 0.95)`, matrix weight decay `0.1`, and gradient clipping
at `1.0`. Benchmark warmup is excluded from timing; optimizer learning-rate
warmup is disabled.

The reference machine is an NVIDIA RTX 5070 12GB with an Intel Core Ultra 7
265F and 32GB host RAM, running PyTorch 2.11.0+cu128. Hardware and runtime fields
in each receipt identify the actual case. Run cases sequentially without other
training or CPU benchmarks competing for the GPU or CPU optimizer.

## Timing, checks, and memory

The main comparison uses **supervised targets per second**. Shifted causal loss
does not supervise the first token of each sequence, so every effective update
contains `32 × 511 = 16,352` targets and `32 × 512 = 16,384` input tokens. The
receipt includes both rates. A projection to one billion targets is
`1e9 / supervised_tokens_per_second`; it excludes startup and corpus work.

CUDA synchronization surrounds the measured interval. Timing includes CPU-to-GPU
input transfer, forward, backward, gradient accumulation, clipping, FP32 CPU Adam
updates, and gradient/weight transfers. Model initialization and conversion are
recorded in `initialization_seconds`; warmup and lazy compilation are recorded in
`warmup_and_compile_seconds`. Corpus preparation, evaluation, checkpoint writes,
and the following validation checks are outside steady-state timing.
Startup timings reflect the machine's existing compiler caches; this was not a
controlled cold-cache comparison.

Every successful case requires finite warmup/final loss, finite final gradient
norm, and an observed decoder matrix parameter update during the measured
interval. The update check samples up to 128 values from each decoder matrix;
embeddings alone cannot satisfy it. The final gradient norm describes accumulated
gradients after clipping. These checks detect broken execution, not adequate
training quality. The same synthetic batch repeats throughout each short run.

**All modes retain BF16 trainable weights and gradients on the GPU, plus FP32
master weights and both Adam moments on the CPU.** The precision names describe
projection arithmetic, not reduced optimizer storage. Before activations and
temporary buffers, this is about 4 bytes per parameter on the GPU and 12 bytes
per parameter on the CPU: approximately 4GB and 12GB at one billion parameters.
BitNet's added norms and quantization intermediates add to those requirements.

Receipts inventory unique retained parameter, gradient, model-buffer, optimizer,
and direct offload-buffer tensor storage, deduplicating aliases. CUDA allocated
and reserved peaks cover initialization, warmup, and measurement, including
temporary execution allocations. Retained storage totals exclude transient
activations, Python objects, and compiler caches; they are not total process RAM.

## Reproduce

Use the same source revision and compatible PyTorch, TorchAO, native low-precision
providers, and local CPU compiler recorded with the receipts. Run from the yyLab
repository with its configured Python environment. The saved per-mode JSON
recipes under `artifacts/sprint-precision-20260911` contain explicit
`model_config` values so later planner changes cannot alter the comparison.
Later receipts also embed a complete `recipe` object, a UTC creation
timestamp, source SHA-256 hashes, and runtime package versions. Save a case's
`recipe` object as JSON and pass it to `--config` to reproduce without the local
artifact directory. The separate `optimizer_schedule` field records the common
schedule used inside measured steps; `recipe.warmup_steps` controls benchmark
warmup.

For the early core runs, the consolidated receipt also includes
`assembled_core_recipes`: their original saved configuration files, attached
during report assembly. These are labeled separately from metadata captured
at run time; missing historical hashes or timestamps are not reconstructed.

For one case, writing a new receipt directory:

```powershell
.venv/Scripts/python.exe -m yylab.train.benchmark_sprint_precision --config artifacts/sprint-precision-20260911/50000000-bf16.json --output artifacts/sprint-precision-reproduction/50m-bf16
```

To repeat all fifteen core cases in separate processes:

```powershell
$precisionModes = @('bf16', 'nvfp4_bf16', 'nvfp4_mxfp8', 'mxfp8', 'bitnet_b158')
$precisionTargets = @(50000000, 500000000, 1000000000)
foreach ($precisionTarget in $precisionTargets) {
    foreach ($precisionMode in $precisionModes) {
        $precisionConfig = "artifacts/sprint-precision-20260911/$precisionTarget-$precisionMode.json"
        $precisionOutput = "artifacts/sprint-precision-reproduction/$precisionTarget-$precisionMode"
        .venv/Scripts/python.exe -m yylab.train.benchmark_sprint_precision --config $precisionConfig --output $precisionOutput
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Case failed: $precisionTarget / $precisionMode. Inspect its benchmark.json."
        }
    }
}
```

`--config` supplies the complete recipe. Ordinary recipe flags do not override it;
the explicit experimental `--cpu-optimizer-min-elements` flag is the exception,
as documented in the [follow-up speed round](SPRINT_SPEED_ROUND2_2026-09-11.md#reproduction).
Without a config, `--help` lists target sizing, precision, microbatch, compilation,
CPU, attention, and timing controls. Each output directory receives incremental
`benchmark.json` and `progress.json`, followed by a final `summary.json`.
Rows retain error/OOM status and the failing stage. A partial comparison exits
nonzero, and a comparison with no successful cases raises after writing its
receipt. Failed modes are never included in the fastest measured selection.

## Further optimization experiments

The fast MXFP8 variants fuse transpose and quantization, avoid unnecessary
padding, and reuse padded backward operands. Each gradient matrix multiplication
still computes its own block scales along its contraction dimension. GPU tests
compare cast bytes and forward/backward results with the original native path,
including the benchmark's unaligned MLP widths.

The INT8 BitNet variant stores ternary codes in one-byte integers and uses native
INT8 forward multiplication. It is not a packed ternary kernel. Forward rescales
INT32 sums in FP32 before rounding to BF16, so its arithmetic differs slightly
from separately rounding both dequantized operands to BF16. Backward keeps the
reference dense BF16 straight-through recipe. Integer products pass an exact
INT64-reference check. Compilation can change activation codes near rounding
boundaries; the numerical check bounds aggregate and maximum output/gradient
errors rather than assuming bitwise equality across fused and eager execution.

The stock all-NVFP4 training kernel requires a stochastic conversion instruction
that this RTX 5070 does not support. The `nvfp4_all` mode therefore fails during
preflight. The separate `nvfp4_all_portable` experiment retains native FP4 matrix
multiplications and TorchAO's Hadamard transforms and scaling, and implements
stochastic rounding to adjacent E2M1 values in software. It uses 24 random bits
per element and is not bitwise equivalent to the hardware instruction's random
layout. See NVIDIA's [PTX conversion documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt).

Portable mode pads operands to 128 internally and crops outputs and gradients;
it preserves the original model dimensions and Parameter objects. It checks six
upstream source hashes before cloning private provider functions, without
modifying the installed package. Its conversion search is bounded to 128 by 128
tiles, two pipeline stages, and four or eight warps. GPU tests check rounding
codes, probabilities, random offsets, native forward/backward execution, and
gradient cropping. The source guard covers the cloned recipe and its direct
transform/cast utilities, not every transitive dependency. Hadamard transforms
still contain BF16 operations; "all NVFP4" means the three converted projection
matrix multiplications. Passing these checks does not establish convergence.
The repeated default-compiled backward test also verifies fresh stochastic
rounding with unchanged weights, inputs, and output gradients. Explicit CUDA
graph replay was not tested.

### Screening results

These short probes guided the final configuration choice. They are not averaged
with the longer confirmation runs above. The effective batch stays 32; the
microbatch is shown where it differs across sizes. Dashes mean not measured.

| Additional mode | 50M, microbatch 32 | 500M | 1B |
|---|---:|---:|---:|
| Fast MXFP8 | 161,205 | — | — |
| Fast NVFP4/MXFP8 | 162,783 | 23,616 at batch 32 | 12,046 at batch 8; 2,469 at batch 16 |
| BitNet with native INT8 forward | 124,855 | 14,893 at batch 8 | 7,799 at batch 8 |
| All-NVFP4 projections with portable rounding | 121,634 | 19,600 at batch 16 | — |

At 500M, the original hybrid reached 22,533 at batch 16 and 22,989 at batch 32;
the fast hybrid reached 23,019 at batch 16. BF16 reached 13,891 at batch 16.
At 1B, the original hybrid reached 12,193 at batch 16. The 500M BF16 search used
batches 8 and 16; batch 32 was not timed. These are the best configurations found
in this bounded search, not an exhaustive search of all execution settings.

The 1B fast-hybrid batch-16 probe completed successfully but encountered GPU
memory pressure: Windows counters recorded 1.80GB of shared GPU allocation,
and measured throughput fell to 2,469. This was a multiple-case process, so its
saved recipe includes both batches 8 and 16; fresh standalone batch-16 behavior
was not established. The original hybrid at batch 16 was the fastest successful
1B screening configuration, but its longer standalone confirmation fell to
3,742 targets/sec. The 500M fast-hybrid batch-32 confirmation also fell from
23,616 in screening to 22,399. Both longer results are retained under
`confirm-1b-hybrid-b16` and `confirm-500m-fast-b32`. These results prompted the
smaller-batch rechecks used in the final comparison. The cause of every
run-to-run difference was not separately profiled; the memory-pressure counter
observations above apply to the specifically identified screening runs.
The longer fast-hybrid 1B batch-8 run also fell to 4,048 targets/sec and is
retained as `confirm-1b-fast-b8`. Its higher peak allocation includes compilation
and warmup; peak counters alone do not establish when the extra memory was used.
The original hybrid was therefore rechecked at batch 8 as well.

The 1B BF16 batch-16 probe recorded 2.07GB of shared GPU allocation and was
stopped after 283 seconds total process time without a completed timed result.
Its receipt is explicitly marked `interrupted`, rather than a CUDA OOM or a
throughput measurement. The consolidated receipt retains both counter
observations. A larger nominal batch is not automatically a faster setting on
this GPU.

INT8 forward improved BitNet-style throughput over its dense BF16 implementation
by 7.5%, 13.7%, and 20.1% at the matched core microbatches. It remained slower
than the native NVFP4/MXFP8 and MXFP8 comparisons. All-NVFP4 projections also
remained slower than the hybrid in the tested 50M and 500M cases. That recipe
includes additional transforms and rounding work; this benchmark does not
isolate each operation's contribution to its runtime.

## Validation

The focused suite passed **121 tests** on this machine, including the native
CUDA checks, with 16 warnings and no skipped tests. Ruff passed for the changed
precision implementation and test files. The consolidated receipt records the
command, test-file hashes, and result. Successful benchmark cases also pass the
finite-loss, finite-gradient, and decoder-update checks described above.

```powershell
.venv/Scripts/python.exe -m pytest -vv tests/test_sprint_precision_benchmark.py tests/test_sprint_precision.py tests/test_bitnet_benchmark.py tests/test_bitnet_int8_benchmark.py tests/test_mxfp8.py tests/test_mxfp8_fast.py tests/test_sprint_nvfp4_training.py tests/test_sprint_nvfp4_portable_cuda.py
```
