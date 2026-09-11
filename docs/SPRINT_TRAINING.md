# Sprint training

Open **Train · Sprint** (`/train/sprint`) in yyLab. Sprint is a new dense causal
language model and training runner focused on complete optimizer-step throughput.
It trains independently of the SPALMER model/optimizer implementation while reusing
yyLab's tokenizer registry, corpus selections, split protections, data packers,
background jobs, checkpoints, chat and export loading.

See [the latest 10M–1B speed experiments](SPRINT_SPEED_ROUND2_2026-09-11.md),
[the native precision comparison](SPRINT_PRECISION_BENCHMARKS_2026-09-11.md), and
[the original RTX 5070 measurements](SPRINT_BENCHMARKS_2026-09-10.md).

## Architecture

- Fixed-depth pre-normalized residual decoder. Every block runs once.
- One packed QKV projection, full multi-head causal PyTorch SDPA, and an output
  projection. Runtime chooses an available attention kernel for the GPU and shape;
  installing PyTorch does not guarantee that FlashAttention is available.
- Rotary position embeddings, RMSNorm, and two-projection tanh-approximate GELU.
  The model API also supports packed gate/up SwiGLU (`mlp_type="swiglu"`).
- Tied input/output embeddings by default. No recurrent depth, expert routing,
  quantization simulation or auxiliary objectives.
- Training returns only the loss. Large CUDA vocabularies use Liger fused linear
  cross entropy so the full token-by-vocabulary logits need not be retained.
  `loss_backend="torch"`, `"liger"`, and `"chunked"` are explicit alternatives.
  The chunked PyTorch path checkpoints head chunks to bound retained logit memory.
  New-model setup exposes Auto, PyTorch, and Liger in **Loss implementation**.
  The selection is shared by training and its benchmark. Full PyTorch loss
  improved measured 10M/50M speed on the tested GPU while using more memory;
  it is an explicit choice, and existing checkpoints retain their saved backend.

The parameter planner accepts arbitrary integer targets, including 500M and 1B,
and computes the exact unique parameter count without allocating the model.
For new targets of 10M parameters and above, the default `shape_profile="throughput"`
uses wider decoders with fewer layers, widths aligned to 256, head dimensions of
128, and MLP widths aligned to 64. It fits within 2% of the requested budget when
these constraints permit. `shape_profile="balanced"` retains the original
proportions; targets below 10M retain their previous sizing with either profile.
Manual shapes and existing checkpoints retain their saved dimensions.
Vocabulary is part of the parameter count and is never silently replaced. Targets
below the supported vocabulary/architecture floor produce an actionable error.
This shape planner is a heuristic. The wider profiles reduce measured training
time and activation memory on the tested GPU, but equal convergence or final
quality at the same parameter count has not been established.

To explore different feed-forward allocations at a fixed total parameter budget:

```powershell
uv run python -m yylab.train.sprint_shapes --target-params 50000000 --ffn-fractions 0.25,0.375,0.5,0.7 --depths 2,4,8 --output artifacts/sprint-shapes-50m.json
```

This arithmetic-only search returns reusable `model_config` objects, exact
embedding/attention/FFN/norm counts, and achieved FFN fractions. Every candidate
stays below the requested total and within the requested tolerance; an empty
result means no matching shapes were found. Fractions include the whole unique
parameter count in their denominator. Sprint is dense: these are feed-forward
budgets, not routed experts. Enter a candidate's dimensions under **Manual
architecture**, or reuse its config in a recipe, then measure it. The ordering
of candidates does not predict speed or language quality.

The experimental precision benchmark also accepts
`--cpu-optimizer-min-elements 262144` with `--cpu-optimizer-kernel compiled`.
This changes the cutoff for using the already prepared compiled CPU Adam kernel
and records actual bucket coverage. It is a disposable benchmark setting;
ordinary training and exact resume keep their existing optimizer dispatch.

## Training workflow

1. Select a registered tokenizer and one or more ready corpora. Choose pretraining
   or chat SFT, training source splits, domains and mixture settings.
2. Enter a parameter target or a manual shape. Choose context, training sequence
   length and a token budget or step limit.
3. Run the disposable throughput benchmark for that shape. It holds effective
   batch and sequence length fixed while varying microbatch, accumulation and
   compilation. Advanced options can also compare CPU threads, optimizer transfer
   bucket sizes and attention implementations. Inspect failures as well as the
   fastest completed candidate.
4. Apply the measured settings and start training. Runs shows progress and loss.
   Finished checkpoints can be selected in Chat and the checkpoint registry.

The default CUDA recipe uses **BF16 weights and gradients**, with **FP32 master
weights and Adam moments offloaded to CPU RAM**. Small updates accumulate in the
FP32 CPU master before weights are copied back to BF16, so rounding the visible
weights does not erase those updates. Gradient transfer and CPU updates use
two bounded pinned buffers: gradient copies and weight writebacks overlap CPU
updates, with events protecting buffer reuse. At the default 32 MiB bucket size,
this uses at most 64 MiB of pinned transfer buffers and 64 MiB of FP32 gradient
scratch. This substantially lowers GPU memory use; CPU memory bandwidth
and PCIe transfers become part of the training step and must be benchmarked.

`cpu_optimizer_kernel="auto"` prepares a compiled CPU update for large BF16
slices when a local C++ compiler is available. It fuses the BF16 conversions with
FP32 Adam/master-weight updates, while small slices use native Adam. Preparation
and alias checks run on disposable bounded buffers before live state is touched.
The worker can activate an already-installed Visual Studio C++ toolchain on
Windows; it does not install one. Auto falls back visibly to native Adam if
preparation fails. `"native"` disables CPU compilation; `"compiled"` requires it.
Runtime and benchmark receipts record the resolved implementation and reason.
Once a live compiled update starts, an error stops the run instead of retrying
and risking a double update. Older checkpoints resume with their native CPU
update; switching implementation requires starting a continuation.

The earlier FP32-weight, fused GPU AdamW path remains an explicit comparison
(`parameter_dtype="float32", optimizer_state_offload="none"`). CPU execution
uses FP32 by default. No simulated FP4/FP8 training or precision experiment matrix
is introduced. Benchmark recommendations retain the precision, optimizer location
and activation-checkpointing settings that produced their measurements.
They also carry the measured CPU thread count, transfer bucket size and attention
implementation. `attention_backend="auto"` preserves PyTorch's dispatch;
`"cudnn"`, `"efficient"`, `"flash"`, and `"math"` explicitly select a backend and
fail if unsupported. Selection is scoped to training/evaluation and is part of
the exact-resume contract. Optimizer kernel, CPU thread count and transfer bucket
size also belong to that contract, since they determine CPU kernel execution and
which slices use each implementation. Omit `cpu_threads` to keep the process's
current setting on a new run; resolved thread counts are saved for resume.
With CPU offload, gradient accumulation amortizes the optimizer transfer/update
over more tokens. The page starts at 32 sequences per optimizer update (4 × 8)
and the benchmark keeps the effective batch fixed at 32 by default. Tune both
memory and update frequency for the intended learning task.

Fixed rows are prepared into a reusable memory-mapped cache (uint16 token IDs
when the vocabulary fits, otherwise uint32; SFT retains labels and masks).
Training uses bounded CPU read-ahead, pinned memory and asynchronous transfers.
The cache identity includes tokenizer contents, corpus snapshots, selections,
packing settings and its row limit. It stores derived token data, not source text.
For budgets larger than one available corpus pass, cached rows are repeated with
deterministic block shuffling. Positive `epochs` can cap passes through that cache.

Losses and gradients are weighted by actual supervised targets when accumulating
unequal or masked batches. Training counts forwarded token slots and supervised
next-token targets separately; sequence-boundary first tokens and ignored SFT
labels do not count as targets. A requested token budget is rounded down to whole
sequence rows, with a smaller final batch when needed. Its exact resolved totals
are saved in the runtime receipt.

Validation uses explicitly selected disjoint source splits. Logging, evaluation
and saves have independent intervals. Checkpoints retain the model, tokenizer,
optimizer, RNG states, schedule, cache identity and exact consumed row cursor.
**Continue checkpoint** starts a new schedule/optimizer. **Resume optimizer state**
continues the original recipe and rejects changes to its data or optimizer
contract; choose continuation when changing the budget or learning schedule.
Outside yyLab, import `yylab.models` before using Hugging Face auto-loading, or
load directly with `SprintForCausalLM.from_pretrained(path)`. The new architecture
requires yyLab's model code; its checkpoint does not pretend to be a stock Llama.

## Benchmark from a terminal

The training runner also accepts a saved yyLab JSON recipe:

```powershell
uv run python -m yylab.train.sprint --config recipe.json --output artifacts/sprint-run
```

```powershell
uv run python -m yylab.train.benchmark_sprint --target-params 50000000 --vocab-size 8192 --seq-len 512 --effective-batch 32 --batch-sizes 8,16,32 --compile-modes eager,default --warmup-steps 10 --measure-steps 100 --output artifacts/sprint-50m-benchmark
```

For a targeted tuning sweep, add `--cpu-thread-counts 8,20`,
`--offload-bucket-sizes-mb 16,32,64`, or `--attention-backends auto,cudnn`.
Candidates form a Cartesian product, so start with a small set. Use
`--shape-profile balanced` to benchmark the original architecture proportions.

The result includes complete forward/backward/clipping/optimizer/transfer time,
supervised targets per second, memory, startup/compile time, candidate errors,
and the fastest measured recipe. Synthetic inputs avoid changing or training an
existing checkpoint. They do **not** measure corpus throughput or learning quality.
The projected time per billion targets excludes preparation, compilation,
evaluation, logging, and checkpoint IO. Final training summaries separately record
preparation time and end-to-end throughput. Repeat the best candidates before a
long run; timing short runs can be noisy, especially on a shared desktop GPU.
Liger runs behind an explicit compilation boundary because the installed library's
custom backward is incompatible with full Dynamo tracing; the decoder still
compiles. Its fused kernels execute natively, with the library's existing per-batch
target-count synchronization. The plain PyTorch loss can compile as a whole.

Experimental precision comparisons use the separate
[disposable benchmark CLI](../yylab/train/benchmark_sprint_precision.py):
`uv run python -m yylab.train.benchmark_sprint_precision --help`.
It can compare `bf16`, `nvfp4_bf16`, `nvfp4_mxfp8`, `mxfp8`, and `bitnet_b158`
on matched model shapes. These adapters are separate from normal training
recipes. See the [precision comparison report](SPRINT_PRECISION_BENCHMARKS_2026-09-11.md)
for methodology, reproduction instructions, and the limits of each precision label.

## Scale and current limits

Sprint currently trains on one CUDA device or CPU. There is no distributed or
sharded optimizer path. With BF16 weights/gradients and CPU optimizer offload,
budget approximately **4 GPU bytes per parameter** and **12 CPU bytes per
parameter** for the FP32 master and two moments, before activations, staging,
workspaces, caches and checkpoints. For 1B parameters that is about **4 GB GPU +
12 GB CPU**. This machine has a 12 GB RTX 5070 and 32 GB system RAM; usable batch
size still depends on remaining memory. Activation checkpointing reduces
activation memory at the cost of extra computation.

The optional FP32 GPU-resident baseline needs approximately **16 GPU bytes per
parameter** before activations: about 8 GB for 500M and 16 GB for 1B. It cannot
fit a 1B run on this card. CPU offload addresses this memory floor, with a
throughput cost from transfers and CPU computation.

Throughput depends on vocabulary, context, batch, hardware, kernel availability
and the quality objective. Sparse SPALMER and dense Sprint do not have equal
active compute at equal total parameter counts. No claim of universal fastest
training, equal model quality or billion-token convergence is made.

Implementation references: [PyTorch 2.11 SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html),
[torch.compile](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html),
[fused AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html),
and [Liger Kernel](https://github.com/linkedin/Liger-Kernel).
