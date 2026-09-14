# Sprint training

Open **Train · Sprint** (`/train/sprint`) in yyLab. Sprint is a new dense causal
language model and training runner focused on complete optimizer-step throughput.
It trains independently of the SPALMER model/optimizer implementation while reusing
yyLab's tokenizer registry, corpus selections, split protections, data packers,
background jobs, checkpoints, chat and export loading.

See [the 150M PLE + KDA billion-token run](SPRINT_150M_PLE_KDA_1B_2026-09-13.md),
[the repetition investigation and repair status](SPRINT_REPETITION_REPAIR_2026-09-13.md),
[the 1B PLE and KDA comparison](SPRINT_PLE_1B_2026-09-13.md),
[the 10M–1B speed experiments](SPRINT_SPEED_ROUND2_2026-09-11.md),
[packing speed and cache reuse](SPRINT_PACKING_2026-09-11.md),
[the native precision comparison](SPRINT_PRECISION_BENCHMARKS_2026-09-11.md), and
[the original RTX 5070 measurements](SPRINT_BENCHMARKS_2026-09-10.md).

## Architecture

- Fixed-depth pre-normalized residual decoder. Every block runs once.
- Standard attention (`attention_type="sdpa"`, the default) uses one packed QKV
  projection, full multi-head causal PyTorch SDPA, and an output
  projection. Runtime chooses an available attention kernel for the GPU and shape;
  installing PyTorch does not guarantee that FlashAttention is available.
- Rotary position embeddings in SDPA layers, RMSNorm, and two-projection tanh-approximate GELU.
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
For standard-attention targets of 10M parameters and above, the default `shape_profile="throughput"`
uses wider decoders with fewer layers, widths aligned to 256, head dimensions of
128, and MLP widths aligned to 128. It fits within 2% of the requested budget when
these constraints permit. `shape_profile="balanced"` retains the original
proportions; targets below 10M retain their previous sizing with either profile.
Manual shapes and existing checkpoints retain their saved dimensions.
Vocabulary is part of the parameter count and is never silently replaced. Targets
below the supported vocabulary/architecture floor produce an actionable error.
This shape planner is a heuristic. The wider profiles reduce measured training
time and activation memory on the tested GPU, but equal convergence or final
quality at the same parameter count has not been established.

### Optional per-layer embeddings (PLE)

For a **New model**, choose **Per-layer embeddings (PLE)** in the model section:

- **No PLE** retains the existing Sprint architecture and remains the default.
- **Regular PLE** adds one independent vocabulary-by-width lookup table and a
  learned scalar gate before each layer after the first. The first input
  embedding remains tied to the output head. This is one table per later layer,
  not the historical four-full-table expansion.
- **QR PLE** uses four smaller quotient/remainder lanes whose lookup vectors
  combine before entering each later layer. The model/recipe API can set another
  positive `ple_lanes` count; the page uses four.

The preview shows total, PLE and feed-forward parameter allocations. With a
standard-attention parameter target, the planner first keeps the ordinary plan's width and layer
count and reduces feed-forward width to pay for PLE. It chooses a smaller shape
if those fixed costs cannot fit. With a manual shape, PLE adds parameters to the
specified backbone. Regular PLE can consume a large part of a small model's
budget, leaving a narrow feed-forward layer. Faster training at the same total
count does not establish equal model capability; no PLE quality study is implied.

The selection travels with training and benchmark recipes as
`model_config.ple_type` (`"none"`, `"full"` or `"qr"`) and
`model_config.ple_lanes` (default `4`). The Python planner accepts these same
keyword arguments. Benchmark recommendations require the same PLE choice and
lane count, and older recipes without these fields retain No PLE. The selected
weight precision, optimizer placement and token budget are unchanged by PLE.
Changing PLE also keeps compatible prepared token data reusable.

Saved models, periodic saves and final checkpoints include PLE weights and
configuration. **Continue checkpoint** and **Resume optimizer state** inherit
the saved choice. They reject changes to PLE type or lane count; start a new
model for those changes. Model loading and generation retain the per-token PLE
lookups, including when standard attention uses its generation cache.

Use the page's complete-step benchmark for the selected architecture. Synthetic
random tokens can differ from real corpus token-frequency skew in lookup cache
behavior and repeated-row gradient accumulation, so their throughput is not a
measured real-corpus training ETA.

### Kimi linear attention with PLE

Choose **Model attention → Kimi linear attention · hybrid 3:1** for a new model,
or set `model_config.attention_type="kda"`. This Sprint hybrid repeats three
Kimi Delta Attention layers followed by one full causal SDPA layer. Layer count
must be divisible by four. Every physical layer still runs once; KDA's recurrence
is across sequence positions and does not add SPALMER recurrent depth reasoning.
No PLE, Regular PLE and QR PLE can all be combined with this attention choice.

`kda_backend="auto"` selects the installed Flash Linear Attention (FLA) CUDA
implementation for supported CUDA execution and the PyTorch reference path on
CPU. `"fla"` explicitly requires the CUDA implementation; `"reference"` selects
the correctness-oriented reference implementation. CUDA training uses BF16
compute with **FP32 stored weights and FP32 GPU AdamW moments**:
`parameter_dtype="float32", optimizer_state_offload="none"`. The page selects
the GPU optimizer preset when KDA is chosen or detected in a continued/resumed
checkpoint. BF16 parameter storage and the CPU-offload preset are rejected for
production KDA training; experimental quantized weight or optimizer adapters are
separate benchmarks. CPU reference training uses FP32 compute.

`kda_disable_recompute=True` retains FLA intermediates for backward. Setting it
to `False` lets FLA recompute some intermediates to reduce retained activation
memory; measure the resulting speed/memory tradeoff. Block activation
checkpointing is a separate setting. The KDA planner includes its gates,
convolution and PLE in the exact parameter budget, can change width and layer
count, and retains at least a width-sized feed-forward layer. Its shape choices
are capacity heuristics rather than measured quality equivalence.

Attention type, KDA backend and recomputation setting travel with model configs,
saved checkpoints and exact-resume contracts. Existing checkpoints with no KDA
fields retain standard SDPA. Continued training inherits the saved architecture,
and incompatible overrides are rejected. Benchmark recommendations match these
attention settings as well as shape and PLE, so a standard-attention measurement
cannot be applied to a KDA model.

KDA generation currently recomputes the complete prefix for each new token.
`generate(use_cache=True)` deliberately selects the uncached path; a supplied
external cache is rejected. Chat and Antidoom avoid constructing an inapplicable
Transformers KV cache and display/expose the uncached behavior. This supports
sampling, beam search, generation scores and padded batches, but does not provide
fast recurrent-cache decoding. Left/right 2D padding and supported block-causal
segments reset sequence state correctly; arbitrary attention biases or
head-dependent masks are rejected.

Saved PLE + KDA models are usable for continued pretraining, masked SFT and the
prepared Antidoom FTPO workflow. Antidoom maps a complete split Q/K/V LoRA request
to Sprint's packed `qkv_proj`; partial split requests fail clearly. The existing
tied-output-head Antidoom adapter remains adapter-only and must not be merged
into the tied base weights. Compatibility tests cover checkpoint and adapter
roundtrips, the FTPO policy/reference objective, mining scores, NLL and chat;
they do not establish downstream task quality. See
[the training tests](../tests/test_sprint_kda_training.py) and
[the posttraining tests](../tests/test_sprint_posttrain.py).

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
4. Apply the measured settings and start training. Runs shows progress, loss and
   generated-text diagnostics. Inspect the saved samples as well as the loss.
   Finished checkpoints can be selected in Chat and the checkpoint registry.

The default standard-attention CUDA recipe uses **BF16 weights and gradients**, with **FP32 master
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

The FP32-weight, fused GPU AdamW path remains an explicit choice and is required for KDA
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
Reuse is automatic under `artifacts/cache/sprint-packed`: model size, learning
rate and optimizer changes do not invalidate data. For legacy stream mixing, a smaller token budget can
read a prefix of a larger compatible cache without copying files. A larger
budget can reuse an already complete source pass. A shorter, budget-truncated
cache is never mistaken for the whole corpus. Logical keys retain the requested
row limit, so exact resume works regardless of which physical cache backs it.
Changing source contents, tokenizer, objective, sequence length, seed, or the
ordered corpus selection/mixing rules still requires different packed rows.

Cold causal packing batches tokenization and writes bounded NumPy row chunks.
Its token order, document EOS boundaries and source-accounting semantics match
the original packer. Progress reports packing tokens/second; runtime receipts
distinguish a new build, exact cache reuse, and compatible cache reuse. Prefix
views report only known prefix statistics and retain full backing-cache
statistics separately. Partial staging files from interrupted packing are not
treated as valid caches. The optional per-document token cache remains useful
across changed mixtures, but preparing entire corpora adds upfront work and disk
usage; it is independent of the always-on packed cache.
For legacy stream budgets larger than one available corpus pass, cached rows are repeated with
deterministic block shuffling. Positive `epochs` can cap passes through that cache.

### Fixed token shares

New causal-training recipes created by the page default to **Fixed token shares**
(`corpus_mix="fixed_token_quota"`). The worker does not reinterpret older saved
`token_balanced`, `round_robin` or `concatenate` recipes. The legacy option is
labeled **Balance while sources last (characters)** because it estimates cost
from text length and redistributes the remaining stream after sources exhaust.
It does not guarantee the final token mixture.

Fixed quotas allocate complete sequence rows using `corpus_weights`: a weight of
3 versus 1 requests 75% versus 25% of input tokens. Omitted selected corpora have
weight 1. Largest-remainder rounding preserves the total budget and differs from
each requested share by less than one sequence. Every row belongs to one source,
so its share of shifted supervised targets is also known exactly.

The packer prepares complete source row pools, samples without replacement before
replay, and globally shuffles the allocated rows before the first update. Changed
weights, budget, seed or replay allowance reuse these pools. A different budget
materializes its own allocation instead of treating a random prefix as exact
new quotas. Changing source contents/selection, tokenizer or sequence length needs
different pools. The optional document token cache can avoid repeated tokenization;
the source-pool cache independently avoids repeated packing of unchanged selections.

**Maximum passes through each source** (`quota_max_source_epochs`) defaults to 1:
each available source row can be selected at most once. A larger finite value
explicitly permits replay. The recipe API also accepts a dictionary of limits by
corpus ID; missing entries default to 1. If any source cannot supply its quota
within its allowance, preparation fails with its required and available counts.
Other sources do not silently replace the missing share. Reduce the weight or
budget, add suitable data, or explicitly change the replay allowance.

`source_max_ratio` belongs to the legacy scheduler and must be unset for fixed
quotas. Chat SFT retains its existing source scheduling; fixed quotas currently
support `causal_lm` only. Explicit `epochs` may be 0 or 1 because the allocation
already contains the whole finite budget and any permitted replay.

The version `sprint-fixed-token-quota-v1`, exact allocations and replay limits
belong to cache identity. Saved row provenance records the source and original
pool position. Final/checkpoint `consumed_data_stats` reports actual consumed
source inputs, shifted targets and repeated positions, including an early stop.
These are unique **pool positions**, not a semantic diversity measure. Configured
exact-text training deduplication is preserved; this mode does not establish
held-out decontamination or repair poor source content.

### Optional cleanup of packed rows

New fixed-share recipes in the page enable **Remove duplicate and extremely
repetitive sequences**. The recipe field is
`quota_row_filter={"enabled": true, "version": "sprint-quota-row-filter-v1"}`;
the version can be omitted and resolves to v1. An absent field or
`{"enabled": false}` preserves the existing unfiltered cache identity. Explicitly
enabling this option outside `fixed_token_quota` is rejected.

The v1 policy removes later exact complete-row matches across the selected
sources, using SHA256 over their token bytes. It also removes rows of at least
256 tokens when at least 90% of their overlapping token 8-gram occurrences are
copies beyond the first occurrence of each distinct 8-gram. The latter check
runs first; rejection reasons are exclusive. These fixed criteria catch obvious
repeated-character, separator and cyclic spans. They do not normalize numbers,
compare paraphrases, crop documents, or establish semantic diversity. Shorter
sequence lengths still receive exact-row deduplication.

Cleanup runs once on the CPU before quota allocation and replay admission. It
derives a separate immutable pool group from reusable unfiltered pools, so
enabling it does not require retokenizing unchanged source documents. Its digest
index uses a disk-backed SQLite table with a bounded page cache. Changed weights,
budget, seed or replay allowance reuse the cleaned pools. The policy participates
in both pool and final-mixture identities; an enabled recipe cannot silently
reuse an old unfiltered cache. Changing it on exact resume changes the data
contract and requires starting a new schedule from a checkpoint instead.

Replay limits apply to the **retained** row count. If cleanup leaves too little
data for a source's requested share, preparation reports that shortage instead
of replacing its quota with another source. A filtered pool's `row_pool_index`
addresses retained rows; `source-N.original_rows.bin` maps them back to original
unfiltered pool row offsets. The receipt records each source's input, retained
and rejected rows and token slots, exclusive reason counts, and up to eight
rejected examples with original offsets and hashes. `raw_data_stats` retains
the original document/rendered-content/EOS counters separately from retained-row
counts. Prefix and completed training statistics carry these removal receipts
under `row_filter`, explicitly scoped to complete source-pool preparation rather
than consumed training tokens.

The [actual 150M payload audit](../artifacts/corpus-duplication-audit-20260913/PACKED_DUPLICATION_FINDINGS.md)
motivates this conservative guard. The guard does not demonstrate that generation
repetition is fixed or that an existing checkpoint has been repaired.

### Materialized fixed allocations

An explicit `fixed_token_mix` recipe instead identifies an already materialized
fixed allocation. Its descriptor records the version, sampling seed, rows per
selected corpus, SHA256 of each full-pass token pool and exact-text deduplication
policy. It is part of packed-cache identity. The worker requires a matching
materialized cache (or a compatible prefix), and fails if it is missing instead
of rebuilding an ordinary concatenated dataset. The requested budget cannot
exceed the descriptor's rows or add repeated epochs. Deliberate small-source
repetition, when desired, must already be represented and counted in the
materialized allocation. Held-out evaluation uses a separate cache and removes
the training-only descriptor. The
[150M run's data receipts](../artifacts/sprint-150m-ple-kda-1b-20260913/preparation-receipt.json)
record this preparation and its exact token accounting.

Losses and gradients are weighted by actual supervised targets when accumulating
unequal or masked batches. Training counts forwarded token slots and supervised
next-token targets separately; sequence-boundary first tokens and ignored SFT
labels do not count as targets. A requested token budget is rounded down to whole
sequence rows, with a smaller final batch when needed. Its exact resolved totals
are saved in the runtime receipt.

### Generated-text and numerical checks

**Check generated text for repetition** is enabled by default. Its own interval
defaults to 500 optimizer updates, independently of loss evaluation and saving.
It also runs at the final save unless that exact step was already checked.
The default four prompts cover prose continuation, an instruction, copying two
words and a simple word problem. Each uses greedy and sampled decoding, up to
64 new tokens, without repetition penalties or ngram blocking. Token duplication,
exact repeated tails, output length, raw probability/entropy, finite logits and
the two simple answer checks are recorded in
`<run>/generation-eval/step-XXXXXXXX.json`. Progress and metrics reference that
receipt and warn about repetition. Runs displays the saved prompts and outputs,
with gradient telemetry and warnings retained for completed/stopped runs.
A short/skipped output is reported as such;
absence of a detected loop is not a passing model-quality score.

Use the checkbox to disable the checks, or set
`"generation_eval": {"enabled": false}` in a recipe. The recipe API also accepts
`interval` and bounded `options` such as `max_new_tokens`, `modes`, `prompts` and
`prompt_format`. Causal training defaults to plain completion; chat SFT uses its
recorded chat template. A chat-format diagnostic on a raw-only checkpoint is
rejected. Explicit role/`[INST]` diagnostic formats do not establish that those
formats occurred in training. The evaluator preserves RNG and model training
modes, uses the eager model, and adds evaluation time rather than optimizer work.

When clipping is enabled, the existing pre-clipping norm is now logged as latest
and window-mean `gradient_norm`, together with `gradient_clip_fraction` and the
latest clipping coefficient. Clipping disabled means no norm sample is collected.
These are diagnostics, not a changed optimizer. Before publishing a checkpoint,
the worker checks weights, FP32 master weights and Adam moments for nonfinite
values and refuses an invalid bundle. Loss/norm checks report failures during
logging; generated nonfinite logits also fail the run. Repetition warnings alone
do not automatically halt a run or certify its checkpoint.

### Checkpoint input format

New Sprint bundles save `prompt_format.json`. Plain causal pretraining establishes
**Text continuation**: the passage is sent literally, without synthetic chat
markers or previous conversation turns. A tokenizer carrying a chat template
does not establish that the weights learned it. Old Sprint causal checkpoints
are interpreted from their training metadata without rewriting their files.

Chat SFT records the actual tokenizer template and its SHA256; inference uses
that saved template. Continued causal pretraining inherits an established parent
chat contract. New chat SFT can upgrade a raw checkpoint, while conflicting
inherited chat templates fail clearly. Exact resume cannot change the saved
format contract. Unknown external/adapter formats remain visibly unverified.

Input-format corrections and generated-text checks do not repair existing model
weights. An experimental generated-repeat unlikelihood helper is available for
bounded studies; it is **not part of ordinary Sprint training or enabled by a UI
switch**. See the [repair report](SPRINT_REPETITION_REPAIR_2026-09-13.md) for the
failed data-only intervention, preliminary targeted improvement and its limits.

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
