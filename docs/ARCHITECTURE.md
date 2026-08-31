# yyLab Architecture

## Layout

```
yylab/
  config.py        paths, settings.json (YYLAB_HOME override)
  db.py            SQLite registry: corpora/models/runs plus review labels and swipe events
  jobs.py          in-process background job manager (progress polling)
  hw.py            NVML/torch GPU probe, CPU/RAM info, triton detection
  artifacts.py     artifact dir layout + helpers
  ingest/          corpora pipeline
    adapters/      filetype adapters (add new ones here)
    pipeline.py    walk → extract → normalize → dedup → shard → tokenize
    dedup.py       sha256 exact + MinHash LSH near-dedup
    domains.py     code/math/science/web/language classification
    ranker.py      feature ranker, grouped validation, safetensors artifact
    tokenize_stats.py  multi-tokenizer counting registry
  review.py        durable swipes + uniform/active-learning queue selection
  mole/            Mixture-of-LoRA-Experts core (pure torch, CPU-testable)
    adaptive.py    exact lane-budget solver + surprise/occupancy controller
    layer.py       MoLELinear + per-block token and legacy prompt routers
    inject.py      model surgery, block-router hooks, stage freezing
    losses.py      route balance + predicted-NLL calibration
    runtime.py     lazy request-scoped adaptive expert residency
    store.py       strict versioned/sharded safetensors save/load
  train/
    planner.py     VRAM-aware auto method picker (full/lora/qlora)
    base_config.py validated scratch architecture, exact parameter count/solver
    low_precision.py state-compatible STE QAT and saved-format restoration
    data.py        shard streaming + EOS-packed sequence batches
    loop.py        generic training loop (metrics, aux loss, saves)
  benchmark/
    models.py      strict immutable suite schema + canonical fingerprints
    scoring.py     deterministic choice/numeric/exact/command/JSON scorers
    backend.py     direct checkpoint/adapter generation + route telemetry
    quality.py     resumable runs, matrices, provenance, reports, comparisons
    performance.py synthetic complete-step joint MoLE performance benchmark
    suites/        frozen benchmark prompt resources (never training data)
  workers/         detached training subprocesses (train.py, launch.py)
  models/loading.py  shared base loader (bnb quant, SDPA, RoPE scaling) + checkpoint scanner
  infer/engine.py  chat inference engine (base + MoLE/LoRA + router, KV settings)
  export/merge.py  merge adapters into standalone HF checkpoints (RL-ready)
  ui/              NiceGUI app shell + pages
```

## Artifact store (`artifacts/`, override with `YYLAB_HOME`)

```
corpora/<id>/stats.json, data/shard-*.jsonl.zst
checkpoints/<id>/      HF model/tokenizer + exact yylab_recipe.json for trained bases
adapters/<id>/mole_weights.safetensors + mole_config.json [+ experts/*.safetensors]
routers/<id>/          (legacy prompt-router snapshots; complete MoLE tensor sets)
runs/<id>/             (training/benchmark configs, logs, metrics, reports)
annotations/*.jsonl.zst append-only review-session receipts
rankers/corpus-stage1/ ranker.safetensors + versioned ranker.json
yylab.sqlite3          registry index
```

Every trainable artifact is plain safetensors + JSON. Native dynamic MoLE
loading uses the small injection/runtime layer in `yylab.mole`; dense LoRA and
compatible static MoLE exports can be consumed as ordinary Hugging Face
checkpoints. Corpora shards are zstd JSONL `{"text","domain","source"}`.

Review labels have two durable forms: unique document annotations and
append-only per-session swipe events in SQLite, plus compressed session
receipts under `annotations/`. The Stage-1 usefulness ranker recomputes a fixed
continuous feature schema from text, adds bounded domain/language categories,
standardizes on the training set, and fits L2-regularized logistic regression.
Its artifact contains coefficients in safetensors and its scaler, category
schema, data fingerprint, and corpus-held-out metrics in JSON. Active review
selection streams unreviewed documents and combines uncertainty (70%), exact
uniform exploration (20%), and confidence audits (10%). Deterministic ranked
components deliberately have null sampling propensity; only the random
component records a probability suitable for unbiased audits. Ranker output
does not participate in ingest admission or the permanent rejected-hash list.

MoLE artifact compatibility is explicit:

- Format v4 (`adaptive.enabled=true`) stores the cheap block-router tensors in
  `mole_weights.safetensors` and one complete global expert lane per
  `experts/expert_NNN.safetensors` shard. The ordered manifest records each
  lane's exact parameter count, tensor bytes, and SHA-256 digest. Loading
  validates every shard's keys/shapes without materializing its tensors; a
  lane is copied to the accelerator only when a request makes it resident.
- Format v3 (`router_type="per_token"`) stores every per-projection expert A/B
  tensor and one `token_router.*` state per adapted transformer block. The
  config records the exact target-layer manifest and frozen `base_ref`.
- Formats v1/v2 retain legacy per-projection gates and v2 shared-prompt router
  state. Missing router-type fields continue to default to the legacy layout.
- Loading derives the expected tensor set from the config and injected model,
  then rejects missing, extra, wrong-shaped, or wrong-base artifacts before
  copying any weights.

### Adaptive surprise occupancy (format v4)

Adaptive MoLE treats expert index `e` as one global lane across every adapted
projection and transformer block. One global active set therefore bounds real
incremental weight occupancy; block-local routers still learn different token
weights inside that set. The architecture solver measures the frozen base and
selected projections, then jointly resolves integer expert count `E` and rank
`r` against an exact total-parameter budget. The default target is 1.25× the
base with one lane near 0.5% of the resolved total. Active bounds are derived
from the achieved (not requested) lane fraction, so rounding can never exceed
the 10% ceiling.

Each block predicts a positive per-lane NLL with
`softplus(surprise_bias - route_utility)`. Language-model loss learns utility;
a smooth-L1 calibration objective anchors route-weighted predictions to the
realized token NLL in nats. Training rotates exploration lanes, samples nested
K widths, and globally ranks the next active pool by predicted NLL. This gives
every disk-resident lane training opportunities without hand-partitioning the
corpus.

Adaptive execution packs only the active `K × r` A/B factors and performs two
GEMMs. It does not evaluate or build activations for the complete bank. Eval
packs are cached until K changes; training rebuilds the pack so gradients reach
live parameters. This is distinct from v3's compatibility path, whose packed
width remains `E × r` even though its mathematical route is top-K.

At inference, the base and predicted-NLL heads load first while expert
parameters remain on `meta`. A bounded bootstrap prompt pass ranks all lanes;
the runtime then loads a request-fixed, memory-capped pool and starts at the 1%
minimum. Teacher-forced prompt NLL may propose another lane and recompute from
scratch with `use_cache=False`; worsening or sub-knee marginal gains roll back.
Once decode starts, residency cannot change. Predictive entropy may adjust K
for the next token only, with hysteresis, while transfers remain exactly zero.
Telemetry reports NLL versus entropy explicitly, occupancy, residency, caps,
recomputations, lane loads, and H2D bytes. The frozen base always executes, so
the displayed 1–10% is incremental expert-parameter occupancy, not total model
FLOPs or physical byte percentage.

## MoLE forward

For token `t`, transformer block `l` computes one sparse route and reuses it
for every adapted projection `p` in that block:

```
(I_l,t, g_l,t) = topK(router_l(h_l,t))
y_l,p,t = W_l,p,frozen(x_l,p,t)
          + Σ_{e∈I_l,t} g_l,t,e · B_l,p,e A_l,p,e x_l,p,t
```

This is the default `router_type="per_token"` architecture:

- One small E-way router is registered on each adapted transformer block. Its
  `[tokens, top_k]` route is shared by q/k/v/o and MLP projections even though
  those projections have different input widths. A token may choose a
  different route in the next block, matching conventional layer-local MoE
  semantics while keeping each expert slot coherent within a block.
- All LoRA experts and block routers train jointly on the same shuffled mixed
  corpus. The optional domain selector is a single global data filter; it
  never assigns a corpus, domain, or document shard to a particular expert.
  Switch-style auxiliary loss encourages balanced token usage.
- A tiny asymmetric router initialization prevents every token from selecting
  the same experts at step zero. Top-1 uses a unit-forward straight-through
  weight so task loss still reaches the gate; top-K routing remains sparse in
  the forward semantics.
- Expert matrices keep their ordinary per-expert parameters and artifact keys,
  but execution concatenates them in rank space. The first GEMM produces
  `[tokens, experts, rank]`, the sparse gate is applied between A and B, and a
  second GEMM reduces the packed rack. This removes per-expert Python launches,
  CUDA scalar readbacks, and the much larger
  `tokens × experts × output_width` activation.
- Packed GEMM width and expert parameter count scale with `E × r`. E8/r8 and
  E4/r16 therefore have equal LoRA parameter/GEMM width. With the default k=2,
  their active mathematical ranks per token are 16 and 32 respectively: E8/r8
  trades per-route rank for twice as many specialization slots. E8/r8/k2 is
  the default; racks up to 32 experts are available as an advanced tradeoff.
- Version-3 artifacts store the complete expert bank plus each block router.
  Version-1/2 per-projection and shared-prompt artifacts remain strict-load
  compatible. Legacy sequential training still forces one coherent expert
  across the model, and the optional legacy stage 2 can replace that bank's
  zero gates with one pooled-prompt router held fixed during cached generation.
- Quantization is stored in `checkpoints.quant_locked` and committed only after
  the complete adapter finishes successfully, preventing both cross-adapter
  drift and failed runs from poisoning the checkpoint.

## Base-model training contract (`train/base_config.py`)

Base training has two initialization modes with intentionally different
contracts:

| Mode | Weight source | Architecture | Supported update methods |
|---|---|---|---|
| `scratch` | Random initialization on the destination device | A complete, validated yyLab architecture recipe plus a selected tokenizer | Full-parameter training only |
| `pretrained` | An existing local or Hugging Face checkpoint | Inherited from that checkpoint | Full continued training, LoRA, or QLoRA |

The version-1 base-job shape is still accepted as `pretrained`, so already
queued continued-pretraining jobs replay with their historical meaning. A
scratch job never loads model weights from the tokenizer source. It loads only
the tokenizer, constructs a Hugging Face `Qwen3Config`, `LlamaConfig`, or
`BitNetConfig`, and calls `AutoModelForCausalLM.from_config` directly in the
requested parameter dtype and on the destination device. This avoids creating
a second full-sized host model during random initialization.

### Scratch architecture and native context

`ScratchModelSpec` exposes the behavior-bearing decoder dimensions rather than
hiding them behind a few size presets:

- family (`qwen3`, `llama`, or `bitnet`), tokenizer-derived or explicit
  vocabulary size, and vocabulary padding multiple;
- hidden and intermediate widths, layer count, attention heads, KV heads, and
  head dimension;
- native `max_position_embeddings` and RoPE theta;
- tied/untied token embeddings, initializer range, RMSNorm epsilon, attention
  dropout, and attention bias.

Native context is a model architecture property. The run's `seq_len` is the
actual training sequence length and may be shorter, but it may not exceed the
native context. Changing native context therefore changes the saved base
contract; it is not silently treated as an inference-only YaRN override.
Validation also requires `hidden_size = attention_heads × head_dim`, attention
heads divisible by KV heads, kernel-aligned dimensions, a tokenizer EOS token,
and a model vocabulary no smaller than the tokenizer. The vocabulary is padded
up to the requested multiple and that resolved value is saved.

The parameter calculator is exact for yyLab's supported Qwen3, Llama, and
BitNet Hugging Face configurations, including tied versus untied embeddings
and family-specific norms. The target-parameter solver changes only the number
of transformer layers (within its configured bounds), rounds to the nearest
achievable layer count, and reports the resulting exact parameter total. It
does not imply that every arbitrary target can be hit exactly or silently
change width, heads, vocabulary, or context to manufacture a match.

### Parameter state, QAT numerics, and optimizer state

The recipe keeps independent axes that are often incorrectly collapsed into a
single "precision" selector:

- `parameter_dtype` chooses the stored/trainable shadow parameters: BF16,
  FP16, or FP32. FP16 scratch training requires CUDA.
- `precision.qat` chooses dense, INT4, NVFP4, or ternary forward numerics.
  INT4 uses grouped symmetric fake quantization. On a supported Blackwell CUDA
  stack, NVFP4 quantizes weights and activations into packed E2M1 payloads with
  E4M3 scales over groups of 16 and dispatches native block-scaled FP4 forward
  GEMMs through TorchAO. Ternary uses BitNet W1.58 weights and INT8 activation
  QAT and is rejected for Qwen3/Llama; BitNet rejects INT4/NVFP4.
- `precision.qat_backend` chooses automatic, required-native, or explicitly
  emulated execution. A required-native request fails preflight if the backend
  is unavailable. NVFP4 forces matching NVFP4 activation quantization, while
  ternary uses the BitNet W1.58/A8 contract.
- `precision.sparsity=paired_4:8` retains two complete two-value pairs from
  each group of eight along weight K. This is NVIDIA's E2M1 sparse-MMA pattern;
  arbitrary scalar 2:4 masks are not compatible with sparse NVFP4 hardware.
- `precision.sparsity_backend` independently chooses automatic, required
  compressed cuSPARSELt execution, or dense masked execution. The compressed
  SM120 lane uses group-32 UE4M3 scales (not TorchAO dense NVFP4's group-16
  payload) and fails a required request unless a real GEMM smoke succeeds.
- `precision.backward_precision` selects BF16 or MXFP8 dgrad/wgrad for native
  NVFP4 autograd. MXFP8 uses E4M3 elements with an E8M0 scale per 32 values;
  `precision.backward_backend` selects native Tensor Core execution or
  format-faithful emulation. Native status requires both gradient GEMMs to run.
- optimizer implementation, state placement (`gpu`, `cpu_paged`, or full
  `cpu_offload`), learning rate, weight decay, betas, epsilon, scheduler, and
  warmup are explicit. Gradient accumulation, clipping/checkpointing, TF32,
  compile mode, cuDNN-priority SDPA, and Liger kernels remain separate runtime
  controls.
- `optimizer.master_weight_storage` selects an FP32 CPU master, the legacy
  dense-residual potentiated E4M3 representation, or adaptive E4M3
  potentiation. Adaptive state stores one real E4M3 byte per weight, one
  power-of-two scale and hysteresis counter per 32-value block, and sparse
  block indices/residual rows only for promoted regions. A block is promoted
  when normalized reconstruction and lost-update error identify it as
  troublesome, capped by `adaptive_max_promoted_fraction`; a lower threshold
  and patience counter demote it after stabilization. Accelerator parameters
  remain BF16 and Adam moments remain FP32.

All QAT modes retain dense BF16 shadow parameters for updates and checkpointing.
Optimizer updates use either those parameters directly or the independently selected
CPU master representation.
INT4 and ternary use yyLab's straight-through fake-quant `QATLinear`. Native
NVFP4 packs each forward's activations and effective weights and invokes
PyTorch's SM120 scaled FP4 GEMM. Compressed NVFP4 instead calls cuSPARSELt 0.9.1
directly with a paired-4:8 E2M1 weight, UE4M3 scales, FP32 accumulation, and BF16
output. Its proven padding contract is M×16, N×32, K×64; yyLab pads and slices
arbitrary Linear shapes around that contract. Custom autograd then computes
dgrad and wgrad in the requested BF16 or MXFP8 lane. The LM head and embeddings
remain high precision. Ordinary AdamW moments remain high precision as well;
selecting an 8-bit, paged, or CPU-offloaded optimizer is independent.

The paired 4:8 recipe applies a dynamic magnitude decision and uses STE
gradients so masked pairs may regrow. With `sparsity_backend=compressed`, that
effective weight is physically compressed for the forward GEMM and metadata
records `native_sparse_gemm=true`. The trainable checkpoint deliberately still
stores BF16 shadow weights, so `compressed_sparse_checkpoint=false`; deployment
packing remains a separate export contract. Dense-masked execution stays
available for portability and saved native recipes may use it when loaded on a
non-native device without rewriting recipe provenance.

### Recipe identity, provenance, and reload

The complete launch recipe is stored in the run's immutable `config_json` and
written alongside every full base checkpoint as `yylab_recipe.json`. The saved
Hugging Face config embeds the resolved scratch initialization, architecture,
precision, training sequence length, and seed under
`yylab_training_recipe`. A finished checkpoint's registry metadata records the
recipe version, scratch/continued mode, parent checkpoint when applicable,
tokenizer source, resolved architecture, precision contract, sequence length,
exact parameter count, and local artifact path. Model weights, tokenizer
assets, and these receipts are saved together; a failed job removes only its
still-pending registry output.

QAT checkpoints intentionally contain high-precision shadow tensors plus a
`yylab_low_precision` metadata block in `config.json`. On reload, yyLab reads
that block before loading weights, refuses to stack bitsandbytes NF4/INT8 over
the QAT base, loads the saved BF16/FP16/FP32 shadow checkpoint, and then
reapplies the recorded `QATLinear` format, backend, and sparsity policy to the
transformer projections. This restores the trained forward numerics without
changing state-dict keys. A generic Hugging Face loader that ignores yyLab
metadata will see only the high-precision shadow model; packed deployment or
compressed sparse export is a separate artifact contract and is not implied by
the QAT checkpoint.

## Training auto-selection (`train/planner.py`)

Inputs: model params and decoder shapes, free VRAM/RAM, sequence length,
method/quant preference, fused-kernel availability, attention shape, and GPU
compute capability. For joint token routing the planner includes all `E × r`
expert parameters, one FP32 E-way gate per transformer block, gradients, and
optimizer state. It tries:

1. LoRA over a BF16 base
2. QLoRA over an NF4/INT8 base

Both use gradient checkpointing. A compatible Qwen3 grouped-query model on
CUDA evaluates microbatches 16 → 8 → 4 → 2 → 1 and selects the largest peak
projection inside an 88%-of-free-VRAM ceiling. Those power-of-two candidates
divide a fixed effective batch of 16 sequences, so accumulation falls as the
resident batch grows without changing tokens per optimizer update or epoch
step count. Other paths retain the conservative batch 2 → 1 profile.
Small racks use fused AdamW on the GPU. `optimizer_state="auto"` switches racks
above the validated fused-optimizer threshold to bitsandbytes
`PagedAdamW8bit`, whose compact unified-memory state can page to CPU RAM; the
UI also exposes explicit GPU and paged-state choices. A full CPU-offload option
keeps master parameters and both Adam moments in system RAM, copying gradients
down and updated parameters back once per optimizer step. FP32 master storage
costs four bytes per parameter. The legacy potentiated-FP8 alternative costs
about 1.625 bytes per parameter because every value owns a residual nibble.
Adaptive FP8 starts near 1.15625 bytes per parameter (one E4M3 byte plus
scale/counter overhead) and adds 20 bytes per promoted 32-value block for its
index and packed residual row. At the default 10% block cap, its upper bound is
about 1.21875 bytes per parameter. This is persistent master-state compression,
not a direct float8 AdamW; optimizer arithmetic and moments remain FP32.
That is the memory-saving part of ZeRO-Offload on this single-GPU system; there
is no useful data-parallel state partition to perform with world size one. The
planner accounts for the reduced VRAM and reports the corresponding RAM state.
Legacy stage-2 prompt-router training uses the same guarded planner.

## Corpora intake pipeline

1. Walk folder (skip hidden/`.git`/`node_modules`, size cap), dispatch each
   file to the first adapter that claims it (by extension).
2. Extract `Doc(text, source, domain)` → NFKC normalize + length/printable
   filters.
3. Exact dedup (sha256, per-run set + optional global `doc_hashes` table).
4. Optional MinHash LSH near-dedup (threshold 0.8, word 5-grams).
5. Domain: adapter hint if trusted (code/web/math/science), else content
   heuristics.
6. Stream to zstd JSONL shards (20k docs each).
7. Token counts per enabled tokenizer, per domain + totals → `stats.json`.

### Adding a filetype adapter

Create `yylab/ingest/adapters/<name>.py`:

```python
from ..records import Doc
from .base import Adapter


class PdfAdapter(Adapter):
    name = "pdf"
    extensions = {".pdf"}
    default_domain = "other"

    def extract(self, path):
        ...
        yield Doc(text=..., source=str(path), domain=self.default_domain)
```

Then register the class in `adapters/loader.py::_ADAPTER_CLASSES`.

## Engine (context/KV) settings

`settings.json → engine`: attention impl (SDPA on Windows), KV cache
(dynamic / quantized / quantized-offloaded / ordinary offloaded /
sliding-window / static), KV quant (int4/int8), max context, RoPE scaling, and
adaptive context-selection thresholds. Consumed by the inference engine
(phase 4) and used as training context defaults.

## Training runtime (phase 2)

Training runs in detached subprocess workers (`python -m yylab.workers.train
--run <id>`), launched from the UI via `workers/launch.py`. Workers stream to
`runs/<id>/train.log`, write `progress.json` (live step plus typed metric) and
`metrics.jsonl`, and update the `runs` table. The Runs page keeps one
client-scoped poller: active runs update every second, while inactive selected
runs back off to five seconds. Log reads track a file offset, append only new
bytes to the terminal, and recover from truncation/replacement with a bounded
tail.

Run kinds:

- `mole_stage1` — the retained run-kind name now dispatches by the strategy in
  its config. Legacy fixed-width runs use `joint_per_token_v1`: load a frozen NF4/INT8/BF16
  base → inject the full expert rack and one router per transformer block →
  jointly optimize experts and routers over one mixed corpus stream → atomically
  save a complete version-3 adapter. Quant lock is committed only after the
  artifact succeeds. A config with no strategy retains the historical
  `sequential_single_resident_v1` behavior, so queued old runs still replay
  deterministically: one forced expert and data slice at a time, followed by
  atomic bank assembly.
  New adaptive runs use `joint_adaptive_surprise_v1`: solve the exact E/r lane
  budget, train globally coherent nested active sets plus calibrated surprise
  heads, and save a sharded version-4 bank. All expert weights are present
  during training, but only the current global K participates in forward and
  backward; paged or explicit CPU-offloaded optimizer state supplies the
  memory-oriented path for the larger disk bank.
- `router_stage2` — compatibility workflow for legacy isolated banks only:
  load base + adapter weights → freeze experts → train one shared prompt router
  (top-K may be changed) → save a version-2 snapshot to `routers/<id>/`. Joint
  per-token adapters already contain routers and are rejected by this worker.
- `base` — either construct a validated Qwen3/Llama/BitNet base from random
  weights and train all parameters, or continue an existing checkpoint. Full
  runs save a new checkpoint row plus exact recipe/provenance; continued
  LoRA/QLoRA uses PEFT (`target_modules="all-linear"`) and saves an adapter row.
  Scratch mode rejects adapter-only methods. Base recipes accept an ordered
  list of corpora and persist that selection, the active domain filter, and the
  corpus-mix policy in run and artifact provenance. New runs use deterministic
  document-level round-robin mixing; legacy configs that omit `corpus_mix`
  retain their historical concatenate replay. Saved QAT bases carry
  reloadable fake-quant metadata and high-precision shadow tensors.
- `benchmark_quality` — load a fresh, worker-local model stack and run the
  frozen MoLoR core suite using base, dynamic learned routing, one forced
  expert, or a rack matrix. It never mutates adapter state.
- `benchmark_train_perf` — inject an ephemeral per-token rack over synthetic
  IDs and time complete optimizer steps after warmup. It never writes an
  adapter artifact or silently resizes an OOM candidate.

Data path: zstd shards → domain filter → corpus mixing/shuffling → tokenize →
EOS-separated packing into fixed-length sequences → micro batches
(`train/data.py`). For new multi-corpus base runs, each corpus has an isolated
bounded shuffle and the mixer visits one eligible document from each
still-active corpus in deterministic selection order, then continues with the
remaining corpora as shorter sources exhaust; therefore a complete epoch still
consumes every eligible document exactly once. Steps prefer the intake
tokenizer counts matching the model family (for example Qwen 2.5 counts for
Qwen2/Qwen3), with a per-domain character fallback for old or incomplete
corpus statistics. The loop stops early when packed data is exhausted.

Checkpoint loading streams weights with `low_cpu_mem_usage=True`; shuffle is
bounded at 128 documents / 64 MB, raw text is chunked at 64K characters, and
tokenization batches at most 8 chunks. Raw tokenizer encodings are never
truncated; packing consumes every ID and only fixed-length sequences reach the
model. Qwen3
training applies Liger RoPE, RMSNorm, SwiGLU, and fused linear cross entropy, so
logits with shape `batch × sequence × vocabulary` are never materialized. On
validated grouped-query shapes, a scoped SDPA policy prioritizes cuDNN while
retaining Flash, memory-efficient, and math fallbacks; it stays active through
backward so checkpoint recomputation uses the same backend. Small adapter/router
parameter sets use fused AdamW. Larger racks automatically use paged 8-bit
optimizer state backed by unified memory, allowing pressure to spill to CPU RAM
without loading another expert bank; full/other large fine-tunes retain the
memory-oriented 8-bit fallback. Explicit `cpu_offload` is slower but keeps all
FP32 master/moment state in RAM; a real E8/r8/k2 CUDA smoke step held 0.454 GB
of optimizer tensors on CPU and verified that none of those state tensors
resided on the GPU.

For the local Qwen3-0.6B shape with all seven target projections, E8/r8 has
40,370,176 expert parameters plus 229,376 block-router parameters. E4/r16 has
the same 40,370,176 expert parameters and 114,688 router parameters. The local
RTX 5070 planner now selects BF16 batch-8 × accumulation-2 for the default
E8/r8/k2, sequence-2048 fast path: 6.6 GB projected peak within its free-VRAM
reserve while preserving the former 32,768 tokens per optimizer update. Batch
16 remains a benchmark-first option because its conservative projection is too
close to WDDM and external-application variability.

A real E8/r8/k2 batch-4, sequence-2048 forward/backward/optimizer smoke step on
that RTX 5070 adapted 196 projections through exactly 28 block routers. It used
2.699 GB peak allocated / 3.074 GB reserved, completed 8,192 tokens in 2.848 s,
and reported normalized route entropy 0.9912 with no dead experts. Auto selected
fused AdamW because the 40,599,552 trainable parameters remain below the paged
threshold.

The former token dispatcher looped over experts and branched on a CUDA scalar
inside every projection. A BF16 RTX 5070 microbenchmark at 4,096 tokens and
1,024-wide projections found the packed rank-space path 4.43× faster for
E4/r16 while reducing peak allocation from 104.4 to 73.3 MiB. Capacity-matched
E8/r8 was 8.13× faster than the old loop, and E32/r2 was 26.86× faster. Output
and gradient equivalence, route reuse across different projection widths,
top-1 gate gradients, stale-route rejection, and 32-expert execution are
covered by CPU tests. Whole-model gains are smaller because frozen base-model
compute still dominates.

The compatibility paths retain their earlier hardware evidence. On the same
RTX 5070, isolated rank-16 Qwen3-0.6B training at sequence 2048 used 2.09 GB
peak allocated / 2.43 GB reserved for a real forward, backward, and fused-AdamW
step; cuDNN-priority SDPA plus full Liger kernels raised isolated throughput
from about 1,983 to 7,150 tokens/s. Legacy shared-prompt dispatch improved a
representative four-expert layer from 6.14 to 0.82 ms and a sixteen-expert
layer from 8.28 to 0.80 ms with equivalent outputs and gradients. An integrated
legacy batch-4 router step used 2.47 GB allocated / 2.77 GB reserved. The older
RTX 2050 (4 GB) validation remains Qwen2.5-0.5B-Instruct NF4 with sequential
E4/r8/k2 expert training followed by shared-prompt stage 2.

## Benchmarks

The Benchmarks page uses the same detached worker, SQLite run history, live
progress, stop control, and incremental log tail as training. Launch is
refused while another run is queued or active so timing and OOM results cannot
be corrupted by a competing yyLab GPU job.

The bundled `molor-core-v1` resource is content-identical to the MoLoR suite
(the canonical JSON fingerprint is unchanged):
smoke 10, quick 30, full 64 across ten weighted categories. Its canonical
SHA-256 is stored with each run. Quality output under `runs/<id>/` is
self-contained: `suite.snapshot.json`, `stack.lock.json`, `run.json`,
append-only `results.jsonl`, `summary.json`, `scores.csv`, `report.md`, and an
optional `comparison.json`. Each result cell binds the suite, complete local
checkpoint/adapter byte hashes, exact generation request, policy, case, and
repetition. Successful cells resume without regeneration; suite, workload,
scorer, or model-byte drift fails closed.

Evaluation has two additive views. `score` remains the frozen Molor v1.2
compatibility result, including its exact-output envelopes. `semantic_score`
answers whether the requested value is supported by the visible completion, so
a correct answer followed by an explanation is not classified as a capability
regression. Numeric conclusions outrank intermediate values, unfinished
expressions are ambiguous, and complete or unfinished `<think>` blocks are
excluded. The UI and baseline regression flag lead with the semantic index and
retain Molor strict alongside it for instruction-format diagnostics. Existing
append-only generations can be evaluated without inference into fingerprinted
`semantic-summary.json`, `semantic-scores.csv`, and `semantic-report.md`
projections. Thinking is off by default to preserve the frozen suite workload.
When selected, the runner records separate hidden-reasoning and final-answer
budgets and uses seeded Qwen thinking-mode sampling. Reaching the reasoning
limit closes that private phase and starts a direct-answer turn with the complete
answer allowance; those reasoning tokens never consume `max_new_tokens`. The
backend pre-opens `<think>` when an adapter omits it and uses the same direct
answer turn when hidden reasoning ends at EOS or closes without an answer.
Prompt, reasoning, continuation, and answer must still fit the checkpoint's
physical `max_position_embeddings` context window; the resolved policy is part
of the workload contract. A generation that nevertheless produces no visible
answer is excluded from scoring and fails the run immediately rather than
appearing as a semantic or strict zero.

Quality benchmarks default to the checkpoint's native context. Explicit YaRN
2× (65,536) and 4× (131,072) modes update both the Qwen RoPE parameters and
`max_position_embeddings` at load time, and the mode is included in baseline
comparability. Static YaRN remains opt-in because it can reduce quality on
short-context prompts.

Because learned per-token experts are anonymous, the yyLab matrix intentionally
does not reuse MoLoR's named-domain route accuracy. It measures base, every
forced expert slot, and the configured dynamic router. The best forced expert
per case forms both semantic and Molor-strict oracles; each oracle's Quality
Index minus the dynamic policy's matching index is reported as router regret.
Forward hooks accumulate token selections and
weights over prompt prefill and every decoding step instead of reporting only
the final cached route.

The training-performance lane exercises the same packed expert rack, shared
block routers, auxiliary loss, gradient checkpointing, Liger fused loss,
cuDNN-priority SDPA, and optimizer-state choices as joint training. Warmup is
excluded and each sample spans data transfer, forward, backward, clipping,
optimizer update, and zeroing. The receipt preserves requested E/r/k,
sequence, batch/accumulation, quantization, and GPU state; steady metrics cover
useful tokens/s, step median/P95, peak CUDA allocation/reservation, process
memory, loss/parameter finiteness, and final route balance. On the local RTX
5070, a BF16 E4/r4/k2 sequence-256 functional smoke used 1.252 GB peak
allocated / 1.271 GB reserved, with finite loss and parameters and no dead
experts. Concurrent desktop GPU load was present, so its throughput is not
recorded as a reference result; the UI's default idle-GPU preflight prevents
that kind of contaminated timing run.

## Inference engine + chat (phase 3)

`yylab/infer/engine.py` holds a singleton `ENGINE` in the GUI process:

- `LoadSpec` = checkpoint ref + quant (adapter quant / checkpoint quant-lock /
  auto) + optional MoLE adapter dir + optional legacy router snapshot dir +
  engine settings. Version-3 adapters include their learned block routers;
  version-1/2 banks may use a separate prompt-router snapshot.
- Version-4 adapters meta-inject their full lane shapes, strictly load only
  routing tensors, and materialize the request-fixed expert pool from verified
  lane shards. Generation calls are serialized because occupancy and route
  caches are model-attached mutable state.
- Load path: `models/loading.load_base` (NF4/INT8 via bitsandbytes, SDPA
  attention, optional RoPE scaling) → inject MoLE + apply safetensors →
  `set_stage("none")` → eval. PEFT path for LoRA adapters.
- Version-3 generation computes a fresh route for each current token at each
  transformer block; cached past tokens already contain their routed K/V state.
  Legacy shared-prompt generation instead pins its first pooled-prompt decision
  across the cached sequence.
- Version-4 prompt preparation uses known-token NLL for expansion and safe
  recomputation. Free generation has no known next-token NLL, so a logits
  processor measures predictive entropy and applies any K change one forward
  later. It never rewrites past KV state or transfers a lane after decode
  begins.
- Engine settings from `settings.json`: attention impl, KV cache
  (dynamic / quantized / quantized-offloaded / ordinary offloaded /
  sliding-window / static), max context budget, RoPE factor, and adaptive
  selection policy. Stock cache kwargs are passed to `generate()`; yyLab's
  quantized-offload object is passed as `past_key_values`. Either path removes
  the failed cache and retries once with a fresh dynamic cache.
- `QuantizedOffloadedCache` fills the composition missing from Transformers'
  stock cache matrix. Keys and values are symmetrically group-quantized on the
  source device before PCIe transfer. INT8 stores one signed byte per value;
  INT4 packs two signed nibbles per byte. FP16 group scales and packed values
  live in amortized, optionally pinned CPU buffers. During attention, only one
  layer's history is transferred and dequantized on the model device. A
  configurable newest-token residual may stay in model dtype. This is a
  full-attention, dynamic, non-compileable cache: it reduces VRAM and host RAM,
  but transfers/dequantizes the active layer's complete history each decode
  step. Hybrid/sliding/linear cache layouts fail closed rather than silently
  becoming full precision.
- Chat load presets resolve native context from the checkpoint or apply static
  YaRN at 2×/4× the trained RoPE length. Before each send, tokenizer-exact
  planning reserves the full answer allowance and always protects every
  system/developer message, the current turn, and a configurable recent tail.
  The `nll_atxy` policy ranks older complete turns cheaply with factorized
  address `A=(T,X,Y)`, lexical overlap, and recency, then evaluates at most a
  bounded shortlist. A candidate is accepted only when its marginal query-NLL
  reduction exceeds both the configured knee and per-token cost; direct query
  echoes are rejected. Query NLL is explicitly an inference-time `R_x` proxy,
  not the answer-side `R_y` teacher used by the ATXY experiments (the correct
  answer is unavailable during live retrieval). ATXY currently uses optional
  exact metadata; automated address assignment and a saved learned expected-
  `R_y` router remain separate scale-up work.
- Selected turns are restored to chronological order and the prompt is rebuilt
  once at contiguous logical positions before prefill. NLL/ATXY therefore
  changes which spans consume the positional budget, not RoPE phases. YaRN
  remains the representation for the final packed length, and cached keys never
  change position during decode. The UI reports prompt/answer budgets,
  evaluated and accepted turns, ATXY candidates, marginal NLL, and any safe
  recency fallback.
- Streaming: `TextIteratorStreamer` in a thread; chat page drains a queue via
  `ui.timer` into message bubbles. Chat template via
  `apply_chat_template` with a plain-text fallback.
- Stop: chat Stop calls `ENGINE.cancel()`, which sets an event checked by a
  `StoppingCriteria` so `generate()` halts on the next token (frees the GPU
  instead of just abandoning the stream).
- KV-cache modes are validated on-device; a failing cache falls back to dynamic
  automatically. Capacity estimates include packed int4/int8 CPU-offload rows
  and their FP16 group-scale overhead.

Validated legacy path on RTX 2050: Qwen2.5-0.5B NF4 + trained MoLE adapter +
router snapshot → coherent streaming replies. Version-3 save/load and inference
hooks use the same strict tensor manifest exercised by training round-trip
tests.

## RL export (phase 4)

`yylab/export/merge.py` produces standalone HF checkpoints when the adapter's
behavior can be represented by one static set of weights:

- **LoRA adapters**: PEFT `merge_and_unload` over the full-precision base.
- **Legacy/static MoLE adapters**: base loaded full-precision on CPU → inject +
  apply adapter (or prompt-router snapshot) → calibrate per-layer usage on
  packed corpus sequences → `materialize_mole` folds
  `Σ_e w_e · scaling · (B_e A_e)` into each base Linear → `save_pretrained`.
  This is exact only when routing is input-independent and the calibrated
  mixture is the actual mixture; otherwise it is explicitly a static
  approximation.
- **Version-3 per-token MoLE adapters**: one dense matrix cannot preserve a
  route that changes by token and block. The lossless artifact is
  `mole_weights.safetensors` + `mole_config.json`, loaded with
  `mole.store.apply_mole_weights` after injection. Any future calibrated dense
  conversion must be labelled as lossy and must average the actual sparse
  top-K routes rather than silently substituting a uniform mixture.
- **Version-4 adaptive MoLE adapters** have the same lossless-native-only
  contract plus a sharded expert bank and runtime occupancy policy. A future
  static export must name and calibrate a fixed resident/active policy; it
  cannot silently discard the surprise controller.

Compatible adapters are exported from Checkpoints → adapter row → "Export
merged"; the detached `export` worker registers the result as a new checkpoint.
The native version-3 adapter is already the portable dynamic-routing artifact.

GGUF conversion stays external (llama.cpp's convert script consumes the
merged HF directory directly).

## Phases

- [x] 1. Scaffold, artifact store, corpora intake + UI, planner, MoLE core
- [x] 2. Training runtime (joint per-token MoLE, legacy router stage 2,
  base training)
- [x] 3. Inference engine + chat (base + adapter + router, KV settings)
- [x] 4. RL export (static-compatible merged checkpoints + native v3 artifacts)
