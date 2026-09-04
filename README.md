# yyLab

Local ML lab GUI for consumer GPUs (RTX A4000 / 5070 / 2050 class):
corpus engineering, checkpoint management, and
**Mixture-of-LoRA-Experts (MoLE / MoQLE)** adapter + router training over
frozen bases, with chat inference.

## Quickstart

```powershell
uv sync
uv run yylab
```

Opens at http://127.0.0.1:8080 (`--port`, `--no-browser` supported).
Artifacts live in `./artifacts` (override with `YYLAB_HOME`).

## Features

- **Corpora intake** — point at a folder; adapter plugins per filetype,
  streaming JSONL/Parquet normalization, exact + bounded-memory MinHash dedup,
  domain tagging, zstd JSONL shards,
  token counts across GPT-2/tiktoken, Llama 3.x, Qwen 2.5, Phi-3, Gemma,
  Mistral tokenizers; mixture charts + live stats table. JSON-encoded message
  columns are retained structurally, accepted-document token counts are
  accumulated inline, and compressed output is not reread after ingest. The
  dataset-library button uses a preservation-oriented fast path for published
  corpora: canonical global dedup only, with MinHash/language/quality cleaning
  reserved for the explicit ingest dialog. When enabled, MinHash retains the
  same 9×13 band decision in a compact fingerprint index rather than keeping
  document keys and candidate sets for every row. Extraction and analysis run
  across parallel worker processes (large JSONL and Parquet files are split
  into chunks) while one coordinator decides in file order, so the output is
  identical to a single-worker run; the worker count is per run or a setting.
  Every extracted document is kept in a per-corpus prepared cache so
  **Re-import** applies changed policies (accepted file types, selection
  profile, language/quality/MinHash cleaning, tokenizers) to an existing corpus
  without extracting unchanged files again; the corpus keeps its id and stays
  usable until the new output is published.
- **Human-trained corpus ranker** — every review swipe is committed immediately
  with its text, provenance, queue component, and model score. A standardized
  logistic Stage-1 model trains from the unique human labels and reports strict
  leave-one-corpus-out metrics. Auto review queues mix 70% uncertainty, 20%
  exact random exploration, and 10% confidence audits while skipping previously
  reviewed documents. The model prioritizes review only; it never rejects or
  quarantines ingest data automatically.
- **Checkpoint registry** — local folders (param count scanned from
  safetensors headers) or HF hub ids; per-checkpoint quant lock.
- **Raw or continued base training** — start from random weights with a Qwen3,
  Llama, or BitNet b1.58 decoder, or continue from an existing checkpoint.
  Scratch recipes expose vocabulary sizing/padding, width, MLP width, layers,
  attention and KV heads, head dimension, native context, RoPE theta,
  embeddings, initialization, normalization, and attention controls. Native
  context and the shorter sequence length used by a particular run are
  independent. An exact parameter calculator and deterministic shape solver
  can vary width, MLP width, attention layout, and depth for genuinely tiny
  models. Targets are continuous from 0.100M upward rather than limited to
  presets, with an exact integer override for non-round counts. It never
  shrinks below the tokenizer vocabulary and reports the
  irreducible embedding floor. Scratch runs may reuse an existing tokenizer or
  train a configurable compact byte-level BPE tokenizer from the selected
  corpora; 2048 entries is a default, not a fixed size, and a ~1M preset is only
  a shortcut.
  Scratch runs are full-parameter runs,
  while continued runs retain full, LoRA, and QLoRA choices. A base run can
  select multiple ordered corpora; new recipes mix their eligible documents by
  deterministic round robin while still consuming every document in each
  epoch, and persist the selected sources and mix provenance with the run. The
  UI defaults source splits to `train + unspecified`, retaining legacy corpora
  while excluding explicit validation/test shards unless selected.
  Runs may instead request an explicit training-token budget; yyLab rounds it
  up to a whole optimizer step, persists requested and scheduled token totals,
  and keeps legacy explicit-step and epoch-derived caps replayable.
- **Tokenizer laboratory** — train reusable byte-level BPE artifacts directly
  from any corpus/split/domain selection without materializing the corpus in
  RAM, or import an HF tokenizer directory, raw `tokenizer.json`, or Hugging
  Face hub repo id into an independent self-contained copy (presets cover
  o200k via gpt-oss, Qwen 3, Qwen 2.5, Llama 3.2, and gated Gemma 3, and the
  SPALMER page offers the same presets). Vocabulary size is arbitrary above the
  chosen byte-coverage floor: 257 for minimal causal, 264 for built-in chat
  roles, or 256 plus a custom special-token list. No normalization (exact byte
  round trip) is the default; NFKC and NFKC+lowercase are explicit options.
  Detached runs report held-out or deterministic-fallback bytes/token,
  vocabulary utilization, round-trip failures, checksums, and corpus
  provenance. Ready artifacts have stable registry IDs for later model recipes.
- **Native SPALMER experiments** — plan and queue scratch 10M, 50M, 100M, or
  exact custom-size SPALMER checkpoints from one or more yyLab corpora while
  reusing any registered or compatible Hugging Face tokenizer. The worker
  preserves the architecture's QR-PLE, 3:1 KDA/MLA cycle, directional branch,
  optional ATXY component, and 200 coherent micro-experts. Expert precision is
  selected as an exact capability-verified pair: the integrated providers are
  NVFP4/NVFP4 and MXFP8/MXFP8 dense per-expert CUDA, plus grouped BF16/BF16.
  Unsupported pairs are omitted and fail closed; there is no fake/reference
  fallback. Trainable masters and gradients remain BF16, Adam moments remain
  FP32 on CPU, and FLA supplies the optimized CUDA KDA path. Executable native
  `.pt` bundles are registered automatically and can be loaded directly in
  Chat or benchmarked. Legacy emulated bundles remain metadata-scannable but
  cannot load. CUDA inference defaults to the
  checkpoint's trained full-pool top-k while independently paging selected
  expert rows through bounded per-layer caches; the older prompt-global
  adaptive-residency controller remains an explicit experimental comparison.
  HF LoRA/MoLE adapters and RoPE/YaRN are intentionally not applied to this
  NoPE recurrent architecture.
- **Explicit base-training precision** — parameter dtype, QAT numerics,
  activation quantization, optimizer implementation/state placement, AdamW
  constants, scheduler/warmup, batching, checkpointing, compile, and fused
  kernels are separate recipe axes. Base training exposes dense BF16/FP16/FP32
  plus NVFP4 only when its native Blackwell path is available; unsupported
  low-precision modes fail closed instead of being emulated. NVFP4 uses
  TorchAO's packed E2M1 values, E4M3 block scales, and native block-scaled FP4
  forward GEMM. Each native precision choice is exposed only after a cached,
  bounded forward/backward kernel smoke succeeds on the active GPU. The SM120
  sparse lane uses cuSPARSELt's
  physically compressed paired-4:8 E2M1 operand, group-32 UE4M3 scales, and
  BF16 output; a real irregular-shape GEMM smoke is required before it is
  reported available. Backward can remain BF16 or use native block-32 MXFP8
  dgrad/wgrad. Trainable shadow parameters remain BF16; full optimizer offload
  keeps an FP32 master and FP32 Adam moments in CPU memory.
  Every trained base keeps its exact recipe and resolved architecture/precision
  provenance. QAT checkpoints save high-precision shadow tensors plus format
  and sparsity metadata so yyLab can restore the same training numerics.
  The native sparse/MXFP8 preset combines compressed sparse NVFP4 forward,
  native MXFP8 backward, and FP32 CPU optimizer offload. The retired adaptive
  and potentiated FP8 master simulations are not runnable.
- **MoLE adapter training** — frozen base with all experts and learned routers
  trained jointly on one mixed corpus stream. The legacy E8/r8/k2 preset makes
  one sparse decision per token and transformer block, then reuses that route
  across the block's adapted attention and MLP projections. An optional domain
  filter applies globally; data is never assigned to experts by hand.
- **Soft LoRA identity routing** — an optional learned, parameter-free identity
  lane competes with the LoRA experts at every token. Selecting it contributes
  exactly zero adapter delta, so the gate can retain the frozen base behavior
  where adaptation is unnecessary; dense top-K can include every expert plus
  identity. This implements [SLIM's](https://arxiv.org/abs/2410.07739)
  LoRA-versus-identity routing core, not its
  separate sliding-clustering or dynamic-merge stages.
- **Adaptive surprise MoLE** — an architecture mode solves expert count and
  rank so each global lane is approximately 0.5% of resolved model parameters,
  then trains nested 1–10% active sets with a calibrated predicted-NLL head.
  The UI shows the exact resolved E/r/K estimate; the worker recomputes it from
  real tensor shapes and saves router state plus independently verified expert
  shards. At inference only a request-fixed, VRAM-capped pool is resident, and
  two packed `K×r` GEMMs execute the active lanes. Prompt NLL can expand and
  safely recompute; decode entropy changes K only for the next token and never
  pages experts while KV cache state is live.
- **Legacy router training** — older sequential expert banks remain loadable
  and can still train a separate shared prompt-level router. New per-token
  adapters carry their routers in the adapter and need no second stage.
- **Offline preference optimization** — a file or recursive shard directory of
  JSONL, JSON, and/or Parquet is normalized to one canonical
  `prompt`/`chosen`/`rejected` contract. Besides direct pair rows, yyLab can
  group candidate traces by an explicit ID or identical structured prompt and
  resolve an explicit chosen/rejected label, finite score, or rank. Exact
  duplicate projections are removed across formats, source splits never pair
  with each other, and a canonical checksum is revalidated by the worker.
  Reasoning and tool context are preserved when the trajectories share the
  exact prompt; divergent tool trajectories and positive-only traces are
  rejected rather than converted into fake preferences. Valid pairs train an
  ordinary LoRA or QLoRA adapter with DPO. Prompt tokens are masked, complete
  assistant turns are scored, and the selected base checkpoint (with the
  adapter disabled) is the frozen reference policy. This remains deliberately
  offline: yyLab does not yet provide reward-model, rollout, PPO, or GRPO
  infrastructure.
- **Auto training method** — architecture-aware planner estimates VRAM and
  selects LoRA/QLoRA plus batch, accumulation, and gradient checkpointing;
  compatible Qwen3/RTX systems choose the largest power-of-two microbatch that
  fits an explicit free-VRAM reserve while holding the effective batch fixed.
- **Bounded-memory runtime** — legacy packed rank-space routing scales its
  adapter activations with total width `experts × rank`; adaptive v4 instead
  scales GEMMs and low-rank activations with active width `K × rank` while its
  larger bank stays in per-lane shards. Streaming corpus packing,
  low-CPU-memory checkpoint loading, exact tokenizer-aware step estimates, and
  an automatic paged 8-bit optimizer-state fallback keep host and device memory
  bounded. An explicit full CPU-offload mode keeps master weights and both
  Adam moments in system RAM when maximum VRAM headroom matters more than PCIe
  transfer speed. Masters, optimizer arithmetic, and moments remain FP32.
  Qwen3 uses cuDNN-priority attention, full Liger kernels, and fused
  linear cross entropy (no sequence × vocabulary logits allocation).
- **Live run terminal** — the selected run's progress, state, errors, and
  bounded log tail update in place without refreshing the page.
- **Benchmarks** — launch the immutable 64-case MoLoR core suite as a detached
  base, learned-routing, forced-expert, or full rack-matrix run. Results include
  a semantic 0–1000 Quality Index, the original Molor strict-output score,
  category scores, forced-expert oracle and dynamic-router regret, exact
  stack/workload fingerprints, resumable JSONL, reports, and fail-closed
  baseline comparisons. Correct explanatory answers retain semantic credit even
  when they miss a terse output envelope. Final-answer length and hidden
  reasoning have independent, fingerprinted budgets: reasoning never reduces
  the requested answer allowance. yyLab pre-opens the private block for adapters
  that omit `<think>` and turns an unclosed/EOS-only thought into a fresh direct
  answer turn; a still-empty final answer fails closed instead of becoming a
  false zero. Native context is the default;
  optional 2×/4× YaRN modes are fingerprinted for longer workloads. A second tab measures the real
  joint per-token MoLE forward/backward/optimizer path on synthetic tokens,
  including throughput, step percentiles, VRAM/RAM, routing health, finiteness,
  and explicit OOM receipts for selectable E/r/k and optimizer-offload modes.
  Benchmark launch is exclusive with yyLab runs and loaded Chat models; timing
  runs also default to an idle-GPU preflight that can be explicitly relaxed.
- **Context & KV cache** — SDPA attention, dynamic/quantized/sliding-window
  caches, RoPE scaling for context extension, and true packed int4/int8 CPU
  offload. Quantized offload stores history and group scales in RAM, then
  transfers and dequantizes only the active attention layer; an optional recent
  full-precision residual trades a little VRAM for fidelity.
- **Chat** — load checkpoint (+ native adapter and, for legacy banks, an
  optional prompt-router snapshot), choose native or YaRN 2×/4× context,
  select chat-template or independent raw-completion prompts, stream generation,
  and preserve instructions, recent turns, and the full requested answer budget
  in chat mode. Raw mode bypasses chat control tokens and rejects oversized
  prompts instead of inventing history or silently compacting them. Under
  pressure, the adaptive chat-context policy uses
  optional factorized addresses `A=(T,X,Y)`, lexical overlap, and recency to
  propose old complete turns, then admits them only when bounded query-NLL gain
  exceeds a configurable token cost. Accepted turns return to chronological
  order and are prefetched once at contiguous YaRN/RoPE positions. Prompt usage,
  evaluated/retrieved turns, ATXY candidates, and marginal NLL are shown live.
  Adaptive artifacts additionally show live active/resident lanes, parameter
  occupancy, NLL or entropy signal, hysteresis decisions, recomputations, and
  transfer counters.
- **Adapter export** — LoRA (including DPO-trained LoRA) and legacy/static MoLE adapters can be folded into a
  standalone HF checkpoint. Learned per-token routing is dynamic and cannot be
  represented exactly by one dense weight matrix, so its lossless artifact is
  the native version-3 safetensors + JSON adapter. Adaptive v4 adds its expert
  shard manifest and runtime policy to that lossless native contract.
- **RL-friendly artifacts** — trainable state is plain safetensors + JSON;
  native MoLE loading uses yyLab's small injection/runtime layer.

Docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Tests / lint

```powershell
uv run pytest
uv run ruff check .
```
