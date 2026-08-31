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
  normalization, exact + MinHash dedup, domain tagging, zstd JSONL shards,
  token counts across GPT-2/tiktoken, Llama 3.x, Qwen 2.5, Phi-3, Gemma,
  Mistral tokenizers; mixture charts + live stats table.
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
  independent. An exact parameter calculator and nearest-layer solver show the
  achieved model size before launch; scratch runs are full-parameter runs,
  while continued runs retain full, LoRA, and QLoRA choices. A base run can
  select multiple ordered corpora; new recipes mix their eligible documents by
  deterministic round robin while still consuming every document in each
  epoch, and persist the selected sources and mix provenance with the run.
- **Explicit base-training precision** — parameter dtype, QAT numerics,
  activation quantization, optimizer implementation/state placement, AdamW
  constants, scheduler/warmup, batching, checkpointing, compile, and fused
  kernels are separate recipe axes. INT4 remains compatibility-oriented STE
  QAT because Blackwell has no native INT4 Tensor Core input path. On Blackwell,
  NVFP4 QAT uses TorchAO's packed E2M1 values, E4M3 block scales, and native
  block-scaled FP4 forward GEMM. The SM120 sparse lane uses cuSPARSELt's
  physically compressed paired-4:8 E2M1 operand, group-32 UE4M3 scales, and
  BF16 output; a real irregular-shape GEMM smoke is required before it is
  reported available. Backward can remain BF16 or use native block-32 MXFP8
  dgrad/wgrad, with explicit format emulation as a portable fallback. Trainable
  shadow parameters remain BF16 and optimizer-master storage is selected
  independently. Ternary W1.58/A8 QAT is available only with the BitNet family.
  Every trained base keeps its exact recipe and resolved architecture/precision
  provenance. QAT checkpoints save high-precision shadow tensors plus format
  and sparsity metadata so yyLab can restore the same training numerics.
  The recommended experimental preset combines compressed sparse NVFP4 forward,
  native MXFP8 backward, and an adaptive FP8 CPU master. Every master weight has
  a real E4M3 byte, but only blocks with high reconstruction or lost-update error
  receive a sparse packed four-bit potentiation overlay. Promotion is capped
  (10% by default) and hysteretic demotion removes the overlay after a block
  stabilizes. BF16 accelerator shadows and FP32 Adam moments remain intact.
- **MoLE adapter training** — frozen base with all experts and learned routers
  trained jointly on one mixed corpus stream. The legacy E8/r8/k2 preset makes
  one sparse decision per token and transformer block, then reuses that route
  across the block's adapted attention and MLP projections. An optional domain
  filter applies globally; data is never assigned to experts by hand.
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
  transfer speed. Masters may remain FP32 or use the experimental potentiated
  FP8 representation; optimizer arithmetic and moments stay FP32 in either
  case. Qwen3 uses cuDNN-priority attention, full Liger kernels, and fused
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
  stream generation, and preserve instructions, recent turns, and the full
  requested answer budget. Under pressure, the adaptive context policy uses
  optional factorized addresses `A=(T,X,Y)`, lexical overlap, and recency to
  propose old complete turns, then admits them only when bounded query-NLL gain
  exceeds a configurable token cost. Accepted turns return to chronological
  order and are prefetched once at contiguous YaRN/RoPE positions. Prompt usage,
  evaluated/retrieved turns, ATXY candidates, and marginal NLL are shown live.
  Adaptive artifacts additionally show live active/resident lanes, parameter
  occupancy, NLL or entropy signal, hysteresis decisions, recomputations, and
  transfer counters.
- **RL export** — LoRA and legacy/static MoLE adapters can be folded into a
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
