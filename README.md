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
- **Checkpoint registry** — local folders (param count scanned from
  safetensors headers) or HF hub ids; per-checkpoint quant lock.
- **MoLE adapter training** — frozen base, K LoRA experts per target module,
  top-K per-token routing, NF4/INT8/FP8 base quant locked per adapter.
- **Router training** — stage-2 training of the router only (top-K
  configurable, load-balance aux loss).
- **Auto training method** — planner estimates VRAM and picks
  full/LoRA/QLoRA + batch/accum/grad-ckpt/offload/compile automatically.
- **Context & KV cache** — SDPA attention, dynamic/quantized/sliding-window
  caches, RoPE scaling for context extension.
- **RL-friendly artifacts** — everything is plain safetensors + JSON.

Docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Tests / lint

```powershell
uv run pytest
uv run ruff check .
```
