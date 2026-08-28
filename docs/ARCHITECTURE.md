# yyLab Architecture

## Layout

```
yylab/
  config.py        paths, settings.json (YYLAB_HOME override)
  db.py            SQLite registry: corpora, checkpoints, adapters, routers, runs, doc_hashes
  jobs.py          in-process background job manager (progress polling)
  hw.py            NVML/torch GPU probe, CPU/RAM info, triton detection
  artifacts.py     artifact dir layout + helpers
  ingest/          corpora pipeline
    adapters/      filetype adapters (add new ones here)
    pipeline.py    walk → extract → normalize → dedup → shard → tokenize
    dedup.py       sha256 exact + MinHash LSH near-dedup
    domains.py     code/math/science/web/language classification
    tokenize_stats.py  multi-tokenizer counting registry
  mole/            Mixture-of-LoRA-Experts core (pure torch, CPU-testable)
    layer.py       MoLELinear = frozen base + K LoRA experts + TopKRouter
    inject.py      model surgery + stage freezing (experts/router/both)
    losses.py      switch-style load-balance aux loss
    store.py       safetensors save/load (RL-friendly format)
  train/planner.py VRAM-aware auto method picker (full/lora/qlora)
  ui/              NiceGUI app shell + pages
```

## Artifact store (`artifacts/`, override with `YYLAB_HOME`)

```
corpora/<id>/stats.json, data/shard-*.jsonl.zst
checkpoints/<id>/      (hub downloads land here)
adapters/<id>/mole_weights.safetensors + mole_config.json
routers/<id>/          (router-only checkpoints)
runs/<id>/             (training configs, logs, metrics)
yylab.sqlite3          registry index
```

Every trainable artifact is plain safetensors + JSON so external RL stacks
(TRL, etc.) can consume them directly; corpora shards are zstd JSONL
`{"text","domain","source"}`.

## MoLE forward

```
y = W_frozen(x) + Σ_{e∈topK(router(x))} g_e(x) · (B_e A_e x)
```

- Router is zero-initialized → uniform routing at init → adapter output equals
  the frozen base exactly at step 0.
- Quantization is locked per checkpoint+adapter at adapter creation time;
  stored in `adapters.quant_locked` to prevent drift between adapters.
- Training stages: `both` (experts+router), `router` (freeze experts),
  `experts` (freeze router).

## Training auto-selection (`train/planner.py`)

Inputs: model params, free VRAM, seq len, method/quant preference.
Ladder tried in order until it fits `0.88 × VRAM`:

1. full bf16 (grad-ckpt, bs 1)
2. LoRA bf16 base
3. QLoRA (NF4/INT8/FP8 base) bs 2 → 1 → +CPU offload

Extras: torch.compile when triton importable, FP8 hint on sm≥8.9,
offload fallback when RAM ≥ 2× model.

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
(dynamic / quantized / sliding-window / static), KV quant (int8/fp8),
max context, RoPE scaling. Consumed by the inference engine (phase 4) and
used as training context defaults.

## Phases

- [x] 1. Scaffold, artifact store, corpora intake + UI, planner, MoLE core
- [ ] 2. Training runtime (MoLE stage 1, router stage 2, base training)
- [ ] 3. Inference engine + chat (base + adapter + router, KV settings)
- [ ] 4. RL export hardening (merged checkpoint export, GGUF)
