# SPALMER continued training and epoch control

Date: 2026-09-03
Status: approved design (minimal scope), awaiting implementation plan

## Goal

Allow a registered SPALMER checkpoint to be the starting point for a new
full-parameter causal-LM training run, and expose the trainer's existing epoch
loop as a control on the Train · SPALMER page for both scratch and continued
runs.

First use: supervised fine-tuning of the `spal50m` base on formatted chat and
reasoning transcripts mixed with pretraining replay, causal LM objective.

## Non-goals

- Chat SFT objective, LoRA, optimizer-state resume.
- Changing architecture or expert numerics when continuing.
- Read-only architecture summaries, schedule prefills, runs-page changes.
- Adding SPALMER to the Train · Base page.

## Background

- `yylab/train/spalmer.py` accepts only `initialization.mode == "scratch"`
  and builds a model from a recomputed plan.
- The trainer config already carries `epochs`; `build_spalmer_batch_source`
  re-iterates the corpora that many times with a reshuffled seed per pass. The
  page hardcodes `epochs = 1`. The stop condition is always `max_steps`.
- A bundle's `model_state` includes potentiation controller state, the
  surprise EMA, and router weights, so loading weights restores them.
- `relocated_spalmer_tokenizer` in `yylab/models/spalmer.py` resolves a moved
  tokenizer artifact by fingerprint and returns a `tokenizer_override` for
  `spalmer.load_checkpoint(path, map_location, tokenizer_override)`, which
  returns `(model, tokenizer, metadata)`.

## Config contract

`initialization` gains a second mode:

```json
{"mode": "scratch", "tokenizer_ref": "tok_..."}
{"mode": "checkpoint", "checkpoint_id": "ckpt_..."}
```

Normalization in checkpoint mode:

- `checkpoint_id` is required and must resolve through the checkpoint registry
  to a directory containing `spalmer.pt`, else `ValueError`.
- No `tokenizer_ref` is required; the tokenizer comes from the bundle.
- The `spalmer` block may contain only the attention preflight keys
  (`attention_backend`, `require_fla`, `allow_reference_attention`), which
  gate the FLA availability check. The KDA backend, expert numerics, and QAT
  backend are baked into the loaded model and are inherited. Any other key
  (`target_parameters`, `num_experts`, `experts`, `model`, `directional`,
  `atxy`, `expert_weight_format`, `expert_qat_backend`, `plan_fingerprint`)
  is a `ValueError` naming the key.
- `objective` stays `causal_lm`, `method` stays `full`.
- `epochs` is validated as an integer >= 1 in both modes.
- `recipe_version` bumps from 2 to 3 (`SPALMER_RECIPE_VERSION`).

## Trainer changes (`yylab/train/spalmer.py`)

- `PreparedSpalmerRun` gains optional `model` (prebuilt when continuing) and
  optional `parent` (`checkpoint_id`, `checkpoint_path`, `run_id`).
- `prepare_spalmer_run` branches on mode. Checkpoint mode: resolve the bundle
  path from the registry, read the tokenizer binding with the existing safe
  memory-mapped load, call `relocated_spalmer_tokenizer`, call
  `spalmer.load_checkpoint` on CPU with the override when produced, take the
  plan and construction configs from the loaded bundle rather than recomputing
  them, and build the training config from the recipe as today. The prepared
  `plan` exposes `to_dict()`, `fingerprint`, and `relative_error` from the
  parent's metadata so checkpoint saving and the run log work unchanged.
- `run_spalmer_base` uses `prepared.model` when present, and logs the parent
  checkpoint ID plus a note that optimizer state and schedule start fresh.
- `_save_bundle_checkpoint` adds `metadata["parent_checkpoint"]` with
  `checkpoint_id`, `checkpoint_path`, and `run_id` when continuing.
- Data exhaustion before `max_steps` is verified during implementation to end
  with a final checkpoint and a logged reason; fixed only if it does not.

## Page changes (`yylab/ui/pages/train_spalmer.py`)

- Section 1 gains a mode select (`Random weights (scratch)` /
  `Continue from checkpoint`) and, in checkpoint mode, a checkpoint select
  listing registry rows whose metadata family is `spalmer`, newest first. The
  tokenizer select is hidden in checkpoint mode.
- Section 2 (architecture) is hidden in checkpoint mode.
- Section 4 gains an `Epochs` number input (default 1, min 1). The estimate
  line multiplies corpus tokens by epochs.
- `build_spalmer_run_config` emits the chosen `initialization` block, omits the
  architecture keys in checkpoint mode, and emits `epochs`.

## Worker

Run kind stays `spalmer_base`; routing is untouched.

## Errors

| Condition | Where | Result |
|---|---|---|
| Unknown checkpoint ID or bundle missing | normalize | ValueError |
| Architecture key present in checkpoint mode | normalize | ValueError naming the key |
| Tokenizer path missing and no fingerprint match | prepare | FileNotFoundError from relocation |
| `epochs` < 1 or non-integer | normalize | ValueError |

## Tests

Fake-API pattern from `tests/test_spalmer_training.py` and the fake `spalmer`
module pattern from `tests/test_spalmer_runtime.py`.

- Normalization: checkpoint mode accepted; unknown ID, architecture keys, and
  bad epochs rejected; epochs passes through.
- Preparation: checkpoint mode calls the fake `load_checkpoint` with the
  override when the recorded path is missing, never calls
  `plan_configuration`, and returns the loaded model and parent info.
- Saving: `parent_checkpoint` metadata present when continuing.
- Config builder: checkpoint-mode recipe has no architecture keys, carries
  `initialization.checkpoint_id` and `epochs`; scratch recipe carries `epochs`.

The continued run on the local GPU is a manual smoke test and doubles as the
peak-memory check at micro-batch 8.
