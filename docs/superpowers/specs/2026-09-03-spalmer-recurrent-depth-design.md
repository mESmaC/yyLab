# SPALMER recurrent-depth latent reasoning — final design

> **Historical recurrent-depth design record.** The recurrent architecture
> remains relevant, but any format-v5/reference/fake-QAT precision or loading
> examples below are superseded and unloadable. Current execution requires
> QR-PLE format v7/v8 plus an exact capability-verified native expert pair.

Date: 2026-09-03
Status: approved design, ready for implementation (SPALMER repo `C:/Users/engin/Desktop/yy/spalmer` main, yyLab `C:/Users/engin/Desktop/yy/yyLab` main checkout — never the `.claude/worktrees` copy).

This document is the winning panel design (Part 2, below, "Design body") plus the normative amendments in Part 1. Where Part 1 and Part 2 disagree, **Part 1 wins**. Line anchors in Part 2 refer to spalmer @54d6b8d and yyLab @73f386c.

## Part 1 — Normative amendments

### A. Numerics of the recurrent core (replaces `RecurrenceAdapter`)

The recurrence module is `LatentRecurrence(nn.Module)`, registered as `backbone.recurrence` (the name must not contain `norm`/`gate`, so the adapter weight decays; sub-module names must contain `norm` so the norms do not, per `training/optim.py:66-76`). Submodules and parameter names:

- `backbone.recurrence.injection_norm.weight` `[d]` — learned RMSNorm applied ONCE per forward to the prelude output `e` → `e_n` (unit RMS).
- `backbone.recurrence.adapter.weight` `[d, 2d]` — `nn.Linear(2d, d, bias=False)` over `cat[s, e_n]`.
- `backbone.recurrence.latent_norm.weight` `[d]` — learned RMSNorm applied to the residual stream after the last core block of every iteration.

Init (`RecurrenceConfig.adapter_init`, default `"identity_mix"`): `adapter.weight = cat([I_d, I_d], dim=1) / sqrt(2) + N(0, initializer_range^2)`; `"random"` → `N(0, 1/sqrt(2d))`. Parameter count of the module: `2*d*d + 2*d`.

Latent numerics: `s` is carried in **fp32** between iterations; the only fp32→residual-dtype cast is at the adapter input (`torch.cat([s.to(wd), e_n], -1)`); after the last core block of an iteration `s = latent_norm(h).float()`. `s0 = randn((B,T,d), fp32, generator) * latent_init_std` (default `latent_init_std = 1.0`, unit-RMS space; `0.0` → zeros). `latent_init` (warm start) is used as given (`.float()`), shape must be `[B,T,d]`. Under bf16 autocast the adapter and blocks run in bf16 as today; `latent_deltas`, the KL probe and the correlation metric are computed in fp32 under `torch.no_grad()`.

`RecurrenceConfig` fields (frozen, slots): `prelude_layers: int`, `core_layers: int`, `coda_layers: int`, `default_steps: int = 8`, `latent_init_std: float = 1.0`, `adapter: Literal["linear_concat"] = "linear_concat"`, `adapter_init: Literal["identity_mix", "random"] = "identity_mix"`. Everything else about `RecurrenceConfig` / `SPALMERConfig` (coercion from a mapping in `__post_init__`, `prelude+core+coda == n_layers`, helpers `core_range`, `effective_depth`, `block_role`, `core_mixer_counts`) is as in Part 2.

Accounting/planning component is named **`recurrence`** (not `recurrence_adapter`): `_COMPONENT_ORDER` gains `"recurrence"` before `"vocab_head"`; `classify_parameter` returns `"recurrence"` when `".recurrence." in name`, and that rule must precede the directional and norm rules; `ParameterBreakdown.recurrence: int = 0` (omitted from `to_dict` when 0, always present in `components`); `RecurrenceScaleConfig.parameter_count(d) = 2*d*d + 2*d`; `count_parameters` adds it when `config.recurrence` is set. Golden `9_908_080` and every existing fingerprint stay byte-identical.

### B. Model-level defaults (in addition to explicit kwargs)

`SPALMERBackbone` gets plain attributes `recurrence_steps_override: int | None = None` and `adaptive_exit_override: AdaptiveExit | None = None`. `SPALMERCausalLM.set_recurrence_defaults(steps: int | None = None, adaptive_exit: AdaptiveExit | None = None)` validates (`steps >= 1`; model must be recurrent, else ValueError) and sets both; `clear_recurrence_defaults()` resets. Resolution order in the backbone: `recurrence_steps` kwarg → override → `config.recurrence.default_steps`; `adaptive_exit` kwarg → override (decode only). Explicit kwargs (Part 2) are still threaded by `generate_tokens`, `choose_inference_residency._evaluate`, `evaluate_model_batches`, the trainer and the yyLab facade; the yyLab facade ALSO calls `set_recurrence_defaults` after load so any bare `model(ids)` call honours the loaded depth. Not thread-safe by design (the yyLab engine serialises generation).

### C. `AdaptiveExit`

Fields: `criterion: Literal["latent_diff", "kl"]`, `threshold: float | None = None` (defaults 0.03 / 1e-3), `min_steps: int = 2` (values < 2 are clamped to 2), `check_every: int = 1`, `state_policy: Literal["fill", "skip"] = "fill"`, `apply_in_prefill: bool = False`. Adaptive exit is an inference policy: if it is supplied while `torch.is_grad_enabled()` is True → `ValueError("adaptive exit is an inference-only policy")`. In prefill it is honoured only when `apply_in_prefill` is True (otherwise ValueError as in Part 2). The criterion is checked at iterations `it+1 >= min_steps` with `(it+1-min_steps) % check_every == 0`.

### D. `RecurrentMixerStates`

`iterations` may contain `None` elements (channel-mixer doubles and stateless mixers); only an empty tuple is rejected. A core channel slot whose elements are all `None` is emitted as `None`. `budget` (output slot length) is the length of the first non-`None` core slot among token OR channel slots; all non-`None` core slots must share that length (ValueError otherwise). `.to()` maps elements that expose a callable `to` and passes the others through.

### E. Training config and sampler

`TrainingConfig` gains (after `telemetry_interval_steps`): `mean_recurrence: float | None = None`, `mean_backprop_depth: int = 4`, `recurrence_sigma: float = 0.5`, `recurrence_sampling: Literal["poisson_lognormal", "uniform", "fixed"] = "poisson_lognormal"`, `max_recurrence: int | None = None` (None resolves to `4 * ceil(mean_recurrence)` inside the sampler). Validation as in Part 2 plus `uniform`: `r = rng.randint(1, max(1, 2*round(mean) - 1))`. `RecurrenceSampler` API: `RecurrenceSampler.from_training_config(config) -> RecurrenceSampler | None`, `.sample(step) -> tuple[int, int]` (r, k_eff), stateless `random.Random(f"spalmer-recurrence:{seed}:{step}")`, `E[r] == mean_recurrence` (rate mean `mean-1`), Knuth/inverse-CDF Poisson. `mean_recurrence <= 1` → always `r = 1`.

### F. Telemetry additions

`BackboneOutput` / `CausalLMOutput` gain `recurrence_steps`, `latent_states` (fp32 `[B,T,d]`, detached), `latent_deltas` (`[iterations_run]` fp32) AND `latent_position_correlation` (0-d fp32: mean over batch of the mean off-diagonal cosine similarity between up to 64 evenly spaced positions of the final latent; `None` when T < 2). `TrainStepMetrics` gains `recurrence_steps`, `backprop_steps`, `effective_depth`, `latent_delta_final: float | None`, `latent_position_correlation: float | None` (means over the step's microbatches). No periodic r=1 vs r̄ loss probe in this change (follow-up). yyLab `_metric_payload` copies all five and the printed progress line appends ` · r={r} k={k} Δ={latent_delta_final:.3f} ρ={corr:.2f}` when present.

### G. yyLab engine / benchmarks: conditional pass-through

`Engine._load_spalmer` passes `recurrence_steps / adaptive_exit / exit_threshold / min_steps / warm_start` to `load_spalmer_runtime` ONLY when at least one `spalmer_recurrence_*`/`spalmer_adaptive_exit`/`spalmer_exit_threshold`/`spalmer_warm_start` key is present in the settings (so existing fake `load()` signatures and non-recurrent loads are untouched). Resolved values are written back into `spec.settings` (`spalmer_recurrence_steps`, `spalmer_adaptive_exit`, `spalmer_exit_threshold`, `spalmer_warm_start`, `spalmer_recurrent=True`) only when the facade reports `recurrent`; flat checkpoints leave settings untouched. `_stack_lock`, `_workload_contract`, `_generation_comparison_fingerprint` and `backend.identity()` add a `spalmer_recurrence` block ONLY when a recurrence setting is active, so every existing spalmer stack/workload/comparison fingerprint and resumable benchmark cell is unchanged.

### H. Facade reproducibility and metadata lookup

The facade builds a `latent_generator = torch.Generator(device).manual_seed(seed)` whenever `generate(seed=...)` is given (same seed as the sampling generator) and passes it to prefill and decode calls; without a seed it passes `None` (global RNG). Chat's `_checkpoint_recurrence(row)` and the checkpoints page helper read, in order, `meta_json["recurrence"]`, `meta_json["recurrence_config"]`, `meta_json["metadata"]["recurrence"]`; `scan_spalmer_checkpoint` stores the block top-level as `recurrence_config` (and `effective_depth`).

### I. Versions

`FORMAT_VERSION = 6` with the Part 2 gating (`required_model -= {"recurrence"}` for `< 6`; `_migrate_experts_config` gate `>= 5`). `SPALMER_RECIPE_VERSION = 4` in yyLab (update `tests/test_spalmer_ui.py` pins). yyLab manifest `checkpoint_format = f"spalmer.prototype.checkpoint.v{getattr(api, 'FORMAT_VERSION', 5)}"`; `SUPPORTED_SPALMER_FORMAT_VERSIONS = frozenset({5, 6})`.

### J. Work packages (disjoint file ownership; all four may run in parallel against this spec)

Shared interfaces every package codes against (names are final): `spalmer.config.RecurrenceConfig` (fields in A), `SPALMERConfig.recurrence`, `spalmer.modeling.AdaptiveExit` (C), `spalmer.modeling.RecurrentMixerStates` (D), backbone/LM forward kwargs `recurrence_steps, backprop_steps, latent_init, adaptive_exit, latent_generator`, output fields (F), `SPALMERCausalLM.set_recurrence_defaults/clear_recurrence_defaults/is_recurrent/recurrence_default_steps`, `spalmer.training.recurrence.RecurrenceSampler` (E), `TrainingConfig` fields (E), `TrainStepMetrics` fields (F), `spalmer.runtime.RecurrenceTrace` + `generate_tokens` kwargs (Part 2), `spalmer.experiment.planning.RecurrenceScaleConfig(prelude_layers, coda_layers, default_steps, latent_init_std)` with `parameter_count`, `plan_configuration(..., recurrence=)`, `presets.build_configs` mapping plan → `RecurrenceConfig`, `spalmer.checkpoint.FORMAT_VERSION`, exports in `spalmer/__init__.py` (`RecurrenceConfig`, `AdaptiveExit`, `RecurrentMixerStates`, `RecurrenceTrace`), `spalmer.experts.residency.choose_inference_residency(..., recurrence_steps=None)`, `spalmer.training.evaluation.evaluate_model_batches(..., recurrence_steps=None)`.

| Package | Owns (only these files) |
|---|---|
| WP-A spalmer model/config/checkpoint/accounting | `src/spalmer/config.py`, `src/spalmer/modeling.py`, `src/spalmer/factory.py`, `src/spalmer/checkpoint.py`, `src/spalmer/experts/accounting.py`, `src/spalmer/__init__.py`; tests: `tests/core/test_modeling.py`, `tests/core/test_checkpoint_metadata.py`, `tests/attention/test_full_vs_recurrent.py` (new case), `tests/memory/test_atxy.py` (ATXY-in-core cases), `tests/experts/test_paged_offload.py` (core-bank counter case), new `tests/core/test_recurrence_model.py` if preferred |
| WP-B spalmer training/planning/runtime/CLI | `src/spalmer/training/config.py`, new `src/spalmer/training/recurrence.py`, `src/spalmer/training/engine.py`, `src/spalmer/training/evaluation.py`, `src/spalmer/training/resume.py`, `src/spalmer/training/__init__.py`, `src/spalmer/runtime.py`, `src/spalmer/__main__.py`, `src/spalmer/experiment/planning.py`, `src/spalmer/experiment/__init__.py`, `src/spalmer/presets.py`, `src/spalmer/experts/residency.py`; tests: `tests/training/*`, `tests/experiment/*`, `tests/core/test_presets.py`, `tests/core/test_cli.py`, `tests/core/test_runtime*.py`, `tests/experts/test_residency.py` |
| WP-C yyLab training bridge + page + registry + docs | `yylab/train/spalmer.py`, `yylab/ui/pages/train_spalmer.py`, `yylab/ui/pages/checkpoints.py`, `tests/test_spalmer_training.py`, `tests/test_spalmer_ui.py`, `docs/ARCHITECTURE.md` |
| WP-D yyLab inference | `yylab/models/spalmer.py`, `yylab/infer/engine.py`, `yylab/ui/pages/chat.py`, `yylab/benchmark/backend.py`, `yylab/benchmark/quality.py`, `yylab/ui/pages/benchmarks.py`, `tests/test_spalmer_runtime.py`, `tests/test_spalmer_engine_integration.py`, benchmark/chat tests |

Cross-package contracts inside yyLab: `scan_spalmer_checkpoint` (WP-D) returns `recurrence_config` and `effective_depth`; the checkpoints page (WP-C) reads them with the H lookup order; the bridge (WP-C) writes `meta_json["recurrence"]` and `effective_depth` for run-produced rows; chat (WP-D) reads them with the same helper.

### K. Acceptance

1. SPALMER: `cd C:/Users/engin/Desktop/yy/spalmer && PYTHONPATH=C:/Users/engin/Desktop/yy/spalmer/src C:/Users/engin/Desktop/yy/yyLab/.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` → all pass (baseline 349 passed, 1 skipped; existing tests unchanged except where the spec says a pin moves); `... python.exe -m ruff check src tests` clean.
2. yyLab: `cd C:/Users/engin/Desktop/yy/yyLab && .venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` (baseline 613) and `.venv/Scripts/python.exe -m ruff check .` clean.
3. CUDA smoke (WP-B, yyLab venv, spalmer source on PYTHONPATH, from the spalmer repo): `python -m spalmer --smoke --steps 20 --layers 4 --recurrent-layers 1 2 1 --mean-recurrence 4 --backprop-depth 2 --output runs/recurrent-smoke.pt` trains and saves a v6 bundle; `python -m spalmer generate runs/recurrent-smoke.pt --prompt "SPALMER routes" --recurrence-steps 6 --adaptive-exit latent_diff` prints text plus the recurrence trace line; the same generate without recurrence flags uses `default_steps`.
4. yyLab worker smoke (orchestrator, after integration): a synthetic-corpus ingest + a 20-step recurrent SPALMER run through `run_spalmer_base` in a throwaway `YYLAB_HOME`, then `load_spalmer_runtime` + a short generation with adaptive exit; continued training from the produced v6 bundle for 5 steps.

## Part 2 — Design body (winning panel design; subordinate to Part 1)

# SPALMER recurrent-depth latent reasoning (Huginn-style) and yyLab integration

Date: 2026-09-03
Status: design (inference / decode-state / product lens), for parallel implementation
Repos: SPALMER `C:/Users/engin/Desktop/yy/spalmer` @54d6b8d (src `src/spalmer`), yyLab `C:/Users/engin/Desktop/yy/yyLab` @73f386c (main checkout, NOT the worktree). All file:line anchors below were read from those revisions.

## Goal

Add an optional depth-recurrent core (prelude P physical blocks, core C physical blocks iterated r times, coda K physical blocks) to `SPALMERCausalLM` with: truncated backprop through the last k iterations, per-optimizer-step sampled r, per-(block, iteration) KDA/MLA decode state, fixed-r or adaptive-exit decoding with well-defined state semantics for exited iterations, warm start, a checkpoint format that records its own default depth, exact parameter/effective-depth accounting, and a yyLab surface (train page, worker bridge, checkpoint registry, inference facade, engine settings, chat, benchmarks) that threads the depth policy everywhere a native call is made. A checkpoint with `recurrence = None` must behave bit-for-bit as today (same state tuples, same layer_metrics, same fingerprints, same golden 9_908_080, same v5 loading).

## Non-goals

- Mixture-of-Recursions routers, per-token learned depth, batch>1 generation in `generate_tokens`/facade (backbone semantics are defined for B>1 but only B=1 and a B=2 unit test are exercised), KV/state compression across iterations (Huginn compress-sN), cross-iteration attention, activation checkpointing (still absent: engine.py:404-410 hook is still unimplemented; memory is bounded by the no_grad prefix + k retained iterations), per-block PLE tables per iteration, changing `n_layers % 4` rules, exact optimizer-state resume in yyLab, validation streams in yyLab.
- Making adaptive exit available in prefill (see D5: prefill always runs the full step budget so that every iteration's KDA/MLA state covers the whole prompt).

## Background (verified contracts)

- `SPALMERBackbone.forward` (modeling.py:213-315) enumerates `self.blocks` once, injects `embeddings.inject(h, ids, layer_index)` for `layer_index>0` (279-280), fires ATXY `inject_values` when `layer_index == atxy.config.injection_layer` (291-299), appends one state and one metrics mapping per block (300-302), and averages per-block auxiliary losses (306-308). `_states_or_none` (665-670) requires `len(states) == len(blocks)`; `len(blocks) == config.n_layers` (193-194); ATXY bound `0 <= injection_layer < n_layers` (200-204).
- `SPALMERBlock.forward` (94-166) dispatches `token_mixer(...)` in prefill or `token_mixer.step(...)` in decode; mixers accept `state=None` (fresh zero state, KDA mixer.py:179-180, MLA mla.py:190-191) and return detached `KDAState` (state.py:26-70, has `.to(device=, dtype=)`) / `MLAState` (mla.py:87-125, has `.to`). `MLAState.latent_kv` grows by T per call (mla.py:219), so one cache can never be shared across iterations (it would attend to its own earlier iterations); `KDAState.recurrent_state` updated r times per token would break the prefill==decode contract enforced by tests/attention/test_full_vs_recurrent.py. Per-iteration states are the only consistent option.
- `SPALMERCausalLM.forward` (530-595) computes loss, `_attach_surprise_telemetry` (681-723: one calibration term per metrics entry carrying `router_scores/expert_ids/routing_weights`, mean at 722), `update_potentiation` (597-641: mean/sum over entries), predictive entropy (580-584). `_PotentiationAccumulator.add` (engine.py:166-203) raises when `len(layer_metrics)` changes between microbatches (194-198). `RouterTelemetryAccumulator` keys by enumerate position (training/evaluation.py:154; telemetry.py:435-440). `rank_nonresident_experts` averages `router_scores` over entries (residency.py:336-361); `choose_inference_residency` (389-511) runs full prefills via `_evaluate` (525-536) which calls `model(prompt_ids[, labels])` with no other kwargs.
- `generate_tokens` (runtime.py:173-243) is B=1, prefill then `model(next_token, token_mixer_states=..., channel_mixer_states=..., execution_mode="decode")` (231-236), threading tuples opaquely. `_select_token` 246-263.
- `ExperimentTrainer._optimizer_step` (engine.py:421-512) calls `self.model(batch.input_ids, labels=..., attention_mask=..., state_reset_mask=...)` at 443-448 inside autocast; `TrainStepMetrics` at 102-117; `seed_everything` at 388. `TrainingConfig` (training/config.py:11-129) is frozen slots, `to_dict=asdict` and is equality-checked on resume (resume.py:295-313). `RNGSnapshot` covers python/torch-cpu/cuda global streams only (state.py:293-328, 450-460).
- Checkpoint: `FORMAT_VERSION = 5`, `_SUPPORTED_FORMAT_VERSIONS = {1,2,3,4,FORMAT_VERSION}` (checkpoint.py:36-37); `load_checkpoint` (460-583) requires the exact `SPALMERConfig` field set (491-495 with `_VERSION_3_MODEL_CONFIG_FIELDS` subtraction at 493), loads `strict=True` for v>=3 (540), and `_migrate_experts_config` (597-612) accepts the current expert schema ONLY when `checkpoint_version == FORMAT_VERSION` (604) — bumping the version without fixing this makes every v5 bundle fail with "unexpected" `_VERSION_5_EXPERT_FIELDS`. `save_checkpoint` (74-127) writes `model_config = model.config.to_dict()` (asdict, so nested dataclasses serialize as dicts). Validators enumerate physical blocks with `token_mixer_for_layer(i)` (738-749, 752-768).
- Planning: `ModelScaleConfig` (planning.py:89-191, `to_dict=asdict` at 190), `ParameterBreakdown` (195-283, `to_dict` 275 = asdict + totals), `count_parameters` (285-358; norms 333, router 335, lm_head 345), `ScalePlan.fingerprint` hashes `config.to_dict()+parameters.to_dict()+memory.to_dict()` (603-611), `plan_configuration` grid (625-694; `n_layers` loop 643, ATXY skip 646, key 675-684). `presets.build_configs` (139-238; `_MODEL_SHAPE_FIELDS` 106-113; SPALMERConfig fields 166-177), `assert_accounting_matches_plan` (242-271). `accounting._COMPONENT_ORDER` (accounting.py:21-32), `classify_parameter` (156-177: the `"_norm."`/`norm.weight` rule at 175 would swallow a recurrence RMSNorm unless a recurrence rule precedes it), `account_parameters` (180-253).
- CLI (`__main__.py`): `_plan_parser` 63-91, `_run_plan` 216-253 (table header 236-239, row 246-253), `_training_parser` 120-172, `_run` 333-477 (layers%4 at 338-339, SPALMERConfig 364-373, `train_token_stream` 425-434, metadata 442-462, summary 470-476), `_generate_parser` 175-213, `_run_generate` 480-532 (`generate_tokens` call 512-519). `train_token_stream` (runtime.py:29-133) calls `model(batch, labels=batch)` at 90.
- yyLab facade `yylab/models/spalmer.py`: `scan_spalmer_checkpoint` 71-149 (`num_layers` from `model_config.n_layers` at 95, `base_cfg` 101-117), `SpalmerCausalLMFacade.__init__` 307-352 (`config.num_hidden_layers` 348), `routing_telemetry` 372-383, `forward` 390-431, `generate` 433-570 (prefill 507/515/523, decode 556-561), `load_spalmer_runtime` 642-804 (signature 642-650, notes 752-792, facade construction 793-804), `_validate_ignored_generation_kwargs` 928-983 (rejects any kwarg not in `inert_defaults`), `_full_pool_routing_telemetry` 1312-1355, `_fixed_residency_telemetry` 1358-1385, `_residency_telemetry` 1388-1439.
- yyLab engine `yylab/infer/engine.py`: `_load_spalmer` 969-1040 (settings 991-1015, `load_spalmer_runtime` call 1022-1033), `_generate_kwargs` 1042-1109 (HF kwargs only; spalmer branch 1065-1069), `_stream` 1200-1229 (`self.model.generate(**gen_kwargs)` 1215), `routing_telemetry` 1185-1193.
- yyLab bridge `yylab/train/spalmer.py`: `CHECKPOINT_EXECUTION_KEYS` 50-52, `_load_spalmer_api` 112-163, `_normalize_checkpoint_execution` 289-308, `_normalize_scratch_architecture` 311-446 (`spalmer["atxy"]` 389, model overrides 425-432), `normalize_spalmer_config` 449-635 (training aliases 546-567), `_scale_components` 665-685, `plan_spalmer_configuration` 688-720 (fingerprint check 714-719), `build_spalmer_bundle` 723-743, `build_spalmer_training_config` 746-801, `_metric_payload` 1117-1184, `_TelemetryWriter.record` 1193-1212, `_prepare_checkpoint_run` 1375-1420 (`format_version != 5` at 1383), `_save_bundle_checkpoint` 1442-1479, manifest 1679-1704 (`checkpoint_format` string 1682), `checkpoint_metadata` 1707-1727, registry UPDATE 1728-1741 (`num_layers = plan.config.n_layers` 1737).
- yyLab page `yylab/ui/pages/train_spalmer.py`: `SPALMER_RECIPE_VERSION = 3` (29), `_cached_plan_json` 128-185 (13 positional args), `plan_spalmer_configuration` 188-223, `build_spalmer_run_config` 300-603 (spalmer block 467-488, training block 489-520, checkpoint branch 521-534), Section 2 726-857, Section 4 943-1062, `update_plan_preview` 1203-1285 (parent branch 1206-1225, layer line 1251-1256, KDA/MLA line 1257-1261), `on_start` 1353-1456 (build call 1407-1453), pending row 1458-1495.
- yyLab chat `yylab/ui/pages/chat.py`: `_routing_telemetry_text` 153-258, controls 319-341, `refresh_adapters` 487-517 (visibility 504-505), load settings 596-606, badge 641-653, `refresh_routing_telemetry` 668-671. Benchmarks: `_quality_job_config` (benchmarks.py:70-141, spalmer block 130-140), `_config_text` spalmer line 300-308, cache-size input 788-799, visibility 1028, job build 1107-1111; `quality.py` `_stack_lock` 123-153 (`spalmer_runtime` 141-151), `_workload_contract` spalmer keys 351-366, `_generation_comparison_fingerprint` 371-420 (`spalmer_expert_placement` 414-419), backend settings 1279-1292; `backend.py` `identity` 275-309, `configure_policy` 311-389, `generate` 391+ (HF kwargs 526-540, telemetry 733-736). Checkpoints page: `_spalmer_base_recipe` 43-81, detail line 252-255.
- Reference: Geiping et al. 2025 (Huginn). prelude → e; s0 ~ N(0,σ²); s_i = core(A(cat[s_{i-1}, e])); coda(s_r) → head; r ~ log-normal Poisson per step (σ=0.5), first r-k iterations no_grad, last k with grad; loss on final logits only; inference fixed r or adaptive exit (KL 1e-3 code default / latent-diff 0.03); per-(iteration, layer) cache; `latest-m4` substitutes the latest same-layer cache entry for tokens that exited earlier; `continuous_compute` warm-starts s0 from the previous token's final latent.

## Config contract

### `RecurrenceConfig` (new, `spalmer/config.py`, next to `PLEConfig`)

```python
@dataclass(frozen=True, slots=True)
class RecurrenceConfig:
    prelude_layers: int                # >= 1 physical blocks before the core
    core_layers: int                   # >= 1 physical blocks iterated r times
    coda_layers: int                   # >= 1 physical blocks after the core
    default_steps: int = 8             # inference r when a caller passes none; >= 1
    latent_init_std: float = 1.0       # sigma of s0 ~ N(0, sigma^2); 0.0 => zeros; must be >= 0 and finite
    adapter: Literal["linear"] = "linear"   # Linear(2d -> d, bias=False) over cat[s, e]

    __post_init__: _require_positive on the three layer counts and default_steps; latent_init_std < 0 or non-finite -> ValueError; adapter != "linear" -> ValueError.
    @property total_layers -> prelude + core + coda
    def core_range(self) -> range: range(prelude_layers, prelude_layers + core_layers)
    def role(self, layer_index) -> Literal["prelude","core","coda"]
    def effective_depth(self, steps: int) -> int: prelude + steps*core + coda
    def to_dict(self) -> asdict(self)
```

### `SPALMERConfig` (config.py:76-155) changes

- New last field `recurrence: RecurrenceConfig | None = None`.
- `__post_init__` (100-120) adds: if `isinstance(self.recurrence, Mapping)`: `object.__setattr__(self, "recurrence", RecurrenceConfig(**self.recurrence))` (checkpoint round-trip: `to_dict()` = `asdict` serializes the nested dataclass as a dict, and `load_checkpoint` reconstructs with `SPALMERConfig(**raw)`); then if `self.recurrence is not None and self.recurrence.total_layers != self.n_layers`: `ValueError("recurrence layers prelude+core+coda=%d must equal n_layers=%d")`.
- New helpers: `block_role(layer_index) -> "prelude"|"core"|"coda"|"stack"` ("stack" when recurrence is None), `core_layer_indices -> tuple[int,...]` (empty when None), `effective_depth(steps: int | None = None) -> int` (n_layers when None; otherwise `recurrence.effective_depth(steps or recurrence.default_steps)`), `core_mixer_counts -> dict[str,int]` counting `token_mixer_for_layer(i)` over the core range (for plan/UI text: "core 3 KDA / 1 MLA").
- `n_layers` keeps meaning physical blocks everywhere (PLE tables, `token_mixer_for_layer`, checkpoint validators, accounting, registry `num_layers`). `token_mixer_pattern` stays by physical index; no new constraint (a (1,2,1) split of n=4 yields a core of 2 KDA blocks and no MLA — the plan preview and CLI print the core composition so this is visible, not silent).
- `to_dict()` unchanged (asdict); with `recurrence=None` the dict gains the key `"recurrence": None` — this is why FORMAT_VERSION bumps (below).

### `AdaptiveExit` (new, `spalmer/modeling.py`)

```python
@dataclass(frozen=True, slots=True)
class AdaptiveExit:
    criterion: Literal["latent_diff", "kl"]
    threshold: float | None = None      # None -> DEFAULT_EXIT_THRESHOLDS[criterion] = {"latent_diff": 0.03, "kl": 1e-3}
    min_steps: int = 2                  # earliest iteration count at which the criterion may fire (criteria compare successive iterations, so 1 is clamped to 2)
    state_policy: Literal["fill", "skip"] = "fill"
    __post_init__: criterion in set else ValueError; threshold is not None and (threshold <= 0 or not finite) -> ValueError; min_steps < 1 -> ValueError
    @property resolved_threshold -> float
```
Exported from `spalmer/__init__.py` together with `RecurrenceConfig`, `RecurrentMixerStates`, `RecurrenceTrace` (runtime).

### `TrainingConfig` (training/config.py:11-129) new fields (defaults mean "not recurrent")

```python
mean_recurrence: float | None = None        # r̄; None => model must have recurrence=None
mean_backprop_depth: int = 4                # k
recurrence_sigma: float = 0.5               # log-normal sigma
recurrence_sampling: Literal["poisson_lognormal", "fixed"] = "poisson_lognormal"
max_recurrence: int | None = None           # hard cap on sampled r; None => uncapped
```
Validation (append to `__post_init__` 50-100): `mean_recurrence is not None and mean_recurrence < 1` -> ValueError; `mean_backprop_depth < 1` -> ValueError; `recurrence_sigma < 0` -> ValueError; `recurrence_sampling` not in set -> ValueError; `max_recurrence is not None and max_recurrence < 1` -> ValueError; `recurrence_sampling == "fixed" and mean_recurrence is not None and mean_recurrence != int(mean_recurrence)` -> ValueError("fixed recurrence sampling requires an integer mean_recurrence"); `max_recurrence is not None and mean_recurrence is not None and max_recurrence < mean_recurrence` -> ValueError. `to_dict` unchanged (asdict). `tokens_per_optimizer_step`/`token_budget` unchanged (tokens are supervised targets, not block passes).

### `RecurrenceScaleConfig` (new, `spalmer/experiment/planning.py`, next to `ATXYScaleConfig` 49-86)

```python
@dataclass(frozen=True, slots=True)
class RecurrenceScaleConfig:
    prelude_layers: int = 1
    coda_layers: int = 1
    default_steps: int = 8
    latent_init_std: float = 1.0
    __post_init__: positive ints; latent_init_std >= 0
    def core_layers(self, n_layers) -> int: n_layers - prelude - coda  (planner guarantees >= 1)
    def parameter_count(self, d_model) -> int: 2*d*d + d           # adapter proj + latent RMSNorm
    def effective_depth(self, n_layers, steps=None) -> int
```
`ModelScaleConfig` gains `recurrence: RecurrenceScaleConfig | None = None` (after `atxy`, planning.py:117). `__post_init__` (119-164) adds: `recurrence is not None and n_layers <= prelude+coda` -> ValueError("recurrence needs at least one core layer"); `recurrence is not None and atxy is not None and prelude <= atxy.injection_layer < n_layers - coda` -> ValueError("ATXY injection_layer must lie outside the recurrent core"). `to_dict` (190): `values = asdict(self); if self.recurrence is None: values.pop("recurrence"); return values` — existing fingerprints stay identical. Add `core_layers` / `effective_depth(steps)` properties.

## Model changes

### `RecurrentMixerStates` (new, `spalmer/modeling.py`, after `BlockOutput`)

```python
@dataclass(frozen=True, slots=True)
class RecurrentMixerStates:
    """Per-iteration token-mixer states of ONE core block: iterations[i] is the state after iteration i."""
    iterations: tuple[Any, ...]
    def __post_init__: if not self.iterations -> ValueError("recurrent mixer states need at least one iteration"); any element is None -> ValueError
    def __len__ / __getitem__ / __iter__
    def to(self, *, device=None, dtype=None) -> RecurrentMixerStates: element.to(device=device, dtype=dtype) when it has a callable `to`, else the element unchanged (keeps test doubles working)
```
Container contract: `token_mixer_states` and `channel_mixer_states` stay tuples of length `n_layers` (one slot per physical block). Prelude/coda slots hold exactly what they hold today. Each core slot holds a `RecurrentMixerStates` (or `None` when no state is threaded). Channel-mixer core slots hold `RecurrentMixerStates` too (MicroExpertChannelMixer is stateless so every element is `None` → to keep the "no None elements" rule, channel core slots are `None` when every iteration returned `None`, else a `RecurrentMixerStates`; test mixers with real channel state get the container). `_states_or_none` (665-670) is replaced by `_validate_states(name, states, config)` which keeps the length check and adds, for core indices: slot is `None` or `RecurrentMixerStates`, else `TypeError("core block %d %s state must be RecurrentMixerStates, got %s")`; a `RecurrentMixerStates` in a non-core slot -> TypeError. Output slot length for a core block = `len(input slot)` when a slot was supplied, else `iterations_run` (see algorithm): prefill with `None` produces `r` states; every decode call returns a slot of the same length it received, so the container never shrinks across a generation.

Why a dataclass rather than a bare tuple: it is type-distinguishable from mixer states (fail-closed validation), carries `.to()` for device moves, and is what the facade/telemetry can introspect (`len(slot)` = state budget) without knowing mixer types.

### `RecurrenceAdapter` (new module class in `spalmer/modeling.py`)

```python
class RecurrenceAdapter(nn.Module):
    def __init__(self, d_model: int, *, norm_eps: float, initializer_range: float):
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)     # Huginn A: R^{2d} -> R^d
        self.latent_norm = RMSNorm(d_model, eps=norm_eps)           # learned; applied to the latent at the end of every iteration
        with torch.no_grad(): self.proj.weight.copy_(torch.cat([torch.eye(d), torch.eye(d)], dim=1)); self.proj.weight.add_(torch.randn_like(...) * initializer_range)
    def inject(self, latent, prelude_output) -> Tensor: return self.proj(torch.cat([latent, prelude_output], dim=-1))
    def normalize(self, latent) -> Tensor: return self.latent_norm(latent)
```
Parameter names: `backbone.recurrence_adapter.proj.weight` [d, 2d] and `backbone.recurrence_adapter.latent_norm.weight` [d]. Init `[I | I] + N(0, initializer_range²)` so that at step 0 the adapter is additive injection (s + e) and nothing random is imposed between s0/e and the core; it is fully learnable (2-D weight decays; the norm weight is no-decay by the `"norm" in name` rule at optim.py:66-76).

### `SPALMERBackbone.__init__` (modeling.py:184-211)

- `rc = config.recurrence`; `self.recurrence_adapter = None if rc is None else RecurrenceAdapter(config.d_model, norm_eps=config.norm_eps, initializer_range=config.initializer_range)`; store `self._core_range = rc.core_range() if rc else range(0)`.
- ATXY check extension after 200-204: `if rc is not None and atxy.config.injection_layer in rc.core_range(): raise ValueError("ATXY injection_layer=%d lies inside the recurrent core [%d, %d)")`.
- Existing length/PLE checks unchanged (`n_layers` physical). PLE gate init stays `1/sqrt(n_layers)` (config.py:52) — documented, not changed.

### `SPALMERBlock.advance_token_mixer` (new method, modeling.py after `forward` 94-166)

```python
def advance_token_mixer(self, hidden_states, *, token_mixer_state, execution_mode, attention_mask=None, state_reset_mask=None, mixer_kwargs=None) -> Any:
    """Run only the token-mixer branch to advance its state; the update is discarded (used by the 'fill' exit policy)."""
    kwargs as in forward; normalized = self.token_norm(hidden_states)
    mixed = self.token_mixer(normalized, state=..., **kwargs) if prefill else step(...)
    return mixed[1] if isinstance(mixed, tuple) else token_mixer_state
```

### Backbone forward — algorithm

Signature (modeling.py:213-224) gains keyword-only: `recurrence_steps: int | None = None`, `backprop_steps: int | None = None`, `latent_init: Tensor | None = None`, `adaptive_exit: AdaptiveExit | None = None`, `latent_generator: torch.Generator | None = None`, `exit_readout: Callable[[Tensor], Tensor] | None = None` (maps final-normed hidden `[B,1,d]` to logits; supplied by `SPALMERCausalLM`, required when `adaptive_exit.criterion == "kl"`). `BackboneOutput` gains `recurrence_steps: int | None = None`, `latent_states: Tensor | None = None`, `latent_deltas: Tensor | None = None`.

Non-recurrent config (`rc is None`): any of `recurrence_steps/backprop_steps/latent_init/adaptive_exit/latent_generator` not None -> `ValueError("model has no recurrent core; recurrence arguments are not accepted")`; then the existing loop (260-315) runs unchanged and the three new output fields stay `None`.

Recurrent config, shapes `B,T,d`; P,C,K = prelude/core/coda; `n = P+C+K`:

```
r  = recurrence_steps if not None else rc.default_steps;  r < 1 -> ValueError
k  = backprop_steps if not None else r;  k < 1 -> ValueError;  k = min(k, r)
grad_start = r - k                                   # iterations [0, grad_start) run under torch.no_grad()
if adaptive_exit is not None and execution_mode != "decode": ValueError("adaptive exit is only available in decode mode; prefill runs every requested iteration")
if adaptive_exit is not None and adaptive_exit.criterion == "kl" and exit_readout is None: ValueError
if latent_init is not None: require shape == (B, T, d) else ValueError
token_states, channel_states = _validate_states(...)   # length n; core slots None | RecurrentMixerStates
for core slots given (decode or prefill): len(slot) < r -> ValueError("core block %d carries %d iteration states but %d recurrence steps were requested; prefill deeper or reduce recurrence_steps")
budget = len(first non-None core token slot) if any else r         # output slot length; all core slots must share one length else ValueError

h = embeddings(ids, layer_index=0); ATXY embed as today (260-264)
# ---- prelude (indices 0..P-1): exactly today's loop body (279-304) incl. PLE inject for index>0 and ATXY inject_values at injection_layer
e = h                                                              # [B,T,d], grad-carrying prelude output
# ---- core PLE looked up ONCE per forward (one stochastic-rounding draw, one unique() sync per core block); re-added every iteration
core_ple = [embeddings(ids, layer_index=P+j) for j in range(C)]    # each [B,T,d]; index P+j >= 1 always since P >= 1
# ---- latent init
if latent_init is not None: s = latent_init.to(dtype=h.dtype)
elif rc.latent_init_std == 0: s = zeros(B,T,d)
else: s = (torch.randn(B,T,d, device=h.device, dtype=torch.float32, generator=latent_generator) * rc.latent_init_std).to(h.dtype)
    # generator None -> global torch RNG (covered by RNGSnapshot); generation passes its own generator
new_core_token = [[] for _ in range(C)]; new_core_channel = [[] for _ in range(C)]
core_aux = [[] for _ in range(C)]; core_metrics = [dict() for _ in range(C)]
deltas = []                                                        # python list of 0-d float32 tensors (no host sync)
exited = zeros(B, bool); exit_step = full((B,), r, long); iteration_inputs = [None]*C; frozen_inputs = [None]*C
prev_log_probs = None; iterations_run = 0
for i in range(r):
    grad_iteration = i >= grad_start
    with (nullcontext() if grad_iteration else torch.no_grad()):
        if grad_iteration and i == grad_start: s = s.detach()      # autograd boundary: the no_grad prefix is a constant
        s_prev = s
        x = self.recurrence_adapter.inject(s, e)                    # [B,T,d]; e carries grad into every grad iteration (prelude learns every step)
        for j in range(C):
            p = P + j
            x = x + core_ple[j]
            if exited.any(): x = torch.where(exited[:, None, None], frozen_inputs[j], x)   # B>1 only; frozen rows keep the block input they saw when they exited
            iteration_inputs[j] = x
            slot = token_states[p]; state_in = None if slot is None else slot.iterations[i]
            cslot = channel_states[p]; cstate_in = None if cslot is None else cslot.iterations[i]
            out = blocks[p](x, token_mixer_state=state_in, channel_mixer_state=cstate_in, execution_mode=..., attention_mask=..., state_reset_mask=..., mixer_kwargs=per_layer_kwargs[p])
            x = out.hidden_states
            new_core_token[j].append(out.token_mixer_state); new_core_channel[j].append(out.channel_mixer_state)
            core_metrics[j] = out.metrics                            # overwritten each iteration -> LAST iteration survives
            if grad_iteration and out.auxiliary_loss is not None: core_aux[j].append(out.auxiliary_loss)
        s = self.recurrence_adapter.normalize(x)                     # learned RMSNorm at the end of EVERY iteration (bounded latent)
        if exited.any(): s = torch.where(exited[:, None, None], s_prev, s)
    iterations_run = i + 1
    with torch.no_grad():
        per_seq_delta = ((s - s_prev).float().norm(dim=-1) / s.float().norm(dim=-1).clamp_min(1e-6)).mean(dim=1)   # [B], mean over positions
        deltas.append(per_seq_delta.mean())                          # 0-d tensor, stays on device
    if adaptive_exit is not None and iterations_run < r and iterations_run >= max(2, adaptive_exit.min_steps):
        if criterion == "latent_diff": signal = per_seq_delta                                             # [B]
        else:  # "kl": run coda on a PROBE copy (coda states are inputs only; outputs discarded), final_norm, readout
            probe_hidden = self.final_norm(self._run_coda(s, token_states, channel_states, ..., collect=False))   # [B,T,d]
            log_probs = F.log_softmax(exit_readout(probe_hidden[:, -1:, :]).float(), dim=-1)                  # [B,1,V]
            signal = full((B,), inf) if prev_log_probs is None else (log_probs.exp() * (log_probs - prev_log_probs)).sum(-1).squeeze(1)   # KL(p_i || p_{i-1})
            prev_log_probs = log_probs
        newly = (signal < adaptive_exit.resolved_threshold) & ~exited                                          # host sync happens on .any()/.all() below (decode only)
        if bool(newly.any()):
            exit_step = torch.where(newly, iterations_run, exit_step)
            for j in range(C): frozen_inputs[j] = iteration_inputs[j] if frozen_inputs[j] is None else torch.where(newly[:, None, None], iteration_inputs[j], frozen_inputs[j])
            exited |= newly
            if bool(exited.all()): break
# ---- deeper iteration states for exited sequences (also covers "decode with fewer steps than the state budget")
if budget > iterations_run:
    for i2 in range(iterations_run, budget):
        for j in range(C):
            p = P + j; state_in = token_states[p].iterations[i2]; cstate_in = channel slot element or None
            if adaptive_exit is None or adaptive_exit.state_policy == "fill":
                fill_input = iteration_inputs[j] if not exited.any() else frozen_inputs[j]     # B=1: the block input of the exit iteration (Huginn latest-mC analogue)
                new_core_token[j].append(blocks[p].advance_token_mixer(fill_input, token_mixer_state=state_in, execution_mode=..., ...))
            else:  # "skip": leave the deeper state untouched (missing this token's contribution)
                new_core_token[j].append(state_in)
            new_core_channel[j].append(cstate_in)
# ---- coda (indices P+C..n-1): today's loop body on h = s, incl. PLE inject and ATXY inject_values; states/metrics/aux collected as today
hidden = final_norm(h)
token_out = prelude states + [RecurrentMixerStates(tuple(new_core_token[j])) for j] + coda states       # len n
channel_out = prelude + [None if all(x is None for x in new_core_channel[j]) else RecurrentMixerStates(tuple(new_core_channel[j]))] + coda
layer_metrics = prelude metrics + core_metrics (one per core block, LAST iteration) + coda metrics          # len n, unchanged contract
aux = prelude aux + [torch.stack(core_aux[j]).mean() for j if core_aux[j]] + coda aux -> torch.stack(...).mean() as today (306-308)
return BackboneOutput(hidden, token_out, channel_out, aux, layer_metrics, recurrence_steps=iterations_run, latent_states=s.detach(), latent_deltas=torch.stack(deltas))   # deltas: [iterations_run] float32
```

Autograd boundaries: no graph is built for iterations `< grad_start` (torch.no_grad); `s.detach()` at `grad_start` is explicit even though `s` already has `requires_grad=False`; `core_ple[j]` and `e` are computed outside no_grad, so the k grad iterations backpropagate into PLE tables and prelude (k contributions each); coda always has gradient; loss is on final logits only. When `torch.is_grad_enabled()` is False (eval), the split is inert. Under the trainer's bf16 autocast (engine.py:442) `s` is bf16; the delta metric and the KL probe are computed in float32. Memory: at most k core-iteration graphs are alive (plus prelude/coda); forward FLOPs are `(P+K) + r*C` block passes.

`_run_coda(...)` is a private helper factored out of the coda loop so the KL probe can call it with `collect=False` (no state/metric/aux appends, no ATXY injection duplicates — ATXY inject_values is idempotent per call but the probe passes `atxy=None`). Coda token-mixer states are pure inputs to the probe; the probe's returned states are discarded, so the real coda pass afterwards sees exactly the states it would have seen without a probe.

Per-sequence semantics (B>1, backbone-level only): once a sequence exits at e its latent is frozen at s_e; in every deeper iteration each core block's token-mixer state for that row is advanced with the block input it saw at iteration e (`frozen_inputs[j]`), which is precisely what the B=1 "fill" tail does with token-mixer-only stepping. Exited rows still pay the full block (no masking is possible: mixers reject non-trivial masks at mixer.py:199-227 / mla.py:273-301) — cost, not correctness.

State semantics summary:
- Prefill (`slot is None`): every iteration i starts from a fresh zero state, `slot_out = r` states; prefill NEVER exits (ValueError if asked), so every iteration's KDA/MLA memory covers the whole prompt — otherwise deeper iterations would start decode with an EMPTY memory (no prompt at all), which is far worse than a missing token.
- Decode: iteration i steps `slot.iterations[i]`; output slot length = input slot length; iterations not run (`iterations_run < budget`) are advanced with the exit-iteration input ("fill", default) or carried untouched ("skip"). "fill" is the KDA/MLA analogue of Huginn `latest-m4` and keeps every iteration's memory aligned with the token stream; "skip" is the cheap ablation.
- Decode with `recurrence_steps > len(slot)` is an error (no memory exists for those iterations). Decode with fewer steps than the slot is allowed and treated exactly like an exit at that step.
- Warm start: `latent_init=[B,1,d]` replaces s0 (no noise).

### `SPALMERCausalLM.forward` (modeling.py:530-595) and `CausalLMOutput`

- Signature gains `recurrence_steps`, `backprop_steps`, `latent_init`, `adaptive_exit`, `latent_generator` (all keyword-only, default None) and forwards them to the backbone plus `exit_readout=(lambda hidden: self.lm_head(hidden)) if adaptive_exit is not None and adaptive_exit.criterion == "kl" else None`.
- `CausalLMOutput` (51-61) gains `recurrence_steps: int | None = None`, `latent_states: Tensor | None = None`, `latent_deltas: Tensor | None = None`, copied from the backbone output. All existing keyword constructions remain valid.
- Loss/calibration/entropy code unchanged: `layer_metrics` still has one entry per physical block, so `_attach_surprise_telemetry` (681-723) calibrates the core's LAST-iteration routing (always inside the grad window because k >= 1), `update_potentiation` (597-641) sees one core vote, `_PotentiationAccumulator` sees a constant length, `RouterTelemetryAccumulator` keys stay physical, `rank_nonresident_experts` ranks final-state routing. `_ExecutionTelemetryAccumulator` maxima now cover only the last iteration per core block (documented).
- New convenience: `SPALMERCausalLM.recurrence_default_steps -> int | None` (config.recurrence.default_steps) and `is_recurrent -> bool`.

### Telemetry aggregation contract (single source of truth)

| Signal | Non-recurrent | Recurrent core block |
|---|---|---|
| `layer_metrics[p]` | per block | LAST iteration's mixer metrics (routing, utilization, quantization error, grouped-padding scalars, directional) |
| auxiliary loss | mean over blocks | per core block: mean over GRAD iterations only; then mean over blocks |
| surprise calibration / attributed NLL | per block | from the last iteration's router scores (grad window) |
| `potentiation_utilization` | per block | last iteration |
| `recurrence_steps` (output) | None | iterations run (B>1: max over rows) |
| `latent_deltas` | None | `[iterations_run]` float32, mean over batch and positions of relative latent change |
| `latent_states` | None | final `[B,T,d]`, detached |

### Expert paging cost in the core (no core change)

`MicroExpertBank` paged execution re-stages rows per `execute_routing` call (bank.py:362-386, 775-881) and `quantization_error` is valid only for the most recent execution (bank.py:557-569); the mixer reads it in the same call (mixer.py:199-210), so per-iteration re-routing is safe. Consequences, documented in `enable_expert_offload` docstring and the yyLab load note: the core bank's `stage_operations/transferred_rows/evicted_rows` counters grow up to r× per forward; `ExpertOffloadTelemetry.layer_count` stays the physical bank count. Guidance: choose `cache_size >= |union of experts selected across iterations|` (UI hint: "with recurrence the per-layer cache should cover the union of routes across iterations; the paged counters count every iteration"). `choose_inference_residency` expansions each cost a full prefill × r.

## Training changes

### `RecurrenceSampler` (new `spalmer/training/recurrence.py`)

```python
@dataclass(frozen=True, slots=True)
class RecurrenceSampler:
    mean_recurrence: float; mean_backprop_depth: int; sigma: float = 0.5
    scheme: Literal["poisson_lognormal", "fixed"] = "poisson_lognormal"; max_recurrence: int | None = None; seed: int = 0
    @classmethod def from_training_config(cls, config: TrainingConfig) -> RecurrenceSampler | None   # None when mean_recurrence is None
    def sample(self, step: int) -> tuple[int, int]:  # (r, k_eff)
        rng = random.Random(f"spalmer-recurrence:{self.seed}:{step}")     # stateless, str seed -> sha512, process independent, needs no run-state slot
        if scheme == "fixed": r = int(mean_recurrence)
        elif mean_recurrence <= 1.0: r = 1
        else:
            mu = math.log(mean_recurrence - 1.0) - sigma**2 / 2          # E[r] == mean_recurrence exactly
            rate = math.exp(rng.gauss(mu, sigma))
            r = _poisson(rng, rate) + 1
        if max_recurrence is not None: r = min(r, max_recurrence)
        k = min(mean_backprop_depth, r)
        return r, k
def _poisson(rng, rate): Knuth for rate <= 500 (L=exp(-rate); k=0; p=1; loop p*=rng.random(); until p<=L; return k), normal approximation max(0, round(gauss(rate, sqrt(rate)))) above.
```
Formula rationale: Huginn samples `rate ~ LogNormal(log(r̄) - σ²/2, σ)`, `r = Poisson(rate)+1`, giving E[r] = r̄ + 1; we shift the mean by one so `mean_recurrence` is the true expectation and `mean_recurrence = 1` degenerates to a plain stack (r = 1 always). Draws never touch the global torch/python RNG, so PLE/QAT stochastic-rounding streams are unchanged relative to a non-recurrent run and resume needs no new run-state payload (r is a pure function of `(config.seed, completed_steps)`). Latent noise uses the global torch RNG (inside `RNGSnapshot`).

### `ExperimentTrainer` (engine.py:372-512)

- `__init__` after `_validate_expert_master_dtype` (395): `self.recurrence = RecurrenceSampler.from_training_config(config)`; `is_recurrent = model.config.recurrence is not None`; `is_recurrent and self.recurrence is None` -> `ValueError("recurrent model requires TrainingConfig.mean_recurrence")`; `not is_recurrent and self.recurrence is not None` -> `ValueError("mean_recurrence was set but the model has no recurrent core")`.
- `_optimizer_step` (421): after `step_index` compute `recurrence = None if self.recurrence is None else self.recurrence.sample(step_index)`; the forward at 443-448 gains `recurrence_steps=recurrence[0], backprop_steps=recurrence[1]` when not None (constant across all microbatches of the step → `_PotentiationAccumulator` invariant holds).
- `TrainStepMetrics` (102-117) gains `recurrence_steps: int | None = None`, `backprop_steps: int | None = None`, `effective_depth: int | None = None` (= `config.effective_depth(r)`), filled at 485-510.
- No change to `_objective`, accumulators, callbacks.

### `train_token_stream` (runtime.py:29-133)

Gains `recurrence: RecurrenceSampler | None = None`; per step `r,k = recurrence.sample(step-1)` and `model(batch, labels=batch, recurrence_steps=r, backprop_steps=k)`; same fail-closed checks as the trainer. `TrainResult` gains `mean_recurrence_steps: float | None = None`.

### Evaluation (training/evaluation.py:41-93)

`evaluate_model_batches` gains `recurrence_steps: int | None = None`, forwarded at 76-81 (full backprop irrelevant under inference_mode). Router telemetry keying (154) is unchanged because entries are physical.

### Resume (resume.py:295-313)

`_validate_authoritative_metadata`: normalise the saved `training_config` through `TrainingConfig(**saved).to_dict()` before equality when `saved` is a mapping lacking the five new keys (legacy run states written before this change), so old run states still resume; any explicit mismatch still raises `RunStateCompatibilityError`.

## Inference changes

### `generate_tokens` (runtime.py:173-243)

New keyword-only parameters: `recurrence_steps: int | None = None` (None → `model.config.recurrence.default_steps`), `adaptive_exit: AdaptiveExit | None = None`, `warm_start: bool = False`, `latent_seed: int | None = None` (None → reuse `seed` for the latent generator), `trace: RecurrenceTrace | None = None`.

```python
@dataclass(slots=True)
class RecurrenceTrace:          # caller-owned mutable receipt (return type of generate_tokens stays Tensor)
    max_steps: int = 0; prefill_steps: int = 0; decode_steps: list[int] = field(default_factory=list); early_exits: int = 0
    @property mean_decode_steps / min_decode_steps / max_decode_steps
```
Behaviour: non-recurrent model with any of these set (steps not None, adaptive_exit not None, warm_start True) -> `ValueError("model has no recurrent core")`. Recurrent: `steps = recurrence_steps or default`; `latent_generator = torch.Generator(device).manual_seed(latent_seed if not None else seed)`; prefill `model(generated, recurrence_steps=steps, latent_generator=g)` (dynamic residency: `choose_inference_residency(model, generated, recurrence_steps=steps)`); each decode call `model(next_token, token_mixer_states=..., channel_mixer_states=..., execution_mode="decode", recurrence_steps=steps, adaptive_exit=adaptive_exit, latent_init=(output.latent_states[:, -1:, :] if warm_start else None), latent_generator=g)`; after each call `trace.decode_steps.append(output.recurrence_steps)`, `early_exits += output.recurrence_steps < steps`. Prefill sets `trace.prefill_steps = steps`.

### `choose_inference_residency` (residency.py:389-511) and `_evaluate` (525-536)

Gain `recurrence_steps: int | None = None` threaded into every `model(...)` call in `_evaluate` (None → model default). Nothing else changes (one router-score entry per block).

### Warm start

`latent_init` is the previous token's final latent `[B,1,d]`; iteration 0 skips noise. It is orthogonal to adaptive exit (Huginn `continuous_compute`, saves 1-2 iterations). Not used in prefill.

## Checkpoint / versioning (`spalmer/checkpoint.py`)

- `FORMAT_VERSION = 6`; `_SUPPORTED_FORMAT_VERSIONS = {1,2,3,4,5,FORMAT_VERSION}` (36-37). New `_VERSION_6_MODEL_CONFIG_FIELDS = {"recurrence"}`.
- `load_checkpoint` 491-495: `if checkpoint_version < 6: required_model -= _VERSION_6_MODEL_CONFIG_FIELDS` (v5 bundles construct with `recurrence=None`). Then: `raw_recurrence = raw_model_config.get("recurrence")`; if it is a mapping, `_require_config_schema("recurrence", raw_recurrence, {f.name for f in fields(RecurrenceConfig)})` before `SPALMERConfig(**raw_model_config)` (post_init coerces the dict).
- `_migrate_experts_config` 604: `if checkpoint_version >= 5:` (equality trap). v5 and v6 expert schemas are identical.
- `save_checkpoint` payload unchanged in shape (`model_config` now contains `"recurrence": {...}|None`). New validator `_validate_model_recurrence(model)` called from `save_checkpoint` (78-82) and `load_checkpoint` (after 570): `(model.config.recurrence is None) == (model.backbone.recurrence_adapter is None)` else ValueError; ATXY-inside-core check is already enforced by the backbone constructor.
- New parameters exist only when recurrence is set, so v5 bundles load `strict=True` unchanged; a v6 recurrent bundle's `model_state` contains the two adapter keys and the binding digest covers them.
- `_load_model_parameter_dtype` (`< 5`), `_load_optional_component_configs` (`< 5`), binding check (`>= 4`) are already inequality-based — leave.
- Metadata: `save_checkpoint` callers (CLI 442-462, yyLab) add a `recurrence` block (see below); the authoritative default depth is `model_config.recurrence.default_steps` (part of the model identity, hashed into `config_sha256`), inference policy (adaptive exit) is NOT part of the checkpoint.

## Planning / accounting / CLI

### `count_parameters` (planning.py:285-358)

`recurrence_adapter = 0 if config.recurrence is None else config.recurrence.parameter_count(d)`; `ParameterBreakdown` (195) gains `recurrence_adapter: int = 0` (last field); `total` includes it; `components` (249-263) gains `"recurrence_adapter": self.recurrence_adapter`; `to_dict` (275-283) pops `"recurrence_adapter"` when 0 so every existing plan's `to_dict()`/fingerprint and the 9_908_080 golden are byte-identical; `dense_parameters` includes it. `estimate_memory` counts it through `total` (bytes-of-parameters only; activations still excluded and now documented as "× k retained core iterations").

### `ScalePlan` (588-622)

New properties `effective_depth` (= `config.effective_depth()` = n_layers when not recurrent) and `block_passes_per_token`; `to_dict` adds `"recurrence": {"prelude_layers","core_layers","coda_layers","default_steps","effective_depth","core_mixers": {"kda": x, "mla": y}}` ONLY when `config.recurrence is not None` (fingerprint is computed from the four hashed parts and stays stable for non-recurrent plans; recurrent plans hash `config.to_dict()` which includes the recurrence block).

### `plan_configuration` / `plan_ladder` (625-725)

Gain `recurrence: RecurrenceScaleConfig | None = None`; grid skips `n_layers <= prelude + coda` and `atxy` injection inside the core; `ModelScaleConfig(..., recurrence=recurrence)`. Search key unchanged (parameter distance; recurrence adds only `2d²+d`). `__all__`/`spalmer/experiment/__init__.py` export `RecurrenceScaleConfig`.

### `presets.build_configs` (139-238)

`_MODEL_SHAPE_FIELDS` (106-113) gains `"recurrence"`. `model_fields` (166-177) gains `"recurrence": None if config.recurrence is None else RecurrenceConfig(prelude_layers=..., core_layers=config.recurrence.core_layers(config.n_layers), coda_layers=..., default_steps=config.recurrence.default_steps, latent_init_std=config.recurrence.latent_init_std)`. `assert_accounting_matches_plan` (242-271) unchanged (components compared 1:1; `expected_per_expert` stays physical).

### `accounting.py`

`_COMPONENT_ORDER` (21-32) gains `"recurrence_adapter"` before `"vocab_head"`; `classify_parameter` (156-177): insert `if ".recurrence_adapter." in name: return "recurrence_adapter"` BEFORE the `.directional_mixer.` rule (it must precede the `norm` rule at 175). `ParameterAccounting` gains `effective_depth: int | None = None` and `per_token_block_passes: int | None = None` (filled from `model.config.effective_depth()` by `account_parameters`); `summary()` prints "effective depth P+r×C+K" when recurrent. `per_token_active_parameters` stays a distinct-parameter count (docstring notes compute = passes × per-pass).

### CLI (`__main__.py`)

- train (`_training_parser` 120-172): `--recurrent-layers PRELUDE CORE CODA` (nargs=3, int; sum must equal `--layers` else `ValueError`), `--mean-recurrence FLOAT` (required when `--recurrent-layers` given), `--backprop-depth INT` (default 4), `--recurrence-sigma FLOAT` (default 0.5), `--default-steps INT` (default `round(mean_recurrence)`), `--max-recurrence INT`. `_run` (333-477): build `RecurrenceConfig` into `SPALMERConfig(recurrence=...)` (364-373), `RecurrenceSampler(...)` passed to `train_token_stream`; metadata (442-462) gains `"recurrence": {prelude, core, coda, default_steps, latent_init_std, mean_recurrence, mean_backprop_depth, recurrence_sigma, mean_recurrence_steps}` or `None`; summary line (471-475) appends `effective_depth=... mean_r=...` when recurrent.
- generate (`_generate_parser` 175-213): `--recurrence-steps INT`, `--adaptive-exit {none,latent_diff,kl}` (default none), `--exit-threshold FLOAT`, `--min-steps INT` (default 2), `--warm-start` (store_true), `--exit-state-policy {fill,skip}` (default fill). `_run_generate` builds `AdaptiveExit` when not none, passes `trace=RecurrenceTrace()`, and prints after the text: `recurrence: steps={max} prefill={p} decode mean={m:.2f} min={} max={} early_exits={n}` when the model is recurrent. Non-recurrent checkpoint + any recurrence flag -> `SystemExit("checkpoint has no recurrent core")`.
- plan (`_plan_parser` 63-91): `--recurrence PRELUDE CODA` (nargs=2) and `--recurrence-steps INT` (default 8) → `RecurrenceScaleConfig`; the table (236-253) gains columns `core` (core_layers or "-") and `eff` (effective depth) ; `--json` includes the `recurrence` block via `to_dict`.

## yyLab changes

### Bridge `yylab/train/spalmer.py`

- `_load_spalmer_api` (112-163): import and expose `RecurrenceConfig` (spalmer.config), `RecurrenceScaleConfig` (planning), `FORMAT_VERSION` (spalmer.checkpoint). Add `SUPPORTED_SPALMER_FORMAT_VERSIONS = frozenset({5, 6})` module constant.
- `_normalize_scratch_architecture` (311-446): after `spalmer["atxy"] = atxy` (389) normalise `spalmer["recurrence"]`: mapping with `enabled` (default False); when enabled require `prelude_layers >= 1`, `coda_layers >= 1`, `default_steps >= 1` (ints), `latent_init_std >= 0` (default 1.0); unknown keys -> ValueError naming them; store canonical dict `{"enabled": bool, ...}`. Do not add `recurrence` to `_MODEL_RUNTIME_FIELDS` (it is a plan field).
- `normalize_spalmer_config` (449-635): `training_aliases` (546-564) gains `"mean_recurrence"`, `"mean_backprop_depth"`, `"recurrence_sigma"`, `"recurrence_sampling"`, `"max_recurrence"` (root ← training). Validation after the aliases: `recurrent = cfg["spalmer"].get("recurrence", {}).get("enabled")` in scratch mode; scratch + recurrent → `mean_recurrence` required (float >= 1), `mean_backprop_depth` int >= 1 (default 4), `recurrence_sigma` float >= 0 (default 0.5), `recurrence_sampling` in set (default poisson_lognormal), `max_recurrence` optional int >= mean; scratch + not recurrent → any of these keys present and not None -> ValueError("recurrence training keys require spalmer.recurrence.enabled"). Checkpoint mode: `_normalize_checkpoint_execution` (289-308) unchanged (architecture inherited; recurrence keys are TRAINING keys and are allowed through `cfg["training"]`/root); consistency with the parent is checked in `_prepare_checkpoint_run`.
- `_scale_components` (665-685) returns a third value: `api.RecurrenceScaleConfig(prelude_layers, coda_layers, default_steps, latent_init_std)` when enabled else None; `plan_spalmer_configuration` (688-720) passes `recurrence=...`.
- `build_spalmer_training_config` (746-801): pass `mean_recurrence`, `mean_backprop_depth`, `recurrence_sigma`, `recurrence_sampling`, `max_recurrence` from cfg (None/defaults when absent).
- `_prepare_checkpoint_run` (1375-1420): `format_version not in SUPPORTED_SPALMER_FORMAT_VERSIONS` -> ValueError("SPALMER continued training requires a format-version 5 or 6 bundle"); after loading: `parent_recurrent = getattr(model.config, "recurrence", None) is not None`; `parent_recurrent and cfg.get("mean_recurrence") is None` -> ValueError("parent checkpoint is recurrent; set training.mean_recurrence"); `not parent_recurrent and cfg.get("mean_recurrence") is not None` -> ValueError("parent checkpoint has no recurrent core; remove recurrence training keys"). `bundle.model = symbols.SPALMERConfig(**payload["model_config"])` already round-trips the nested dict.
- `_metric_payload` (1117-1184): add `"recurrence_steps"`, `"backprop_steps"`, `"effective_depth"` (from `TrainStepMetrics`, None for non-recurrent) and append ` · r={r} k={k}` to `msg` when present; `_TelemetryWriter.record` prints `r=` too.
- `_save_bundle_checkpoint` (1442-1479): `metadata["recurrence"] = _recurrence_metadata(prepared)` = `{"prelude_layers","core_layers","coda_layers","default_steps","latent_init_std","effective_depth","mean_recurrence","mean_backprop_depth","recurrence_sigma","recurrence_sampling","max_recurrence"}` or None (model side from `prepared.bundle.model.recurrence`, training side from `prepared.training`).
- Manifest (1679-1704): `"checkpoint_format": f"spalmer.prototype.checkpoint.v{int(api.FORMAT_VERSION)}"`, plus `"recurrence": <same block>`. `checkpoint_metadata` (1707-1727): `"recurrence": <block>`, `"effective_depth": int`. Registry `num_layers` stays `plan.config.n_layers` (physical, 1737).

### Page `yylab/ui/pages/train_spalmer.py`

- `SPALMER_RECIPE_VERSION = 4` (29).
- `_cached_plan_json` (128-185) gains 5 trailing positional args `recurrence_enabled: bool, recurrence_prelude: int, recurrence_coda: int, recurrence_default_steps: int, recurrence_latent_init_std: float`, imports `RecurrenceScaleConfig`, passes `recurrence=`; `plan_spalmer_configuration` (188-223) gains `recurrence: dict[str, Any]` and normalises `(enabled, prelude or 1, coda or 1, default_steps or 8, 1.0)` into the cache key.
- Section 2 (726-857) gains, after ATXY controls: switch `recurrence_enabled` ("Recurrent depth (Huginn-style latent reasoning)", default False) and a `field_grid(3)` `recurrence_prelude` (number, default 1, min 1), `recurrence_coda` (default 1, min 1), `recurrence_default_steps` ("Default inference steps", default 8, min 1) inside `recurrence_controls` column (visible iff enabled) + hint text about r× compute and paging cache union.
- Section 4 (943-1062) gains `recurrence_training_controls` grid (visible iff recurrence enabled, or in checkpoint mode iff the parent meta_json `recurrence` block is not None): `mean_recurrence` (default 8.0, min 1), `backprop_depth` (default 4, min 1), `recurrence_sigma` (default 0.5, min 0). `update_initialization_controls` (1125-1130) updates visibility from `_checkpoint_recurrence(parent)`.
- `update_plan_preview` (1203-1285): pass `recurrence=current_recurrence()`; add a line "recurrent: prelude P · core C (x KDA / y MLA) · coda K · effective depth P+C×steps+K = N at steps S · adapter {params}" when `plan.get("recurrence")`; parent branch prints effective depth from `meta.get("recurrence")`.
- `build_spalmer_run_config` (300-603) gains kwargs `recurrence: dict[str, Any]` (default `{"enabled": False}`), `mean_recurrence: float | None = None`, `backprop_depth: int | None = None`, `recurrence_sigma: float | None = None`, `parent_recurrent: bool = False`. Validation: scratch+enabled → prelude/coda >= 1, `int(resolved_plan["config"]["n_layers"]) > prelude + coda`, default_steps >= 1, mean_recurrence >= 1, backprop_depth >= 1, sigma >= 0, and `resolved_plan.get("recurrence")` must be present (else ValueError "resolved SPALMER plan is not recurrent"); scratch+disabled with any training recurrence value set -> ValueError; checkpoint mode: training recurrence keys emitted iff `parent_recurrent`. Emits `spalmer["recurrence"] = {"enabled", "prelude_layers", "coda_layers", "default_steps", "latent_init_std": 1.0}` (scratch), `training["mean_recurrence"|"mean_backprop_depth"|"recurrence_sigma"|"recurrence_sampling"="poisson_lognormal"]` and root aliases `mean_recurrence`, `mean_backprop_depth`, `recurrence_sigma` when recurrent.
- `on_start` (1353-1456): pass the new kwargs; pending row metadata (1467-1475) gains `"recurrence": config["spalmer"].get("recurrence")` (scratch) / parent block (checkpoint).
- New helper `_checkpoint_recurrence(row) -> dict | None` reading `meta_json["recurrence"]`; `_parent_plan` untouched.

### Facade `yylab/models/spalmer.py`

- `scan_spalmer_checkpoint` (71-149): `recurrence = model_config.get("recurrence")` (dict or None); `base_cfg["spalmer_recurrence_config"] = recurrence`; `base_cfg["effective_depth"] = num_layers if recurrence is None else prelude + default_steps*core + coda`; return dict gains `"recurrence_config"`, `"effective_depth"`. `num_layers` stays physical. Accept `format_version in {5, 6}` wherever a version is checked (none in this file today; `_prepare_checkpoint_run` is the only gate).
- `SpalmerCausalLMFacade.__init__` (307-352) gains keyword-only `recurrence_steps: int | None = None`, `adaptive_exit: str = "none"`, `exit_threshold: float | None = None`, `min_steps: int = 2`, `warm_start: bool = False`, `exit_state_policy: str = "fill"`. `recurrence_config = getattr(model.config, "recurrence", None)`; `self.recurrent = recurrence_config is not None`; if not recurrent and (recurrence_steps is not None or adaptive_exit != "none" or warm_start): `ValueError("SPALMER checkpoint has no recurrent core; clear the recurrence settings")`; `adaptive_exit not in {"none","latent_diff","kl"}` -> ValueError; `recurrence_steps is not None and recurrence_steps < 1` -> ValueError; `exit_threshold is not None and exit_threshold <= 0` -> ValueError. Store `self.recurrence_steps = recurrence_steps or (recurrence_config.default_steps if recurrent else None)`, `self.adaptive_exit`, `self.exit_threshold`, `self.min_steps`, `self.warm_start`, `self.exit_state_policy`, `self.last_recurrence_telemetry: dict | None = None`. `self.config` gains `spalmer_recurrence=(dict of the config or None)`, `effective_depth=int`, `recurrence_steps=self.recurrence_steps` (num_hidden_layers stays physical).
- `_native_recurrence_kwargs(self, *, decode: bool) -> dict`: `{}` when not recurrent; else `{"recurrence_steps": self.recurrence_steps}` plus, when decode: `adaptive_exit=AdaptiveExit(criterion, threshold, min_steps, state_policy)` (lazy `from spalmer.modeling import AdaptiveExit`) when `self.adaptive_exit != "none"`.
- `forward` (390-431): `kwargs.setdefault("recurrence_steps", self.recurrence_steps)` when recurrent (both branches), so NLL scoring / context compaction use the loaded depth.
- `generate` (433-570): explicit new parameters `recurrence_steps: int | None = None, adaptive_exit: str | None = None, exit_threshold: float | None = None, min_steps: int | None = None, warm_start: bool | None = None` (each None → facade default; explicit values validated like `__init__`; non-recurrent + any non-None → ValueError). They are named parameters, so `_validate_ignored_generation_kwargs` (928-983) never sees them; `inert_defaults` unchanged. Prefill (507/523) → `self.model(generated, attention_mask=mask, **prefill_kwargs)`; dynamic path (515) → `choose_inference_residency(self.model, generated, recurrence_steps=steps)` (only when recurrent; kwarg omitted otherwise so the fake/legacy signature still works); decode (556-561) → adds `**decode_kwargs` and `latent_init=output.latent_states[:, -1:, :]` when warm start and `output.latent_states is not None`. After every call, record `steps_run = getattr(output, "recurrence_steps", None)` into a local list; on return set `self.last_recurrence_telemetry = {"recurrence_steps": steps, "adaptive_exit": crit, "exit_threshold": resolved or None, "min_steps", "warm_start", "exit_state_policy", "prefill_steps": steps, "decode_tokens": n, "mean_decode_steps", "min_decode_steps", "max_decode_steps", "early_exits", "effective_depth_mean": P + mean*C + K, "per_token_steps": list (capped at 4096 entries)}` (None when not recurrent).
- `routing_telemetry` (372-383): every branch returns a dict; when `self.last_recurrence_telemetry` is not None, `value["recurrence"] = dict(...)`. Benchmarks pick it up through `Engine.routing_telemetry` (backend.py:733-736) automatically.
- `load_spalmer_runtime` (642-650) gains `recurrence_steps: int | None = None, adaptive_exit: str = "none", exit_threshold: float | None = None, min_steps: int = 2, warm_start: bool = False, exit_state_policy: str = "fill"`; forwards to the facade; notes (752-792) add "Recurrent depth: prelude P · core C × {steps} · coda K (effective depth N); adaptive exit {crit} (threshold t) / fixed; warm start on/off; the paged expert cache is re-staged every iteration — size it to the union of routes." when recurrent; validation errors surface from the facade before any GPU placement (construct the facade arguments first, i.e. check `getattr(model.config, "recurrence", None)` right after `load_checkpoint`).

### Engine `yylab/infer/engine.py`

`_load_spalmer` (969-1040): read `spalmer_recurrence_steps` (int; 0/None → None), `spalmer_adaptive_exit` (str, default "none"), `spalmer_exit_threshold` (float; 0/None → None), `spalmer_warm_start` (bool, default False); negative steps/threshold -> ValueError; pass `recurrence_steps=, adaptive_exit=, exit_threshold=, warm_start=` to `load_spalmer_runtime` (always, explicitly). After load write the RESOLVED values back: `settings.update(spalmer_recurrence_steps=facade.recurrence_steps, spalmer_adaptive_exit=facade.adaptive_exit, spalmer_exit_threshold=facade.exit_threshold, spalmer_warm_start=facade.warm_start, spalmer_recurrent=facade.recurrent)` before `replace(spec, ...)` so chat/benchmarks read effective policy. `_generate_kwargs` unchanged (policy is load-time; per-call kwargs exist only for programmatic callers). No settings.json defaults are added (spalmer keys are per-load, as today for `spalmer_expert_cache_size`).

### Chat `yylab/ui/pages/chat.py`

- Controls next to `spalmer_cache_size` (326-341): `spalmer_recurrence_steps` number ("SPALMER recurrence steps (0 = checkpoint default)", min 0, max 256), `spalmer_adaptive_exit` select `{"none": "Fixed depth", "latent_diff": "Adaptive · latent change", "kl": "Adaptive · output KL"}`, `spalmer_exit_threshold` number ("Exit threshold (0 = criterion default)", min 0, step 1e-4), `spalmer_warm_start` checkbox ("Warm-start latent from previous token"); all hidden by default. `refresh_adapters` (487-517): visible iff spalmer AND `_checkpoint_recurrence(row)` is not None (helper reading meta_json `recurrence`); when visible set the steps input to the checkpoint's `default_steps` as placeholder text.
- Load settings (596-606): add the four keys (0/None conventions) — for non-recurrent checkpoints emit `spalmer_recurrence_steps=None`, `spalmer_adaptive_exit="none"`, `spalmer_warm_start=False`.
- Badge (641-653): append ` · depth {resolved['spalmer_recurrence_steps']}` + (` adaptive {crit}` if not none) + (` warm` if warm start) when `resolved.get("spalmer_recurrent")`.
- `_routing_telemetry_text` (153-258): after the paging/resident section, if `isinstance(telemetry.get("recurrence"), dict)`: append ` · recurrence r≤{steps} {crit or 'fixed'}: mean {mean:.1f} steps/token ({min}–{max}), {early_exits} early exits` (and ` · warm start` when set). Unknown keys still ignored.

### Benchmarks

- `benchmarks.py` `_quality_job_config` (70-141): kwargs `spalmer_recurrence_steps: int | None = None`, `spalmer_adaptive_exit: str = "none"`, `spalmer_exit_threshold: float | None = None`, `spalmer_warm_start: bool = False`; spalmer branch (130-140) validates (`steps <= 0` -> ValueError, exit in set, threshold > 0 when set) and adds the four keys. Inputs next to `spalmer_cache_size` (788-799): steps number, exit select, threshold number; visibility at 1028 tied to `spalmer and recurrent`; job build (1107-1111) passes them. `_config_text` (300-308): append ` · depth {steps or 'checkpoint'}{' adaptive '+crit if crit!='none'}` for spalmer.
- `quality.py`: `_stack_lock` `spalmer_runtime` (141-151) gains `recurrence_steps`, `adaptive_exit`, `exit_threshold`, `warm_start`; `_workload_contract` (351-366) gains the same four keys (None for non-spalmer); `_generation_comparison_fingerprint` (414-419): when `workload_contract.get("spalmer_recurrence_steps") is not None or workload_contract.get("spalmer_adaptive_exit") not in (None, "none")` add `fingerprint_value["spalmer_recurrence"] = {steps, adaptive_exit, exit_threshold, warm_start}` — depth changes compute, so it is part of the stack like expert placement; existing non-recurrent fingerprints are unchanged. Backend settings (1279-1292) forward the four keys.
- `backend.py` `identity` (275-309): `value["recurrence"] = {...four keys...}` when `spalmer_recurrence_steps` in settings; `configure_policy` (311-339) already forwards `self.settings` in `LoadSpec`. `generate` (391+) unchanged (HF kwargs only; policy comes from the load).

### Checkpoints page `yylab/ui/pages/checkpoints.py`

`_spalmer_base_recipe` (43-81): append ` · recurrent P/C/K × {default_steps}` when `metadata.get("recurrence")` (registry receipt) or `metadata["metadata"]["recurrence"]` (scanned bundle) is a dict. Detail line (252-255): after `{num_layers} layers` add ` (effective {effective_depth})` when present. Add flow (`scan_checkpoint` → `scan_spalmer_checkpoint`) stores the scan dict as meta_json already; nothing else to change.

### Docs

`docs/ARCHITECTURE.md` SPALMER section (~727): add a "Recurrent depth" subsection (config block, training keys, settings keys, telemetry keys, registry semantics: `num_layers` physical, `effective_depth` in meta). This spec is saved as `docs/superpowers/specs/2026-09-03-spalmer-recurrent-depth-design.md`.

## Errors table

| Condition | Where | Result |
|---|---|---|
| prelude+core+coda != n_layers | `SPALMERConfig.__post_init__` | ValueError |
| any RecurrenceConfig count < 1, default_steps < 1, latent_init_std < 0/non-finite, adapter != linear | `RecurrenceConfig.__post_init__` | ValueError |
| ATXY injection_layer inside core | `SPALMERBackbone.__init__`, `ModelScaleConfig.__post_init__`, planner grid (skip) | ValueError / candidate skipped |
| recurrence kwargs on a non-recurrent model | `SPALMERBackbone.forward` | ValueError("model has no recurrent core…") |
| `recurrence_steps < 1` or `backprop_steps < 1` | backbone | ValueError |
| adaptive exit requested with execution_mode != decode | backbone | ValueError |
| kl criterion without readout | backbone | ValueError (internal; LM always supplies it) |
| core slot not `RecurrentMixerStates`/None, or container in a non-core slot | `_validate_states` | TypeError |
| core slots with unequal lengths | `_validate_states` | ValueError |
| decode `recurrence_steps > len(core slot)` | backbone | ValueError("…prefill deeper or reduce recurrence_steps") |
| `latent_init` shape != [B,T,d] | backbone | ValueError |
| `RecurrentMixerStates(())` or None element | container `__post_init__` | ValueError |
| AdaptiveExit bad criterion/threshold/min_steps | `AdaptiveExit.__post_init__` | ValueError |
| TrainingConfig recurrence fields out of range / fixed sampling with non-integer mean / max < mean | `TrainingConfig.__post_init__` | ValueError |
| recurrent model without `mean_recurrence`, or `mean_recurrence` with non-recurrent model | `ExperimentTrainer.__init__`, `train_token_stream`, yyLab `_prepare_checkpoint_run` | ValueError |
| v<6 bundle | `load_checkpoint` | loads with `recurrence=None` (required set minus `recurrence`) |
| v5 bundle expert schema after bump | `_migrate_experts_config` (`>= 5`) | accepted |
| recurrence block with wrong keys | `load_checkpoint` `_require_config_schema("recurrence")` | ValueError |
| adapter presence != config.recurrence | `_validate_model_recurrence` (save/load) | ValueError |
| recurrent plan with `n_layers <= prelude+coda` | `ModelScaleConfig` / planner | ValueError / skipped; no candidates -> existing "no valid" ValueError |
| model_overrides containing `recurrence` | `presets._reject_plan_overrides` | ValueError |
| CLI `--recurrent-layers` sum != `--layers`, missing `--mean-recurrence` | `__main__._run` | ValueError |
| CLI generate recurrence flags on non-recurrent checkpoint | `_run_generate` | SystemExit |
| yyLab scratch recipe: recurrence training keys without `spalmer.recurrence.enabled`; enabled without mean_recurrence; bad ranges; unknown recurrence keys; plan not recurrent | `normalize_spalmer_config` / `build_spalmer_run_config` | ValueError naming the key |
| yyLab checkpoint mode: parent recurrent but no mean_recurrence, or vice versa | `_prepare_checkpoint_run` | ValueError |
| yyLab bundle format not in {5,6} | `_prepare_checkpoint_run` | ValueError |
| facade/engine: recurrence settings on a non-recurrent checkpoint; steps < 1; threshold <= 0; unknown criterion | facade `__init__`/`generate`, `Engine._load_spalmer` | ValueError (before GPU placement) |
| benchmark config `spalmer_recurrence_steps <= 0`, bad criterion | `_quality_job_config` | ValueError |

## Tests

### SPALMER (`tests/`, run with the command in the brief; baseline 349 passed)

Fixtures: extend `tests/core/test_modeling.py::_tiny_model` with a `recurrence=RecurrenceConfig(1,2,1)` variant on the existing TinyTokenMixer/RoutedChannelMixer doubles (n_layers=4), and a real-mixer fixture (`build_spalmer_model` with d=16, 4 layers, 4 experts, reference backend, CPU) for state tests.

core/test_modeling.py
1. `test_non_recurrent_config_keeps_flat_contract` — existing assertions unchanged (`(1,1,1,1)` states, one metrics entry per block, `recurrence_steps is None`).
2. `test_recurrent_prefill_returns_per_iteration_core_states` — r=3 prefill: slots 0 and 3 are mixer states, slots 1-2 are `RecurrentMixerStates` of len 3; `len(layer_metrics)==4`; `latent_deltas.shape==(3,)`; `latent_states.shape==(B,T,d)`; `recurrence_steps==3`.
3. `test_recurrent_decode_threads_iteration_states_and_keeps_budget` — prefill r=4 then decode r=4 and r=2: output core slot length stays 4 both times; with r=2 iterations 2-3 are advanced by `advance_token_mixer` ("fill": states differ from input) and with `AdaptiveExit(state_policy="skip")`-less path use a direct `model(..., recurrence_steps=2)` to check fill; separately `adaptive_exit=AdaptiveExit("latent_diff", threshold=1e9, min_steps=2, state_policy="skip")` exits at 2 and leaves iterations 2-3 identical to the input.
4. `test_decode_rejects_more_steps_than_state_budget` — ValueError message.
5. `test_adaptive_exit_is_decode_only_and_respects_min_steps` — prefill with adaptive_exit raises; decode with threshold=inf and min_steps=3 reports `recurrence_steps==3`.
6. `test_kl_exit_probes_head_without_mutating_coda_states` — kl criterion: coda output states equal a run without exit; readout called; exit happens when logits are constant (TinyTokenMixer returning zeros) at iteration 2.
7. `test_per_sequence_exit_freezes_latent_rows` (B=2) — row 0 exits (custom mixer makes its delta 0 after iteration 2), row 1 continues; `latent_states[0]` equals the iteration-2 latent; `recurrence_steps == r`.
8. `test_latent_init_replaces_noise_and_generator_is_honoured` — same generator seed → identical logits; `latent_init` given → deterministic without generator; `latent_init_std=0` → zeros.
9. `test_truncated_backprop_grad_flow` — r=4,k=2 with labels: `recurrence_adapter.proj.weight.grad` non-None; a hook on the core block counts backward passes == 2; PLE table of the core block and prelude weights receive grad; with k=4 hook counts 4; aux loss equals mean over grad iterations (compare with a manual run).
10. `test_core_layer_metrics_are_last_iteration_and_constant_length` — RoutedChannelMixer double tagging iteration index in metrics; entry for core blocks carries the last index; `_attach_surprise_telemetry` output length 4.
11. `test_recurrence_kwargs_rejected_without_recurrence` — five kwargs each raise.
12. `test_recurrent_mixer_states_container_validation_and_to` — empty/None element errors; `.to(dtype=float16)` maps elements with `.to` and passes others.
13. `test_atxy_injection_inside_core_is_rejected` and `test_atxy_outside_core_fires_once` (count `inject_values` calls with a stub).
14. `test_core_ple_is_looked_up_once_per_forward` — spy on `AlternatingPLE.forward`: calls == n_layers (not P + r*C + K) in training mode; identical PLE tensor re-added each iteration.

attention/test_full_vs_recurrent.py (new case) 15. `test_recurrent_core_prefill_matches_decode` — real KDA/MLA mixers, r=3: prefill logits at position t vs token-by-token decode from a t-1 prefill agree (atol 1e-4 fp32) for both `fill`-irrelevant fixed r; and with `warm_start` disabled.

core/test_runtime*.py 16. `test_generate_tokens_threads_recurrence_and_trace` — recurrent tiny model: `trace.prefill_steps==steps`, `len(trace.decode_steps)==max_new_tokens-1`, early_exits counted with an always-exiting AdaptiveExit; warm start passes `latent_init` (spy on forward). 17. `test_generate_tokens_rejects_recurrence_args_on_flat_model`. 18. `test_train_token_stream_samples_r_per_step` (sampler fixed scheme → forward called with `recurrence_steps==mean`).

training/test_recurrence.py 19. `test_sampler_is_stateless_and_seed_step_deterministic` (same (seed,step) → same (r,k); different step → different; no global RNG consumption: `torch.rand(1)` before/after equal). 20. `test_sampler_mean_and_bounds` — 20k draws: |mean(r) - mean_recurrence| < 0.15 for r̄=8, σ=0.5; k == min(depth, r); max_recurrence cap; mean 1 → always 1; fixed scheme. 21. `test_training_config_recurrence_validation` (each rule).

training/test_engine_contract.py 22. `test_trainer_samples_recurrence_once_per_optimizer_step` — fake model records kwargs; grad_accum=2: both microbatches receive identical (r,k); `TrainStepMetrics.recurrence_steps/backprop_steps/effective_depth` set; `_PotentiationAccumulator` does not raise. 23. `test_trainer_fails_closed_on_recurrence_mismatch` (both directions). 24. `test_resume_accepts_legacy_training_config_without_recurrence_keys`.

core/test_checkpoint_metadata.py 25. `test_v5_bundle_loads_with_recurrence_none_and_current_expert_schema` — save with `FORMAT_VERSION` monkeypatched to 5 semantics (write a v5 payload fixture: strip `recurrence`, `format_version=5`) → loads, `config.recurrence is None`, no adapter. 26. `test_v6_recurrent_bundle_round_trips_adapter_and_default_steps` — nested dict round-trip, `recurrence_adapter` keys in state, strict load, binding digest stable. 27. `test_recurrence_schema_is_exact` (extra key → ValueError). 28. `test_adapter_presence_must_match_config`.

experiment/test_planning.py 29. `test_recurrence_adds_adapter_and_keeps_fingerprints` — golden 9_908_080 and `to_dict()` unchanged when None; recurrent plan total = golden + 2d²+d; `to_dict()` has `recurrence` block; ATXY-in-core and `n_layers<=P+K` rejected; planner skips too-shallow grid points. core/test_presets.py 30. `test_build_configs_passes_recurrence_and_protects_it` (override `recurrence` rejected; `assert_accounting_matches_plan` passes with a built recurrent model; `classify_parameter` returns `recurrence_adapter` for both keys). core/test_cli.py 31. `test_training_recurrence_flags_build_config_and_metadata` (smoke run, tiny), 32. `test_generate_recurrence_flags_and_trace_line` (fake `generate_tokens` captures kwargs; prints line), 33. `test_plan_recurrence_columns`. experts/test_residency.py 34. `test_choose_inference_residency_threads_recurrence_steps` (spy `_evaluate`). experts/test_paged_offload.py 35. `test_paged_core_bank_counters_scale_with_iterations` (r=3 forward: core bank stage_operations == 3 × single-pass; other banks 1×; quantization_error never raises).

### yyLab (`tests/`, baseline 613 passed)

test_spalmer_runtime.py: add `_RecurrentFakeSpalmerModel` (config.recurrence=SimpleNamespace(prelude_layers=1, core_layers=2, coda_layers=1, default_steps=4); forward records kwargs, returns `recurrence_steps=kwargs.get("recurrence_steps")` and `latent_states=zeros(1,T,4)`).
36. `test_facade_defaults_recurrence_steps_from_checkpoint_and_threads_every_call` — prefill kwargs `{"recurrence_steps": 4}`, decode kwargs carry `recurrence_steps` and no `adaptive_exit`; `last_recurrence_telemetry["decode_tokens"]`, `routing_telemetry()["recurrence"]` present.
37. `test_facade_adaptive_exit_and_warm_start_kwargs` — `adaptive_exit="latent_diff", exit_threshold=0.05, warm_start=True`: decode kwargs contain an `AdaptiveExit` with criterion/threshold (spalmer fake module provides `spalmer.modeling.AdaptiveExit` via sys.modules) and `latent_init` of shape [1,1,4]; early_exits counted when the fake returns fewer steps.
38. `test_facade_rejects_recurrence_settings_for_flat_checkpoints` and `test_facade_per_call_overrides_validate` (steps<1, bad criterion, threshold<=0 → ValueError; unknown kwarg still "unsupported non-default").
39. `test_facade_forward_injects_default_recurrence_steps`.
40. `test_load_spalmer_runtime_recurrence_notes_and_config_fields` (notes mention effective depth; `config.effective_depth`, `config.spalmer_recurrence`).
41. `test_dynamic_generation_passes_recurrence_steps_to_residency_controller` (fake `choose_inference_residency` asserts kwarg only for recurrent model; legacy fake without the kwarg still works for flat models).
42. `test_scan_reports_recurrence_config_and_effective_depth` (v6 payload with nested dict; v5 payload → None, `num_layers` physical).

test_spalmer_engine_integration.py 43. update both fake `load()` signatures to accept the four new kwargs; `test_engine_maps_recurrence_settings_and_writes_back_resolved_values` (settings 0/None conventions; `spec.settings["spalmer_recurrent"]`).

test_spalmer_training.py 44. `test_normalize_scratch_recurrence_block_and_training_keys` (accepted, canonical, aliases; rejections: keys without enabled, missing mean, unknown key, bad ranges). 45. `test_planner_passes_recurrence_scale_config` (fake api gains `RecurrenceScaleConfig`; `calls["recurrence"].values == {...}`; None when disabled). 46. `test_training_config_carries_recurrence_fields`. 47. `test_prepare_checkpoint_run_accepts_v6_and_checks_parent_recurrence` (fake load returns a model whose `config.recurrence` is set/None; both mismatch directions raise; v6 accepted; v7 rejected). 48. `test_metric_payload_and_checkpoint_metadata_include_recurrence` (extend `test_run_writes_native_artifact_metadata_without_real_training`: fake api gains `FORMAT_VERSION=6`, manifest `checkpoint_format == "spalmer.prototype.checkpoint.v6"`, `recurrence` block present/None, `metrics.jsonl` rows carry `recurrence_steps`).

test_spalmer_ui.py 49. `test_spalmer_ui_config_emits_recurrence_recipe` (`recipe_version == 4`; `spalmer.recurrence` and training keys; root aliases; `normalize_spalmer_config` accepts it). 50. `test_spalmer_ui_config_rejects_inconsistent_recurrence` (enabled but plan without `recurrence`; prelude+coda >= n_layers; training keys when disabled). 51. `test_spalmer_ui_checkpoint_mode_recurrence_keys_follow_parent` (`parent_recurrent=True` emits training keys only; False emits none). 52. `test_plan_preview_cache_key_includes_recurrence` (monkeypatched `_cached_plan_json` receives the five trailing args).

Benchmarks/chat: 53. `test_quality_job_config_carries_recurrence_and_fingerprint_separates_depths` (two workload contracts differing only in steps have different comparison fingerprints; non-recurrent contract fingerprint unchanged vs a stored value computed before the change). 54. `test_routing_telemetry_text_renders_recurrence_line` (chat helper). 55. `test_spalmer_base_recipe_mentions_recurrence` (checkpoints page helper with registry and scanned shapes).

Manual GPU smoke (not automated): train a 10M (1,6,1) model for 200 steps with r̄=8,k=4, micro 8 × seq 512 on the RTX 5070 (peak VRAM noted), then `spalmer generate` with `--adaptive-exit latent_diff` and chat with warm start; verify paged-offload counters and per-token step telemetry.
