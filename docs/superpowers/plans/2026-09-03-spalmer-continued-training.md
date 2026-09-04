# SPALMER Continued Training and Epoch Control Implementation Plan

> **Historical implementation record — not the current precision contract.**
> Its format-v5/reference/fake-QAT examples are retired and unloadable. Current
> yyLab continues only QR-PLE format-v7/v8 bundles whose exact native expert
> precision pair passes capability validation; do not replay the obsolete code
> snippets in this plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a registered SPALMER checkpoint start a new full-parameter causal-LM run, and expose the trainer's epoch loop on the Train · SPALMER page.

**Architecture:** The recipe's `initialization` block gains a `checkpoint` mode. The trainer's preparation step loads the parent bundle through SPALMER's own loader (with the fingerprint-based tokenizer relocation added earlier today), takes the plan and construction configs from the bundle instead of recomputing them, and hands the prebuilt model to the run. The page gains a mode select, a parent checkpoint select, and an epochs input.

**Tech Stack:** Python 3.12, PyTorch, the `spalmer` package (installed in `.venv`), NiceGUI, SQLite via `yylab.db`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-spalmer-continued-training-design.md`

## Global Constraints

- Run kind stays `spalmer_base`; `yylab/workers/train.py` routing is untouched.
- `objective` stays `causal_lm`; `method` stays `full`.
- Checkpoint mode inherits architecture. Only `attention_backend`, `require_fla`, and `allow_reference_attention` are accepted in the `spalmer` block; every other key is a `ValueError` naming the key.
- Continued runs start with fresh optimizer state and a fresh schedule; the run log must say so.
- `SPALMER_RECIPE_VERSION` bumps from 2 to 3.
- No SQLite schema change.
- Tests run with `python -m pytest`, lint with `python -m ruff check`. Do not reformat untouched code.
- Commit after each task with a short imperative subject and the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File map

| File | Responsibility in this plan |
|---|---|
| `yylab/models/spalmer.py` | Public `load_spalmer_checkpoint_payload` helper (safe memory-mapped payload read) reused by scanning, tokenizer relocation, and the trainer. |
| `yylab/train/spalmer.py` | Checkpoint-mode normalization, `_prepare_checkpoint_run`, `ParentPlan`, `PreparedSpalmerRun.model` / `.parent`, parent metadata on save, run wiring. |
| `yylab/ui/pages/train_spalmer.py` | `build_spalmer_run_config` checkpoint mode and `epochs`; mode select, parent select, epochs input, hidden tokenizer and architecture controls. |
| `tests/test_spalmer_runtime.py` | Payload helper test. |
| `tests/test_spalmer_training.py` | Normalization, preparation, and run tests for checkpoint mode. |
| `tests/test_spalmer_ui.py` | Config builder and page helper tests. |

---

### Task 1: Public checkpoint payload reader

**Files:**
- Modify: `yylab/models/spalmer.py` (around `scan_spalmer_checkpoint` at line 48 and `_checkpoint_tokenizer_binding` at line 561)
- Test: `tests/test_spalmer_runtime.py`

**Interfaces:**
- Produces: `load_spalmer_checkpoint_payload(ref: str | Path) -> dict[str, Any]` — resolves a bundle path (directory or `.pt`), loads it with `weights_only=True, mmap=True`, validates the `format` field, returns the raw payload dict.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_spalmer_runtime.py`:

```python
def test_load_spalmer_checkpoint_payload_reads_binding_without_building(tmp_path) -> None:
    from yylab.models.spalmer import load_spalmer_checkpoint_payload

    checkpoint = _write_checkpoint(tmp_path / "run" / "spalmer.pt")

    payload = load_spalmer_checkpoint_payload(tmp_path / "run")

    assert payload["format"] == "spalmer.prototype.checkpoint"
    assert payload["tokenizer_binding"] == {"kind": "spalmer_vocab", "vocab_size": 32}
    assert payload["model_config"]["d_model"] == 8
    assert checkpoint.is_file()
    (tmp_path / "bad.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(Exception):
        load_spalmer_checkpoint_payload(tmp_path / "bad.pt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spalmer_runtime.py::test_load_spalmer_checkpoint_payload_reads_binding_without_building -q`
Expected: FAIL with `ImportError: cannot import name 'load_spalmer_checkpoint_payload'`

- [ ] **Step 3: Implement the helper and reuse it**

In `yylab/models/spalmer.py`, insert before `scan_spalmer_checkpoint`:

```python
def load_spalmer_checkpoint_payload(ref: str | Path) -> dict[str, Any]:
    """Load a trusted bundle's payload without constructing its model.

    ``weights_only=True`` constrains unpickling to PyTorch's safe allowlist and
    ``mmap=True`` avoids materializing tensors that the caller never touches.
    """

    path = resolve_spalmer_checkpoint(ref)
    payload = torch.load(
        str(path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("SPALMER checkpoint payload must be a plain dictionary")
    if payload.get("format") != SPALMER_FORMAT:
        raise ValueError("not a SPALMER checkpoint")
    return payload
```

Replace the body of `scan_spalmer_checkpoint` from `path = resolve_spalmer_checkpoint(ref)` through the `if payload.get("format") != SPALMER_FORMAT:` check with:

```python
    path = resolve_spalmer_checkpoint(ref)
    payload = load_spalmer_checkpoint_payload(path)
```

Replace the body of `_checkpoint_tokenizer_binding` with:

```python
    payload = load_spalmer_checkpoint_payload(path)
    return _optional_plain_dict(payload.get("tokenizer_binding"), "tokenizer_binding")
```

- [ ] **Step 4: Run the runtime tests**

Run: `python -m pytest tests/test_spalmer_runtime.py -q`
Expected: all pass (30 tests).

- [ ] **Step 5: Commit**

```bash
git add yylab/models/spalmer.py tests/test_spalmer_runtime.py
git commit -m "Expose SPALMER checkpoint payload reader" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Checkpoint-mode normalization

**Files:**
- Modify: `yylab/train/spalmer.py` (`normalize_spalmer_config`, lines 234–546)
- Test: `tests/test_spalmer_training.py`

**Interfaces:**
- Produces: module constants `INITIALIZATION_SCRATCH = "scratch"`, `INITIALIZATION_CHECKPOINT = "checkpoint"`, `CHECKPOINT_EXECUTION_KEYS`.
- Produces: normalized `cfg["initialization"]` in checkpoint mode: `{"mode": "checkpoint", "checkpoint_id": str, "checkpoint_ref": str}` where `checkpoint_ref` is the resolved `spalmer.pt` path; `cfg["tokenizer_ref"]` is `None`; `cfg["spalmer"]` holds only `attention_backend` and `require_fla`; `cfg` has no `target_parameters`.
- Produces: helper `_normalize_scratch_architecture(cfg: dict, spalmer: dict) -> None` (the existing architecture block, moved verbatim).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spalmer_training.py`:

```python
def _parent_row(bundle: Path, *, family: str = "spalmer") -> dict:
    return {
        "id": "ckpt_parent",
        "name": "spal50m",
        "kind": "base",
        "ref": str(bundle),
        "path": str(bundle.parent),
        "params_b": 0.05,
        "hidden_size": 192,
        "num_layers": 4,
        "meta_json": json.dumps({"model_family": family, "plan": {"fingerprint": "parent-plan"}}),
    }


def _checkpoint_config(**updates):
    config = _config(
        initialization={"mode": "checkpoint", "checkpoint_id": "ckpt_parent"},
        spalmer={"attention_backend": "reference"},
    )
    config.update(updates)
    return config


def test_normalize_spalmer_config_checkpoint_mode_inherits_architecture(tmp_path, monkeypatch):
    bundle = tmp_path / "parent" / "spalmer.pt"
    bundle.parent.mkdir()
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))

    normalized = bridge.normalize_spalmer_config(_checkpoint_config(epochs=3), require_runtime=True)

    assert normalized["initialization"] == {
        "mode": "checkpoint",
        "checkpoint_id": "ckpt_parent",
        "checkpoint_ref": str(bundle.resolve()),
    }
    assert normalized["tokenizer_ref"] is None
    assert normalized["spalmer"] == {"attention_backend": "reference", "require_fla": False}
    assert "target_parameters" not in normalized
    assert normalized["epochs"] == 3


@pytest.mark.parametrize(
    ("row_factory", "updates", "message"),
    [
        (lambda bundle: None, {}, "unknown parent checkpoint"),
        (lambda bundle: _parent_row(bundle, family="hf"), {}, "not a SPALMER checkpoint"),
        (
            lambda bundle: _parent_row(bundle),
            {"spalmer": {"attention_backend": "reference", "num_experts": 100}},
            "num_experts",
        ),
        (lambda bundle: _parent_row(bundle), {"epochs": 0}, "epochs must be positive"),
        (
            lambda bundle: _parent_row(bundle),
            {"initialization": {"mode": "checkpoint"}},
            "initialization.checkpoint_id",
        ),
    ],
)
def test_normalize_spalmer_config_checkpoint_mode_rejections(
    tmp_path, monkeypatch, row_factory, updates, message
):
    bundle = tmp_path / "parent" / "spalmer.pt"
    bundle.parent.mkdir()
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): row_factory(bundle))

    with pytest.raises(ValueError, match=message):
        bridge.normalize_spalmer_config(_checkpoint_config(**updates))


def test_normalize_spalmer_config_checkpoint_mode_requires_bundle_file(tmp_path, monkeypatch):
    missing = tmp_path / "parent" / "spalmer.pt"
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(missing))

    with pytest.raises(ValueError, match="bundle is unavailable"):
        bridge.normalize_spalmer_config(_checkpoint_config())


def test_normalize_spalmer_config_rejects_unknown_initialization_mode():
    with pytest.raises(ValueError, match="initialization mode"):
        bridge.normalize_spalmer_config(_config(initialization={"mode": "lora"}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spalmer_training.py -q -k "checkpoint_mode or initialization_mode"`
Expected: FAIL. The first test fails with `the first SPALMER base-training path supports scratch initialization only`.

- [ ] **Step 3: Add constants and helpers**

In `yylab/train/spalmer.py`, after `TOKENIZER_CONTRACT_VERSION = 1` add:

```python
INITIALIZATION_SCRATCH = "scratch"
INITIALIZATION_CHECKPOINT = "checkpoint"
CHECKPOINT_EXECUTION_KEYS = frozenset(
    {"attention_backend", "require_fla", "allow_reference_attention"}
)
```

After `_normalize_optional_component` (before `normalize_spalmer_config`) add:

```python
def _loads_json_object(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_checkpoint_initialization(initialization: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a registry checkpoint ID to a SPALMER bundle for continued training."""

    from ..models.spalmer import resolve_spalmer_checkpoint

    checkpoint_id = str(initialization.get("checkpoint_id") or "").strip()
    if not checkpoint_id:
        raise ValueError("SPALMER continued training requires initialization.checkpoint_id")
    row = db.q1("SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,))
    if row is None:
        raise ValueError(f"unknown parent checkpoint: {checkpoint_id}")
    meta = _loads_json_object(row.get("meta_json"))
    family = str(meta.get("model_family") or meta.get("family") or "").strip().casefold()
    if family != MODEL_FAMILY:
        raise ValueError(f"parent checkpoint {checkpoint_id} is not a SPALMER checkpoint")
    try:
        bundle_path = resolve_spalmer_checkpoint(str(row.get("ref") or row.get("path") or ""))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"parent checkpoint {checkpoint_id} bundle is unavailable: {exc}"
        ) from exc
    return {
        "mode": INITIALIZATION_CHECKPOINT,
        "checkpoint_id": checkpoint_id,
        "checkpoint_ref": str(bundle_path),
    }


def _normalize_checkpoint_execution(spalmer: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only attention preflight keys; the loaded bundle owns everything else."""

    forbidden = sorted(set(spalmer) - CHECKPOINT_EXECUTION_KEYS)
    if forbidden:
        raise ValueError(
            "SPALMER continued training inherits its architecture from the parent "
            "checkpoint; remove " + ", ".join(forbidden)
        )
    attention_backend = str(spalmer.get("attention_backend") or "auto").strip().casefold()
    if attention_backend not in {"auto", "fla", "reference"}:
        raise ValueError("attention_backend must be 'auto', 'fla', or 'reference'")
    require_fla = _bool(
        spalmer.get("require_fla"),
        default=not _bool(spalmer.get("allow_reference_attention"), default=False),
    )
    return {"attention_backend": attention_backend, "require_fla": require_fla}
```

- [ ] **Step 4: Move the scratch architecture block into a helper**

In `normalize_spalmer_config`, the architecture block starts at the line `spalmer = _mapping(cfg.get("spalmer"), "spalmer")` (currently line 319) and ends at `cfg["target_parameters"] = spalmer["target_parameters"]` (currently line 455). Cut the lines **after** `spalmer = _mapping(...)` through `cfg["target_parameters"] = ...` (that is, from `target = spalmer.get("target_parameters", cfg.get("target_parameters"))` to `cfg["target_parameters"] = spalmer["target_parameters"]`, inclusive) and paste them, unchanged and dedented by four spaces, as the body of a new function placed immediately before `normalize_spalmer_config`:

```python
def _normalize_scratch_architecture(cfg: dict[str, Any], spalmer: dict[str, Any]) -> None:
    """Validate the plan-driving architecture recipe for scratch initialization."""

    # <moved body; first line is `target = spalmer.get("target_parameters", cfg.get("target_parameters"))`
    #  and last line is `cfg["target_parameters"] = spalmer["target_parameters"]`>
```

Use this script to do the move mechanically, then inspect the diff:

```python
from pathlib import Path
p = Path("yylab/train/spalmer.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.startswith('    target = spalmer.get("target_parameters"'))
end = next(i for i, l in enumerate(lines) if l.startswith('    cfg["target_parameters"] = spalmer["target_parameters"]'))
body = [l[4:] if l.startswith("    ") else l for l in lines[start:end + 1]]
del lines[start:end + 1]
func_start = next(i for i, l in enumerate(lines) if l.startswith("def normalize_spalmer_config("))
helper = (
    "def _normalize_scratch_architecture(cfg: dict[str, Any], spalmer: dict[str, Any]) -> None:\n"
    '    """Validate the plan-driving architecture recipe for scratch initialization."""\n\n'
    + "".join("    " + l if l.strip() else l for l in body)
    + "\n\n"
)
lines.insert(func_start, helper)
p.write_text("".join(lines), encoding="utf-8")
```

- [ ] **Step 5: Rewrite the initialization block and call site**

Replace the block from `initialization = _mapping(cfg.get("initialization"), "initialization")` through `cfg["tokenizer_ref"] = tokenizer_ref` with:

```python
    initialization = _mapping(cfg.get("initialization"), "initialization")
    mode = str(initialization.get("mode") or INITIALIZATION_SCRATCH).strip().casefold()
    if mode == INITIALIZATION_SCRATCH:
        tokenizer_ref = str(
            initialization.get("tokenizer_ref")
            or cfg.get("tokenizer_ref")
            or _mapping(cfg.get("spalmer"), "spalmer").get("tokenizer_ref")
            or ""
        ).strip()
        if not tokenizer_ref:
            raise ValueError("SPALMER scratch training requires a registered or direct HF tokenizer")
        initialization.update(mode=INITIALIZATION_SCRATCH, tokenizer_ref=tokenizer_ref)
        cfg["tokenizer_ref"] = tokenizer_ref
    elif mode == INITIALIZATION_CHECKPOINT:
        initialization = _normalize_checkpoint_initialization(initialization)
        cfg["tokenizer_ref"] = None
    else:
        raise ValueError(f"unsupported SPALMER initialization mode: {mode!r}")
    cfg["initialization"] = initialization
```

Then replace the line `spalmer = _mapping(cfg.get("spalmer"), "spalmer")` (now directly followed by `training = _mapping(cfg.get("training"), "training")`) with:

```python
    spalmer = _mapping(cfg.get("spalmer"), "spalmer")
    if mode == INITIALIZATION_CHECKPOINT:
        cfg["spalmer"] = _normalize_checkpoint_execution(spalmer)
        cfg.pop("target_parameters", None)
    else:
        _normalize_scratch_architecture(cfg, spalmer)
```

- [ ] **Step 6: Run the training tests**

Run: `python -m pytest tests/test_spalmer_training.py -q`
Expected: all pass, including the new checkpoint-mode tests and the existing scratch tests (the moved block must behave identically).

- [ ] **Step 7: Lint and commit**

Run: `python -m ruff check yylab/train/spalmer.py tests/test_spalmer_training.py`

```bash
git add yylab/train/spalmer.py tests/test_spalmer_training.py
git commit -m "Accept checkpoint initialization in SPALMER recipes" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Prepare a run from a parent bundle

**Files:**
- Modify: `yylab/train/spalmer.py` (`PreparedSpalmerRun` line 95, `_load_spalmer_api` line 105, `prepare_spalmer_run` line 1263)
- Test: `tests/test_spalmer_training.py`

**Interfaces:**
- Consumes: `load_spalmer_checkpoint_payload`, `relocated_spalmer_tokenizer` from `yylab.models.spalmer`; normalized `cfg["initialization"]["checkpoint_ref"]` from Task 2.
- Produces: `PreparedSpalmerRun` fields `model: Any = None`, `parent: dict[str, Any] | None = None`.
- Produces: `ParentPlan` with attributes `target_parameters`, `relative_error`, `fingerprint`, `config` (namespace), `parameters` (namespace), and `to_dict()`.
- Produces: `_load_spalmer_api()` namespace gains `load_checkpoint`, `SPALMERConfig`, `KDAConfig`, `MLAConfig`, `MicroExpertsConfig`, `DirectionalConfig`, `ATXYConfig`.
- Produces: `_prepare_checkpoint_run(cfg, symbols) -> PreparedSpalmerRun`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_spalmer_training.py`:

```python
def _parent_bundle(path: Path, *, tokenizer_source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spalmer.prototype.checkpoint",
            "format_version": 5,
            "model_parameter_dtype": "bfloat16",
            "model_config": {"vocab_size": 32, "d_model": 8, "n_layers": 4},
            "kda_config": {"hidden_size": 8, "num_heads": 2},
            "mla_config": {"hidden_size": 8, "num_heads": 2},
            "experts_config": {"num_experts": 200, "expert_weight_format": "mxfp4"},
            "directional_config": {"d_model": 8, "enabled": True},
            "atxy_config": None,
            "tokenizer_binding": {
                "kind": "huggingface",
                "identity": {
                    "backend": "huggingface",
                    "source": tokenizer_source,
                    "artifact_fingerprint": "artifact-expected",
                    "vocab_size": 32,
                    "special_tokens": {"bos_id": None, "eos_id": 0, "eod_id": None, "pad_id": 0},
                    "revision": None,
                    "fingerprint": "identity-expected",
                },
            },
            "model_state": {"w": torch.zeros(2, dtype=torch.bfloat16)},
            "metadata": {
                "run_id": "run_parent",
                "plan": {
                    "target_parameters": 50_000_000,
                    "relative_error": 0.0001,
                    "fingerprint": "parent-plan",
                    "config": {"num_experts": 200, "expert_weight_format": "mxfp4", "d_model": 8, "n_layers": 4},
                    "parameters": {"total": 50_006_136},
                },
            },
        },
        path,
    )
    return path


def _checkpoint_api(load_checkpoint, **extra):
    return SimpleNamespace(
        load_checkpoint=load_checkpoint,
        SPALMERConfig=_ConfigObject,
        KDAConfig=_ConfigObject,
        MLAConfig=_ConfigObject,
        MicroExpertsConfig=_ConfigObject,
        DirectionalConfig=_ConfigObject,
        ATXYConfig=_ConfigObject,
        TrainingConfig=lambda **values: SimpleNamespace(**values),
        fla_available=lambda: True,
        plan_configuration=lambda *_args, **_kwargs: pytest.fail("plan must not be recomputed"),
        build_configs=lambda *_args, **_kwargs: pytest.fail("bundle must not be rebuilt"),
        **extra,
    )


def test_prepare_spalmer_run_continues_from_parent_bundle(tmp_path, monkeypatch):
    bundle = _parent_bundle(tmp_path / "parent" / "spalmer.pt", tokenizer_source=str(tmp_path / "gone"))
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    override = SimpleNamespace(identity=SimpleNamespace(source="relocated"))
    seen_bindings = []
    monkeypatch.setattr(
        "yylab.models.spalmer.relocated_spalmer_tokenizer",
        lambda binding: seen_bindings.append(binding) or override,
    )
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    calls = {}

    def load_checkpoint(path, **kwargs):
        calls["path"] = Path(path)
        calls["kwargs"] = kwargs
        return model, tokenizer, {"run_id": "run_parent", "plan": torch.load(path, weights_only=True)["metadata"]["plan"]}

    prepared = bridge.prepare_spalmer_run(_checkpoint_config(), api=_checkpoint_api(load_checkpoint))

    assert prepared.model is model
    assert prepared.tokenizer is tokenizer
    assert calls["path"] == bundle.resolve()
    assert calls["kwargs"] == {"map_location": "cpu", "tokenizer_override": override}
    assert seen_bindings[0]["kind"] == "huggingface"
    assert prepared.plan.fingerprint == "parent-plan"
    assert prepared.plan.config.num_experts == 200
    assert prepared.plan.parameters.total == 50_006_136
    assert prepared.plan.to_dict()["target_parameters"] == 50_000_000
    assert prepared.bundle.kda.hidden_size == 8
    assert prepared.bundle.model.to_dict()["d_model"] == 8
    assert prepared.bundle.directional.enabled is True
    assert prepared.bundle.atxy is None
    assert prepared.parent == {
        "checkpoint_id": "ckpt_parent",
        "checkpoint_path": str(bundle.resolve()),
        "run_id": "run_parent",
        "plan_fingerprint": "parent-plan",
    }
    assert prepared.training.max_steps == 2


def test_prepare_spalmer_run_skips_override_when_tokenizer_path_exists(tmp_path, monkeypatch):
    present = tmp_path / "tok"
    present.mkdir()
    (present / "tokenizer.json").write_text("{}", encoding="utf-8")
    bundle = _parent_bundle(tmp_path / "parent" / "spalmer.pt", tokenizer_source=str(present))
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    monkeypatch.setattr("yylab.models.spalmer.relocated_spalmer_tokenizer", lambda binding: None)
    calls = {}

    def load_checkpoint(path, **kwargs):
        calls["kwargs"] = kwargs
        return _FakeModel(), _FakeTokenizer(), {"run_id": "run_parent", "plan": {"fingerprint": "parent-plan"}}

    prepared = bridge.prepare_spalmer_run(_checkpoint_config(), api=_checkpoint_api(load_checkpoint))

    assert calls["kwargs"] == {"map_location": "cpu"}
    assert prepared.parent["run_id"] == "run_parent"


def test_prepare_spalmer_run_requires_parent_plan_metadata(tmp_path, monkeypatch):
    bundle = _parent_bundle(tmp_path / "parent" / "spalmer.pt", tokenizer_source=str(tmp_path / "gone"))
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    monkeypatch.setattr("yylab.models.spalmer.relocated_spalmer_tokenizer", lambda binding: None)
    api = _checkpoint_api(lambda path, **kwargs: (_FakeModel(), _FakeTokenizer(), {"run_id": "x"}))

    with pytest.raises(ValueError, match="scale plan"):
        bridge.prepare_spalmer_run(_checkpoint_config(), api=api)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spalmer_training.py -q -k "continues_from_parent or skips_override or parent_plan_metadata"`
Expected: FAIL. `prepare_spalmer_run` calls `resolve_spalmer_tokenizer`, which raises `SPALMER tokenizer reference is empty`.

- [ ] **Step 3: Extend the prepared-run dataclass and the API namespace**

Replace `PreparedSpalmerRun`:

```python
@dataclass(frozen=True, slots=True)
class PreparedSpalmerRun:
    """Validated construction inputs shared by the worker and checkpoint code."""

    config: dict[str, Any]
    tokenizer: Any
    plan: Any
    bundle: Any
    training: Any
    model: Any = None
    parent: dict[str, Any] | None = None
```

In `_load_spalmer_api`, add to the `try` block:

```python
        from spalmer import load_checkpoint, save_checkpoint
        from spalmer.attention import KDAConfig, MLAConfig
        from spalmer.config import SPALMERConfig
        from spalmer.directional import DirectionalConfig
        from spalmer.experts import MicroExpertsConfig
        from spalmer.memory import ATXYConfig
```

(replacing the existing `from spalmer import save_checkpoint` line) and add to the returned namespace:

```python
        ATXYConfig=ATXYConfig,
        DirectionalConfig=DirectionalConfig,
        KDAConfig=KDAConfig,
        MLAConfig=MLAConfig,
        MicroExpertsConfig=MicroExpertsConfig,
        SPALMERConfig=SPALMERConfig,
        load_checkpoint=load_checkpoint,
```

- [ ] **Step 4: Add `ParentPlan` and `_prepare_checkpoint_run`**

Insert before `prepare_spalmer_run`:

```python
class ParentPlan:
    """Read-only view of the scale plan recorded in a parent bundle."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        self._plan = dict(plan)
        self.target_parameters = int(self._plan.get("target_parameters") or 0)
        self.relative_error = float(self._plan.get("relative_error") or 0.0)
        self.fingerprint = str(self._plan.get("fingerprint") or "")
        self.config = SimpleNamespace(**_mapping(self._plan.get("config"), "plan.config"))
        self.parameters = SimpleNamespace(
            **_mapping(self._plan.get("parameters"), "plan.parameters")
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self._plan)


def _optional_config(factory: Callable[..., Any], value: object, name: str) -> Any:
    return None if value is None else factory(**_mapping(value, name))


def _prepare_checkpoint_run(cfg: dict[str, Any], symbols: SimpleNamespace) -> PreparedSpalmerRun:
    """Load a parent bundle as the starting point of a new full-parameter run."""

    from ..models.spalmer import load_spalmer_checkpoint_payload, relocated_spalmer_tokenizer

    initialization = _mapping(cfg.get("initialization"), "initialization")
    checkpoint_path = Path(str(initialization["checkpoint_ref"]))
    payload = load_spalmer_checkpoint_payload(checkpoint_path)
    if int(payload.get("format_version") or 0) != 5:
        raise ValueError(
            "SPALMER continued training requires a format-version-5 bundle; "
            f"found {payload.get('format_version')!r}"
        )
    binding = payload.get("tokenizer_binding")
    override = relocated_spalmer_tokenizer(binding if isinstance(binding, Mapping) else None)
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if override is not None:
        load_kwargs["tokenizer_override"] = override
    model, tokenizer, metadata = symbols.load_checkpoint(checkpoint_path, **load_kwargs)
    metadata = dict(metadata or {})
    plan_payload = metadata.get("plan")
    if not isinstance(plan_payload, Mapping) or not plan_payload:
        raise ValueError("parent checkpoint metadata does not record a SPALMER scale plan")
    plan = ParentPlan(plan_payload)
    bundle = SimpleNamespace(
        plan=plan,
        model=symbols.SPALMERConfig(**_mapping(payload.get("model_config"), "model_config")),
        kda=symbols.KDAConfig(**_mapping(payload.get("kda_config"), "kda_config")),
        mla=symbols.MLAConfig(**_mapping(payload.get("mla_config"), "mla_config")),
        experts=symbols.MicroExpertsConfig(
            **_mapping(payload.get("experts_config"), "experts_config")
        ),
        directional=_optional_config(
            symbols.DirectionalConfig, payload.get("directional_config"), "directional_config"
        ),
        atxy=_optional_config(symbols.ATXYConfig, payload.get("atxy_config"), "atxy_config"),
    )
    training = build_spalmer_training_config(cfg, api=symbols)
    _validate_fast_attention(cfg, symbols)
    parent = {
        "checkpoint_id": str(initialization.get("checkpoint_id") or ""),
        "checkpoint_path": str(checkpoint_path),
        "run_id": str(metadata.get("run_id") or ""),
        "plan_fingerprint": plan.fingerprint,
    }
    return PreparedSpalmerRun(cfg, tokenizer, plan, bundle, training, model=model, parent=parent)
```

Replace `prepare_spalmer_run`:

```python
def prepare_spalmer_run(
    config: Mapping[str, Any],
    *,
    api: SimpleNamespace | None = None,
) -> PreparedSpalmerRun:
    """Resolve tokenizer, plan, construction bundle, and runtime config."""

    cfg = normalize_spalmer_config(config, require_runtime=True)
    symbols = api or _load_spalmer_api()
    if cfg["initialization"]["mode"] == INITIALIZATION_CHECKPOINT:
        return _prepare_checkpoint_run(cfg, symbols)
    tokenizer = resolve_spalmer_tokenizer(cfg, api=symbols)
    plan = plan_spalmer_configuration(cfg, int(tokenizer.vocab_size), api=symbols)
    bundle = build_spalmer_bundle(cfg, tokenizer, plan, api=symbols)
    training = build_spalmer_training_config(cfg, api=symbols)
    _validate_fast_attention(cfg, symbols)
    return PreparedSpalmerRun(cfg, tokenizer, plan, bundle, training)
```

- [ ] **Step 5: Run the training tests**

Run: `python -m pytest tests/test_spalmer_training.py -q`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

Run: `python -m ruff check yylab/train/spalmer.py tests/test_spalmer_training.py`

```bash
git add yylab/train/spalmer.py tests/test_spalmer_training.py
git commit -m "Prepare SPALMER runs from a parent checkpoint" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Run with the loaded model and record the parent

**Files:**
- Modify: `yylab/train/spalmer.py` (`_save_bundle_checkpoint` line 1280, `run_spalmer_base` line 1318 onward)
- Test: `tests/test_spalmer_training.py`

**Interfaces:**
- Consumes: `PreparedSpalmerRun.model` and `.parent` from Task 3.
- Produces: bundle `metadata["parent_checkpoint"]`, manifest `"parent_checkpoint"`, registry `meta_json["initialization"]` and `meta_json["parent_checkpoint_id"]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_spalmer_training.py`. It mirrors `test_run_writes_native_artifact_metadata_without_real_training` but with a prebuilt model and parent:

```python
class _FakeLoadedModel(_FakeModel):
    def __init__(self):
        self.moves = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self


def test_run_uses_loaded_model_and_records_parent(tmp_path, monkeypatch):
    output = tmp_path / "checkpoint"
    bundle_path = tmp_path / "parent" / "spalmer.pt"
    bundle_path.parent.mkdir()
    bundle_path.write_bytes(b"bundle")
    batches = _FakeBatches()
    training = _ConfigObject(
        max_steps=2,
        sequence_length=8,
        seed=0,
        deterministic_algorithms=False,
        device="cpu",
        optimizer_update_chunk_size=1_048_576,
    )
    experts = _ConfigObject(
        expert_activation_format="mxfp8",
        expert_master_dtype="bfloat16",
        expert_qat_backend="reference",
    )
    bundle = SimpleNamespace(
        build=lambda: pytest.fail("continued runs must not build a fresh model"),
        model=_ConfigObject(vocab_size=4096, d_model=64, n_layers=4),
        kda=_ConfigObject(backend="reference"),
        mla=_ConfigObject(num_heads=4),
        experts=experts,
        directional=None,
        atxy=None,
    )
    loaded = _FakeLoadedModel()
    parent = {
        "checkpoint_id": "ckpt_parent",
        "checkpoint_path": str(bundle_path),
        "run_id": "run_parent",
        "plan_fingerprint": "plan-fingerprint",
    }
    prepared = bridge.PreparedSpalmerRun(
        _checkpoint_config(), _FakeTokenizer(), _FakePlan(), bundle, training,
        model=loaded, parent=parent,
    )
    saved = {}

    def save_checkpoint(path, *_args, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock-spalmer-checkpoint")
        saved[path] = kwargs["metadata"]
        return path

    api = SimpleNamespace(
        CausalBatch=_CausalBatch,
        ExperimentTrainer=_FakeTrainer,
        assert_accounting_matches_plan=lambda *_args: pytest.fail("accounting is not replanned"),
        save_checkpoint=save_checkpoint,
        seed_everything=lambda *_args, **_kwargs: None,
    )
    database_calls = []
    monkeypatch.setattr(bridge, "_load_spalmer_api", lambda: api)
    monkeypatch.setattr(bridge, "prepare_spalmer_run", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(bridge, "build_spalmer_batch_source", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(bridge, "checkpoint_dir", lambda _checkpoint_id: output)
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle_path))
    monkeypatch.setattr(
        bridge,
        "corpus_training_preflight",
        lambda *_args, **_kwargs: {
            "errors": [],
            "warnings": [],
            "provenance": {"schema_version": 3, "corpus_ids": ["alpha"]},
        },
    )
    monkeypatch.setattr(
        bridge.db,
        "execute",
        lambda sql, args=(): database_calls.append((sql, args)) or 1,
    )

    summary = bridge.run_spalmer_base(_checkpoint_config(), tmp_path / "run")

    assert loaded.moves, "loaded model must be moved to the training device"
    assert saved[output / "spalmer.pt"]["parent_checkpoint"] == parent
    manifest = json.loads((output / "spalmer_manifest.json").read_text("utf-8"))
    assert manifest["parent_checkpoint"] == parent
    assert manifest["data_provenance"]["initialization"] == "checkpoint"
    registry_meta = json.loads(database_calls[-1][1][6])
    assert registry_meta["initialization"] == "checkpoint"
    assert registry_meta["parent_checkpoint_id"] == "ckpt_parent"
    assert summary["parent_checkpoint"] == parent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spalmer_training.py::test_run_uses_loaded_model_and_records_parent -q`
Expected: FAIL at `bundle.build` ("continued runs must not build a fresh model").

- [ ] **Step 3: Wire the loaded model into the run**

In `run_spalmer_base`, replace the block from `destination_device = torch.device(prepared.training.device)` through `api.assert_accounting_matches_plan(plan, accounting)` with:

```python
    destination_device = torch.device(prepared.training.device)
    if prepared.model is not None:
        parent = dict(prepared.parent or {})
        model = prepared.model.to(destination_device)
        print(
            "SPALMER continued training: "
            f"parent={parent.get('checkpoint_id')}; plan={plan.fingerprint[:16]}…; "
            "optimizer moments and learning-rate schedule start fresh",
            flush=True,
        )
    elif destination_device.type == "cuda":
        # Construct directly on the destination to avoid a second, transient
        # full-sized host copy before ExperimentTrainer takes ownership.
        with torch.device(destination_device):
            model = prepared.bundle.build()
    else:
        model = prepared.bundle.build()
    accounting = model.parameter_accounting()
    if prepared.model is None:
        api.assert_accounting_matches_plan(plan, accounting)
```

Directly after `data_provenance["objective"] = OBJECTIVE_CAUSAL_LM` add:

```python
    data_provenance["initialization"] = str(cfg["initialization"]["mode"])
```

In `_save_bundle_checkpoint`, after the `metadata = {...}` literal add:

```python
    if prepared.parent:
        metadata["parent_checkpoint"] = dict(prepared.parent)
```

In the `manifest = {...}` literal add, after `"plan_fingerprint": plan.fingerprint,`:

```python
        "parent_checkpoint": dict(prepared.parent or {}) or None,
```

In `checkpoint_metadata`, replace `"initialization": "scratch",` with:

```python
        "initialization": str(cfg["initialization"]["mode"]),
        "parent_checkpoint_id": (prepared.parent or {}).get("checkpoint_id"),
```

In the returned summary dict add, after `"plan_fingerprint": plan.fingerprint,`:

```python
        "parent_checkpoint": dict(prepared.parent or {}) or None,
```

- [ ] **Step 4: Run the training tests**

Run: `python -m pytest tests/test_spalmer_training.py -q`
Expected: all pass. The existing scratch run test still sees `"initialization": "scratch"` and `parent_checkpoint` as `None`.

- [ ] **Step 5: Lint and commit**

Run: `python -m ruff check yylab/train/spalmer.py tests/test_spalmer_training.py`

```bash
git add yylab/train/spalmer.py tests/test_spalmer_training.py
git commit -m "Run SPALMER continued training from a loaded bundle" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Recipe builder for checkpoint mode and epochs

**Files:**
- Modify: `yylab/ui/pages/train_spalmer.py` (`SPALMER_RECIPE_VERSION` line 29, `build_spalmer_run_config` lines 269–476)
- Test: `tests/test_spalmer_ui.py`

**Interfaces:**
- Produces: `build_spalmer_run_config(..., initialization_mode: str = "scratch", parent_checkpoint_id: str | None = None, epochs: int = 1)`.
- Produces: checkpoint-mode recipe with `initialization == {"mode": "checkpoint", "checkpoint_id": ...}`, `spalmer` limited to attention keys, no `architecture` or `precision` keys, `tokenizer["source"] == "checkpoint"`, and `epochs` in both modes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spalmer_ui.py`:

```python
def test_spalmer_ui_config_carries_epochs_for_scratch_runs(monkeypatch) -> None:
    config = _run_config(monkeypatch, epochs=3)

    assert config["recipe_version"] == 3
    assert config["epochs"] == 3
    assert config["initialization"] == {"mode": "scratch", "tokenizer_ref": "tok_4096"}


def test_spalmer_ui_config_defaults_to_one_epoch(monkeypatch) -> None:
    assert _run_config(monkeypatch)["epochs"] == 1


def test_spalmer_ui_config_rejects_non_positive_epochs(monkeypatch) -> None:
    with pytest.raises(ValueError, match="epochs"):
        _run_config(monkeypatch, epochs=0)


def test_spalmer_ui_config_continues_from_checkpoint(monkeypatch) -> None:
    config = _run_config(
        monkeypatch,
        initialization_mode="checkpoint",
        parent_checkpoint_id="ckpt_parent",
        tokenizer_ref="",
        tokenizer_source="checkpoint",
        tokenizer_registry_id=None,
        resolved_plan={},
        target_parameters=0,
        attention_backend="reference",
        epochs=2,
    )

    assert config["initialization"] == {"mode": "checkpoint", "checkpoint_id": "ckpt_parent"}
    assert set(config["spalmer"]) <= {"attention_backend", "require_fla", "allow_reference_attention"}
    assert config["spalmer"]["attention_backend"] == "reference"
    assert "architecture" not in config
    assert "precision" not in config
    assert config["tokenizer"] == {"source": "checkpoint", "registry_id": None, "ref": "", "preview_vocab_size": None}
    assert config["epochs"] == 2
    assert config["max_steps"] > 0


def test_spalmer_ui_checkpoint_mode_requires_parent(monkeypatch) -> None:
    with pytest.raises(ValueError, match="parent checkpoint"):
        _run_config(monkeypatch, initialization_mode="checkpoint", parent_checkpoint_id=None)


def test_spalmer_ui_checkpoint_mode_normalizes(monkeypatch, tmp_path) -> None:
    from yylab import db
    from yylab.train.spalmer import normalize_spalmer_config

    bundle = tmp_path / "parent" / "spalmer.pt"
    bundle.parent.mkdir()
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(
        db,
        "q1",
        lambda sql, args=(): {
            "id": "ckpt_parent",
            "ref": str(bundle),
            "path": str(bundle.parent),
            "meta_json": '{"model_family": "spalmer"}',
        },
    )
    config = _run_config(
        monkeypatch,
        initialization_mode="checkpoint",
        parent_checkpoint_id="ckpt_parent",
        tokenizer_ref="",
        tokenizer_source="checkpoint",
        tokenizer_registry_id=None,
        resolved_plan={},
        target_parameters=0,
        attention_backend="reference",
    )

    normalized = normalize_spalmer_config(config, require_runtime=True)

    assert normalized["initialization"]["checkpoint_ref"] == str(bundle.resolve())
    assert normalized["spalmer"] == {"attention_backend": "reference", "require_fla": False}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spalmer_ui.py -q`
Expected: the new tests FAIL with `TypeError: build_spalmer_run_config() got an unexpected keyword argument 'epochs'` (and `initialization_mode`).

- [ ] **Step 3: Implement**

Set `SPALMER_RECIPE_VERSION = 3`.

Add the three keyword parameters to `build_spalmer_run_config` after `data_preflight_warnings: Iterable[str] = (),`:

```python
    initialization_mode: str = "scratch",
    parent_checkpoint_id: str | None = None,
    epochs: int = 1,
```

Replace the top of the body (from `name = str(checkpoint_name or "").strip()` through the `if len(fingerprint) != 64:` check) with:

```python
    name = str(checkpoint_name or "").strip()
    reference = str(tokenizer_ref or "").strip()
    mode = str(initialization_mode or "scratch").strip().casefold()
    if mode not in {"scratch", "checkpoint"}:
        raise ValueError(f"unsupported initialization mode: {initialization_mode!r}")
    continuing = mode == "checkpoint"
    parent_id = str(parent_checkpoint_id or "").strip()
    if not name:
        raise ValueError("checkpoint name is required")
    if continuing and not parent_id:
        raise ValueError("choose a parent checkpoint to continue from")
    if not continuing and not reference:
        raise ValueError("choose a registered tokenizer or enter a tokenizer reference")
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1")
    selected_corpora = _ordered_values(corpora)
    if not selected_corpora:
        raise ValueError("choose at least one training corpus")
    normalized_domains = _ordered_values(domains)
    normalized_splits = normalize_source_splits(splits)
    mix = normalize_corpus_mix(corpus_mix)
    target = int(target_parameters)
    vocabulary = int(tokenizer_preview_vocab_size)
    fingerprint = ""
    if not continuing:
        if target < 1 or vocabulary < 1:
            raise ValueError("target parameters and tokenizer preview vocabulary must be positive")
        if int(resolved_plan.get("target_parameters") or 0) != target:
            raise ValueError("resolved SPALMER plan does not match the requested target")
        planned_config = dict(resolved_plan.get("config") or {})
        if int(planned_config.get("vocab_size") or 0) != vocabulary:
            raise ValueError("resolved SPALMER plan does not match the preview vocabulary")
        if str(planned_config.get("expert_weight_format") or "") != expert_weight_format:
            raise ValueError("resolved SPALMER plan does not match the expert weight format")
        fingerprint = str(resolved_plan.get("fingerprint") or "")
        if len(fingerprint) != 64:
            raise ValueError("resolved SPALMER plan is missing its fingerprint")
```

Replace the QAT capability check:

```python
    attention_status = attention_backend_status(attention_backend)
    if attention_backend in {"auto", "fla"} and not attention_status["available"]:
        raise ValueError(str(attention_status["detail"]))
    qat_status = qat_backend_status(expert_weight_format, expert_qat_backend)
    if expert_qat_backend == "native" and not qat_status["native_w4a8_available"]:
        raise ValueError(str(qat_status["detail"]))
```

with:

```python
    attention_status = attention_backend_status(attention_backend)
    if attention_backend in {"auto", "fla"} and not attention_status["available"]:
        raise ValueError(str(attention_status["detail"]))
    qat_status = None
    if not continuing:
        qat_status = qat_backend_status(expert_weight_format, expert_qat_backend)
        if expert_qat_backend == "native" and not qat_status["native_w4a8_available"]:
            raise ValueError(str(qat_status["detail"]))
```

Replace the `tokenizer = {...}` literal and the `return {...}` with:

```python
    if continuing:
        initialization = {"mode": "checkpoint", "checkpoint_id": parent_id}
        tokenizer = {
            "source": "checkpoint",
            "registry_id": None,
            "ref": "",
            "preview_vocab_size": None,
        }
        spalmer_block: dict[str, Any] = {
            "attention_backend": attention_backend,
            "require_fla": attention_backend != "reference",
            "allow_reference_attention": attention_backend == "reference",
        }
        architecture_keys: dict[str, Any] = {}
    else:
        initialization = {"mode": "scratch", "tokenizer_ref": reference}
        tokenizer = {
            "source": tokenizer_source,
            "registry_id": tokenizer_registry_id,
            "ref": reference,
            "preview_vocab_size": vocabulary,
        }
        spalmer_block = spalmer
        architecture_keys = {
            "architecture": resolved_plan,
            "precision": {
                "recipe": "spalmer_w4a8_bf16_master_v1",
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "expert_weight_format": expert_weight_format,
                "expert_activation_format": "mxfp8",
                "expert_master_dtype": "bfloat16",
                "expert_qat_backend": expert_qat_backend,
                "ple_quant_bits": 4,
                "ple_stochastic_rounding": True,
            },
        }
    return {
        "recipe_version": SPALMER_RECIPE_VERSION,
        "architecture_type": "spalmer_v1",
        "model_family": "spalmer",
        "method": "full",
        "objective": "causal_lm",
        "checkpoint_name": name,
        "output_name": name,
        "out_checkpoint_id": str(out_checkpoint_id),
        "initialization": initialization,
        "tokenizer": tokenizer,
        "spalmer": spalmer_block,
        **architecture_keys,
        "training": training,
        "corpora": selected_corpora,
        "domains": normalized_domains or None,
        "splits": normalized_splits or None,
        "corpus_mix": mix,
        "cross_corpus_dedup": bool(cross_corpus_dedup),
        "corpus_weights": {},
        "source_max_ratio": source_cap,
        "data_preflight_warnings": list(data_preflight_warnings),
        "seq_len": int(sequence_length),
        "micro_batch": int(micro_batch),
        "grad_accum": int(gradient_accumulation),
        "max_steps": resolved_steps,
        "epochs": int(epochs),
        "seed": int(seed),
        "shuffle_buffer": 128,
        "shuffle_buffer_bytes": 64 * 2**20,
        "tokenize_chunk": 8,
        "lr": float(learning_rate),
        "min_lr_ratio": float(minimum_lr_ratio),
        "warmup_steps": warmup,
        "weight_decay": float(weight_decay),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(epsilon),
        "grad_clip": None if gradient_clip is None else float(gradient_clip),
        "auxiliary_loss_weight": float(auxiliary_loss_weight),
        "surprise_calibration_weight": float(surprise_calibration_weight),
        "log_every": int(log_every),
        "save_every": int(save_every),
        "activation_checkpointing": False,
        **budget,
    }
```

The existing `spalmer = {...}` literal (with `"capabilities": {"attention": attention_status, "expert_qat": qat_status}`) stays as is for scratch mode; in checkpoint mode `qat_status` is `None` and that literal is still built but not emitted.

- [ ] **Step 4: Run the UI tests**

Run: `python -m pytest tests/test_spalmer_ui.py -q`
Expected: all pass, including the four pre-existing tests.

- [ ] **Step 5: Lint and commit**

Run: `python -m ruff check yylab/ui/pages/train_spalmer.py tests/test_spalmer_ui.py`

```bash
git add yylab/ui/pages/train_spalmer.py tests/test_spalmer_ui.py
git commit -m "Emit checkpoint-mode and epoch fields in SPALMER recipes" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Page controls

**Files:**
- Modify: `yylab/ui/pages/train_spalmer.py` (page function `train_spalmer_page`, lines 553 onward)
- Test: `tests/test_spalmer_ui.py`

**Interfaces:**
- Consumes: `build_spalmer_run_config` keyword parameters from Task 5.
- Produces: module helpers `_spalmer_checkpoint_rows() -> list[dict[str, Any]]` and `_parent_plan(meta: Mapping[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing tests for the helpers**

Append to `tests/test_spalmer_ui.py`:

```python
def test_spalmer_checkpoint_rows_filter_by_family(monkeypatch) -> None:
    from yylab import db

    rows = [
        {"id": "a", "name": "spal", "params_b": 0.05, "meta_json": '{"model_family": "spalmer"}'},
        {"id": "b", "name": "scan", "params_b": 0.05, "meta_json": '{"family": "spalmer"}'},
        {"id": "c", "name": "hf", "params_b": 0.5, "meta_json": '{"model_family": "hf"}'},
        {"id": "d", "name": "broken", "params_b": 0.5, "meta_json": "not json"},
    ]
    monkeypatch.setattr(db, "q", lambda sql, args=(): rows)

    assert [row["id"] for row in train_spalmer._spalmer_checkpoint_rows()] == ["a", "b"]


def test_parent_plan_reads_nested_and_flat_metadata() -> None:
    flat = {"plan": {"fingerprint": "f", "config": {"vocab_size": 8192}}}
    nested = {"metadata": {"plan": {"fingerprint": "n"}}}

    assert train_spalmer._parent_plan(flat)["fingerprint"] == "f"
    assert train_spalmer._parent_plan(nested)["fingerprint"] == "n"
    assert train_spalmer._parent_plan({}) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spalmer_ui.py -q -k "checkpoint_rows or parent_plan"`
Expected: FAIL with `AttributeError: module ... has no attribute '_spalmer_checkpoint_rows'`.

- [ ] **Step 3: Add the module helpers**

Change the import `from collections.abc import Iterable` to `from collections.abc import Iterable, Mapping`. After `_resolve_target` add:

```python
def _spalmer_checkpoint_rows() -> list[dict[str, Any]]:
    """Registered base checkpoints whose metadata declares the SPALMER family."""

    rows = db.q(
        "SELECT id,name,params_b,hidden_size,num_layers,meta_json,created_at "
        "FROM checkpoints WHERE kind='base' ORDER BY created_at DESC"
    )
    selected: list[dict[str, Any]] = []
    for row in rows:
        try:
            meta = json.loads(row.get("meta_json") or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        family = str(meta.get("model_family") or meta.get("family") or "").strip().casefold()
        if family == "spalmer":
            selected.append(dict(row))
    return selected


def _parent_plan(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Read a recorded scale plan from run-produced or scan-registered metadata."""

    plan = meta.get("plan")
    if not isinstance(plan, Mapping):
        nested = meta.get("metadata")
        plan = nested.get("plan") if isinstance(nested, Mapping) else None
    return dict(plan) if isinstance(plan, Mapping) else {}
```

- [ ] **Step 4: Run the helper tests**

Run: `python -m pytest tests/test_spalmer_ui.py -q`
Expected: all pass.

- [ ] **Step 5: Add the initialization controls to section 1**

In `train_spalmer_page`, after `corpora = corpus_options()` add:

```python
        parent_rows = {row["id"]: row for row in _spalmer_checkpoint_rows()}
        parent_choices = {
            checkpoint_id: (
                f"{row['name']} · "
                f"{_format_parameters(int(round(float(row.get('params_b') or 0) * 1e9)))}"
            )
            for checkpoint_id, row in parent_rows.items()
        }
        first_parent_id = next(iter(parent_choices), None)
```

Change `form_section("1", "Output and tokenizer")` to `form_section("1", "Initialization and output")`. After the `checkpoint_name` input add:

```python
                initialization_mode = (
                    ui.select(
                        {
                            "scratch": "Random weights (scratch)",
                            "checkpoint": "Continue from checkpoint",
                        },
                        value="scratch",
                        label="Initialization",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                parent_checkpoint = (
                    ui.select(
                        parent_choices,
                        value=first_parent_id,
                        label="Parent SPALMER checkpoint",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                parent_hint = hint(
                    "Architecture, expert numerics, and tokenizer are inherited from the parent. "
                    "Optimizer moments and the learning-rate schedule start fresh."
                )
                with ui.column().classes("w-full gap-2") as tokenizer_controls:
```

and indent the existing `registered_tokenizer`, `direct_tokenizer_ref`, `preview_vocab`, and their `hint(...)` call by four spaces so they sit inside `tokenizer_controls`. Note `hint` returns `None`; make it return the label so `parent_hint.visible` can be toggled: in `yylab/ui/components.py` change `hint` to `return ui.label(text).classes("text-xs text-gray-500")`.

- [ ] **Step 6: Wrap section 2 so it can be hidden**

Immediately before `form_section("2", "Architecture and execution")` insert `with ui.column().classes("w-full") as architecture_section:` and indent every line from `form_section("2", ...)` through the violet `ui.label(...).classes("text-xs text-violet-300")` that ends section 2 (the line before `ui.separator().classes("my-3")` preceding `form_section("3", "Training data")`) by four spaces. Use this script:

```python
from pathlib import Path
p = Path("yylab/ui/pages/train_spalmer.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if 'form_section("2", "Architecture and execution")' in l)
sec3 = next(i for i, l in enumerate(lines) if 'form_section("3", "Training data")' in l)
end = sec3 - 1
while not lines[end].strip():
    end -= 1
assert "ui.separator()" in lines[end]
end -= 1  # keep the separator outside the wrapper
indent = " " * (len(lines[start]) - len(lines[start].lstrip()))
wrapped = [indent + 'with ui.column().classes("w-full") as architecture_section:\n']
wrapped += [("    " + l) if l.strip() else l for l in lines[start:end + 1]]
lines[start:end + 1] = wrapped
p.write_text("".join(lines), encoding="utf-8")
```

- [ ] **Step 7: Add the epochs input to section 4**

After the first `with field_grid(3):` block in section 4 (the one holding `sequence_length`, `token_budget`, `max_steps`), add:

```python
                with field_grid(3):
                    epochs = (
                        ui.number("Epochs (passes over the data)", value=1, min=1, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
```

In `update_budget_summary`, change the summary assignment to:

```python
            budget_summary.text = (
                f"{int(budget['max_steps']):,} optimizer steps × "
                f"{int(budget['tokens_per_optimizer_step']):,} tokens = "
                f"{int(budget['scheduled_training_tokens']):,} scheduled tokens; "
                f"the corpora are re-shuffled and replayed for {int(epochs.value or 1)} "
                f"epoch(s) until the step cap is reached."
            )
```

- [ ] **Step 8: Mode switching and plan preview**

After `selected_tokenizer` add:

```python
        def continuing() -> bool:
            return str(initialization_mode.value or "scratch") == "checkpoint"

        def selected_parent() -> dict[str, Any] | None:
            return parent_rows.get(str(parent_checkpoint.value or ""))

        def update_initialization_controls() -> None:
            parent_checkpoint.visible = continuing()
            parent_hint.visible = continuing()
            tokenizer_controls.visible = not continuing()
            architecture_section.visible = not continuing()
            update_plan_preview()
```

At the top of `update_plan_preview`, inside `with plan_preview:` and before the `try:`, add:

```python
                if continuing():
                    parent = selected_parent()
                    if parent is None:
                        ui.label("Register a SPALMER checkpoint to continue from.").classes(
                            "text-sm text-red-300"
                        )
                        return
                    meta = json.loads(parent.get("meta_json") or "{}")
                    parent_plan = _parent_plan(meta)
                    parameters = int(round(float(parent.get("params_b") or 0) * 1e9))
                    ui.label(
                        f"Inherited from {parent['name']}: {_format_parameters(parameters)} parameters · "
                        f"d={int(parent.get('hidden_size') or 0):,} · "
                        f"{int(parent.get('num_layers') or 0)} layers"
                    ).classes("text-sm text-gray-200")
                    ui.label(
                        f"Plan {str(parent_plan.get('fingerprint') or '')[:16]}… · tokenizer and "
                        "expert numerics come from the bundle."
                    ).classes("text-[11px] text-gray-500")
                    return
```

- [ ] **Step 9: Branch `on_start`**

Replace the start of `on_start` (from `selected_corpora = _ordered_values(corpora_select.value)` through the registered-tokenizer vocabulary check ending with `return`) with:

```python
            selected_corpora = _ordered_values(corpora_select.value)
            parent = selected_parent() if continuing() else None
            if continuing() and parent is None:
                ui.notify("Choose a parent SPALMER checkpoint.", type="negative")
                return
            tokenizer_ref, tokenizer_source, tokenizer_registry_id = ("", "checkpoint", None)
            if not continuing():
                tokenizer_ref, tokenizer_source, tokenizer_registry_id = selected_tokenizer()
                if tokenizer_source == "registered":
                    tokenizer_row = tokenizer_rows.get(str(tokenizer_registry_id or ""))
                    if tokenizer_row is None:
                        ui.notify("Choose a ready registered tokenizer.", type="negative")
                        return
                    catalogued_vocab = int(tokenizer_row.get("vocab_size") or 0)
                    if catalogued_vocab and catalogued_vocab != int(preview_vocab.value or 0):
                        ui.notify(
                            f"Registered tokenizer vocabulary is {catalogued_vocab:,}; "
                            "reset the planning vocabulary before starting.",
                            type="negative",
                            multi_line=True,
                        )
                        return
```

Replace the `try:` block that computes `plan` and `config` so it reads:

```python
            try:
                if continuing():
                    plan = _parent_plan(json.loads(parent.get("meta_json") or "{}"))
                else:
                    plan = plan_spalmer_configuration(
                        target_parameters=requested_target(),
                        vocab_size=int(preview_vocab.value or 0),
                        expert_weight_format=str(expert_weight_format.value or "mxfp4"),
                        directional=current_directional(),
                        atxy=current_atxy(),
                    )
                checkpoint_id = db.new_id("ckpt")
                config = build_spalmer_run_config(
                    checkpoint_name=str(checkpoint_name.value or ""),
                    tokenizer_ref=tokenizer_ref,
                    tokenizer_source=tokenizer_source,
                    tokenizer_registry_id=tokenizer_registry_id,
                    tokenizer_preview_vocab_size=int(preview_vocab.value or 0),
                    target_scale="inherited" if continuing() else str(target_scale.value or "10M"),
                    target_parameters=0 if continuing() else requested_target(),
                    resolved_plan=plan,
                    ...  # every existing keyword argument unchanged ...
                    out_checkpoint_id=checkpoint_id,
                    data_preflight_warnings=data_check["warnings"],
                    initialization_mode="checkpoint" if continuing() else "scratch",
                    parent_checkpoint_id=str(parent["id"]) if parent else None,
                    epochs=int(epochs.value or 1),
                )
```

(Keep every existing keyword argument between `resolved_plan=plan,` and `out_checkpoint_id=checkpoint_id,` exactly as it is today; only the three new keywords and the two `continuing()` expressions are added.)

Replace the pending-metadata block:

```python
            planned_config = dict(plan["config"])
            planned_parameters = int(dict(plan["parameters"])["total"])
            pending_metadata = {
```

with:

```python
            if continuing():
                planned_config = {
                    "d_model": int(parent.get("hidden_size") or 0),
                    "n_layers": int(parent.get("num_layers") or 0),
                }
                planned_parameters = int(round(float(parent.get("params_b") or 0) * 1e9))
            else:
                planned_config = dict(plan["config"])
                planned_parameters = int(dict(plan["parameters"])["total"])
            pending_metadata = {
                "initialization": config["initialization"],
```

and change `"plan_fingerprint": plan["fingerprint"],` to `"plan_fingerprint": str(plan.get("fingerprint") or ""),` and `"target_parameters": int(plan["target_parameters"]),` to `"target_parameters": int(plan.get("target_parameters") or planned_parameters),`.

- [ ] **Step 10: Wire events and initial state**

In the event wiring block add:

```python
        initialization_mode.on_value_change(lambda _: update_initialization_controls())
        parent_checkpoint.on_value_change(lambda _: update_plan_preview())
        epochs.on_value_change(lambda _: update_budget_summary())
```

and after `update_budget_summary()` at the very end add `update_initialization_controls()`.

- [ ] **Step 11: Verify the page imports and renders**

Run: `python -c "import yylab.ui.pages.train_spalmer as m; print(m.SPALMER_RECIPE_VERSION)"`
Expected: `3`

Run: `python -m pytest tests/test_spalmer_ui.py tests/test_spalmer_training.py tests/test_spalmer_runtime.py -q`
Expected: all pass.

Run the app (`python main.py`), open `/train/spalmer`, switch Initialization to "Continue from checkpoint", confirm the tokenizer and architecture controls disappear, the parent select lists `spal50m`, and the plan panel shows the inherited summary. Switch back and confirm the scratch controls return.

- [ ] **Step 12: Lint and commit**

Run: `python -m ruff check yylab/ui/pages/train_spalmer.py yylab/ui/components.py tests/test_spalmer_ui.py`

```bash
git add yylab/ui/pages/train_spalmer.py yylab/ui/components.py tests/test_spalmer_ui.py
git commit -m "Add continue-from-checkpoint and epochs controls to Train SPALMER" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Full verification and smoke run

**Files:**
- No source changes expected.

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q -p no:cacheprovider --deselect tests/test_potentiated_fp8.py::test_native_nvfp4_sparse_step_with_potentiated_master`
Expected: all pass (the deselected test needs a Blackwell GPU).

- [ ] **Step 2: Lint**

Run: `python -m ruff check yylab tests`
Expected: `All checks passed!`

- [ ] **Step 3: Dry preparation against the real checkpoint**

Run a preparation-only probe (no training) to confirm the bundle loads through the trainer path with the relocated tokenizer:

```python
from yylab.train.spalmer import prepare_spalmer_run
cfg = {
    "model_family": "spalmer", "method": "full", "objective": "causal_lm",
    "initialization": {"mode": "checkpoint", "checkpoint_id": "ckpt_20260903_083337_bec7f0"},
    "spalmer": {"attention_backend": "auto"},
    "corpora": ["<any registered corpus id>"], "corpus_mix": "token_balanced",
    "seq_len": 512, "micro_batch": 8, "grad_accum": 4, "max_steps": 10, "epochs": 1,
    "seed": 42, "log_every": 1, "save_every": 5, "out_checkpoint_id": "ckpt_probe",
    "device": "cuda", "require_cuda": True,
}
prepared = prepare_spalmer_run(cfg)
print(prepared.parent, prepared.plan.fingerprint, type(prepared.model).__name__, prepared.tokenizer.vocab_size)
```

Expected: prints the parent record, the plan fingerprint from the bundle, `SPALMERCausalLM`, and `8192`.

- [ ] **Step 4: Report**

Summarize the verification results and hand over the run settings.
