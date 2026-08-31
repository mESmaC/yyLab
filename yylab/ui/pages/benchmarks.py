from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nicegui import ui

from ... import db
from ...artifacts import run_dir
from ...benchmark.models import BenchmarkSuite, load_suite
from ...hw import best_gpu
from ...infer.engine import ENGINE
from ...workers.launch import launch_worker
from ..components import empty_state, field_grid, form_section, hint, status_badge
from ..shell import ACCENT, panel, shell, stat_card
from .runs import ACTIVE_RUN_STATUSES, _read_log_chunk, _stop

BENCHMARK_RUN_KINDS = frozenset({"benchmark_quality", "benchmark_train_perf"})
QUALITY_KIND = "benchmark_quality"
TRAIN_PERF_KIND = "benchmark_train_perf"
BUNDLED_SUITE_PATH = Path(__file__).resolve().parents[2] / "benchmark" / "suites" / "molor-core-v1.json"
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
QUALITY_CONTEXT_MODES = {
    "native": ("none", 1.0, None),
    "yarn_2x": ("yarn", 2.0, 65_536),
    "yarn_4x": ("yarn", 4.0, 131_072),
}


class ActiveRunError(RuntimeError):
    """A GPU-consuming run already owns the local execution slot."""


def _json_object(value: str | bytes | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _resolve_quant(checkpoint: dict, requested: str, adapter: dict | None = None) -> str:
    if adapter and adapter.get("quant"):
        return str(adapter["quant"])
    if checkpoint.get("quant_locked"):
        return str(checkpoint["quant_locked"])
    if requested != "auto":
        return requested
    gpu = best_gpu()
    params_b = float(checkpoint.get("params_b") or 0.0)
    if gpu and params_b and params_b * 2.3 > gpu.vram_free_gb:
        return "nf4"
    return "bf16"


def _quality_job_config(
    checkpoint: dict,
    adapter: dict | None,
    *,
    quant: str,
    policy: str,
    profile: str,
    repetitions: int,
    label: str,
    categories: list[str] | None = None,
    limit: int | None = None,
    baseline_run_id: str | None = None,
    max_new_tokens: int | None = None,
    enable_thinking: bool = False,
    max_thinking_tokens: int = 1024,
    context_mode: str = "native",
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Build the stable worker contract without training-artifact status fields."""

    if context_mode not in QUALITY_CONTEXT_MODES:
        raise ValueError(f"unknown benchmark context mode: {context_mode}")
    rope_scaling, rope_factor, max_context_tokens = QUALITY_CONTEXT_MODES[context_mode]

    return {
        "checkpoint_id": checkpoint["id"],
        "checkpoint_ref": checkpoint["ref"],
        "quant": quant,
        # Deliberately not named adapter_id: generic training failure cleanup
        # must never mutate an already-trained benchmark subject.
        "adapter_benchmark_id": adapter["id"] if adapter else None,
        "adapter_path": adapter.get("path") if adapter else None,
        "adapter_arch": adapter.get("arch") if adapter else None,
        "num_experts": int(adapter.get("num_experts") or 0) if adapter else 0,
        "policy": policy,
        "profile": profile,
        "repetitions": repetitions,
        "label": label,
        "categories": categories or None,
        "limit": limit,
        "baseline_run_id": baseline_run_id,
        "max_new_tokens": max_new_tokens,
        "enable_thinking": enable_thinking,
        "max_thinking_tokens": max_thinking_tokens,
        "context_mode": context_mode,
        "rope_scaling": rope_scaling,
        "rope_factor": rope_factor,
        "max_context_tokens": max_context_tokens,
        "continue_on_error": continue_on_error,
    }


def _train_perf_job_config(
    checkpoint: dict,
    *,
    quant: str,
    num_experts: int,
    rank: int,
    top_k: int,
    seq_len: int,
    micro_batch: int,
    grad_accum: int,
    warmup_steps: int,
    measured_steps: int,
    optimizer_state: str,
    lr: float,
    target_modules: list[str],
    grad_ckpt: bool,
    liger_fused_ce: bool,
    cudnn_sdpa: bool,
    label: str,
    require_idle_gpu: bool = True,
) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint["id"],
        "checkpoint_ref": checkpoint["ref"],
        "quant": quant,
        "num_experts": num_experts,
        "rank": rank,
        "top_k": top_k,
        "seq_len": seq_len,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "optimizer_state": optimizer_state,
        "lr": lr,
        "target_modules": target_modules,
        "grad_ckpt": grad_ckpt,
        "liger_fused_ce": liger_fused_ce,
        "cudnn_sdpa": cudnn_sdpa,
        "require_idle_gpu": require_idle_gpu,
        "max_existing_gpu_gb": 1.0,
        "label": label,
    }


def _launch_benchmark(kind: str, config: dict[str, Any]) -> tuple[str, int]:
    if kind not in BENCHMARK_RUN_KINDS:
        raise ValueError(f"unknown benchmark run kind: {kind}")
    if ENGINE.is_loaded():
        raise ActiveRunError(
            "Chat currently has a model loaded on the GPU. Unload it from Chat before "
            "starting a benchmark so the measurement has exclusive memory and compute."
        )
    active = db.q1(
        "SELECT id, kind, status FROM runs WHERE status IN ('queued','running') ORDER BY created_at LIMIT 1"
    )
    if active:
        raise ActiveRunError(
            f"Run {active['id']} ({active['kind']}) is already {active['status']}. "
            "Wait for it to finish or stop it from Runs before starting a benchmark."
        )
    rid = db.new_id("run")
    db.execute(
        "INSERT INTO runs(id, kind, status, created_at, config_json) VALUES(?,?,?,?,?)",
        (rid, kind, "queued", db.now_iso(), json.dumps(config)),
    )
    try:
        pid = launch_worker(rid)
    except Exception as exc:
        db.execute(
            "UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=?",
            (db.now_iso(), f"worker launch failed: {exc}", rid),
        )
        raise
    db.execute("UPDATE runs SET metrics_json=? WHERE id=?", (json.dumps({"pid": pid}), rid))
    return rid, pid


def _benchmark_runs() -> list[dict]:
    placeholders = ",".join("?" for _ in BENCHMARK_RUN_KINDS)
    return db.q(
        f"SELECT * FROM runs WHERE kind IN ({placeholders}) ORDER BY created_at DESC",
        tuple(sorted(BENCHMARK_RUN_KINDS)),
    )


def _number(payload: dict[str, Any], *paths: str) -> float | None:
    for dotted in paths:
        value: Any = payload
        for key in dotted.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _summary_path(run_id: str, metrics: dict[str, Any]) -> Path:
    for key in ("summary_path", "result_path", "receipt_path"):
        candidate = metrics.get(key)
        if isinstance(candidate, str) and candidate:
            return Path(candidate)
    return run_dir(run_id) / "summary.json"


def _summary_for(run: dict) -> dict[str, Any]:
    metrics = _json_object(run.get("metrics_json"))
    paths = (
        run_dir(run["id"]) / "semantic-summary.json",
        _summary_path(run["id"], metrics),
    )
    for path in paths:
        if not path.is_file():
            continue
        try:
            parsed = json.loads(path.read_text("utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except (OSError, json.JSONDecodeError):
            pass
    return metrics


def _progress_text(run: dict) -> str:
    if run["status"] not in ACTIVE_RUN_STATUSES:
        return ""
    path = run_dir(run["id"]) / "progress.json"
    if not path.is_file():
        return "starting…"
    try:
        progress = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return "starting…"
    stage = str(progress.get("stage") or "benchmark")
    step = progress.get("step", 0)
    total = progress.get("max_steps", progress.get("total", "?"))
    message = str(progress.get("msg") or progress.get("message") or "")
    return f"{stage} {step}/{total}" + (f" · {message}" if message else "")


def _quality_contract_text(config: dict[str, Any]) -> str:
    answer = config.get("max_new_tokens")
    answer_text = str(int(answer)) if isinstance(answer, (int, float)) and answer else "suite"
    if config.get("enable_thinking"):
        thinking_text = (
            f"think {int(config['max_thinking_tokens'])}"
            if config.get("max_thinking_tokens")
            else "legacy shared-cap"
        )
    else:
        thinking_text = "no-think"
    context = str(config.get("context_mode") or "native").replace("_", " ")
    return f"answer {answer_text} · {thinking_text} · {context}"


def _history_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in _benchmark_runs():
        config = _json_object(run.get("config_json"))
        metrics = _json_object(run.get("metrics_json"))
        if run["kind"] == QUALITY_KIND:
            summary = _summary_for(run)
            canonical_quality = _number(summary, "quality_index")
            semantic_quality = _number(summary, "semantic_quality_index")
            if semantic_quality is not None:
                result = f"{semantic_quality:.1f} semantic"
                if canonical_quality is not None:
                    result += f" · {canonical_quality:.1f} Molor"
            else:
                result = f"{canonical_quality:.1f}/1000" if canonical_quality is not None else ""
            mode = (
                f"{summary.get('primary_policy') or config.get('policy') or ''} · "
                f"{_quality_contract_text(config)}"
            )
        else:
            speed = _number(metrics, "useful_tokens_per_second")
            status = metrics.get("status")
            result = (
                str(status).upper()
                if status in {"oom", "busy_gpu"}
                else (f"{speed:,.0f} tok/s" if speed is not None else "")
            )
            mode = f"E{config.get('num_experts', '?')}/r{config.get('rank', '?')}"
        rows.append(
            {
                "id": run["id"],
                "kind": "Quality" if run["kind"] == QUALITY_KIND else "Training perf",
                "label": config.get("label") or run["id"],
                "mode": mode,
                "status": run["status"],
                "result": result,
                "progress": _progress_text(run),
                "created": run["created_at"],
            }
        )
    return rows


def _policy_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    policies = summary.get("policies")
    if not isinstance(policies, dict):
        return []
    rows = []
    for name, value in policies.items():
        result = value if isinstance(value, dict) else {}
        canonical_quality = _number(result, "quality_index")
        semantic_quality = _number(result, "semantic_quality_index", "quality_index")
        coverage = result.get("coverage")
        if isinstance(coverage, dict):
            coverage_text = f"{coverage.get('scored', 0)}/{coverage.get('expected', '?')}"
        else:
            coverage_text = ""
        speed = _number(result, "performance.tokens_per_second_median")
        rows.append(
            {
                "policy": str(name),
                "semantic_quality": (round(semantic_quality, 1) if semantic_quality is not None else None),
                "canonical_quality": (round(canonical_quality, 1) if canonical_quality is not None else None),
                "coverage": coverage_text,
                "speed": round(speed, 1) if speed is not None else None,
            }
        )
    return rows


def _category_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    categories = summary.get("categories")
    if not isinstance(categories, dict):
        primary = summary.get("primary_policy")
        policies = summary.get("policies")
        selected = policies.get(primary) if isinstance(policies, dict) else None
        categories = selected.get("categories") if isinstance(selected, dict) else None
    if not isinstance(categories, dict):
        return []
    rows = []
    for name, value in categories.items():
        result = value if isinstance(value, dict) else {}
        canonical_score = _number(result, "score_percent")
        if canonical_score is None:
            canonical_mean = _number(result, "mean_score")
            canonical_score = canonical_mean * 100 if canonical_mean is not None else None
        semantic_score = _number(result, "semantic_score_percent")
        if semantic_score is None:
            semantic_mean = _number(result, "semantic_mean_score")
            semantic_score = semantic_mean * 100 if semantic_mean is not None else canonical_score
        rows.append(
            {
                "category": result.get("label") or name,
                "semantic_score": round(semantic_score, 1) if semantic_score is not None else None,
                "canonical_score": (round(canonical_score, 1) if canonical_score is not None else None),
                "semantic_passed": result.get("semantic_passed", result.get("passed", "")),
                "canonical_passed": result.get("passed", ""),
                "scored": result.get("scored", ""),
            }
        )
    return rows


def _render_summary(container, run: dict, summary: dict[str, Any]) -> None:
    container.clear()
    if not summary or (run["status"] in ACTIVE_RUN_STATUSES and set(summary) == {"pid"}):
        return
    with container:
        if run["kind"] == QUALITY_KIND:
            primary = str(summary.get("primary_policy") or "—")
            canonical_quality = _number(
                summary,
                "quality_index",
                f"policies.{primary}.quality_index",
            )
            semantic_quality = _number(
                summary,
                "semantic_quality_index",
                f"policies.{primary}.semantic_quality_index",
                "quality_index",
                f"policies.{primary}.quality_index",
            )
            coverage = summary.get("coverage")
            if not isinstance(coverage, dict):
                policies = summary.get("policies")
                selected = policies.get(primary) if isinstance(policies, dict) else None
                coverage = selected.get("coverage") if isinstance(selected, dict) else None
            coverage_text = "—"
            if isinstance(coverage, dict):
                coverage_text = f"{coverage.get('scored', 0)}/{coverage.get('expected', '?')}"
            speed = _number(
                summary,
                "performance.tokens_per_second_median",
                f"policies.{primary}.performance.tokens_per_second_median",
            )
            comparison = summary.get("comparison")
            semantic_delta = (
                _number(comparison, "semantic_quality_index_delta", "quality_index_delta")
                if isinstance(comparison, dict)
                else None
            )
            config = _json_object(run.get("config_json"))
            answer_total = _number(
                summary,
                "performance.answer_tokens_total",
                f"policies.{primary}.performance.answer_tokens_total",
            )
            thinking_total = _number(
                summary,
                "performance.thinking_tokens_total",
                f"policies.{primary}.performance.thinking_tokens_total",
            )
            thinking_exhaustions = _number(
                summary,
                "performance.thinking_budget_exhaustions",
                f"policies.{primary}.performance.thinking_budget_exhaustions",
            )
            with ui.row().classes("w-full gap-3 flex-wrap"):
                stat_card(
                    "Semantic correctness",
                    f"{semantic_quality:.1f}/1000" if semantic_quality is not None else "—",
                )
                stat_card(
                    "Molor strict score",
                    f"{canonical_quality:.1f}/1000" if canonical_quality is not None else "—",
                )
                stat_card("Primary policy", primary)
                stat_card("Coverage", coverage_text)
                stat_card("Median speed", f"{speed:.1f} tok/s" if speed is not None else "—")
                stat_card("Answer cap", str(config.get("max_new_tokens") or "suite"))
                stat_card(
                    "Thinking",
                    (
                        (
                            f"{int(config['max_thinking_tokens'])} tokens"
                            if config.get("max_thinking_tokens")
                            else "legacy shared answer cap"
                        )
                        if config.get("enable_thinking")
                        else "off"
                    ),
                )
                stat_card("Context", str(config.get("context_mode") or "native").replace("_", " "))
                if answer_total is not None and thinking_total is not None:
                    stat_card(
                        "Generated tokens",
                        f"{int(answer_total):,} answer · {int(thinking_total):,} thinking",
                    )
                if thinking_exhaustions:
                    stat_card("Reasoning caps reached", f"{int(thinking_exhaustions):,}")
                if semantic_delta is not None:
                    stat_card("Semantic baseline delta", f"{semantic_delta:+.1f}")

            interval = summary.get("semantic_quality_index_ci") or summary.get("quality_index_ci")
            if isinstance(interval, dict) and interval.get("lower") is not None:
                ui.label(
                    f"Semantic 95% stratified bootstrap interval: {float(interval['lower']):.1f}–"
                    f"{float(interval['upper']):.1f}"
                ).classes("text-xs text-gray-400")

            matrix = summary.get("matrix")
            semantic_regret = (
                _number(matrix, "semantic_dynamic_regret", "dynamic_regret")
                if isinstance(matrix, dict)
                else None
            )
            if semantic_regret is not None:
                ui.label(
                    f"Semantic dynamic-router regret to the best forced expert: "
                    f"{semantic_regret:+.1f} index points. "
                    "Expert slots are anonymous; this is not a domain-route score."
                ).classes("text-xs text-gray-400")
            if isinstance(comparison, dict):
                ui.label(
                    f"Semantic paired baseline: "
                    f"{comparison.get('semantic_wins', comparison.get('wins', 0))} wins · "
                    f"{comparison.get('semantic_ties', comparison.get('ties', 0))} ties · "
                    f"{comparison.get('semantic_losses', comparison.get('losses', 0))} losses"
                ).classes("text-xs text-gray-400")
                if comparison.get("semantic_regression", comparison.get("regression")):
                    ui.label(
                        "Semantic regression flag: correctness fell by at least 5 index points."
                    ).classes("text-xs text-red-400")

            policies = _policy_rows(summary)
            if policies:
                ui.label("Policy comparison").classes("text-sm font-semibold mt-2")
                ui.table(
                    columns=[
                        {"name": "policy", "label": "Policy", "field": "policy"},
                        {
                            "name": "semantic_quality",
                            "label": "Semantic index",
                            "field": "semantic_quality",
                        },
                        {
                            "name": "canonical_quality",
                            "label": "Molor strict",
                            "field": "canonical_quality",
                        },
                        {"name": "coverage", "label": "Coverage", "field": "coverage"},
                        {"name": "speed", "label": "Median tok/s", "field": "speed"},
                    ],
                    rows=policies,
                    row_key="policy",
                ).classes("w-full")
            categories = _category_rows(summary)
            if categories:
                ui.label("Category scores").classes("text-sm font-semibold mt-2")
                ui.table(
                    columns=[
                        {"name": "category", "label": "Category", "field": "category"},
                        {
                            "name": "semantic_score",
                            "label": "Semantic %",
                            "field": "semantic_score",
                        },
                        {
                            "name": "canonical_score",
                            "label": "Molor strict %",
                            "field": "canonical_score",
                        },
                        {
                            "name": "semantic_passed",
                            "label": "Semantic passed",
                            "field": "semantic_passed",
                        },
                        {
                            "name": "canonical_passed",
                            "label": "Strict passed",
                            "field": "canonical_passed",
                        },
                        {"name": "scored", "label": "Scored", "field": "scored"},
                    ],
                    rows=categories,
                    row_key="category",
                ).classes("w-full")
        else:
            useful = _number(summary, "useful_tokens_per_second")
            median = _number(summary, "step_seconds_median")
            p95 = _number(summary, "step_seconds_p95")
            peak_alloc = _number(summary, "peak_allocated_gb", "peak_cuda_allocated_gb")
            peak_reserved = _number(summary, "peak_reserved_gb", "peak_cuda_reserved_gb")
            with ui.row().classes("w-full gap-3 flex-wrap"):
                stat_card("Useful throughput", f"{useful:,.0f} tok/s" if useful is not None else "—")
                stat_card("Median step", f"{median:.3f} s" if median is not None else "—")
                stat_card("P95 step", f"{p95:.3f} s" if p95 is not None else "—")
                stat_card(
                    "Peak VRAM",
                    f"{peak_alloc:.2f}/{peak_reserved:.2f} GB"
                    if peak_alloc is not None and peak_reserved is not None
                    else "—",
                )
            status = summary.get("status")
            if status:
                ui.label(f"Benchmark receipt: {status}").classes("text-xs text-gray-400")
            if summary.get("error"):
                ui.label(str(summary["error"])).classes("text-xs text-red-400 whitespace-pre-wrap")

        artifacts = [
            ("Report", run_dir(run["id"]) / "report.md"),
            ("Summary JSON", run_dir(run["id"]) / "summary.json"),
            ("Results JSONL", run_dir(run["id"]) / "results.jsonl"),
            ("Scores CSV", run_dir(run["id"]) / "scores.csv"),
            ("Comparison", run_dir(run["id"]) / "comparison.json"),
            ("Receipt", run_dir(run["id"]) / "benchmark-receipt.json"),
        ]
        available = [(label, path) for label, path in artifacts if path.is_file()]
        if available:
            with ui.row().classes("w-full gap-2 flex-wrap mt-2"):
                for label, path in available:
                    ui.button(
                        f"Download {label}",
                        icon="download",
                        on_click=lambda _event=None, artifact=path: ui.download(artifact),
                    ).props("dense flat no-caps")


def _load_bundled_suite() -> BenchmarkSuite:
    return load_suite(BUNDLED_SUITE_PATH)


@ui.page("/benchmarks")
def benchmarks_page() -> None:
    suite = _load_bundled_suite()
    state: dict[str, Any] = {
        "selected": None,
        "table_signature": None,
        "detail_signature": None,
        "log_path": None,
        "log_offset": None,
        "log_identity": None,
        "log_mtime_ns": None,
        "log_has_content": False,
    }
    detail_refs: dict[str, Any] = {}

    with shell("Benchmarks", "/benchmarks"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Benchmarks").classes("text-xl font-semibold")
                ui.label(
                    "Run immutable quality evaluations or synthetic training hot-path measurements."
                ).classes("text-xs text-gray-500")
            ui.link("Open all runs", "/runs").classes("text-sm").style(f"color: {ACCENT}")

        with ui.tabs().classes("w-full") as tabs:
            quality_tab = ui.tab("Quality", icon="fact_check")
            performance_tab = ui.tab("Training performance", icon="speed")

        with ui.tab_panels(tabs, value=quality_tab).classes("w-full bg-transparent p-0"):
            with ui.tab_panel(quality_tab).classes("p-0"):
                with panel():
                    form_section("1", "Model stack")
                    checkpoints = {
                        row["id"]: row["name"]
                        for row in db.q("SELECT * FROM checkpoints WHERE kind!='pending' ORDER BY name")
                    }
                    with field_grid(3):
                        quality_ckpt = (
                            ui.select(checkpoints, label="Checkpoint")
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        quality_adapter = (
                            ui.select({"": "(none)"}, label="Adapter")
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        quality_quant = (
                            ui.select(
                                {"auto": "Auto", "bf16": "BF16", "nf4": "NF4", "int8": "INT8"},
                                value="auto",
                                label="Quantization",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )

                    form_section("2", "Evaluation")
                    ui.label(
                        f"Suite: {suite.title} · {len(suite.cases)} cases · {suite.fingerprint[:12]}…"
                    ).classes("text-xs text-gray-400")
                    with field_grid(3):
                        profile = (
                            ui.select(
                                {
                                    name: f"{name} · {len(case_ids)} cases"
                                    for name, case_ids in suite.profiles.items()
                                },
                                value="quick",
                                label="Profile",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        policy = (
                            ui.select(
                                {
                                    "base": "Base only",
                                    "adapter": "Adapter (LoRA / default route)",
                                    "dynamic": "Learned per-token routing",
                                    "matrix": "Base + dynamic + every expert",
                                    "expert": "One forced expert",
                                },
                                value="dynamic",
                                label="Policy",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        forced_expert = (
                            ui.number("Expert index", value=0, min=0, max=31, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    forced_expert.visible = False
                    categories = (
                        ui.select(
                            suite.category_labels,
                            label="Categories (all when empty)",
                            multiple=True,
                        )
                        .props("dense outlined use-chips")
                        .classes("w-full")
                    )
                    with field_grid(3):
                        repetitions = (
                            ui.number("Repetitions", value=1, min=1, max=20, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        case_limit = (
                            ui.number(
                                "Case limit (0 = profile)", value=0, min=0, max=len(suite.cases), step=1
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        max_new = (
                            ui.number("Max answer tokens (0 = suite)", value=0, min=0, max=4096, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(2):
                        thinking_budget = (
                            ui.number("Thinking token budget", value=1024, min=64, max=32768, step=64)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        context_mode = (
                            ui.select(
                                {
                                    "native": "Native checkpoint context (recommended)",
                                    "yarn_2x": "YaRN 2× · 65,536 tokens",
                                    "yarn_4x": "YaRN 4× · 131,072 tokens",
                                },
                                value="native",
                                label="Context extension",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    thinking_budget.visible = False
                    baselines = {"": "(none)"}
                    for row in db.q(
                        "SELECT id, config_json FROM runs WHERE kind=? AND status='done' "
                        "ORDER BY created_at DESC",
                        (QUALITY_KIND,),
                    ):
                        if not (run_dir(row["id"]) / "summary.json").is_file():
                            continue
                        cfg = _json_object(row["config_json"])
                        baselines[row["id"]] = (
                            f"{cfg.get('label') or row['id']} · "
                            f"{cfg.get('profile', '?')}/{cfg.get('policy', '?')} · "
                            f"{_quality_contract_text(cfg)}"
                        )
                    with field_grid(2):
                        baseline = (
                            ui.select(baselines, value="", label="Baseline run")
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        quality_label = (
                            ui.input("Run label", value="quality-benchmark")
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(2):
                        enable_thinking = ui.checkbox("Enable thinking", value=False)
                        continue_errors = ui.checkbox("Continue after individual case errors", value=True)
                    hint(
                        "Thinking stays off by default for frozen-suite compatibility. "
                        "When enabled, its separate budget never reduces the final-answer limit; "
                        "yyLab uses Qwen's seeded thinking-mode sampling controls. Native context "
                        "is best for short cases; select YaRN only when the combined budgets need it."
                    )
                    quality_count = ui.label("").classes("text-xs text-gray-400")
                    ui.label(
                        "Holdout warning: benchmark prompts must never be added to training corpora."
                    ).classes("text-xs text-amber-400")
                    quality_start = (
                        ui.button("Start quality benchmark", icon="play_arrow")
                        .props("no-caps unelevated")
                        .classes("w-full mt-3")
                    )

            with ui.tab_panel(performance_tab).classes("p-0"):
                with panel():
                    form_section("1", "Synthetic training stack")
                    with field_grid(2):
                        perf_ckpt = (
                            ui.select(checkpoints, label="Checkpoint")
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        perf_quant = (
                            ui.select(
                                {"auto": "Auto", "bf16": "BF16", "nf4": "NF4", "int8": "INT8"},
                                value="bf16",
                                label="Quantization",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(3):
                        perf_experts = (
                            ui.number("Experts", value=8, min=1, max=32, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        perf_rank = (
                            ui.number("Rank", value=8, min=1, max=128, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        perf_top_k = (
                            ui.number("Top-K", value=2, min=1, max=32, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with ui.row().classes("w-full gap-2"):
                        ui.button(
                            "E8/r8",
                            on_click=lambda: (
                                perf_experts.set_value(8),
                                perf_rank.set_value(8),
                                perf_top_k.set_value(2),
                            ),
                        ).props("dense flat no-caps")
                        ui.button(
                            "E4/r16",
                            on_click=lambda: (
                                perf_experts.set_value(4),
                                perf_rank.set_value(16),
                                perf_top_k.set_value(2),
                            ),
                        ).props("dense flat no-caps")
                        ui.button(
                            "E16/r4",
                            on_click=lambda: (
                                perf_experts.set_value(16),
                                perf_rank.set_value(4),
                                perf_top_k.set_value(2),
                            ),
                        ).props("dense flat no-caps")

                    form_section("2", "Measured workload")
                    with field_grid(3):
                        perf_seq = (
                            ui.select([256, 512, 1024, 2048, 4096, 8192], value=2048, label="Sequence length")
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        perf_micro = (
                            ui.number("Micro batch", value=4, min=1, max=32, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        perf_accum = (
                            ui.number("Grad accumulation", value=4, min=1, max=256, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(3):
                        warmup_steps = (
                            ui.number("Warmup steps", value=2, min=0, max=50, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        measured_steps = (
                            ui.number("Measured steps", value=5, min=1, max=200, step=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        optimizer = (
                            ui.select(
                                {
                                    "auto": "Auto",
                                    "gpu": "GPU fused/8-bit",
                                    "cpu_paged": "Paged optimizer",
                                    "cpu_offload": "Full CPU offload",
                                },
                                value="auto",
                                label="Optimizer state",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(2):
                        perf_lr = (
                            ui.number("Learning rate", value=0.0003, min=0.000001, max=0.01, step=0.00001)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        perf_label = (
                            ui.input("Run label", value="training-performance")
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    perf_targets = (
                        ui.select(
                            list(TARGET_MODULES),
                            value=list(TARGET_MODULES),
                            multiple=True,
                            label="Target modules",
                        )
                        .props("dense outlined use-chips")
                        .classes("w-full")
                    )
                    with ui.row().classes("w-full gap-5"):
                        perf_grad_ckpt = ui.checkbox("Gradient checkpointing", value=True)
                        perf_liger = ui.checkbox("Liger fused CE", value=True)
                        perf_cudnn = ui.checkbox("cuDNN-priority SDPA", value=True)
                        perf_idle_gpu = ui.checkbox("Require idle GPU", value=True)
                    hint(
                        "This uses deterministic synthetic tokens only. Warmup is excluded from measured "
                        "throughput. Idle-GPU mode skips the run when existing GPU use exceeds 1 GB."
                    )
                    perf_start = (
                        ui.button("Start training benchmark", icon="play_arrow")
                        .props("no-caps unelevated")
                        .classes("w-full mt-3")
                    )

        def checkpoint(value: str | None) -> dict | None:
            return db.q1("SELECT * FROM checkpoints WHERE id=?", (value,)) if value else None

        def adapter(value: str | None) -> dict | None:
            return db.q1("SELECT * FROM adapters WHERE id=?", (value,)) if value else None

        def refresh_quality_adapters(_event=None) -> None:
            options = {"": "(none)"}
            if quality_ckpt.value:
                for row in db.q(
                    "SELECT * FROM adapters WHERE checkpoint_id=? AND status='trained' "
                    "ORDER BY created_at DESC",
                    (quality_ckpt.value,),
                ):
                    options[row["id"]] = f"{row['name']} · E{row['num_experts']}/r{row['rank']}"
            quality_adapter.options = options
            quality_adapter.value = ""
            quality_adapter.update()

        def update_quality_controls(_event=None) -> None:
            forced_expert.visible = policy.value == "expert"
            thinking_budget.visible = bool(enable_thinking.value)
            selected_adapter = adapter(quality_adapter.value)
            max_expert = max(0, int((selected_adapter or {}).get("num_experts") or 1) - 1)
            forced_expert.set_value(min(int(forced_expert.value or 0), max_expert))
            forced_expert.props(f"max={max_expert}")
            selected = suite.select(
                profile=str(profile.value or "quick"),
                categories=list(categories.value or []),
                limit=int(case_limit.value) if int(case_limit.value or 0) > 0 else None,
            )
            policies = (
                int((selected_adapter or {}).get("num_experts") or 0) + 2 if policy.value == "matrix" else 1
            )
            generations = len(selected) * int(repetitions.value or 1) * policies
            quality_count.text = (
                f"{len(selected)} cases × {int(repetitions.value or 1)} repetitions"
                + (f" × {policies} policies" if policies > 1 else "")
                + f" = {generations} generations"
            )

        def start_quality() -> None:
            ckpt = checkpoint(quality_ckpt.value)
            selected_adapter = adapter(quality_adapter.value)
            selected_policy = str(policy.value or "dynamic")
            problems = []
            if not ckpt:
                problems.append("choose a checkpoint")
            if selected_policy != "base" and not selected_adapter:
                problems.append("choose an adapter for this policy")
            if (
                selected_policy in {"dynamic", "matrix", "expert"}
                and selected_adapter
                and selected_adapter.get("arch") != "mole"
            ):
                problems.append("learned and forced routing policies require a MoLE adapter")
            if not str(quality_label.value or "").strip():
                problems.append("enter a run label")
            if problems:
                ui.notify("Fix: " + ", ".join(problems), type="warning")
                return
            worker_policy = (
                f"expert:{int(forced_expert.value or 0)}" if selected_policy == "expert" else selected_policy
            )
            resolved_quant = _resolve_quant(ckpt, str(quality_quant.value or "auto"), selected_adapter)
            config = _quality_job_config(
                ckpt,
                selected_adapter,
                quant=resolved_quant,
                policy=worker_policy,
                profile=str(profile.value or "quick"),
                repetitions=int(repetitions.value or 1),
                label=str(quality_label.value).strip(),
                categories=list(categories.value or []) or None,
                limit=int(case_limit.value) if int(case_limit.value or 0) > 0 else None,
                baseline_run_id=str(baseline.value) if baseline.value else None,
                max_new_tokens=int(max_new.value) if int(max_new.value or 0) > 0 else None,
                enable_thinking=bool(enable_thinking.value),
                max_thinking_tokens=int(thinking_budget.value or 1024),
                context_mode=str(context_mode.value or "native"),
                continue_on_error=bool(continue_errors.value),
            )
            try:
                rid, pid = _launch_benchmark(QUALITY_KIND, config)
            except ActiveRunError as exc:
                ui.notify(str(exc), type="warning", multi_line=True)
                return
            ui.notify(f"Quality benchmark queued · pid {pid}", type="positive")
            refresh_history(force=True)
            render_detail(rid)

        def start_performance() -> None:
            ckpt = checkpoint(perf_ckpt.value)
            problems = []
            if not ckpt:
                problems.append("choose a checkpoint")
            if int(perf_top_k.value or 1) > int(perf_experts.value or 1):
                problems.append("top-K must be ≤ experts")
            if not perf_targets.value:
                problems.append("choose target modules")
            if not str(perf_label.value or "").strip():
                problems.append("enter a run label")
            if problems:
                ui.notify("Fix: " + ", ".join(problems), type="warning")
                return
            config = _train_perf_job_config(
                ckpt,
                quant=_resolve_quant(ckpt, str(perf_quant.value or "auto")),
                num_experts=int(perf_experts.value or 8),
                rank=int(perf_rank.value or 8),
                top_k=int(perf_top_k.value or 2),
                seq_len=int(perf_seq.value or 2048),
                micro_batch=int(perf_micro.value or 4),
                grad_accum=int(perf_accum.value or 4),
                warmup_steps=int(warmup_steps.value if warmup_steps.value is not None else 2),
                measured_steps=int(measured_steps.value or 5),
                optimizer_state=str(optimizer.value or "auto"),
                lr=float(perf_lr.value or 3e-4),
                target_modules=list(perf_targets.value),
                grad_ckpt=bool(perf_grad_ckpt.value),
                liger_fused_ce=bool(perf_liger.value),
                cudnn_sdpa=bool(perf_cudnn.value),
                label=str(perf_label.value).strip(),
                require_idle_gpu=bool(perf_idle_gpu.value),
            )
            try:
                rid, pid = _launch_benchmark(TRAIN_PERF_KIND, config)
            except ActiveRunError as exc:
                ui.notify(str(exc), type="warning", multi_line=True)
                return
            ui.notify(f"Training benchmark queued · pid {pid}", type="positive")
            refresh_history(force=True)
            render_detail(rid)

        quality_ckpt.on_value_change(refresh_quality_adapters)
        for control in (
            profile,
            policy,
            quality_adapter,
            categories,
            repetitions,
            case_limit,
            enable_thinking,
        ):
            control.on_value_change(update_quality_controls)
        quality_start.on_click(start_quality)
        perf_start.on_click(start_performance)
        update_quality_controls()

        ui.separator().classes("my-2")
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Benchmark history").classes("text-lg font-semibold")
            refresh_button = ui.button("Refresh", icon="refresh").props("dense flat no-caps")
        history_holder = ui.column().classes("w-full")
        with history_holder:
            history_table = ui.table(
                columns=[
                    {"name": "label", "label": "Label", "field": "label", "align": "left"},
                    {"name": "kind", "label": "Kind", "field": "kind"},
                    {"name": "mode", "label": "Mode", "field": "mode"},
                    {"name": "status", "label": "Status", "field": "status"},
                    {"name": "result", "label": "Result", "field": "result"},
                    {"name": "progress", "label": "Progress", "field": "progress", "align": "left"},
                    {"name": "created", "label": "Created", "field": "created"},
                ],
                rows=[],
                row_key="id",
                pagination=20,
            ).classes("w-full")
            history_table.add_slot(
                "body-cell-status",
                """
                <q-td :props="props">
                  <q-badge :color="props.value === 'done' ? 'green' : props.value === 'failed' ? 'red' : props.value === 'running' ? 'primary' : props.value === 'queued' ? 'teal' : 'grey'">
                    {{ props.value }}
                  </q-badge>
                </q-td>
                """,
            )
            with ui.column().classes("w-full") as history_empty:
                empty_state(
                    "science",
                    "No benchmark runs yet",
                    "Start a quality or training-performance benchmark above.",
                )

        detail = ui.column().classes("w-full")

        def stop_benchmark(run_id: str) -> None:
            run = db.q1("SELECT status, metrics_json FROM runs WHERE id=?", (run_id,))
            if not run or run["status"] not in ACTIVE_RUN_STATUSES:
                return
            metadata = _json_object(run.get("metrics_json"))
            if metadata.get("pid"):
                _stop(int(metadata["pid"]))
            db.execute(
                "UPDATE runs SET status='cancelled', finished_at=?, error=? WHERE id=?",
                (db.now_iso(), "cancelled by user", run_id),
            )
            ui.notify("Benchmark stopped", type="warning")
            refresh_history(force=True)
            refresh_detail(force=True)

        def refresh_history(*, force: bool = False) -> None:
            rows = _history_rows()
            signature = tuple(
                (
                    row["id"],
                    row["status"],
                    row["result"],
                    row["progress"],
                    row["created"],
                )
                for row in rows
            )
            if force or signature != state["table_signature"]:
                history_table.update_rows(rows, clear_selection=False)
                history_table.visible = bool(rows)
                history_empty.visible = not rows
                state["table_signature"] = signature

        def render_detail(run_id: str | None) -> None:
            detail.clear()
            detail_refs.clear()
            state["selected"] = run_id
            state["detail_signature"] = None
            state["log_path"] = None
            state["log_offset"] = None
            state["log_identity"] = None
            state["log_mtime_ns"] = None
            state["log_has_content"] = False
            if not run_id:
                return
            run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
            if not run:
                state["selected"] = None
                return
            with detail, panel():
                with ui.row().classes("w-full items-center justify-between"):
                    header = ui.label("").classes("font-semibold")
                    with ui.row().classes("items-center gap-2"):
                        badge_holder = ui.row().classes("items-center")
                        stop_button = ui.button(
                            "Stop",
                            icon="stop",
                            on_click=lambda: stop_benchmark(run_id),
                        ).props("dense flat no-caps color=negative")
                progress_label = ui.label("").classes("text-xs text-gray-400")
                summary_box = ui.column().classes("w-full gap-3")
                with ui.expansion("Configuration", icon="tune").classes("w-full"):
                    ui.label(json.dumps(_json_object(run.get("config_json")), indent=2)).classes(
                        "text-xs font-mono whitespace-pre-wrap text-gray-300"
                    )
                with ui.expansion("Log (live)", value=run["status"] in ACTIVE_RUN_STATUSES).classes("w-full"):
                    log_placeholder = ui.label("Waiting for log output…").classes("text-xs text-gray-500")
                    log_view = (
                        ui.log(max_lines=500)
                        .classes("w-full h-72 text-xs font-mono text-gray-400 rounded p-2")
                        .style("background: #0d0d12")
                    )
                    log_view.visible = False
                error_label = ui.label("").classes("text-xs text-red-400 whitespace-pre-wrap")
            detail_refs.update(
                header=header,
                badge=badge_holder,
                stop=stop_button,
                progress=progress_label,
                summary=summary_box,
                log=log_view,
                log_placeholder=log_placeholder,
                error=error_label,
            )
            refresh_detail(force=True)

        def refresh_detail(*, force: bool = False) -> None:
            run_id = state["selected"]
            if not run_id or not detail_refs:
                return
            run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
            if not run:
                render_detail(None)
                return
            config = _json_object(run.get("config_json"))
            progress_text = _progress_text(run)
            summary = _summary_for(run)
            signature = (
                run["status"],
                progress_text,
                run.get("error") or "",
                run.get("metrics_json") or "",
            )
            if force or signature != state["detail_signature"]:
                detail_refs["header"].text = (
                    f"{config.get('label') or run_id} · "
                    f"{'Quality' if run['kind'] == QUALITY_KIND else 'Training performance'}"
                )
                detail_refs["badge"].clear()
                with detail_refs["badge"]:
                    status_badge(run["status"])
                detail_refs["stop"].visible = run["status"] in ACTIVE_RUN_STATUSES
                detail_refs["progress"].text = progress_text
                detail_refs["progress"].visible = bool(progress_text)
                error = str(run.get("error") or "")[-3000:]
                detail_refs["error"].text = error
                detail_refs["error"].visible = bool(error)
                _render_summary(detail_refs["summary"], run, summary)
                state["detail_signature"] = signature

            log_path = Path(run["log_path"]) if run.get("log_path") else run_dir(run_id) / "train.log"
            path_key = str(log_path)
            if state["log_path"] != path_key:
                detail_refs["log"].clear()
                state["log_path"] = path_key
                state["log_offset"] = None
                state["log_identity"] = None
                state["log_mtime_ns"] = None
                state["log_has_content"] = False
            chunk, offset, identity, mtime_ns, reset = _read_log_chunk(
                log_path,
                offset=state["log_offset"],
                identity=state["log_identity"],
                mtime_ns=state["log_mtime_ns"],
            )
            if reset:
                detail_refs["log"].clear()
                state["log_has_content"] = False
            if chunk:
                detail_refs["log"].push(chunk)
                state["log_has_content"] = True
            state["log_offset"] = offset
            state["log_identity"] = identity
            state["log_mtime_ns"] = mtime_ns
            detail_refs["log"].visible = state["log_has_content"]
            detail_refs["log_placeholder"].visible = not state["log_has_content"]
            detail_refs["log_placeholder"].text = (
                "Waiting for log output…"
                if run["status"] in ACTIVE_RUN_STATUSES
                else "No log output was written."
            )

        history_table.on("row-click", lambda event: render_detail(event.args[1]["id"]))
        refresh_button.on_click(lambda: (refresh_history(force=True), refresh_detail(force=True)))

        def tick() -> None:
            refresh_history()
            refresh_detail()

        refresh_history(force=True)
        ui.timer(1.0, tick, immediate=False)
