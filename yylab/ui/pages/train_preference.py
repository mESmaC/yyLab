from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from ... import db
from ...models.bitsandbytes_capabilities import (
    available_bitsandbytes_quantizations,
    bitsandbytes_compute_precision,
)
from ...models.families import checkpoint_family, hf_compatible_checkpoints
from ...models.precision_capabilities import (
    available_dense_precisions,
    model_execution_device,
    require_model_precision,
)
from ...train.preference import (
    PREFERENCE_OBJECTIVE,
    PREFERENCE_RUN_KIND,
    DPOConfig,
    load_preference_dataset,
    optimizer_steps,
    preference_dataset_stats,
)
from ...workers.launch import launch_preference_worker
from ..components import field_grid, form_section, hint
from ..shell import ACCENT, panel, shell


def build_preference_run_config(**values: object) -> DPOConfig:
    """Typed UI boundary used by the page and regression tests."""

    return DPOConfig.from_dict(values)


@ui.page("/train/preference")
def train_preference_page() -> None:
    model_device = model_execution_device()
    dense_precisions = available_dense_precisions(model_device)
    policy_precision_options = {
        precision: f"LoRA · {precision.upper()}"
        for precision in dense_precisions
    }
    policy_precision_options.update(
        (
            quantization,
            f"QLoRA · {quantization.upper()} · "
            f"{bitsandbytes_compute_precision(quantization).upper()} compute",
        )
        for quantization in available_bitsandbytes_quantizations(model_device)
    )
    with shell("Train · Preference", "/train/preference"):
        ui.label("Offline preference optimization").classes("text-xl font-semibold")
        ui.label(
            "Train a LoRA or QLoRA policy from pair rows or normalized trace shards with DPO. "
            "JSONL, JSON, and Parquet normalize to the same canonical pair contract."
        ).classes("text-sm text-gray-400 -mt-2")

        checkpoint_rows = db.q(
            "SELECT * FROM checkpoints WHERE kind!='pending' ORDER BY created_at DESC"
        )
        checkpoint_rows = hf_compatible_checkpoints(checkpoint_rows)
        checkpoints = {
            row["id"]: (
                f"{row['name']} ({float(row.get('params_b') or 0):g}B"
                f"{', locked ' + row['quant_locked'] if row.get('quant_locked') else ''})"
            )
            for row in checkpoint_rows
        }

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[600px] shrink-0"):
                form_section("1", "Policy and pair data")
                if not checkpoints:
                    ui.label("Add or train a base checkpoint before starting DPO.").classes(
                        "text-sm text-amber-400"
                    )
                checkpoint = (
                    ui.select(checkpoints, label="Base checkpoint / reference policy")
                    .props("dense outlined")
                    .classes("w-full")
                )
                with field_grid(2):
                    dataset_path = (
                        ui.input("Preference file or shard directory")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    dataset_format = (
                        ui.select(
                            {
                                "auto": "Auto-detect rows",
                                "pairs": "Chosen/rejected pair rows",
                                "traces": "Labelled/scored trace rows",
                            },
                            value="auto",
                            label="Record contract",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                hint(
                    "A file or directory may mix JSONL/JSON and Parquet shards. Pair rows may use "
                    "string or message-array branches. Trace rows need an explicit chosen/rejected "
                    "label, score, or rank; unlabelled traces belong in chat SFT."
                )
                validation = (
                    ui.label("Dataset has not been validated.")
                    .classes("text-xs text-gray-400 px-3 py-2 rounded-md w-full")
                    .style(f"background: {ACCENT}10")
                )
                validate_button = ui.button("Validate pairs", icon="fact_check").props(
                    "dense flat no-caps"
                )

                ui.separator().classes("my-3")
                form_section("2", "DPO recipe", "Assistant responses only; prompts are masked")
                with field_grid(3):
                    quant = (
                        ui.select(
                            policy_precision_options,
                            value=(
                                "bf16"
                                if "bf16" in dense_precisions
                                else next(iter(policy_precision_options), None)
                            ),
                            label="Policy loading",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    rank = (
                        ui.select([8, 16, 32, 64], value=16, label="LoRA rank")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    beta = (
                        ui.number("DPO beta", value=0.1, min=0.001, max=2, step=0.01)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(3):
                    max_length = (
                        ui.select([512, 1024, 2048, 4096], value=1024, label="Max tokens")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    epochs = (
                        ui.number("Epochs", value=1, min=1, max=20)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    learning_rate = (
                        ui.select(
                            [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                            value=5e-5,
                            label="Learning rate",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with ui.expansion("Advanced", icon="tune").classes("w-full"):
                    with field_grid(3):
                        micro_batch = (
                            ui.number("Micro batch (pairs)", value=1, min=1, max=32)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        grad_accum = (
                            ui.number("Gradient accumulation", value=8, min=1, max=256)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        max_steps = (
                            ui.number("Max optimizer steps (0 = all)", value=0, min=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(3):
                        dropout = (
                            ui.number("LoRA dropout", value=0.05, min=0, max=0.95, step=0.01)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        smoothing = (
                            ui.number("Label smoothing", value=0, min=0, max=0.49, step=0.01)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        seed = (
                            ui.number("Shuffle seed", value=42, min=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    gradient_checkpointing = ui.checkbox(
                        "Gradient checkpointing", value=True
                    )

                ui.separator().classes("my-3")
                form_section("3", "Output adapter")
                adapter_name = (
                    ui.input("Adapter name (optional)")
                    .props("dense outlined")
                    .classes("w-full")
                )
                start_button = (
                    ui.button("Start offline DPO", icon="play_arrow")
                    .props("no-caps unelevated")
                    .classes("w-full mt-3")
                )
                start_button.enabled = bool(checkpoints)
                hint("Runs in a dedicated background worker; progress and logs appear on Runs.")

            with ui.column().classes("flex-1 gap-4 min-w-[360px]"):
                with panel():
                    ui.label("What this run does").classes("text-base font-semibold mb-2")
                    ui.label(
                        "For each pair, yyLab renders the same prompt twice with the checkpoint's "
                        "chat template, masks prompt tokens, and scores the complete chosen and "
                        "rejected assistant turns. It caches reference scores with the adapter "
                        "disabled, then updates only LoRA weights."
                    ).classes("text-sm text-gray-300")
                    ui.label(
                        "This is offline preference optimization. It does not collect rollouts, "
                        "fit a reward model, or implement PPO/GRPO."
                    ).classes("text-sm text-amber-300 mt-2")
                with panel():
                    ui.label("Accepted canonical pair").classes("text-base font-semibold mb-2")
                    ui.code(
                        '{"id":"tone-1","prompt":[{"role":"system","content":"Be concise."},'
                        '{"role":"user","content":"Define entropy."}],'
                        '"chosen":"A measure of uncertainty.",'
                        '"rejected":"Entropy is definitely just randomness."}'
                    ).classes("w-full text-xs")
                    ui.label("Normalized candidate traces").classes(
                        "text-base font-semibold mt-3 mb-2"
                    )
                    ui.code(
                        '{"pair_id":"tone-1","messages":[{"role":"user",'
                        '"content":"Define entropy."},{"role":"assistant",'
                        '"content":"A measure of uncertainty."}],"reward":1.0}\n'
                        '{"pair_id":"tone-1","messages":[{"role":"user",'
                        '"content":"Define entropy."},{"role":"assistant",'
                        '"content":"Just randomness."}],"reward":0.0}'
                    ).classes("w-full text-xs")

        def selected_checkpoint() -> dict | None:
            if not checkpoint.value:
                return None
            return db.q1("SELECT * FROM checkpoints WHERE id=?", (checkpoint.value,))

        def sync_quant_lock() -> None:
            row = selected_checkpoint()
            locked = str((row or {}).get("quant_locked") or "")
            quant.enabled = not bool(locked)
            if locked:
                quant.value = locked

        def validate_pairs(*, notify: bool = True) -> tuple[object, dict] | None:
            try:
                dataset = load_preference_dataset(
                    str(dataset_path.value or ""),
                    data_format=str(dataset_format.value or "auto"),
                )
                stats = preference_dataset_stats(dataset.examples)
            except (OSError, TypeError, ValueError) as exc:
                validation.text = f"Invalid dataset: {exc}"
                validation.classes(replace="text-xs text-red-300 px-3 py-2 rounded-md w-full")
                if notify:
                    ui.notify(str(exc), type="negative", multi_line=True)
                return None
            provenance = dataset.provenance
            split_summary = ", ".join(
                f"{name} {count:,}"
                for name, count in dict(provenance["split_counts"]).items()
            )
            validation.text = (
                f"{stats['pairs']:,} unique pairs from {provenance['source_rows']:,} rows / "
                f"{len(provenance['source_files']):,} files · "
                f"{provenance['duplicate_pairs']:,} exact duplicates removed · "
                f"{stats['multi_turn_prompts']:,} multi-turn · splits: {split_summary}"
            )
            validation.classes(replace="text-xs text-green-300 px-3 py-2 rounded-md w-full")
            if notify:
                ui.notify("Preference source normalized and validated.", type="positive")
            return dataset, stats

        def on_start() -> None:
            row = selected_checkpoint()
            if row is None:
                ui.notify("Choose a base checkpoint.", type="warning")
                return
            if checkpoint_family(row) == "spalmer":
                ui.notify(
                    "Preference training requires a Hugging Face-compatible base checkpoint.",
                    type="negative",
                )
                return
            validated = validate_pairs(notify=False)
            if validated is None:
                ui.notify("Fix the preference dataset before starting.", type="negative")
                return
            dataset, _stats = validated
            locked = str(row.get("quant_locked") or "")
            requested_quant = str(quant.value or "bf16")
            if locked and requested_quant != locked:
                ui.notify(
                    f"Checkpoint is quant-locked to {locked}; DPO requested {requested_quant}.",
                    type="negative",
                )
                return
            try:
                require_model_precision(requested_quant)
            except (RuntimeError, ValueError) as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return
            adapter_id = db.new_id("adp")
            run_id = db.new_id("run")
            name = str(adapter_name.value or "").strip() or f"{row['name']}-dpo-r{int(rank.value)}"
            try:
                config = build_preference_run_config(
                    checkpoint_id=row["id"],
                    checkpoint_ref=row["ref"],
                    adapter_id=adapter_id,
                    adapter_name=name,
                    dataset_path=str(Path(str(dataset_path.value)).expanduser().resolve()),
                    dataset_format=str(dataset_format.value or "auto"),
                    dataset_checksum=str(dataset.provenance["canonical_sha256"]),
                    quant=requested_quant,
                    rank=int(rank.value or 16),
                    alpha=float(2 * int(rank.value or 16)),
                    dropout=float(dropout.value or 0),
                    beta=float(beta.value or 0.1),
                    label_smoothing=float(smoothing.value or 0),
                    max_length=int(max_length.value or 1024),
                    epochs=int(epochs.value or 1),
                    max_steps=int(max_steps.value or 0) or None,
                    micro_batch=int(micro_batch.value or 1),
                    grad_accum=int(grad_accum.value or 8),
                    learning_rate=float(learning_rate.value or 5e-5),
                    seed=int(seed.value or 0),
                    gradient_checkpointing=bool(gradient_checkpointing.value),
                )
            except (TypeError, ValueError) as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return
            adapter_metadata = {
                "objective": PREFERENCE_OBJECTIVE,
                "algorithm": "dpo",
                "dataset_path": config.dataset_path,
                "dataset_format": config.dataset_format,
                "dataset_checksum": config.dataset_checksum,
                "dataset_provenance": dict(dataset.provenance),
            }
            with db.transaction() as connection:
                connection.execute(
                    "INSERT INTO adapters(id, name, created_at, checkpoint_id, arch, quant, "
                    "num_experts, top_k, rank, target_modules_json, status, meta_json) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        adapter_id,
                        name,
                        db.now_iso(),
                        row["id"],
                        "lora",
                        requested_quant,
                        1,
                        1,
                        config.rank,
                        json.dumps(["all-linear"]),
                        "training",
                        json.dumps(adapter_metadata),
                    ),
                )
                connection.execute(
                    "INSERT INTO runs(id, kind, status, created_at, config_json) VALUES(?,?,?,?,?)",
                    (
                        run_id,
                        PREFERENCE_RUN_KIND,
                        "queued",
                        db.now_iso(),
                        json.dumps(config.to_dict()),
                    ),
                )
            try:
                pid = launch_preference_worker(run_id)
            except OSError as exc:
                db.execute(
                    "UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=?",
                    (db.now_iso(), str(exc), run_id),
                )
                db.execute(
                    "UPDATE adapters SET status='failed' WHERE id=?",
                    (adapter_id,),
                )
                ui.notify(f"Could not launch DPO worker: {exc}", type="negative")
                return
            db.execute(
                "UPDATE runs SET metrics_json=? WHERE id=?",
                (json.dumps({"pid": pid}), run_id),
            )
            steps = optimizer_steps(len(dataset.examples), config)
            ui.notify(
                f"Offline DPO started — {name} ({len(dataset.examples):,} unique pairs, "
                f"up to {steps:,} steps, pid {pid}). See Runs.",
                type="positive",
                multi_line=True,
            )

        validate_button.on_click(validate_pairs)
        checkpoint.on_value_change(lambda _: sync_quant_lock())
        start_button.on_click(on_start)
