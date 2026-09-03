from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nicegui import ui

from ... import db
from ...ingest.domains import DOMAINS
from ...tokenizer_registry import (
    TOKENIZER_ACTION_IMPORT,
    TOKENIZER_ACTION_TRAIN,
    TOKENIZER_RUN_KIND,
    create_tokenizer_record,
    get_tokenizer,
    hub_preset,
    hub_preset_options,
    list_tokenizers,
    mark_tokenizer_failed,
    parse_tokenizer_source,
)
from ...train.compact_tokenizer import CompactTokenizerSpec
from ...train.data import normalize_source_splits
from ...workers.launch import launch_tokenizer_worker
from ..components import dir_browser, empty_state, hint, status_badge
from ..shell import panel, shell


def build_tokenizer_train_config(
    *,
    tokenizer_id: str,
    corpora: list[str],
    splits: list[str],
    evaluation_splits: list[str],
    domains: list[str] | None,
    vocab_size: int,
    min_frequency: int,
    max_documents: int,
    max_training_bytes: int,
    special_token_profile: str,
    lowercase: bool = False,
    normalization: str = "none",
    custom_special_tokens: list[str] | None = None,
    evaluation_max_documents: int = 512,
    evaluation_max_bytes: int = 8 * 2**20,
    seed: int = 0,
) -> dict[str, Any]:
    if not corpora:
        raise ValueError("select at least one corpus")
    spec = CompactTokenizerSpec(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        max_documents=int(max_documents),
        max_training_bytes=int(max_training_bytes),
        lowercase=bool(lowercase),
        normalization=str(normalization),
        special_token_profile=str(special_token_profile),
        custom_special_tokens=tuple(custom_special_tokens or ()),
    ).validate()
    return {
        "action": TOKENIZER_ACTION_TRAIN,
        "tokenizer_id": tokenizer_id,
        "corpora": list(dict.fromkeys(corpora)),
        "splits": normalize_source_splits(splits) or ["train", "unspecified"],
        "evaluation_splits": normalize_source_splits(evaluation_splits),
        "domains": list(dict.fromkeys(domains or ())) or None,
        "corpus_mix": "round_robin" if len(corpora) > 1 else "concatenate",
        "spec": spec.to_dict(),
        "evaluation_max_documents": max(1, int(evaluation_max_documents)),
        "evaluation_max_bytes": max(0, int(evaluation_max_bytes)),
        "seed": int(seed),
    }


def build_tokenizer_import_config(
    *,
    tokenizer_id: str,
    source: str,
    special_token_profile: str = "auto",
    custom_special_tokens: list[str] | None = None,
    hub_revision: str = "",
) -> dict[str, Any]:
    source_kind, source_ref = parse_tokenizer_source(source)
    if source_kind == "path":
        source_path = Path(source_ref)
        if source_path.is_file() and source_path.name.casefold() != "tokenizer.json":
            raise ValueError(
                "select an HF tokenizer directory, a raw tokenizer.json file, or a Hugging Face repo id"
            )
    profile = str(special_token_profile or "auto").strip().casefold()
    if profile not in {"auto", "causal", "chat", "custom"}:
        raise ValueError("import profile must be auto, causal, chat, or custom")
    custom = [value for value in custom_special_tokens or () if value]
    if profile == "custom" and not custom:
        raise ValueError("custom import profile requires at least one special token")
    return {
        "action": TOKENIZER_ACTION_IMPORT,
        "tokenizer_id": tokenizer_id,
        "source": source_ref,
        "source_kind": source_kind,
        "hub_revision": str(hub_revision or "").strip(),
        "special_token_profile": profile,
        "custom_special_tokens": custom,
    }


def _special_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        values = value.replace(",", "\n").splitlines()
    else:
        values = list(value or ())
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _launch_registered_tokenizer(
    *,
    name: str,
    kind: str,
    config: dict[str, Any],
    corpora: list[str] | None = None,
    splits: list[str] | None = None,
) -> tuple[str, str, int]:
    tokenizer_id = str(config["tokenizer_id"])
    run_id = db.new_id("run")
    create_tokenizer_record(
        tokenizer_id=tokenizer_id,
        name=name,
        kind=kind,
        corpora=corpora,
        splits=splits,
        config=config,
    )
    db.execute(
        "INSERT INTO runs(id,kind,status,created_at,config_json) VALUES(?,?,?,?,?)",
        (run_id, TOKENIZER_RUN_KIND, "queued", db.now_iso(), json.dumps(config)),
    )
    try:
        pid = launch_tokenizer_worker(run_id)
    except OSError as exc:
        db.execute(
            "UPDATE runs SET status='failed',finished_at=?,error=? WHERE id=?",
            (db.now_iso(), str(exc), run_id),
        )
        mark_tokenizer_failed(tokenizer_id, str(exc))
        raise
    db.execute("UPDATE runs SET metrics_json=? WHERE id=?", (json.dumps({"pid": pid}), run_id))
    return tokenizer_id, run_id, pid


@ui.page("/tokenizers")
def tokenizers_page():
    state: dict[str, Any] = {"selected": None}

    with shell("Tokenizers", "/tokenizers"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Tokenizers").classes("text-xl font-semibold")
            with ui.row().classes("gap-2"):
                import_button = ui.button("Import", icon="drive_folder_upload").props("outline no-caps")
                train_button = ui.button("Train byte BPE", icon="add").props("no-caps")

        table_container = ui.column().classes("w-full")
        detail_container = ui.column().classes("w-full")

        def rows() -> list[dict[str, Any]]:
            values = []
            for row in list_tokenizers(parsed=True):
                evaluation = row["stats"].get("evaluation", {})
                bytes_per_token = evaluation.get("bytes_per_token")
                values.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "status": row["status"],
                        "kind": row["kind"],
                        "vocab": f"{int(row['vocab_size'] or 0):,}" if row["vocab_size"] else "—",
                        "compression": f"{bytes_per_token:.3f}" if bytes_per_token is not None else "—",
                        "created": row["created_at"],
                    }
                )
            return values

        def select(tokenizer_id: str) -> None:
            state["selected"] = tokenizer_id
            render_detail()

        def refresh() -> None:
            table_container.clear()
            with table_container:
                values = rows()
                if not values:
                    empty_state(
                        "token",
                        "No tokenizer artifacts",
                        "Train a streaming byte-level BPE over one or more corpora, or import an existing "
                        "Hugging Face tokenizer directory/raw tokenizer.json.",
                        "Train tokenizer",
                        train_dialog.open,
                    )
                    return
                table = ui.table(
                    columns=[
                        {"name": "name", "label": "Name", "field": "name", "align": "left", "sortable": True},
                        {"name": "status", "label": "Status", "field": "status", "sortable": True},
                        {"name": "kind", "label": "Kind", "field": "kind"},
                        {"name": "vocab", "label": "Vocabulary", "field": "vocab", "sortable": True},
                        {"name": "compression", "label": "Eval bytes/token", "field": "compression"},
                        {"name": "created", "label": "Created", "field": "created", "sortable": True},
                    ],
                    rows=values,
                    row_key="id",
                    pagination=12,
                ).classes("w-full")
                table.add_slot(
                    "body-cell-status",
                    """<q-td :props="props"><q-badge :color="props.value === 'ready' ? 'green' :
                    props.value === 'failed' ? 'red' : 'primary'" outline>{{ props.value }}</q-badge></q-td>""",
                )
                table.on("row-click", lambda event: select(event.args[1]["id"]))

        def render_detail() -> None:
            detail_container.clear()
            if not state["selected"]:
                return
            row = get_tokenizer(state["selected"], parsed=True)
            if not row:
                return
            with detail_container, panel():
                with ui.row().classes("w-full items-center gap-3"):
                    ui.label(row["name"]).classes("text-base font-semibold")
                    status_badge(row["status"])
                ui.label(row["path"] or "Artifact path pending").classes("text-xs text-gray-500 break-all")
                if row["error"]:
                    ui.label(row["error"]).classes("text-xs text-red-400 whitespace-pre-wrap")
                stats = row["stats"]
                if stats:
                    evaluation = stats.get("evaluation", {})
                    with ui.row().classes("w-full gap-6 mt-2"):
                        ui.label(f"Vocabulary: {int(row['vocab_size'] or 0):,}").classes("text-sm")
                        if evaluation.get("tokens") is not None:
                            ui.label(
                                f"Evaluation: {evaluation.get('tokens', 0):,} tokens · "
                                f"{evaluation.get('bytes_per_token') or 0:.3f} bytes/token"
                            ).classes("text-sm")
                        ui.label(
                            f"Round-trip failures: {evaluation.get('round_trip_failures', 0):,}"
                        ).classes("text-sm")
                    checksum = stats.get("tokenizer_payload_sha256")
                    if checksum:
                        ui.label(f"SHA-256: {checksum}").classes("text-xs text-gray-500 font-mono")
                with ui.expansion("Configuration and statistics", icon="data_object").classes("w-full mt-2"):
                    ui.code(json.dumps({"config": row["config"], "stats": stats}, indent=2)).classes(
                        "w-full text-xs max-h-96 overflow-auto"
                    )

        ready_corpora = {
            row["id"]: row["name"]
            for row in db.q(
                "SELECT id,name FROM corpora WHERE status='ready' AND COALESCE(kind,'ingested')!='rejected' "
                "ORDER BY name"
            )
        }

        with ui.dialog() as train_dialog, ui.card().classes("w-[760px] max-w-[95vw] p-5"):
            ui.label("Train streaming byte-level BPE").classes("text-lg font-semibold")
            hint(
                "The corpus is streamed into the tokenizer trainer; yyLab does not materialize the selected "
                "corpora in RAM. Vocabulary size is arbitrary within the selected byte/special-token floor."
            )
            train_name = ui.input("Artifact name", value="byte-bpe-experiment").props("outlined dense").classes("w-full")
            train_corpora = ui.select(ready_corpora, label="Training corpora", multiple=True).props(
                "outlined dense use-chips"
            ).classes("w-full")
            with ui.row().classes("w-full gap-3"):
                train_splits = ui.select(
                    ["train", "validation", "test", "unspecified"],
                    value=["train", "unspecified"],
                    label="Training source splits",
                    multiple=True,
                ).props("outlined dense use-chips").classes("flex-1")
                evaluation_splits = ui.select(
                    ["validation", "test", "train", "unspecified"],
                    value=["validation", "test"],
                    label="Held-out evaluation splits",
                    multiple=True,
                ).props("outlined dense use-chips").classes("flex-1")
            train_domains = ui.select(DOMAINS, label="Domains (all when empty)", multiple=True).props(
                "outlined dense use-chips clearable"
            ).classes("w-full")
            with ui.row().classes("w-full gap-3"):
                profile = ui.select(
                    {"causal": "Minimal causal (>=257)", "chat": "Chat roles (>=264)", "custom": "Custom"},
                    value="causal",
                    label="Special-token profile",
                ).props("outlined dense").classes("w-56")
                vocab = ui.number("Vocabulary size", value=4096, min=257, step=1).props("outlined dense").classes("w-44")
                min_frequency = ui.number("Minimum frequency", value=2, min=1, step=1).props(
                    "outlined dense"
                ).classes("w-44")
            custom_tokens = ui.textarea(
                "Custom special tokens (comma or newline separated; first is shared EOS/PAD)",
                value="<|endoftext|>",
            ).props("outlined dense").classes("w-full")
            custom_tokens.visible = False

            def sync_profile() -> None:
                custom_tokens.visible = profile.value == "custom"
                floor = 264 if profile.value == "chat" else (
                    256 + max(1, len(_special_tokens(custom_tokens.value)))
                    if profile.value == "custom"
                    else 257
                )
                vocab.props(f"min={floor}")
                if int(vocab.value or 0) < floor:
                    vocab.set_value(floor)

            profile.on_value_change(lambda _: sync_profile())
            custom_tokens.on_value_change(lambda _: sync_profile())
            with ui.row().classes("w-full gap-3"):
                max_documents = ui.number("Max training documents", value=100_000, min=1, step=1000).props(
                    "outlined dense"
                ).classes("flex-1")
                max_mib = ui.number("Max training MiB (0 = unlimited)", value=0, min=0, step=16).props(
                    "outlined dense"
                ).classes("flex-1")
                normalization = ui.select(
                    {
                        "none": "None (byte-lossless)",
                        "nfkc": "NFKC",
                        "nfkc_lowercase": "NFKC + lowercase",
                    },
                    value="none",
                    label="Normalization",
                ).props("outlined dense").classes("w-52")
            with ui.row().classes("w-full gap-3"):
                eval_documents = ui.number("Evaluation documents", value=512, min=1, step=1).props(
                    "outlined dense"
                ).classes("flex-1")
                eval_mib = ui.number("Evaluation MiB", value=8, min=0, step=1).props("outlined dense").classes(
                    "flex-1"
                )
                seed = ui.number("Seed", value=0, step=1).props("outlined dense").classes("w-32")
            train_summary = ui.label("").classes("text-xs text-gray-400")

            def update_train_summary() -> None:
                selected_count = len(train_corpora.value or ())
                byte_cap = float(max_mib.value or 0)
                cap_text = f"{byte_cap:g} MiB" if byte_cap else "unlimited bytes"
                train_summary.text = (
                    f"Will train {int(vocab.value or 0):,}-entry {profile.value} byte BPE from "
                    f"{selected_count} corpus source{'s' if selected_count != 1 else ''}, "
                    f"up to {int(max_documents.value or 0):,} documents / {cap_text}; "
                    f"normalization={normalization.value}."
                )

            for summary_input in (
                train_corpora,
                vocab,
                profile,
                max_documents,
                max_mib,
                normalization,
            ):
                summary_input.on_value_change(lambda _: update_train_summary())
            update_train_summary()
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=train_dialog.close).props("flat no-caps")
                start_train = ui.button("Start training", icon="play_arrow").props("no-caps")

        with ui.dialog() as import_dialog, ui.card().classes("w-[680px] max-w-[95vw] p-5"):
            ui.label("Import tokenizer").classes("text-lg font-semibold")
            hint(
                "Pick a large hub tokenizer preset, type a Hugging Face repo id (owner/name), choose a "
                "local tokenizer directory, or give the full path to a raw tokenizer.json. yyLab "
                "downloads or copies it into a self-contained artifact and never mutates the source."
            )
            import_preset = ui.select(
                {"": "Custom source", **hub_preset_options()},
                value="",
                label="Hugging Face preset",
            ).props("outlined dense").classes("w-full")
            import_preset_note = ui.label("").classes("text-xs text-teal-300")
            import_preset_note.visible = False
            import_name = ui.input("Artifact name", value="imported-tokenizer").props("outlined dense").classes("w-full")
            with ui.row().classes("w-full items-end gap-2"):
                import_source = ui.input(
                    "Source: Hugging Face repo id, tokenizer directory, or tokenizer.json", value=""
                ).props("outlined dense").classes("flex-1")
                import_browser = dir_browser(
                    "Select tokenizer directory", lambda path: import_source.set_value(path)
                )
                ui.button(icon="folder", on_click=import_browser.open).props("outline dense")
            import_revision = ui.input(
                "Hub revision (optional branch, tag, or commit)", value=""
            ).props("outlined dense").classes("w-full")
            hint(
                "Gated repos (for example Gemma) need the Hugging Face token saved under Settings. "
                "The catalogued vocabulary is the size SPALMER will plan with."
            )

            def apply_import_preset() -> None:
                preset = hub_preset(str(import_preset.value or ""))
                if preset is None:
                    import_preset_note.visible = False
                    return
                import_source.value = preset.repo_id
                import_source.update()
                import_name.value = preset.label
                import_name.update()
                import_preset_note.text = preset.note
                import_preset_note.visible = True

            import_preset.on_value_change(lambda _: apply_import_preset())
            import_profile = ui.select(
                {
                    "auto": "Auto / preserve HF mappings",
                    "causal": "Minimal causal",
                    "chat": "Chat roles",
                    "custom": "Custom",
                },
                value="auto",
                label="Special-token handling",
            ).props("outlined dense").classes("w-full")
            import_custom = ui.textarea(
                "Custom special tokens (first is shared EOS/PAD)", value="<|endoftext|>"
            ).props("outlined dense").classes("w-full")
            import_custom.visible = False
            import_profile.on_value_change(
                lambda _: setattr(import_custom, "visible", import_profile.value == "custom")
            )
            ui.label(
                "The source is loaded read-only, saved into yyLab's artifact store, and reloaded "
                "before it becomes selectable."
            ).classes("text-xs text-gray-400")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=import_dialog.close).props("flat no-caps")
                start_import = ui.button("Import", icon="content_copy").props("no-caps")

        def on_train() -> None:
            tokenizer_id = db.new_id("tok")
            try:
                config = build_tokenizer_train_config(
                    tokenizer_id=tokenizer_id,
                    corpora=list(train_corpora.value or ()),
                    splits=list(train_splits.value or ()),
                    evaluation_splits=list(evaluation_splits.value or ()),
                    domains=list(train_domains.value or ()),
                    vocab_size=int(vocab.value or 0),
                    min_frequency=int(min_frequency.value or 0),
                    max_documents=int(max_documents.value or 0),
                    max_training_bytes=int(float(max_mib.value or 0) * 2**20),
                    lowercase=False,
                    normalization=str(normalization.value or "none"),
                    special_token_profile=str(profile.value),
                    custom_special_tokens=_special_tokens(custom_tokens.value),
                    evaluation_max_documents=int(eval_documents.value or 0),
                    evaluation_max_bytes=int(float(eval_mib.value or 0) * 2**20),
                    seed=int(seed.value or 0),
                )
                _, run_id, pid = _launch_registered_tokenizer(
                    name=str(train_name.value or "").strip(),
                    kind="byte_bpe",
                    config=config,
                    corpora=config["corpora"],
                    splits=config["splits"],
                )
            except (OSError, TypeError, ValueError) as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return
            train_dialog.close()
            refresh()
            ui.notify(f"Tokenizer training started (run {run_id}, pid {pid}). See Runs.", type="positive")

        def on_import() -> None:
            tokenizer_id = db.new_id("tok")
            try:
                config = build_tokenizer_import_config(
                    tokenizer_id=tokenizer_id,
                    source=str(import_source.value or ""),
                    special_token_profile=str(import_profile.value),
                    custom_special_tokens=_special_tokens(import_custom.value),
                    hub_revision=str(import_revision.value or ""),
                )
                _, run_id, pid = _launch_registered_tokenizer(
                    name=str(import_name.value or "").strip(),
                    kind="imported",
                    config=config,
                )
            except (OSError, TypeError, ValueError) as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return
            import_dialog.close()
            refresh()
            ui.notify(f"Tokenizer import started (run {run_id}, pid {pid}). See Runs.", type="positive")

        start_train.on_click(on_train)
        start_import.on_click(on_import)
        train_button.on_click(train_dialog.open)
        import_button.on_click(import_dialog.open)
        refresh()

        def poll() -> None:
            refresh()
            if state["selected"]:
                render_detail()

        ui.timer(3.0, poll)
