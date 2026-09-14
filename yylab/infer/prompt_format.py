"""Checkpoint-owned prompt formats, independent of model names or architecture.

A tokenizer shipping a chat template does not establish that model weights
were trained with it. Old metadata is interpreted read-only; saved contracts
are authoritative and invalid contracts never fall back to invented roles.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_FORMAT_FILENAME = "prompt_format.json"
PROMPT_FORMAT_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class PromptFormat:
    default_mode: str = "chat"
    supported_modes: tuple[str, ...] = ("chat", "raw")
    source: str = "unknown"
    chat_template: str | None = None
    note: str = (
        "This checkpoint has no saved input-format record. Conversation mode keeps its previous "
        "behavior, but its training format is unverified. Text continuation is also available."
    )

    def resolve_mode(self, requested: str | None = None) -> str:
        mode = requested or self.default_mode
        if mode not in {"chat", "raw"}:
            raise ValueError(f"unknown prompt mode: {mode}")
        # A caller may explicitly test an unverified chat format. This does
        # not add it to established choices or change the checkpoint default.
        return mode

    def options(self) -> dict[str, str]:
        labels = {"raw": "Text continuation", "chat": "Conversation (trained format)"}
        if self.source == "unknown":
            labels["chat"] = "Conversation (format unverified)"
        return {mode: labels[mode] for mode in self.supported_modes}


def _template(tokenizer: Any) -> str:
    getter = getattr(tokenizer, "get_chat_template", None)
    value = getter() if callable(getter) else getattr(tokenizer, "chat_template", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("chat-trained checkpoint requires its exact saved chat template")
    return value


def _template_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def validate_prompt_format_contract(value: Any) -> PromptFormat:
    if not isinstance(value, Mapping) or value.get("schema") != PROMPT_FORMAT_SCHEMA:
        raise ValueError("unsupported or invalid checkpoint prompt-format schema")
    objective = value.get("objective")
    if objective not in {"causal_lm", "chat_sft"}:
        raise ValueError("checkpoint prompt-format objective must be causal_lm or chat_sft")
    modes = value.get("supported_modes")
    if not isinstance(modes, list) or modes not in [["raw"], ["chat", "raw"]]:
        raise ValueError("invalid checkpoint prompt-format supported_modes")
    default = value.get("default_mode")
    if default not in modes:
        raise ValueError("checkpoint prompt-format default_mode is not supported")
    if value.get("add_special_tokens") is not False:
        raise ValueError("checkpoint prompt formats must disable duplicate special-token insertion")
    template = value.get("chat_template")
    if "chat" in modes:
        if not isinstance(template, str) or not template.strip():
            raise ValueError("chat-trained checkpoint requires its exact saved chat template")
        if value.get("chat_template_sha256") != _template_hash(template):
            raise ValueError("checkpoint prompt-format chat-template checksum mismatch")
        note = "Conversation mode uses the exact format recorded during chat training."
    else:
        if template is not None or value.get("chat_template_sha256") is not None:
            raise ValueError("raw-only checkpoint cannot advertise a trained chat template")
        note = (
            "This run records text-continuation training. Enter a passage for it to continue; each prompt "
            "is independent. A trained conversation format is not recorded."
        )
    return PromptFormat(default, tuple(modes), "saved_contract", template, note)


def build_prompt_format_contract(config: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Build a contract for a newly saved training bundle, never an old one.

    A caller continuing a verified chat checkpoint may explicitly supply its
    inherited ``prompt_format_contract``; causal pretraining alone establishes
    no new role format. No inference is made from a generic tokenizer template.
    """

    objective = str(config.get("objective") or "causal_lm")
    if objective not in {"causal_lm", "chat_sft"}:
        raise ValueError(f"unsupported prompt-format training objective: {objective}")
    inherited = config.get("prompt_format_contract")
    previous = validate_prompt_format_contract(inherited) if inherited is not None else None
    if previous is not None and objective == "causal_lm":
        return dict(inherited)
    chat = objective == "chat_sft"
    template = _template(tokenizer) if chat else None
    if previous is not None and previous.chat_template is not None and previous.chat_template != template:
        raise ValueError(
            "chat_sft tokenizer template differs from the inherited prompt-format contract; "
            "restore the recorded template or explicitly select a matching new training contract"
        )
    result = {
        "schema": PROMPT_FORMAT_SCHEMA,
        "objective": objective,
        "default_mode": "chat" if chat else "raw",
        "supported_modes": ["chat", "raw"] if chat else ["raw"],
        "add_special_tokens": False,
        "chat_template": template,
        "chat_template_sha256": _template_hash(template) if template else None,
    }
    validate_prompt_format_contract(result)
    return result


def load_prompt_format(reference: str | Path, tokenizer: Any = None) -> PromptFormat:
    """Resolve local metadata without writing files or downloading remote refs."""

    directory = Path(reference).expanduser()
    contract = directory / PROMPT_FORMAT_FILENAME
    if contract.is_file():
        try:
            value = json.loads(contract.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read checkpoint prompt-format record: {contract}") from exc
        return validate_prompt_format_contract(value)
    legacy = directory / "sprint_training.json"
    if not legacy.is_file():
        return PromptFormat()
    try:
        state = json.loads(legacy.read_text(encoding="utf-8"))
        config = state.get("config", {}) if isinstance(state, dict) else {}
        objective = config.get("objective") if isinstance(config, dict) else None
    except (OSError, ValueError):
        return PromptFormat(note="The older training record cannot be read; its conversation format is unverified.")
    if objective == "causal_lm":
        contract = build_prompt_format_contract(config, tokenizer)
        resolved = validate_prompt_format_contract(contract)
        return PromptFormat(
            resolved.default_mode, resolved.supported_modes, "legacy_training_objective",
            resolved.chat_template, resolved.note,
        )
    if objective == "chat_sft" and tokenizer is not None:
        try:
            contract = build_prompt_format_contract(config, tokenizer)
        except ValueError:
            return PromptFormat(note="Chat training is recorded, but its saved conversation format is unavailable.")
        resolved = validate_prompt_format_contract(contract)
        return PromptFormat(
            resolved.default_mode, resolved.supported_modes, "legacy_training_objective",
            resolved.chat_template,
            "Chat training is recorded. Conversation mode uses this checkpoint's saved tokenizer format.",
        )
    return PromptFormat()


def checkpoint_prompt_format_contract(reference: str | Path, tokenizer: Any) -> dict[str, Any] | None:
    """Read a parent's established format for a new training run, without edits.

    Explicit bundle metadata wins. A legacy training objective plus its saved
    tokenizer establishes the same limited inference as ``load_prompt_format``;
    an unknown checkpoint or generic tokenizer alone establishes no contract.
    """

    path = Path(reference).expanduser() / PROMPT_FORMAT_FILENAME
    resolved = load_prompt_format(reference, tokenizer)
    if resolved.source == "unknown":
        return None
    if path.is_file():
        # Keep additional provenance fields intact when inheriting a bundle.
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_prompt_format_contract(value)
        return value
    template = resolved.chat_template
    value = {
        "schema": PROMPT_FORMAT_SCHEMA,
        "objective": "chat_sft" if template is not None else "causal_lm",
        "default_mode": resolved.default_mode,
        "supported_modes": list(resolved.supported_modes),
        "add_special_tokens": False,
        "chat_template": template,
        "chat_template_sha256": _template_hash(template) if template else None,
    }
    validate_prompt_format_contract(value)
    return value


def resolve_prompt_format(reference: str, *, adapter_path: str | None = None, tokenizer: Any = None) -> PromptFormat:
    # An adapter may add chat training to a base model. Its own explicit record
    # wins; without a record its format is unknown, not forcibly inherited.
    resolved = load_prompt_format(adapter_path or reference, tokenizer)
    if adapter_path and resolved.source == "unknown":
        return PromptFormat(note="This adapter has no saved input-format record; its conversation format is unverified.")
    return resolved
