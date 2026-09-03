from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db
from .artifacts import tokenizer_dir

TOKENIZER_RUN_KIND = "tokenizer"
TOKENIZER_ACTION_TRAIN = "train"
TOKENIZER_ACTION_IMPORT = "import"
TOKENIZER_ACTIONS = frozenset({TOKENIZER_ACTION_TRAIN, TOKENIZER_ACTION_IMPORT})


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: object, fallback: Any) -> Any:
    try:
        return json.loads(str(value or ""))
    except (TypeError, ValueError):
        return fallback


def create_tokenizer_record(
    *,
    name: str,
    kind: str,
    corpora: list[str] | None = None,
    splits: list[str] | None = None,
    config: dict[str, Any] | None = None,
    tokenizer_id: str | None = None,
    status: str = "queued",
) -> str:
    """Create the durable registry row before an isolated worker starts."""

    resolved_name = str(name or "").strip()
    if not resolved_name:
        raise ValueError("tokenizer name is required")
    tokenizer_id = tokenizer_id or db.new_id("tok")
    db.execute(
        "INSERT INTO tokenizers(id,name,created_at,status,kind,vocab_size,path,corpora_json,"
        "splits_json,config_json,stats_json,error) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            tokenizer_id,
            resolved_name,
            db.now_iso(),
            status,
            str(kind or "byte_bpe"),
            0,
            str(tokenizer_dir(tokenizer_id)),
            _json(list(corpora or ())),
            _json(list(splits or ())),
            _json(dict(config or {})),
            "{}",
            "",
        ),
    )
    return tokenizer_id


def get_tokenizer(tokenizer_id: str, *, parsed: bool = False) -> dict[str, Any] | None:
    row = db.q1("SELECT * FROM tokenizers WHERE id=?", (str(tokenizer_id),))
    return parse_tokenizer_row(row) if row and parsed else row


def list_tokenizers(*, ready_only: bool = False, parsed: bool = False) -> list[dict[str, Any]]:
    clause = " WHERE status='ready'" if ready_only else ""
    rows = db.q(f"SELECT * FROM tokenizers{clause} ORDER BY created_at DESC")
    return [parse_tokenizer_row(row) for row in rows] if parsed else rows


def parse_tokenizer_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(row)
    parsed["corpora"] = _loads(row.get("corpora_json"), [])
    parsed["splits"] = _loads(row.get("splits_json"), [])
    parsed["config"] = _loads(row.get("config_json"), {})
    parsed["stats"] = _loads(row.get("stats_json"), {})
    return parsed


def tokenizer_options() -> dict[str, str]:
    """Return ready registry artifacts in the mapping format NiceGUI selects use."""

    return {
        row["id"]: f"{row['name']} · {int(row.get('vocab_size') or 0):,} tokens"
        for row in list_tokenizers(ready_only=True)
    }


# -- Hugging Face hub sources ---------------------------------------------------

HUB_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class HubTokenizerPreset:
    """A well-known large hub tokenizer that yyLab can import or reference directly.

    ``vocab_size`` is the size SPALMER and the registry will report for the
    current hub revision (``len(tokenizer)``, which equals the highest id plus
    one for these tokenizers).  ``None`` means it is only known after import.
    """

    repo_id: str
    label: str
    vocab_size: int | None
    family: str
    gated: bool = False
    note: str = ""

    @property
    def option_label(self) -> str:
        size = f"{self.vocab_size:,} tokens" if self.vocab_size else "vocabulary verified on import"
        gated = " · needs HF token" if self.gated else ""
        return f"{self.label} · {size}{gated}"


HF_TOKENIZER_PRESETS: tuple[HubTokenizerPreset, ...] = (
    HubTokenizerPreset(
        "openai/gpt-oss-20b",
        "OpenAI o200k (gpt-oss, o200k_harmony)",
        200_019,
        "o200k",
        note=(
            "The o200k byte-level BPE behind GPT-4o and gpt-oss with Harmony control tokens; "
            "strong English, code, and multilingual compression. Apache-2.0, ungated."
        ),
    ),
    HubTokenizerPreset(
        "Qwen/Qwen3-0.6B",
        "Qwen 3",
        151_669,
        "qwen",
        note="151K byte-level BPE with broad multilingual and code coverage. Apache-2.0, ungated.",
    ),
    HubTokenizerPreset(
        "Qwen/Qwen2.5-0.5B",
        "Qwen 2.5",
        151_665,
        "qwen",
        note="Same base vocabulary as Qwen 3 with fewer control tokens. Apache-2.0, ungated.",
    ),
    HubTokenizerPreset(
        "NousResearch/Llama-3.2-1B",
        "Llama 3.2 (ungated mirror)",
        128_256,
        "llama3",
        note="Llama 3 128K tiktoken-derived BPE from an ungated mirror of the tokenizer files.",
    ),
    HubTokenizerPreset(
        "google/gemma-3-1b-pt",
        "Gemma 3",
        None,
        "gemma",
        gated=True,
        note=(
            "262K SentencePiece-derived vocabulary, the largest common open tokenizer. "
            "Accept Google's terms on the hub and save an HF token under Settings first."
        ),
    ),
)


def hub_preset(repo_id: str) -> HubTokenizerPreset | None:
    for preset in HF_TOKENIZER_PRESETS:
        if preset.repo_id == repo_id:
            return preset
    return None


def hub_preset_options() -> dict[str, str]:
    return {preset.repo_id: preset.option_label for preset in HF_TOKENIZER_PRESETS}


def is_hub_reference(value: object) -> bool:
    raw = str(value or "").strip()
    if raw.startswith("hf:"):
        raw = raw[3:].strip()
    if not raw or "\\" in raw or raw.startswith(("./", "../", "/")):
        return False
    return bool(HUB_REPO_RE.match(raw))


def parse_tokenizer_source(value: object) -> tuple[str, str]:
    """Classify an import source as ``("path", resolved)`` or ``("hub", repo_id)``.

    Anything that exists locally is a path.  Otherwise an ``owner/name`` id
    (optionally prefixed with ``hf:``) is a Hugging Face hub reference.
    """

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("tokenizer source is required")
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return "path", str(candidate.resolve())
    if is_hub_reference(raw):
        return "hub", raw[3:].strip() if raw.startswith("hf:") else raw
    raise FileNotFoundError(
        f"tokenizer source does not exist locally and is not a Hugging Face repo id: {raw}"
    )


def resolve_tokenizer_artifact(value: str | Path, *, require_ready: bool = True) -> Path:
    """Resolve a registry ID or a direct local tokenizer path.

    Registry IDs are the stable value stored in training recipes. Direct paths
    remain accepted for programmatic/legacy recipes.
    """

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("tokenizer artifact is required")
    row = get_tokenizer(raw)
    if row:
        if require_ready and row["status"] != "ready":
            raise ValueError(f"tokenizer {raw} is {row['status']}, not ready")
        path = Path(str(row.get("path") or tokenizer_dir(raw))).expanduser().resolve()
    else:
        path = Path(raw).expanduser().resolve()
    if not path.is_dir() or not (path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"tokenizer artifact is incomplete: {path}")
    return path


def artifact_checksum(path: str | Path) -> str:
    """Hash HF tokenizer payload bytes with stable relative-path framing.

    ``yylab_tokenizer.json`` is intentionally excluded because it stores this
    digest. The checksum is therefore reproducible both before and after the
    yyLab metadata manifest is written.
    """

    root = Path(path).resolve()
    digest = hashlib.sha256()
    for file_path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name != "yylab_tokenizer.json"
    ):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def mark_tokenizer_ready(
    tokenizer_id: str,
    *,
    path: str | Path,
    kind: str,
    vocab_size: int,
    config: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    db.execute(
        "UPDATE tokenizers SET status='ready',kind=?,vocab_size=?,path=?,config_json=?,"
        "stats_json=?,error='' WHERE id=?",
        (
            kind,
            int(vocab_size),
            str(Path(path).resolve()),
            _json(config),
            _json(stats),
            tokenizer_id,
        ),
    )


def mark_tokenizer_failed(tokenizer_id: str, error: str) -> None:
    db.execute(
        "UPDATE tokenizers SET status='failed',error=? WHERE id=?",
        (str(error)[-4000:], tokenizer_id),
    )
