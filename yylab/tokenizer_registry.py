from __future__ import annotations

import hashlib
import json
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
