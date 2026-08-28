from __future__ import annotations

import json
import os
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def home_dir() -> Path:
    return Path(os.environ.get("YYLAB_HOME") or (REPO_ROOT / "artifacts"))


class Paths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.corpora = root / "corpora"
        self.checkpoints = root / "checkpoints"
        self.adapters = root / "adapters"
        self.routers = root / "routers"
        self.runs = root / "runs"
        self.cache = root / "cache"
        self.db_file = root / "yylab.sqlite3"
        self.settings_file = root / "settings.json"

    def ensure(self) -> None:
        for p in (
            self.root,
            self.corpora,
            self.checkpoints,
            self.adapters,
            self.routers,
            self.runs,
            self.cache,
        ):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths(home_dir())
PATHS.ensure()
os.environ.setdefault("HF_HOME", str(PATHS.cache / "hf"))

DEFAULT_SETTINGS: dict = {
    "hf_token": "",
    "default_quant": "nf4",
    "tokenizers_enabled": {},
    "engine": {
        "attn_implementation": "sdpa",
        "cache_implementation": "dynamic",
        "kv_quant": "none",
        "sliding_window": 4096,
        "max_context_tokens": 8192,
        "prefill_chunk_size": 8192,
        "rope_scaling": "none",
        "rope_factor": 2.0,
    },
}

_settings_lock = threading.Lock()


def get_settings() -> dict:
    out = json.loads(json.dumps(DEFAULT_SETTINGS))
    if PATHS.settings_file.exists():
        try:
            stored = json.loads(PATHS.settings_file.read_text("utf-8"))
        except Exception:
            stored = {}
        for k, v in stored.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            else:
                out[k] = v
    return out


def update_settings(**kv) -> dict:
    with _settings_lock:
        current = {}
        if PATHS.settings_file.exists():
            try:
                current = json.loads(PATHS.settings_file.read_text("utf-8"))
            except Exception:
                current = {}
        for k, v in kv.items():
            if isinstance(v, dict) and isinstance(current.get(k), dict):
                current[k].update(v)
            else:
                current[k] = v
        PATHS.settings_file.write_text(json.dumps(current, indent=2), "utf-8")
    return get_settings()


def reload_home() -> None:
    new = Paths(home_dir())
    new.ensure()
    PATHS.__dict__.update(new.__dict__)
    os.environ["HF_HOME"] = str(PATHS.cache / "hf")
