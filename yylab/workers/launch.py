from __future__ import annotations

import subprocess
import sys

from ..artifacts import run_dir
from ..config import REPO_ROOT


def launch_worker(run_id: str) -> int:
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    fh = (rd / "train.log").open("ab")
    kwargs: dict = {
        "cwd": str(REPO_ROOT),
        "stdout": fh,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen([sys.executable, "-m", "yylab.workers.train", "--run", run_id], **kwargs)
    return proc.pid
