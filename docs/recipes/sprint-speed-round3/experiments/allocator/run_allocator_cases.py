"""Serial allocator-wrapper cases; reserve a lock, never precreate its output.

Arguments are saved recipe names (without .json). --dry-run performs only
standard-library admission checks. Every child gets a fresh Python process.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_allocator import validate

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
WRAPPER = BASE / "benchmark_allocator.py"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(names):
    if len(names) != len(set(names)):
        raise ValueError("duplicate saved recipe names")
    jobs = []
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"invalid saved recipe name: {name}")
        recipe, output = BASE / f"{name}.json", BASE / name
        log, receipt, lock = (BASE / f"{name}{suffix}" for suffix in (".log", "-process.json", ".allocator.lock"))
        if not recipe.is_file() or any(path.exists() for path in (output, log, receipt, lock)):
            raise ValueError(f"need saved recipe and fresh output/log/receipt/lock: {name}")
        validate(json.loads(recipe.read_text(encoding="utf-8-sig")))
        jobs.append({"name": name, "recipe": recipe, "output": output, "log": log,
                     "receipt": receipt, "lock": lock, "recipe_sha256": sha256(recipe),
                     "command": [sys.executable, "-u", str(WRAPPER), "--config", str(recipe), "--output", str(output)]})
    return jobs


def run_one(job):
    with job["lock"].open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "name": job["name"]}, stream)
    started, child, interrupted = time.perf_counter(), None, False
    record = {"name": job["name"], "harness": "allocator", "command": job["command"], "exit_code": None,
              "recipe_sha256": job["recipe_sha256"], "wrapper_sha256": sha256(WRAPPER),
              "orchestrator_sha256": sha256(Path(__file__)),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    print(json.dumps({"case": job["name"], "stage": "starting"}), flush=True)
    try:
        if job["output"].exists() or job["receipt"].exists() or sha256(job["recipe"]) != job["recipe_sha256"]:
            raise RuntimeError("destination appeared or recipe changed after preflight")
        with job["log"].open("xb") as stream:
            child = subprocess.Popen(job["command"], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            record["pid"] = child.pid
            try:
                record["exit_code"] = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                record["exit_code"] = child.returncode
                raise
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            record.update(wall_seconds=time.perf_counter() - started,
                          completed_at_utc=datetime.now(timezone.utc).isoformat())
            if job["log"].is_file():
                record["log_sha256"] = sha256(job["log"])
            try:
                record["recipe_unchanged"] = sha256(job["recipe"]) == job["recipe_sha256"]
            except OSError:
                record["recipe_unchanged"] = False
            with job["receipt"].open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(record), flush=True)
        finally:
            job["lock"].unlink(missing_ok=True)
    if interrupted:
        raise KeyboardInterrupt
    return record["exit_code"] == 0 and record["recipe_unchanged"] and "error" not in record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        jobs = prepare(args.names)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps([{"name": job["name"], "command": job["command"]} for job in jobs], indent=2))
        return 0
    execution_state, previous, failed = None, None, []
    if sys.platform == "win32":
        execution_state = ctypes.windll.kernel32.SetThreadExecutionState
        execution_state.argtypes, execution_state.restype = [ctypes.c_uint], ctypes.c_uint
        previous = execution_state(0x80000001)
    try:
        for job in jobs:
            if not run_one(job):
                failed.append(job["name"])
    finally:
        if execution_state is not None:
            execution_state(previous or 0x80000000)
    print(json.dumps({"status": "failed" if failed else "ok", "failed_cases": failed}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
