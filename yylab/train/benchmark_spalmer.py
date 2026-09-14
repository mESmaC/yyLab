"""Run registered native training with isolated benchmark instrumentation.

The production worker, trainer, optimizer and packer remain the execution path.
Only this process installs the temporary trainer factory and diagnostic wrappers.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch

from .. import db
from ..artifacts import run_dir
from . import spalmer as bridge


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def summarize_performance(rows):
    normal = [r for r in rows if not r["warmup"] and not r["diagnostic"]]
    seconds = sum(r["training_seconds"] for r in normal)
    tokens = sum(r["target_tokens"] for r in normal)
    diagnostic = [r for r in rows if r["diagnostic"]]
    phases = Counter()
    for row in diagnostic:
        phases.update(row["phase_seconds"])
    return {
        "completed_steps": len(rows),
        "measured_steps": len(normal),
        "measured_target_tokens": tokens,
        "measured_training_seconds": seconds,
        "target_tokens_per_second": tokens / seconds if seconds else None,
        "median_step_seconds": float(np.median([r["training_seconds"] for r in normal])) if normal else None,
        "peak_cuda_allocated_bytes": max((r["peak_cuda_allocated_bytes"] for r in rows), default=0),
        "peak_cuda_reserved_bytes": max((r["peak_cuda_reserved_bytes"] for r in rows), default=0),
        "peak_process_working_set_bytes": max((r["peak_working_set_bytes"] for r in rows), default=0),
        "callback_seconds": sum(r["callback_seconds"] for r in rows),
        "diagnostic_steps": len(diagnostic),
        "diagnostic_phase_seconds": dict(phases),
        "diagnostic_total_seconds": sum(r["training_seconds"] for r in diagnostic),
        "measured_recurrence_counts": dict(Counter(r["recurrence_steps"] for r in normal)),
        "measurement": (
            "Synchronized complete optimizer steps; warmup and diagnostic steps excluded from "
            "throughput; validation/checkpoint/logging callbacks excluded and timed separately. "
            "Working-set peak includes preparation and validation. Process IO is OS-accounted, "
            "not physical NVMe bytes. Phase diagnostics synchronize each boundary and perturb speed."
        ),
    }


def compare_pairs(before, after):
    if len(before) != len(after):
        raise ValueError("paired benchmark lengths differ")
    for left, right in zip(before, after, strict=True):
        for key in ("step", "input_sha256", "target_tokens", "recurrence_steps", "learning_rate"):
            if left[key] != right[key]:
                raise ValueError(f"paired benchmark differs at step {left['step']}: {key}")
    cpu, gpu = summarize_performance(before), summarize_performance(after)
    return {
        "matched_steps": len(before),
        "cpu": cpu,
        "gpu": gpu,
        "gpu_speedup": gpu["target_tokens_per_second"] / cpu["target_tokens_per_second"],
        "max_abs_training_nll_difference": max(
            abs(a["model_loss"] - b["model_loss"]) for a, b in zip(before, after, strict=True)
        ),
        "scope": "One paired CPU-offload then GPU-resident trial; not an order-balanced speed study.",
    }


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def tensor_state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def freeze_validation(cfg, directory, api):
    from .recurrent_eval_data import prepare_view

    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    else:
        tokenizer = bridge.resolve_spalmer_tokenizer(cfg, api=api)
        tokens, manifest = prepare_view(
            tokenizer,
            cfg["benchmark"]["validation_sources"],
            documents=32,
            pool=2048,
        )
        directory.mkdir(parents=True)
        np.save(directory / "tokens.npy", tokens, allow_pickle=False)
        manifest["token_file_sha256"] = hashlib.sha256((directory / "tokens.npy").read_bytes()).hexdigest()
        write_json(directory / "manifest.json", manifest)
    if hashlib.sha256((directory / "tokens.npy").read_bytes()).hexdigest() != manifest["token_file_sha256"]:
        raise ValueError("benchmark validation token checksum mismatch")
    tokenizer = bridge.resolve_spalmer_tokenizer(cfg, api=api)
    if tokenizer.identity.fingerprint != manifest["tokenizer"]["fingerprint"]:
        raise ValueError("benchmark validation tokenizer mismatch")
    return torch.from_numpy(np.load(directory / "tokens.npy", allow_pickle=False).astype(np.int64)), manifest


def benchmark_trainer(api, cfg, rd, tokens, manifest, rows):
    from spalmer.training.depth_evaluation import measure_depths

    process = psutil.Process()
    limit = cfg["benchmark"]["stop_after_steps"]
    is_probe = cfg["benchmark"]["role"] == "cpu_probe"

    def validate(model, step, tokens_seen):
        started = time.perf_counter()
        records = measure_depths(model, tokens, depths=[2, 4, 8], seeds=[101], batch_size=16)
        losses = {}
        for depth in (2, 4, 8):
            selected = [r for r in records if r["depth"] == depth]
            by_source = {}
            for record in selected:
                for offset, nll in enumerate(record["nll_sums"]):
                    source = manifest["references"][record["row_start"] + offset]["source"]
                    by_source.setdefault(source, []).append(nll / record["targets_per_row"])
            losses[depth] = {source: float(np.mean(values)) for source, values in by_source.items()}
            losses[depth]["overall"] = float(np.mean([v for values in by_source.values() for v in values]))
        result = {
            "step": step,
            "training_tokens": tokens_seen,
            "latent_seeds": [101],
            "token_file_sha256": manifest["token_file_sha256"],
            "results": losses,
            "seconds": time.perf_counter() - started,
            "records": records,
        }
        write_json(rd / f"validation-{step:05d}.json", result)
        print("VALIDATION " + json.dumps({"step": step, "nll": losses}), flush=True)
        return losses

    class MeasuredTrainer(api.ExperimentTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            initial = tensor_state_hash(self.model)
            write_json(
                rd / "initial-state.json",
                {
                    "state_sha256": initial,
                    "parameters": sum(p.numel() for p in self.model.parameters()),
                    "model_config": self.model.config.to_dict(),
                },
            )
            expected_path = cfg["benchmark"].get("paired_initial_state")
            if expected_path:
                expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
                if initial != expected["state_sha256"]:
                    raise ValueError("CPU/GPU control initial states differ")
            if not is_probe:
                validate(self.model, 0, 0)

        def run(self):
            self.progress.started_at = time.time()
            self.model.train()
            while self.progress.completed_steps < min(limit, self.config.max_steps):
                yield self._optimizer_step()

        def _optimizer_step(self):
            step = self.progress.completed_steps + 1
            diagnostic = 21 <= step <= 40
            self._phases = Counter()
            self._digest = hashlib.sha256()
            self._callback_seconds = 0.0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._io_before = process.io_counters()
            self._step_start = time.perf_counter()

            def timed(name, function, *, hash_batch=False):
                def call(*args, **kwargs):
                    if diagnostic:
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = function(*args, **kwargs)
                    if diagnostic:
                        torch.cuda.synchronize()
                        self._phases[name] += time.perf_counter() - started
                    if hash_batch:
                        self._digest.update(result.input_ids.numpy().tobytes())
                    return result

                return call

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        self.batches,
                        "next_batch",
                        timed("input_wait", self.batches.next_batch, hash_batch=True),
                    )
                )
                if diagnostic:
                    stack.enter_context(
                        patch.object(self.model, "forward", timed("forward", self.model.forward))
                    )
                    stack.enter_context(
                        patch.object(self.optimizer, "step", timed("optimizer", self.optimizer.step))
                    )
                    stack.enter_context(
                        patch.object(torch.Tensor, "backward", timed("backward", torch.Tensor.backward))
                    )
                    stack.enter_context(
                        patch.object(api.CausalBatch, "to", timed("transfer", api.CausalBatch.to))
                    )
                metrics = super()._optimizer_step()
            row = {
                **self._end_snapshot,
                "step": step,
                "warmup": step <= 20,
                "diagnostic": diagnostic,
                "target_tokens": self.config.tokens_per_optimizer_step,
                "input_sha256": self._digest.hexdigest(),
                "model_loss": metrics.model_loss,
                "objective": metrics.objective,
                "recurrence_steps": metrics.recurrence_steps,
                "learning_rate": metrics.learning_rate,
                "phase_seconds": dict(self._phases),
                "callback_seconds": self._callback_seconds,
            }
            rows.append(row)
            append_json(rd / "performance.jsonl", row)
            if step == 1 or step % 10 == 0:
                summary = summarize_performance(rows)
                write_json(rd / "benchmark-progress.json", summary)
                rate = summary["target_tokens_per_second"]
                rate_text = f"{rate:,.0f} target tok/s" if rate else "warming/profiling"
                print(
                    f"BENCHMARK step={step}/{limit} {rate_text}; "
                    f"step={row['training_seconds']:.3f}s; "
                    f"peak VRAM={row['peak_cuda_allocated_bytes'] / 2**30:.2f} GiB; "
                    f"RAM={row['rss_bytes'] / 2**30:.2f} GiB",
                    flush=True,
                )
            return metrics

        def _dispatch_callbacks(self, metrics):
            torch.cuda.synchronize()
            ended = time.perf_counter()
            memory = process.memory_info()
            io = process.io_counters()
            self._end_snapshot = {
                "training_seconds": ended - self._step_start,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "rss_bytes": memory.rss,
                "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
                "process_read_bytes": io.read_bytes - self._io_before.read_bytes,
                "process_write_bytes": io.write_bytes - self._io_before.write_bytes,
            }
            started = time.perf_counter()
            super()._dispatch_callbacks(metrics)
            if not is_probe and metrics.step % 250 == 0:
                validate(self.model, metrics.step, metrics.tokens_seen)
            self._callback_seconds = time.perf_counter() - started

    return MeasuredTrainer


def run_case(run_id, validation_directory):
    row = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if row is None or row["status"] != "queued":
        raise ValueError(f"run {run_id} must exist and be queued")
    cfg = json.loads(row["config_json"])
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "performance.jsonl").exists():
        raise ValueError("refusing to append to a previous benchmark")
    write_json(rd / "config.json", cfg)
    db.execute(
        "UPDATE runs SET status='running',started_at=?,log_path=? WHERE id=?",
        (
            db.now_iso(),
            str(rd / "train.log"),
            run_id,
        ),
    )
    rows = []
    try:
        with (rd / "train.log").open("a", encoding="utf-8", buffering=1) as logfile:
            with (
                contextlib.redirect_stdout(Tee(sys.stdout, logfile)),
                contextlib.redirect_stderr(Tee(sys.stderr, logfile)),
            ):
                print(f"START {cfg['checkpoint_name']}: {cfg['benchmark']['role']}", flush=True)
                api = bridge._load_spalmer_api()
                tokens, manifest = freeze_validation(cfg, validation_directory, api)
                started = time.perf_counter()
                api.ExperimentTrainer = benchmark_trainer(api, cfg, rd, tokens, manifest, rows)
                with patch.object(bridge, "_load_spalmer_api", return_value=api):
                    summary = bridge.run_spalmer_base(cfg, rd)
                summary["benchmark"] = summarize_performance(rows)
                summary["benchmark"]["run_wall_seconds"] = time.perf_counter() - started
                write_json(rd / "benchmark-summary.json", summary)
                db.execute(
                    "UPDATE runs SET status='done',finished_at=?,metrics_json=? WHERE id=?",
                    (
                        db.now_iso(),
                        json.dumps(summary),
                        run_id,
                    ),
                )
                print("COMPLETE " + json.dumps(summary["benchmark"]), flush=True)
    except BaseException as exc:
        error = traceback.format_exc()
        (rd / "failure.txt").write_text(error, encoding="utf-8")
        db.execute(
            "UPDATE runs SET status='failed',finished_at=?,error=? WHERE id=?",
            (
                db.now_iso(),
                str(exc),
                run_id,
            ),
        )
        raise
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cpu-probe", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    cpu = run_case(args.cpu_probe, args.validation)
    gc.collect()
    torch.cuda.empty_cache()
    gpu = run_case(args.run, args.validation)
    comparison = compare_pairs(cpu, gpu[: len(cpu)])
    write_json(run_dir(args.run) / "optimizer-placement-comparison.json", comparison)
    print("PAIRED OPTIMIZER COMPARISON " + json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
