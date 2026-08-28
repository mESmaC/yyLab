from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import Optimizer


@dataclass
class LoopOpts:
    micro_batch: int = 1
    grad_accum: int = 16
    max_steps: int = 100
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int | None = None
    device: str = "cuda"
    run_dir: Path | None = None
    notes: list[str] = field(default_factory=list)


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = 0.0, cuda: bool = True) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cuda:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception:  # noqa: BLE001
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_loop(
    model: torch.nn.Module,
    batches: Iterator[dict],
    optimizer: Optimizer,
    scheduler,
    opts: LoopOpts,
    aux_fn: Callable[[], torch.Tensor] | None = None,
    save_fn: Callable[[int], None] | None = None,
) -> dict:
    metrics_path = opts.run_dir / "metrics.jsonl" if opts.run_dir else None
    progress_path = opts.run_dir / "progress.json" if opts.run_dir else None
    if opts.run_dir:
        opts.run_dir.mkdir(parents=True, exist_ok=True)

    def write_progress(stage: str, step: int, loss: float, msg: str = "") -> None:
        if progress_path is None:
            return
        progress_path.write_text(
            json.dumps({"stage": stage, "step": step, "max_steps": opts.max_steps, "loss": loss, "msg": msg}),
            "utf-8",
        )

    def log_metrics(rec: dict) -> None:
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    device = torch.device(opts.device)
    model.train()
    micro_step = 0
    opt_step = 0
    window_loss = 0.0
    window_aux = 0.0
    window_micro = 0
    last_loss = 0.0
    tokens = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    def flush_pending() -> None:
        nonlocal opt_step
        if opts.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], opts.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1

    for batch in batches:
        ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss / opts.grad_accum
        aux_val = 0.0
        if aux_fn is not None:
            aux = aux_fn() / opts.grad_accum
            aux_val = aux.item()
            loss = loss + aux
        loss.backward()
        window_loss += out.loss.item()
        window_aux += aux_val
        window_micro += 1
        last_loss = out.loss.item()
        tokens += ids.numel()
        micro_step += 1

        if micro_step % opts.grad_accum != 0:
            continue
        flush_pending()

        if opt_step % opts.log_every == 0 or opt_step == opts.max_steps:
            elapsed = time.time() - t0
            rec = {
                "step": opt_step,
                "loss": round(window_loss / max(window_micro, 1), 4),
                "aux": round(window_aux / max(window_micro, 1), 4),
                "tokens": tokens,
                "tok_per_s": round(tokens / max(elapsed, 1e-6), 1),
                "elapsed_s": round(elapsed, 1),
            }
            log_metrics(rec)
            write_progress("train", opt_step, rec["loss"], f'{rec["tok_per_s"]} tok/s')
            window_loss = 0.0
            window_aux = 0.0
            window_micro = 0
        if opts.save_every and save_fn and opt_step % opts.save_every == 0:
            save_fn(opt_step)
        if opt_step >= opts.max_steps:
            break

    if micro_step % opts.grad_accum != 0 and opt_step < opts.max_steps:
        flush_pending()

    write_progress("finalizing", opt_step, last_loss, "saving")
    return {
        "steps": opt_step,
        "tokens": tokens,
        "elapsed_s": round(time.time() - t0, 1),
    }
