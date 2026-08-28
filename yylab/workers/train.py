from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .. import db
from ..artifacts import adapter_dir, checkpoint_dir, router_dir, run_dir, write_json
from ..mole.inject import MoLEConfig, inject_mole, set_stage
from ..mole.losses import aux_loss
from ..mole.store import apply_mole_weights, load_mole_weights, save_mole
from ..train.data import PackedBatches, estimate_steps, iter_corpus_docs
from ..train.loop import LoopOpts, make_optimizer, train_loop


def log(msg: str) -> None:
    print(msg, flush=True)


def load_base(ref: str, quant: str):
    from ..models.loading import load_base as _load_base

    model, tokenizer, cuda, quant, notes = _load_base(ref, quant)
    model.config.use_cache = False
    return model, tokenizer, cuda, quant, notes


def build_batches(cfg: dict, tokenizer, max_steps: int) -> PackedBatches:
    docs = iter_corpus_docs(cfg["corpora"], cfg.get("domains") or None)

    def tokenize(texts: list[str]) -> list[list[int]]:
        enc = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        return enc["input_ids"]

    return PackedBatches(
        docs,
        tokenize,
        seq_len=cfg["seq_len"],
        eos_token_id=tokenizer.eos_token_id,
        micro_batch=cfg["micro_batch"],
        max_batches=max_steps * cfg["grad_accum"],
    )


def loop_opts(cfg: dict, rd: Path, cuda: bool) -> LoopOpts:
    return LoopOpts(
        micro_batch=cfg["micro_batch"],
        grad_accum=cfg["grad_accum"],
        max_steps=cfg["max_steps"],
        log_every=cfg.get("log_every", 5),
        save_every=cfg.get("save_every"),
        device="cuda" if cuda else "cpu",
        run_dir=rd,
    )


def enforce_quant_lock(checkpoint_id: str, quant: str) -> None:
    row = db.q1("SELECT quant_locked FROM checkpoints WHERE id=?", (checkpoint_id,))
    if not row:
        return
    locked = row["quant_locked"]
    if locked and locked != quant:
        raise RuntimeError(
            f"Checkpoint quantization is locked to '{locked}'; adapter requested '{quant}'. "
            "Quant cannot drift between adapters on the same checkpoint."
        )
    if not locked:
        db.execute("UPDATE checkpoints SET quant_locked=? WHERE id=?", (quant, checkpoint_id))


def enable_grad_ckpt(model) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def run_mole(cfg: dict, rd: Path) -> dict:
    import torch
    from transformers.optimization import get_cosine_schedule_with_warmup

    enforce_quant_lock(cfg["checkpoint_id"], cfg["quant"])
    model, tokenizer, cuda, quant, notes = load_base(cfg["checkpoint_ref"], cfg["quant"])
    for n in notes:
        log(f"note: {n}")

    mole_cfg = MoLEConfig(
        num_experts=cfg["num_experts"],
        rank=cfg["rank"],
        alpha=cfg.get("alpha") or cfg["rank"] * 2.0,
        top_k=cfg["top_k"],
        target_modules=tuple(cfg["target_modules"]),
    )
    inject_mole(model, mole_cfg, expert_dtype=torch.bfloat16 if cuda else torch.float32)
    set_stage(model, "both")
    if cfg.get("grad_ckpt", True):
        enable_grad_ckpt(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"injected MoLE: E={mole_cfg.num_experts} r={mole_cfg.rank} k={mole_cfg.top_k}, trainable={trainable:,}")

    optimizer = make_optimizer(model, cfg["lr"], cuda=cuda)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, cfg["max_steps"] // 20), num_training_steps=cfg["max_steps"]
    )
    batches = build_batches(cfg, tokenizer, cfg["max_steps"])
    a_dir = adapter_dir(cfg["adapter_id"])

    def save(_step: int) -> None:
        save_mole(a_dir, model, mole_cfg, base_ref=cfg["checkpoint_ref"])

    result = train_loop(
        model,
        batches,
        optimizer,
        scheduler,
        loop_opts(cfg, rd, cuda),
        aux_fn=lambda: aux_loss(model, mole_cfg.top_k, cfg.get("aux_weight", 0.01)),
        save_fn=save,
    )
    save(0)
    db.execute(
        "UPDATE adapters SET status='trained', path=? WHERE id=?",
        (str(a_dir), cfg["adapter_id"]),
    )
    return {"result": result, "docs_seen": batches.docs_seen, "tokens": batches.tokens_seen}


def run_router(cfg: dict, rd: Path) -> dict:
    import torch
    from transformers.optimization import get_cosine_schedule_with_warmup

    a_row = db.q1("SELECT * FROM adapters WHERE id=?", (cfg["adapter_id"],))
    if not a_row:
        raise RuntimeError("adapter not found")
    quant = a_row["quant"]
    enforce_quant_lock(cfg["checkpoint_id"], quant)
    model, tokenizer, cuda, quant, notes = load_base(cfg["checkpoint_ref"], quant)
    for n in notes:
        log(f"note: {n}")

    cfg_path = Path(a_row["path"]) / "mole_config.json"
    saved = json.loads(cfg_path.read_text("utf-8"))
    mole_cfg = MoLEConfig.from_dict(saved["config"])
    if cfg.get("top_k"):
        mole_cfg.top_k = min(cfg["top_k"], mole_cfg.num_experts)
    inject_mole(model, mole_cfg, expert_dtype=torch.bfloat16 if cuda else torch.float32)
    n = apply_mole_weights(model, load_mole_weights(a_row["path"]))
    log(f"loaded adapter weights into {n} expert matrices")
    set_stage(model, "router")
    if cfg.get("grad_ckpt", True):
        enable_grad_ckpt(model)

    optimizer = make_optimizer(model, cfg["lr"], cuda=cuda)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, cfg["max_steps"] // 20), num_training_steps=cfg["max_steps"]
    )
    batches = build_batches(cfg, tokenizer, cfg["max_steps"])
    r_dir = router_dir(cfg["router_id"])

    def save(_step: int) -> None:
        save_mole(r_dir, model, mole_cfg, base_ref=cfg["checkpoint_ref"], extra_meta={"stage": "router"})

    result = train_loop(
        model,
        batches,
        optimizer,
        scheduler,
        loop_opts(cfg, rd, cuda),
        aux_fn=lambda: aux_loss(model, mole_cfg.top_k, cfg.get("aux_weight", 0.01)),
        save_fn=save,
    )
    save(0)
    db.execute("UPDATE routers SET status='trained', path=? WHERE id=?", (str(r_dir), cfg["router_id"]))
    return {"result": result, "docs_seen": batches.docs_seen, "tokens": batches.tokens_seen}


def run_base(cfg: dict, rd: Path) -> dict:
    from transformers.optimization import get_cosine_schedule_with_warmup

    method = cfg["method"]
    quant = cfg["quant"] if method == "qlora" else "bf16"
    model, tokenizer, cuda, quant, notes = load_base(cfg["checkpoint_ref"], quant)
    for n in notes:
        log(f"note: {n}")

    if method in {"lora", "qlora"}:
        from peft import LoraConfig, get_peft_model

        peft_cfg = LoraConfig(
            r=cfg.get("rank", 32),
            lora_alpha=cfg.get("alpha", 64),
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
    if cfg.get("grad_ckpt", True):
        enable_grad_ckpt(model)

    optimizer = make_optimizer(model, cfg["lr"], cuda=cuda)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, cfg["max_steps"] // 20), num_training_steps=cfg["max_steps"]
    )
    batches = build_batches(cfg, tokenizer, cfg["max_steps"])

    def save(_step: int) -> None:
        if method == "full":
            out = checkpoint_dir(cfg["out_checkpoint_id"])
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
        else:
            out = adapter_dir(cfg["adapter_id"])
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)

    result = train_loop(
        model, batches, optimizer, scheduler, loop_opts(cfg, rd, cuda), save_fn=save
    )
    save(0)

    if method == "full":
        out = checkpoint_dir(cfg["out_checkpoint_id"])
        params = sum(p.numel() for p in model.parameters()) / 1e9
        db.execute(
            "UPDATE checkpoints SET kind='base', source='local', ref=?, arch=?, params_b=?, path=?,"
            " hidden_size=?, num_layers=?, meta_json=? WHERE id=?",
            (
                str(out),
                type(model).__name__,
                round(params, 3),
                str(out),
                int(getattr(model.config, "hidden_size", 0) or 0),
                int(getattr(model.config, "num_hidden_layers", 0) or 0),
                json.dumps({"derived_from": cfg["checkpoint_ref"]}),
                cfg["out_checkpoint_id"],
            ),
        )
    else:
        db.execute(
            "UPDATE adapters SET status='trained', path=? WHERE id=?",
            (str(adapter_dir(cfg["adapter_id"])), cfg["adapter_id"]),
        )
    return {"result": result, "docs_seen": batches.docs_seen, "tokens": batches.tokens_seen}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_id = args.run

    run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        raise SystemExit(f"run {run_id} not found")
    cfg = json.loads(run["config_json"] or "{}")
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "config.json", cfg)

    db.execute(
        "UPDATE runs SET status='running', started_at=?, log_path=? WHERE id=?",
        (db.now_iso(), str(rd / "train.log"), run_id),
    )
    log(f"yylab worker: run={run_id} kind={run['kind']}")

    if not cfg.get("max_steps"):
        cfg["max_steps"] = estimate_steps(
            cfg["corpora"], cfg.get("domains"), cfg.get("epochs", 1),
            cfg["seq_len"], cfg["micro_batch"], cfg["grad_accum"],
        )
        log(f"estimated {cfg['max_steps']} optimizer steps")

    try:
        if run["kind"] == "mole_stage1":
            summary = run_mole(cfg, rd)
        elif run["kind"] == "router_stage2":
            summary = run_router(cfg, rd)
        elif run["kind"] == "base":
            summary = run_base(cfg, rd)
        else:
            raise RuntimeError(f"unknown run kind: {run['kind']}")
        db.execute(
            "UPDATE runs SET status='done', finished_at=?, metrics_json=? WHERE id=?",
            (db.now_iso(), json.dumps(summary), run_id),
        )
        log("done")
    except Exception as e:  # noqa: BLE001
        err = f"{e}\n{traceback.format_exc()}"
        log(err)
        db.execute(
            "UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=?",
            (db.now_iso(), err[-4000:], run_id),
        )
        if cfg.get("adapter_id"):
            db.execute("UPDATE adapters SET status='failed' WHERE id=?", (cfg["adapter_id"],))
        if cfg.get("router_id"):
            db.execute("UPDATE routers SET status='failed' WHERE id=?", (cfg["router_id"],))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
