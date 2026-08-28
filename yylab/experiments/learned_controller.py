from __future__ import annotations

import argparse
import random
import statistics

import numpy as np
import torch
from torch import nn

from ..infer.engine import ENGINE, LoadSpec
from .predictive_gain import DEFAULT_ST_MODEL, NEUTRAL, _st_model

TRAPS = [
    "Sensors are calibrated regularly and their readings are logged by the unit.",
    "The probe is checked each shift and its output is recorded for review.",
    "Readings from the unit are reviewed and the sensor status is noted carefully.",
]


class UsefulnessHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, zx: torch.Tensor, zc: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([zx, zc, zx * zc], dim=-1)
        return self.net(inp).squeeze(-1)


def _build_sample(tokenizer, rng: random.Random, n_fillers: int, hard: bool) -> tuple[str, list[str]]:
    num = round(rng.uniform(1.0, 9.999), 3)
    rounded = int(round(num))
    chan = rng.randint(1, 16)
    if hard:
        correct = f"The potential across node {chan} hit {num} V in the late trial."
        trap = "The reading came in around the expected volts."
        x = f"The reading came in around {rounded} volts."
    else:
        correct = f"The voltage reading on channel {chan} was {num} volts during the overnight test."
        trap = TRAPS[rng.randrange(len(TRAPS))]
        x = rng.choice(
            [
                f"That sensor came in at about {rounded} volts.",
                f"The probe settled near {rounded} volts.",
                f"It read roughly {rounded} volts.",
            ]
        )
    fillers = [NEUTRAL[rng.randrange(len(NEUTRAL))] for _ in range(n_fillers)]
    memory = [correct, trap] + fillers
    return x, memory


def run_learned_controller(
    checkpoint_ref: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quant: str = "nf4",
    st_model_name: str = DEFAULT_ST_MODEL,
    n_train: int = 40,
    n_eval: int = 20,
    n_fillers: int = 3,
    epochs: int = 400,
    seed: int = 0,
    hard: bool = False,
) -> dict:
    ENGINE.load(LoadSpec(ref=checkpoint_ref, quant=quant, settings={"cache_implementation": "dynamic"}))
    tok = ENGINE.tokenizer
    st = _st_model(st_model_name)
    rng = random.Random(seed)

    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    samples = [_build_sample(tok, rng, n_fillers, hard) for _ in range(n_train + n_eval)]

    zx_list, zc_list, y_list = [], [], []
    for i in range(n_train):
        x, memory = samples[i]
        base = ENGINE.nll_ids([], ids(x))
        zx = st.encode([x], normalize_embeddings=True)[0]
        for c in memory:
            delta = base - ENGINE.nll_ids(ids(c), ids(x))
            zc = st.encode([c], normalize_embeddings=True)[0]
            zx_list.append(zx)
            zc_list.append(zc)
            y_list.append(delta)

    dim = len(zx_list[0])
    ZX = torch.tensor(np.array(zx_list), dtype=torch.float32)
    ZC = torch.tensor(np.array(zc_list), dtype=torch.float32)
    Y = torch.tensor(np.array(y_list), dtype=torch.float32)

    head = UsefulnessHead(dim)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(head(ZX, ZC), Y)
        loss.backward()
        opt.step()
    head.eval()
    train_mse = float(loss.item())

    cos_delta, learn_delta, oracle_delta = [], [], []
    cos_picks_correct = cos_picks_trap = 0
    learn_picks_correct = learn_picks_trap = 0
    disagree = 0
    disagree_learn_better = 0

    for i in range(n_train, n_train + n_eval):
        x, memory = samples[i]
        xids = ids(x)
        base = ENGINE.nll_ids([], xids)
        zx = torch.tensor(st.encode([x], normalize_embeddings=True)[0], dtype=torch.float32)
        mem_emb = torch.tensor(st.encode(memory, normalize_embeddings=True), dtype=torch.float32)

        chunk_deltas = [base - ENGINE.nll_ids(ids(c), xids) for c in memory]

        cos_scores = mem_emb @ zx
        with torch.no_grad():
            learn_scores = head(zx.unsqueeze(0).repeat(len(memory), 1), mem_emb)
        cos_top = int(cos_scores.argmax())
        learn_top = int(learn_scores.argmax())

        cos_delta.append(chunk_deltas[cos_top])
        learn_delta.append(chunk_deltas[learn_top])
        oracle_delta.append(max(chunk_deltas))
        if cos_top == 0:
            cos_picks_correct += 1
        if cos_top == 1:
            cos_picks_trap += 1
        if learn_top == 0:
            learn_picks_correct += 1
        if learn_top == 1:
            learn_picks_trap += 1
        if cos_top != learn_top:
            disagree += 1
            if chunk_deltas[learn_top] > chunk_deltas[cos_top]:
                disagree_learn_better += 1

    ENGINE.unload()

    return {
        "hard": hard,
        "train_mse": train_mse,
        "n_train": n_train,
        "n_eval": n_eval,
        "mean_cos_delta": statistics.mean(cos_delta),
        "mean_learn_delta": statistics.mean(learn_delta),
        "mean_oracle_delta": statistics.mean(oracle_delta),
        "cos_picks_correct": cos_picks_correct,
        "cos_picks_trap": cos_picks_trap,
        "learn_picks_correct": learn_picks_correct,
        "learn_picks_trap": learn_picks_trap,
        "disagree": disagree,
        "disagree_learn_better": disagree_learn_better,
    }


def _print_report(r: dict) -> None:
    mode = "HARD (usefulness decoupled from similarity)" if r["hard"] else "standard"
    print(f"\nLearned controller (ΔNLL-distilled) vs cosine retrieval — {mode}")
    print(f"train samples={r['n_train']}  eval samples={r['n_eval']}  train MSE={r['train_mse']:.4f}")
    print("-" * 60)
    print(f"mean ΔNLL — cosine selection : {r['mean_cos_delta']:+.4f}")
    print(f"mean ΔNLL — learned selection: {r['mean_learn_delta']:+.4f}")
    print(f"mean ΔNLL — oracle (best)    : {r['mean_oracle_delta']:+.4f}")
    print("-" * 60)
    n = r["n_eval"]
    print(f"cosine picked the predictive fact : {r['cos_picks_correct']}/{n}   picked the trap: {r['cos_picks_trap']}/{n}")
    print(f"learned picked the predictive fact: {r['learn_picks_correct']}/{n}   picked the trap: {r['learn_picks_trap']}/{n}")
    print(f"they disagreed on {r['disagree']}/{n}; learned was better on {r['disagree_learn_better']} of those")
    print("-" * 60)
    if r["mean_learn_delta"] > r["mean_cos_delta"]:
        print("=> learned (ΔNLL-distilled) retrieval beat cosine on measured usefulness.")
    else:
        print("=> cosine matched/beat the learned controller here.")


def main() -> None:
    ap = argparse.ArgumentParser(description="ΔNLL-distilled learned controller experiment")
    ap.add_argument("--ref", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--st-model", default=DEFAULT_ST_MODEL)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--n-fillers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard", action="store_true")
    args = ap.parse_args()
    result = run_learned_controller(
        checkpoint_ref=args.ref,
        quant=args.quant,
        st_model_name=args.st_model,
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_fillers=args.n_fillers,
        epochs=args.epochs,
        seed=args.seed,
        hard=args.hard,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
