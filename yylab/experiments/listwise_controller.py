from __future__ import annotations

import argparse
import random
import statistics

import numpy as np
import torch
import torch.nn.functional as F

from ..infer.engine import ENGINE, LoadSpec
from .learned_controller import UsefulnessHead
from .predictive_gain import DEFAULT_ST_MODEL, NEUTRAL, _st_model

CAND_TYPES = ["fact", "echo", "wrong_value", "wrong_channel", "unrelated"]

X_STYLES = [
    "What voltage did channel {chan} record overnight?",
    "Report the voltage that channel {chan} captured during the night shift.",
    "State the overnight voltage measured on channel {chan}.",
    "Give the voltage reading from channel {chan} for the overnight run.",
]
Y_STYLES = [
    "Channel {chan} recorded {num} volts overnight.",
    "The overnight voltage on channel {chan} was {num} volts.",
    "Channel {chan} measured {num} volts during the night.",
    "On channel {chan} the overnight reading was {num} volts.",
]
ECHO_STYLES = [
    "You asked what voltage channel {chan} recorded overnight.",
    "The question concerns the overnight voltage of channel {chan}.",
    "Someone requested the overnight voltage for channel {chan}.",
    "The overnight voltage of channel {chan} was the topic of the query.",
]

TRAIN_STYLES = [0, 1]
EVAL_STYLES = [2, 3]


def _build_sample(rng: random.Random, style: int) -> tuple[str, str, dict]:
    chan = rng.randint(1, 16)
    chan2 = chan
    while chan2 == chan:
        chan2 = rng.randint(1, 16)
    num = round(rng.uniform(1.0, 9.999), 3)
    num2 = num
    while abs(num2 - num) < 0.5:
        num2 = round(rng.uniform(1.0, 9.999), 3)
    x = X_STYLES[style].format(chan=chan)
    y = Y_STYLES[style].format(chan=chan, num=num)
    candidates = {
        "fact": Y_STYLES[style].format(chan=chan, num=num),
        "echo": ECHO_STYLES[style].format(chan=chan),
        "wrong_value": Y_STYLES[style].format(chan=chan, num=num2),
        "wrong_channel": Y_STYLES[style].format(chan=chan2, num=num),
        "unrelated": NEUTRAL[rng.randrange(len(NEUTRAL))],
    }
    return x, y, candidates


def _measure_rewards(tok, cand_texts: dict, x: str, y: str) -> tuple[list[float], float]:
    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    xids, yids = ids(x), ids(y)
    nll_y_x = ENGINE.nll_ids(xids, yids)
    rys = []
    for t in CAND_TYPES:
        rys.append(nll_y_x - ENGINE.nll_ids(ids(cand_texts[t]) + xids, yids))
    return rys, nll_y_x


def run_listwise(
    checkpoint_ref: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quant: str = "nf4",
    st_model_name: str = DEFAULT_ST_MODEL,
    n_train: int = 40,
    n_eval: int = 20,
    epochs: int = 300,
    tau_q: float = 0.5,
    tau_p: float = 1.0,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    ENGINE.load(LoadSpec(ref=checkpoint_ref, quant=quant, settings={"cache_implementation": "dynamic"}))
    tok = ENGINE.tokenizer
    st = _st_model(st_model_name)
    rng = random.Random(seed)

    train_rows = []
    for _ in range(n_train):
        style = TRAIN_STYLES[rng.randrange(len(TRAIN_STYLES))]
        x, y, cands = _build_sample(rng, style)
        rys, _ = _measure_rewards(tok, cands, x, y)
        zx = st.encode([x], normalize_embeddings=True)[0]
        zcs = np.array([st.encode([cands[t]], normalize_embeddings=True)[0] for t in CAND_TYPES])
        train_rows.append((zx, zcs, np.array(rys)))

    dim = train_rows[0][0].shape[0]
    head = UsefulnessHead(dim)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    order = list(range(len(train_rows)))
    final_loss = float("nan")
    for _ in range(epochs):
        rng.shuffle(order)
        epoch_loss = 0.0
        for idx in order:
            zx, zcs, rys = train_rows[idx]
            zx_t = torch.tensor(zx, dtype=torch.float32)
            zcs_t = torch.tensor(zcs, dtype=torch.float32)
            rys_t = torch.tensor(rys, dtype=torch.float32)
            scores = head(zx_t.unsqueeze(0).repeat(len(CAND_TYPES), 1), zcs_t)
            log_p = F.log_softmax(scores / tau_p, dim=-1)
            q = F.softmax(rys_t / tau_q, dim=-1)
            loss = F.kl_div(log_p, q, reduction="sum")
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        final_loss = epoch_loss / len(train_rows)
    head.eval()

    cap_learn, cap_cos, cap_rand = [], [], []
    hit_learn = hit_cos = 0
    pick_counts = {t: 0 for t in CAND_TYPES}

    for _ in range(n_eval):
        style = EVAL_STYLES[rng.randrange(len(EVAL_STYLES))]
        x, y, cands = _build_sample(rng, style)
        rys, _ = _measure_rewards(tok, cands, x, y)
        zx = st.encode([x], normalize_embeddings=True)[0]
        zcs = np.array([st.encode([cands[t]], normalize_embeddings=True)[0] for t in CAND_TYPES])

        zx_t = torch.tensor(zx, dtype=torch.float32)
        zcs_t = torch.tensor(zcs, dtype=torch.float32)
        with torch.no_grad():
            scores = head(zx_t.unsqueeze(0).repeat(len(CAND_TYPES), 1), zcs_t).numpy()
        cos_scores = zcs @ zx

        oracle = max(rys)
        if oracle <= 1e-9:
            continue
        learn_top = int(np.argmax(scores))
        cos_top = int(np.argmax(cos_scores))
        rand_top = rng.randrange(len(CAND_TYPES))
        cap_learn.append(rys[learn_top] / oracle)
        cap_cos.append(rys[cos_top] / oracle)
        cap_rand.append(rys[rand_top] / oracle)
        if CAND_TYPES[learn_top] == "fact":
            hit_learn += 1
        if CAND_TYPES[cos_top] == "fact":
            hit_cos += 1
        pick_counts[CAND_TYPES[learn_top]] += 1

    ENGINE.unload()

    n_used = len(cap_learn)
    return {
        "n_train": n_train,
        "n_eval": n_used,
        "final_kl": final_loss,
        "capture_learn": statistics.mean(cap_learn),
        "capture_cos": statistics.mean(cap_cos),
        "capture_rand": statistics.mean(cap_rand),
        "fact_hit_learn": hit_learn,
        "fact_hit_cos": hit_cos,
        "pick_counts": pick_counts,
    }


def _print_report(r: dict) -> None:
    n = r["n_eval"]
    print("\nListwise R_y controller vs cosine vs random")
    print(f"train={r['n_train']}  eval={n}  final KL={r['final_kl']:.4f}")
    print("-" * 52)
    print(f"mean R_y oracle-capture — learned : {r['capture_learn']:.3f}")
    print(f"mean R_y oracle-capture — cosine  : {r['capture_cos']:.3f}")
    print(f"mean R_y oracle-capture — random  : {r['capture_rand']:.3f}")
    print("-" * 52)
    print(f"fact hit rate — learned: {r['fact_hit_learn']}/{n}   cosine: {r['fact_hit_cos']}/{n}")
    print("learned selections by type:", r["pick_counts"])
    print("-" * 52)
    if r["capture_learn"] > r["capture_cos"]:
        print("=> learned R_y controller beat cosine on answer-side oracle-capture.")
    else:
        print("=> learned controller did not beat cosine here.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Listwise R_y controller experiment")
    ap.add_argument("--ref", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--st-model", default=DEFAULT_ST_MODEL)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tau-q", type=float, default=0.5)
    ap.add_argument("--tau-p", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = run_listwise(
        checkpoint_ref=args.ref,
        quant=args.quant,
        st_model_name=args.st_model,
        n_train=args.n_train,
        n_eval=args.n_eval,
        epochs=args.epochs,
        tau_q=args.tau_q,
        tau_p=args.tau_p,
        lr=args.lr,
        seed=args.seed,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
