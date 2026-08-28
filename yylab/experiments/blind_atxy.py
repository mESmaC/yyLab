from __future__ import annotations

import argparse
import random
import statistics

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..infer.engine import ENGINE, LoadSpec
from .listwise_controller import EVAL_STYLES, TRAIN_STYLES, X_STYLES, Y_STYLES
from .predictive_gain import DEFAULT_ST_MODEL, NEUTRAL, _st_model

CAND_TYPES = ["fact", "wrong_channel", "wrong_condition", "unrelated"]


class BlindAddressHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, zx, e_t, e_x, e_y) -> torch.Tensor:
        inp = torch.cat([zx, e_t, e_x, e_y], dim=-1)
        return self.net(inp).squeeze(-1)


def _build_sample(rng: random.Random, style: int) -> tuple[str, str, dict, dict]:
    chan = rng.randint(1, 16)
    chan2 = chan
    while chan2 == chan:
        chan2 = rng.randint(1, 16)
    num = round(rng.uniform(1.0, 9.999), 3)
    num3 = num
    while abs(num3 - num) < 0.5:
        num3 = round(rng.uniform(1.0, 9.999), 3)
    x = X_STYLES[style].format(chan=chan)
    y = Y_STYLES[style].format(chan=chan, num=num)
    payloads = {
        "fact": Y_STYLES[style].format(chan=chan, num=num),
        "wrong_channel": Y_STYLES[style].format(chan=chan2, num=num),
        "wrong_condition": f"Channel {chan} recorded {num3} volts during the daytime.",
        "unrelated": NEUTRAL[rng.randrange(len(NEUTRAL))],
    }
    addresses = {
        "fact": ("voltage", chan, "overnight"),
        "wrong_channel": ("voltage", chan2, "overnight"),
        "wrong_condition": ("voltage", chan, "daytime"),
        "unrelated": ("neutral", 0, "none"),
    }
    return x, y, payloads, addresses


class _AddressEmbedder:
    def __init__(self, st) -> None:
        self.st = st
        self.cache: dict = {}

    def embed(self, addr: tuple) -> tuple:
        if addr not in self.cache:
            t, xval, yval = addr
            e_t = self.st.encode([f"table {t}"], normalize_embeddings=True)[0]
            e_x = self.st.encode([f"channel {xval}"], normalize_embeddings=True)[0]
            e_y = self.st.encode([f"condition {yval}"], normalize_embeddings=True)[0]
            self.cache[addr] = (e_t, e_x, e_y)
        return self.cache[addr]


def _measure_ry(tok, payloads: dict, x: str, y: str) -> list[float]:
    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    xids, yids = ids(x), ids(y)
    nll_y_x = ENGINE.nll_ids(xids, yids)
    return [nll_y_x - ENGINE.nll_ids(ids(payloads[t]) + xids, yids) for t in CAND_TYPES]


def run_blind_atxy(
    checkpoint_ref: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quant: str = "nf4",
    st_model_name: str = DEFAULT_ST_MODEL,
    n_train: int = 60,
    n_eval: int = 24,
    epochs: int = 300,
    tau_q: float = 0.5,
    tau_p: float = 1.0,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    ENGINE.load(LoadSpec(ref=checkpoint_ref, quant=quant, settings={"cache_implementation": "dynamic"}))
    tok = ENGINE.tokenizer
    st = _st_model(st_model_name)
    addr_emb = _AddressEmbedder(st)
    rng = random.Random(seed)

    train_rows = []
    for _ in range(n_train):
        style = TRAIN_STYLES[rng.randrange(len(TRAIN_STYLES))]
        x, y, payloads, addresses = _build_sample(rng, style)
        rys = _measure_ry(tok, payloads, x, y)
        zx = st.encode([x], normalize_embeddings=True)[0]
        factors = np.array([np.concatenate(addr_emb.embed(addresses[t])) for t in CAND_TYPES])
        train_rows.append((zx, factors, np.array(rys)))

    dim = train_rows[0][0].shape[0]
    head = BlindAddressHead(dim)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    order = list(range(len(train_rows)))
    final_loss = float("nan")
    for _ in range(epochs):
        rng.shuffle(order)
        epoch_loss = 0.0
        for idx in order:
            zx, factors, rys = train_rows[idx]
            zx_t = torch.tensor(zx, dtype=torch.float32)
            fac_t = torch.tensor(factors, dtype=torch.float32)
            n = fac_t.shape[0]
            fac_dim = fac_t.shape[1] // 3
            e_t, e_x, e_y = fac_t[:, :fac_dim], fac_t[:, fac_dim : 2 * fac_dim], fac_t[:, 2 * fac_dim :]
            scores = head(zx_t.unsqueeze(0).repeat(n, 1), e_t, e_x, e_y)
            log_p = F.log_softmax(scores / tau_p, dim=-1)
            q = F.softmax(torch.tensor(rys, dtype=torch.float32) / tau_q, dim=-1)
            loss = F.kl_div(log_p, q, reduction="sum")
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        final_loss = epoch_loss / len(train_rows)
    head.eval()

    cap_blind, cap_cos, cap_rand = [], [], []
    hit_blind = hit_cos = 0
    pick_counts = {t: 0 for t in CAND_TYPES}

    for _ in range(n_eval):
        style = EVAL_STYLES[rng.randrange(len(EVAL_STYLES))]
        x, y, payloads, addresses = _build_sample(rng, style)
        rys = _measure_ry(tok, payloads, x, y)
        zx = st.encode([x], normalize_embeddings=True)[0]

        factors = np.array([np.concatenate(addr_emb.embed(addresses[t])) for t in CAND_TYPES])
        fac_t = torch.tensor(factors, dtype=torch.float32)
        fac_dim = fac_t.shape[1] // 3
        e_t, e_x, e_y = fac_t[:, :fac_dim], fac_t[:, fac_dim : 2 * fac_dim], fac_t[:, 2 * fac_dim :]
        with torch.no_grad():
            blind_scores = head(
                torch.tensor(zx, dtype=torch.float32).unsqueeze(0).repeat(len(CAND_TYPES), 1), e_t, e_x, e_y
            ).numpy()

        payload_embs = np.array([st.encode([payloads[t]], normalize_embeddings=True)[0] for t in CAND_TYPES])
        cos_scores = payload_embs @ zx

        oracle = max(rys)
        if oracle <= 1e-9:
            continue
        blind_top = int(np.argmax(blind_scores))
        cos_top = int(np.argmax(cos_scores))
        rand_top = rng.randrange(len(CAND_TYPES))
        cap_blind.append(rys[blind_top] / oracle)
        cap_cos.append(rys[cos_top] / oracle)
        cap_rand.append(rys[rand_top] / oracle)
        if CAND_TYPES[blind_top] == "fact":
            hit_blind += 1
        if CAND_TYPES[cos_top] == "fact":
            hit_cos += 1
        pick_counts[CAND_TYPES[blind_top]] += 1

    ENGINE.unload()

    return {
        "n_train": n_train,
        "n_eval": len(cap_blind),
        "final_kl": final_loss,
        "capture_blind": statistics.mean(cap_blind),
        "capture_cos": statistics.mean(cap_cos),
        "capture_rand": statistics.mean(cap_rand),
        "fact_hit_blind": hit_blind,
        "fact_hit_cos": hit_cos,
        "pick_counts": pick_counts,
    }


def _print_report(r: dict) -> None:
    n = r["n_eval"]
    print("\nBlind ATXY controller (address factors only, NO payload) ")
    print(f"train={r['n_train']}  eval={n}  final KL={r['final_kl']:.4f}")
    print("-" * 56)
    print(f"mean R_y oracle-capture — blind addr : {r['capture_blind']:.3f}")
    print(f"mean R_y oracle-capture — cosine payl: {r['capture_cos']:.3f}")
    print(f"mean R_y oracle-capture — random     : {r['capture_rand']:.3f}")
    print("-" * 56)
    print(f"fact hit rate — blind: {r['fact_hit_blind']}/{n}   cosine-payload: {r['fact_hit_cos']}/{n}")
    print("blind selections by type:", r["pick_counts"])
    print("-" * 56)
    if r["capture_blind"] > r["capture_cos"]:
        print("=> BLIND address routing beat payload-cosine. Thesis supported.")
    elif r["capture_blind"] > 0.6:
        print("=> blind routing is strong (payload-free), near content-aware. Thesis largely supported.")
    else:
        print("=> blind routing underperformed; address structure or training needs work.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind ATXY address-routing controller")
    ap.add_argument("--ref", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--st-model", default=DEFAULT_ST_MODEL)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tau-q", type=float, default=0.5)
    ap.add_argument("--tau-p", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = run_blind_atxy(
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
