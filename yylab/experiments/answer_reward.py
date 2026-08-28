from __future__ import annotations

import argparse
import random
import statistics

from ..infer.engine import ENGINE, LoadSpec
from .predictive_gain import DEFAULT_ST_MODEL, NEUTRAL, _st_model

CAND_TYPES = ["fact", "echo", "wrong_value", "wrong_channel", "unrelated"]


def _build_sample(rng: random.Random) -> tuple[str, str, dict]:
    chan = rng.randint(1, 16)
    chan2 = chan
    while chan2 == chan:
        chan2 = rng.randint(1, 16)
    num = round(rng.uniform(1.0, 9.999), 3)
    num2 = num
    while abs(num2 - num) < 0.5:
        num2 = round(rng.uniform(1.0, 9.999), 3)
    x = f"What voltage did channel {chan} record overnight?"
    y = f"Channel {chan} recorded {num} volts overnight."
    candidates = {
        "fact": f"Channel {chan} recorded {num} volts overnight.",
        "echo": f"You asked what voltage channel {chan} recorded overnight.",
        "wrong_value": f"Channel {chan} recorded {num2} volts overnight.",
        "wrong_channel": f"Channel {chan2} recorded {num} volts overnight.",
        "unrelated": NEUTRAL[rng.randrange(len(NEUTRAL))],
    }
    return x, y, candidates


def run_answer_reward(
    checkpoint_ref: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quant: str = "nf4",
    st_model_name: str = DEFAULT_ST_MODEL,
    n_samples: int = 20,
    seed: int = 0,
) -> dict:
    ENGINE.load(LoadSpec(ref=checkpoint_ref, quant=quant, settings={"cache_implementation": "dynamic"}))
    tok = ENGINE.tokenizer
    st = _st_model(st_model_name)
    rng = random.Random(seed)

    def ids(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    per_type = {t: {"ry": [], "rx": [], "cos": []} for t in CAND_TYPES}
    echo_beats_fact_ry = 0
    echo_beats_fact_rx = 0
    cos_prefers_echo_over_fact = 0
    cos_ry_capture = []

    for _ in range(n_samples):
        x, y, candidates = _build_sample(rng)
        xids, yids = ids(x), ids(y)
        nll_y_given_x = ENGINE.nll_ids(xids, yids)
        nll_x_empty = ENGINE.nll_ids([], xids)
        zx = st.encode([x], normalize_embeddings=True)[0]

        ry_by_type, cos_by_type = {}, {}
        for t in CAND_TYPES:
            c = candidates[t]
            cids = ids(c)
            ry = nll_y_given_x - ENGINE.nll_ids(cids + xids, yids)
            rx = nll_x_empty - ENGINE.nll_ids(cids, xids)
            zc = st.encode([c], normalize_embeddings=True)[0]
            cos = float((zc * zx).sum())
            per_type[t]["ry"].append(ry)
            per_type[t]["rx"].append(rx)
            per_type[t]["cos"].append(cos)
            ry_by_type[t] = ry
            cos_by_type[t] = cos

        if ry_by_type["echo"] > ry_by_type["fact"]:
            echo_beats_fact_ry += 1
        if per_type["echo"]["rx"][-1] > per_type["fact"]["rx"][-1]:
            echo_beats_fact_rx += 1
        if cos_by_type["echo"] > cos_by_type["fact"]:
            cos_prefers_echo_over_fact += 1

        best = max(ry_by_type.values())
        cos_pick = max(CAND_TYPES, key=lambda t: cos_by_type[t])
        base_ry = 0.0
        denom = best - base_ry
        capture = (ry_by_type[cos_pick] - base_ry) / denom if denom > 1e-9 else 1.0
        cos_ry_capture.append(capture)

    ENGINE.unload()

    rows = {}
    for t in CAND_TYPES:
        rows[t] = {
            "mean_ry": statistics.mean(per_type[t]["ry"]),
            "mean_rx": statistics.mean(per_type[t]["rx"]),
            "mean_cos": statistics.mean(per_type[t]["cos"]),
        }
    return {
        "rows": rows,
        "n_samples": n_samples,
        "echo_beats_fact_ry": echo_beats_fact_ry,
        "echo_beats_fact_rx": echo_beats_fact_rx,
        "cos_prefers_echo_over_fact": cos_prefers_echo_over_fact,
        "mean_cos_ry_capture": statistics.mean(cos_ry_capture),
    }


def _print_report(r: dict) -> None:
    n = r["n_samples"]
    print("\nAnswer-side reward (R_y) vs input-side (R_x) vs cosine")
    print(f"samples={n}")
    print(f"{'candidate':<15}{'mean R_y':>10}{'mean R_x':>10}{'cos(x,c)':>10}")
    print("-" * 47)
    for t in CAND_TYPES:
        row = r["rows"][t]
        print(f"{t:<15}{row['mean_ry']:>10.4f}{row['mean_rx']:>10.4f}{row['mean_cos']:>10.4f}")
    print("-" * 47)
    print(f"echo beat fact on R_y in {r['echo_beats_fact_ry']}/{n} samples (want 0)")
    print(f"echo beat fact on R_x in {r['echo_beats_fact_rx']}/{n} samples (the old failure)")
    print(f"cosine preferred echo over fact in {r['cos_prefers_echo_over_fact']}/{n} samples")
    print(f"cosine's R_y oracle-capture: {r['mean_cos_ry_capture']:.2f} (1.0 = picked the best)")
    print("-" * 47)
    fact_ry = r["rows"]["fact"]["mean_ry"]
    echo_ry = r["rows"]["echo"]["mean_ry"]
    if fact_ry > echo_ry and r["echo_beats_fact_ry"] == 0:
        print("=> answer-side reward rejects the echo: fact R_y >> echo R_y. Objective fixed.")
    else:
        print("=> echo still competitive under R_y; needs investigation.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer-side (x/y) reward characterization")
    ap.add_argument("--ref", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--st-model", default=DEFAULT_ST_MODEL)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = run_answer_reward(
        checkpoint_ref=args.ref,
        quant=args.quant,
        st_model_name=args.st_model,
        n_samples=args.samples,
        seed=args.seed,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
