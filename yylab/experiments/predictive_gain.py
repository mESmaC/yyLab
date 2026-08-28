from __future__ import annotations

import argparse
import random
import statistics

from ..infer.engine import ENGINE, LoadSpec

NEUTRAL = [
    "The team reviewed the maintenance schedule and confirmed the checklist was up to date.",
    "Ambient temperature in the lab stayed within the expected range throughout the shift.",
    "A routine audit of the logging pipeline found no missing entries or corrupted rows.",
    "The vendor confirmed the shipment would arrive before the end of the week.",
    "Shift notes were archived and the previous operator signed off without incident.",
    "A calibration reminder was posted to the shared board for the next session.",
    "The backup job completed overnight and the integrity check passed cleanly.",
    "Documentation for the procedure was updated to reflect the latest revision.",
]

STRATEGIES = ["empty", "recent", "retrieved", "full_history", "oracle"]


def _filler_ids(tokenizer, target_tokens: int) -> list[int]:
    ids: list[int] = []
    i = 0
    while len(ids) < target_tokens:
        ids += tokenizer(NEUTRAL[i % len(NEUTRAL)], add_special_tokens=False)["input_ids"]
        i += 1
    return ids[:target_tokens]


def _make_sample(tokenizer, rng: random.Random, filler_tokens: int) -> dict:
    num = round(rng.uniform(1.0, 9.999), 3)
    chan = rng.randint(1, 16)
    fact = f"The voltage reading on channel {chan} was {num} volts during the overnight test."
    x = f"Therefore, the voltage reading on channel {chan} was {num} volts."
    fact_ids = tokenizer(fact, add_special_tokens=False)["input_ids"]
    x_ids = tokenizer(x, add_special_tokens=False)["input_ids"]
    filler = _filler_ids(tokenizer, filler_tokens)
    return {"fact_ids": fact_ids, "H_ids": fact_ids + filler, "x_ids": x_ids}


def _retrieve_ids(h_ids: list[int], x_ids: list[int], chunk_size: int, top_k: int) -> list[int]:
    chunks = [tuple(h_ids[i : i + chunk_size]) for i in range(0, len(h_ids), chunk_size)]
    if not chunks:
        return []
    xset = set(x_ids)
    order = {c: i for i, c in enumerate(chunks)}
    top = sorted(chunks, key=lambda c: (-len(set(c) & xset), order[c]))[:top_k]
    out: list[int] = []
    for c in sorted(top, key=lambda c: order[c]):
        out += list(c)
    return out


def run_predictive_gain(
    checkpoint_ref: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quant: str = "nf4",
    n_samples: int = 8,
    filler_tokens: int = 1200,
    recent_k: int = 256,
    chunk_size: int = 64,
    top_k: int = 4,
    seed: int = 0,
) -> dict:
    ENGINE.load(LoadSpec(ref=checkpoint_ref, quant=quant, settings={"cache_implementation": "dynamic"}))
    tok = ENGINE.tokenizer
    rng = random.Random(seed)

    deltas = {s: [] for s in STRATEGIES}
    nlls = {s: [] for s in STRATEGIES}
    prefix_lens = {s: [] for s in STRATEGIES}

    for _ in range(n_samples):
        sample = _make_sample(tok, rng, filler_tokens)
        x = sample["x_ids"]
        prefixes = {
            "empty": [],
            "recent": sample["H_ids"][-recent_k:],
            "retrieved": _retrieve_ids(sample["H_ids"], x, chunk_size, top_k),
            "full_history": sample["H_ids"],
            "oracle": sample["fact_ids"],
        }
        base = ENGINE.nll_ids([], x)
        for name in STRATEGIES:
            prefix = prefixes[name]
            nll_val = base if name == "empty" else ENGINE.nll_ids(prefix, x)
            deltas[name].append(base - nll_val)
            nlls[name].append(nll_val)
            prefix_lens[name].append(len(prefix))

    ENGINE.unload()

    rows = []
    for name in STRATEGIES:
        rows.append(
            {
                "strategy": name,
                "mean_delta": statistics.mean(deltas[name]),
                "mean_nll": statistics.mean(nlls[name]),
                "mean_prefix_tokens": statistics.mean(prefix_lens[name]),
            }
        )
    return {"rows": rows, "config": {"n_samples": n_samples, "filler_tokens": filler_tokens}}


def _print_report(result: dict) -> None:
    base = next(r["mean_nll"] for r in result["rows"] if r["strategy"] == "empty")
    print("\nPredictive gain  (Δ = NLL(X|empty) − NLL(X|C); higher = context helped more)")
    print(f"{'strategy':<14}{'mean NLL':>10}{'Δ gain':>10}{'prefix tok':>12}")
    print("-" * 48)
    for r in result["rows"]:
        print(f'{r["strategy"]:<14}{r["mean_nll"]:>10.4f}{r["mean_delta"]:>10.4f}{r["mean_prefix_tokens"]:>12.0f}')
    print("-" * 48)
    rec = next(r["mean_delta"] for r in result["rows"] if r["strategy"] == "retrieved")
    rec_recent = next(r["mean_delta"] for r in result["rows"] if r["strategy"] == "recent")
    rec_oracle = next(r["mean_delta"] for r in result["rows"] if r["strategy"] == "oracle")
    print(f"baseline NLL(X|empty) = {base:.4f}")
    print(f"retrieved vs recent  : {rec:+.4f} vs {rec_recent:+.4f}")
    print(f"retrieved vs oracle  : {rec:+.4f} vs {rec_oracle:+.4f}  (oracle = ceiling)")
    if rec > rec_recent:
        print("=> relevance retrieval beat recency: memory thesis supported on this sample.")
    else:
        print("=> recency matched/beat retrieval here (needs tweaking, not necessarily wrong).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predictive-gain memory experiment")
    ap.add_argument("--ref", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--filler-tokens", type=int, default=1200)
    ap.add_argument("--recent-k", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = run_predictive_gain(
        checkpoint_ref=args.ref,
        quant=args.quant,
        n_samples=args.samples,
        filler_tokens=args.filler_tokens,
        recent_k=args.recent_k,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        seed=args.seed,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
