from __future__ import annotations

import argparse
import random
import statistics

import numpy as np

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

STRATEGIES = ["empty", "recent", "lex", "sem", "full_history", "oracle"]
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_ST = {"model": None}


def _filler_ids(tokenizer, target_tokens: int) -> list[int]:
    ids: list[int] = []
    i = 0
    while len(ids) < target_tokens:
        ids += tokenizer(NEUTRAL[i % len(NEUTRAL)], add_special_tokens=False)["input_ids"]
        i += 1
    return ids[:target_tokens]


def _make_sample(tokenizer, rng: random.Random, filler_tokens: int, mode: str) -> dict:
    num = round(rng.uniform(1.0, 9.999), 3)
    chan = rng.randint(1, 16)
    fact = f"The voltage reading on channel {chan} was {num} volts during the overnight test."
    if mode == "exact":
        x = f"Therefore, the voltage reading on channel {chan} was {num} volts."
    elif mode == "paraphrase":
        x = rng.choice(
            [
                f"So sensor {chan} logged {num} V.",
                f"Probe {chan} measured {num} V overall.",
                f"Unit {chan} recorded {num} V.",
            ]
        )
    else:  # deanchor: no exact number, no channel referent in X
        rounded = int(round(num))
        x = rng.choice(
            [
                f"That sensor came in at about {rounded} volts.",
                f"The probe settled near {rounded} volts.",
                f"It read roughly {rounded} volts.",
            ]
        )
    fact_ids = tokenizer(fact, add_special_tokens=False)["input_ids"]
    x_ids = tokenizer(x, add_special_tokens=False)["input_ids"]
    filler = _filler_ids(tokenizer, filler_tokens)
    return {"fact_ids": fact_ids, "H_ids": fact_ids + filler, "x_ids": x_ids}


def _chunks(ids: list[int], chunk_size: int) -> list[tuple]:
    return [tuple(ids[i : i + chunk_size]) for i in range(0, len(ids), chunk_size)]


def _retrieve_lex(h_ids: list[int], x_ids: list[int], chunk_size: int, top_k: int) -> list[int]:
    chunks = _chunks(h_ids, chunk_size)
    if not chunks:
        return []
    xset = set(x_ids)
    order = {c: i for i, c in enumerate(chunks)}
    top = sorted(chunks, key=lambda c: (-len(set(c) & xset), order[c]))[:top_k]
    out: list[int] = []
    for c in sorted(top, key=lambda c: order[c]):
        out += list(c)
    return out


def _cosine_np(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _st_model(model_name: str):
    if _ST["model"] is None:
        from sentence_transformers import SentenceTransformer

        _ST["model"] = SentenceTransformer(model_name, device="cpu")
    return _ST["model"]


def _retrieve_sem(h_ids, x_ids, chunk_size, top_k, embed_fn) -> list[int]:
    chunks = _chunks(h_ids, chunk_size)
    if not chunks:
        return []
    x_emb = embed_fn(list(x_ids))
    scored = []
    for i, c in enumerate(chunks):
        c_emb = embed_fn(list(c))
        scored.append((_cosine_np(x_emb, c_emb), i, c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[:top_k]
    top.sort(key=lambda t: t[1])
    out: list[int] = []
    for _, _, c in top:
        out += list(c)
    return out


def _build_embed_fn(retriever: str, tokenizer, st_model_name: str, cache: dict):
    if retriever == "st":
        model = _st_model(st_model_name)

        def embed_fn(ids):
            key = tuple(ids)
            if key not in cache:
                text = tokenizer.decode(ids, skip_special_tokens=True)
                cache[key] = model.encode(text, normalize_embeddings=True)
            return cache[key]

        return embed_fn

    def embed_fn(ids):
        key = tuple(ids)
        if key not in cache:
            cache[key] = ENGINE.embed_ids(list(ids)).numpy()
        return cache[key]

    return embed_fn


def run_predictive_gain(
    checkpoint_ref: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quant: str = "nf4",
    mode: str = "paraphrase",
    retriever: str = "st",
    st_model: str = DEFAULT_ST_MODEL,
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
    emb_cache: dict = {}
    embed_fn = _build_embed_fn(retriever, tok, st_model, emb_cache)

    deltas = {s: [] for s in STRATEGIES}
    nlls = {s: [] for s in STRATEGIES}
    prefix_lens = {s: [] for s in STRATEGIES}

    for _ in range(n_samples):
        sample = _make_sample(tok, rng, filler_tokens, mode)
        x = sample["x_ids"]
        prefixes = {
            "empty": [],
            "recent": sample["H_ids"][-recent_k:],
            "lex": _retrieve_lex(sample["H_ids"], x, chunk_size, top_k),
            "sem": _retrieve_sem(sample["H_ids"], x, chunk_size, top_k, embed_fn),
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
    return {
        "rows": rows,
        "mode": mode,
        "retriever": retriever,
        "config": {"n_samples": n_samples, "filler_tokens": filler_tokens},
    }


def _print_report(result: dict) -> None:
    base = next(r["mean_nll"] for r in result["rows"] if r["strategy"] == "empty")
    print(f"\nPredictive gain — mode={result['mode']}  sem-retriever={result['retriever']}")
    print("(Δ = NLL(X|empty) − NLL(X|C); higher = context helped more)")
    print(f"{'strategy':<14}{'mean NLL':>10}{'Δ gain':>10}{'prefix tok':>12}")
    print("-" * 48)
    for r in result["rows"]:
        print(f'{r["strategy"]:<14}{r["mean_nll"]:>10.4f}{r["mean_delta"]:>10.4f}{r["mean_prefix_tokens"]:>12.0f}')
    print("-" * 48)

    def d(name):
        return next(r["mean_delta"] for r in result["rows"] if r["strategy"] == name)

    print(f"baseline NLL(X|empty) = {base:.4f}")
    print(f"lex (token-overlap)  : {d('lex'):+.4f}")
    print(f"sem ({result['retriever']:<14}): {d('sem'):+.4f}")
    print(f"recent               : {d('recent'):+.4f}")
    print(f"full_history         : {d('full_history'):+.4f}")
    print(f"oracle (ceiling)     : {d('oracle'):+.4f}")
    if d("sem") > d("lex"):
        print("=> semantic retrieval beat token-overlap.")
    else:
        print("=> token-overlap matched/beat semantic.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predictive-gain memory experiment")
    ap.add_argument("--ref", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--mode", default="paraphrase", choices=["exact", "paraphrase", "deanchor"])
    ap.add_argument("--retriever", default="st", choices=["st", "llm"])
    ap.add_argument("--st-model", default=DEFAULT_ST_MODEL)
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
        mode=args.mode,
        retriever=args.retriever,
        st_model=args.st_model,
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
