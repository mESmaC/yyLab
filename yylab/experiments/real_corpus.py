from __future__ import annotations

import argparse
import random
import statistics

from ..infer.engine import ENGINE, LoadSpec
from .predictive_gain import (
    STRATEGIES,
    _build_embed_fn,
    _retrieve_lex,
    _retrieve_sem,
)


def _load_corpus_ids(tokenizer, dataset_name: str, max_tokens: int) -> list[int]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, "wikitext-2-raw-v1", split="train")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens]


def _carve_sample(ids, rng: random.Random, len_a: int, len_x: int, len_m: int) -> dict:
    n = len(ids)
    half = n // 2
    p = rng.randint(0, half - len_a - len_x)
    a = ids[p : p + len_a]
    x = ids[p + len_a : p + len_a + len_x]
    q = rng.randint(half, n - len_m)
    m = ids[q : q + len_m]
    return {"A": a, "x": x, "M": m, "H": a + m}


def run_real_corpus(
    checkpoint_ref: str = "Qwen/Qwen2.5-0.5B-Instruct",
    quant: str = "nf4",
    dataset_name: str = "Salesforce/wikitext",
    retriever: str = "st",
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_samples: int = 8,
    len_a: int = 150,
    len_x: int = 40,
    len_m: int = 1000,
    recent_k: int = 256,
    chunk_size: int = 64,
    top_k: int = 4,
    seed: int = 0,
    max_tokens: int = 300000,
) -> dict:
    ENGINE.load(LoadSpec(ref=checkpoint_ref, quant=quant, settings={"cache_implementation": "dynamic"}))
    tok = ENGINE.tokenizer

    ids = _load_corpus_ids(tok, dataset_name, max_tokens)
    rng = random.Random(seed)
    emb_cache: dict = {}
    embed_fn = _build_embed_fn(retriever, tok, st_model, emb_cache)

    deltas = {s: [] for s in STRATEGIES}
    nlls = {s: [] for s in STRATEGIES}
    prefix_lens = {s: [] for s in STRATEGIES}

    for i in range(n_samples):
        sample = _carve_sample(ids, rng, len_a, len_x, len_m)
        x = sample["x"]
        prefixes = {
            "empty": [],
            "recent": sample["H"][-recent_k:],
            "lex": _retrieve_lex(sample["H"], x, chunk_size, top_k),
            "sem": _retrieve_sem(sample["H"], x, chunk_size, top_k, embed_fn),
            "full_history": sample["H"],
            "oracle": sample["A"],
        }
        base = ENGINE.nll_ids([], x)
        for name in STRATEGIES:
            prefix = prefixes[name]
            nll_val = base if name == "empty" else ENGINE.nll_ids(prefix, x)
            deltas[name].append(base - nll_val)
            nlls[name].append(nll_val)
            prefix_lens[name].append(len(prefix))
        if i == 0:
            print("\n[sample 0] X (held-out next segment):")
            print("  " + tok.decode(x)[:160].replace("\n", " "))
            print("[sample 0] A (relevant segment, oracle):")
            print("  " + tok.decode(sample["A"])[:160].replace("\n", " "))

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
    return {"rows": rows, "retriever": retriever, "dataset": dataset_name, "n_samples": n_samples}


def _print_report(result: dict) -> None:
    base = next(r["mean_nll"] for r in result["rows"] if r["strategy"] == "empty")
    print(f"\nReal-corpus predictive gain — dataset={result['dataset']}  sem-retriever={result['retriever']}")
    print("(A=relevant segment, M=distractor, X=continuation of A; oracle=A)")
    print(f"{'strategy':<14}{'mean NLL':>10}{'Δ gain':>10}{'prefix tok':>12}")
    print("-" * 48)
    for r in result["rows"]:
        print(f'{r["strategy"]:<14}{r["mean_nll"]:>10.4f}{r["mean_delta"]:>10.4f}{r["mean_prefix_tokens"]:>12.0f}')
    print("-" * 48)

    def d(name):
        return next(r["mean_delta"] for r in result["rows"] if r["strategy"] == name)

    print(f"baseline NLL(X|empty) = {base:.4f}")
    print(f"recent (distractor)  : {d('recent'):+.4f}")
    print(f"lex (token-overlap)  : {d('lex'):+.4f}")
    print(f"sem ({result['retriever']:<14}): {d('sem'):+.4f}")
    print(f"full_history (A+M)   : {d('full_history'):+.4f}")
    print(f"oracle (A only)      : {d('oracle'):+.4f}")
    best = max(["lex", "sem"], key=lambda s: d(s))
    if d(best) > d("recent") and d(best) > 0:
        print(f"=> retrieval ({best}) beat the distractor recency on real text.")
    else:
        print("=> retrieval did not clearly beat recency on real text.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-corpus predictive-gain experiment")
    ap.add_argument("--ref", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--retriever", default="st", choices=["st", "llm"])
    ap.add_argument("--st-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--len-a", type=int, default=150)
    ap.add_argument("--len-x", type=int, default=40)
    ap.add_argument("--len-m", type=int, default=1000)
    ap.add_argument("--recent-k", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=300000)
    args = ap.parse_args()
    result = run_real_corpus(
        checkpoint_ref=args.ref,
        quant=args.quant,
        dataset_name=args.dataset,
        retriever=args.retriever,
        st_model=args.st_model,
        n_samples=args.samples,
        len_a=args.len_a,
        len_x=args.len_x,
        len_m=args.len_m,
        recent_k=args.recent_k,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
