"""Arithmetic-only exploration of dense Sprint shapes; this never runs training.

All candidates use tied embeddings. FFN fractions refer to unique parameters in
all decoder FFNs divided by the complete model, including vocabulary and norms.
For each width/depth/fraction, try adjacent aligned FFN widths and the largest
FFN fitting the target. Every result must meet both requested tolerances.
Ordering describes shapes, not measured speed or expected model quality.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

DEFAULT_FFN_FRACTIONS = (0.25, 0.375, 0.45, 0.5, 0.625, 0.70, 0.75)
DEFAULT_DEPTHS = (1, 2, 3, 4, 6, 8, 12)
MAX_SEARCH_COMBINATIONS = 16_384


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fraction(value, name, *, zero=False):
    try:
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError
        result = Fraction(str(value))
        if not (0 <= result < 1 if zero else 0 < result < 1):
            raise ValueError
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be finite and {'in [0, 1)' if zero else 'between 0 and 1'}") from exc
    return result


def _nonempty_sequence(values, name):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a nonempty sequence")
    return values


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def sprint_parameter_breakdown(
    vocab_size: int, hidden_size: int, intermediate_size: int, num_hidden_layers: int, *, mlp_type="gelu",
) -> dict[str, int]:
    """Exact unique parameters for bias-free, tied-embedding Sprint."""
    for name, value in (("vocab_size", vocab_size), ("hidden_size", hidden_size),
                        ("intermediate_size", intermediate_size), ("num_hidden_layers", num_hidden_layers)):
        _positive_int(value, name)
    if mlp_type not in {"gelu", "swiglu"}:
        raise ValueError("mlp_type must be gelu or swiglu")
    components = {
        "embeddings": vocab_size * hidden_size,
        "attention": num_hidden_layers * 4 * hidden_size**2,
        "ffn": num_hidden_layers * (3 if mlp_type == "swiglu" else 2) * hidden_size * intermediate_size,
        "norms": (2 * num_hidden_layers + 1) * hidden_size,
    }
    return {**components, "total": sum(components.values())}


def enumerate_sprint_shapes(
    target_params: int, vocab_size: int = 8192, *,
    ffn_fractions: Sequence = DEFAULT_FFN_FRACTIONS, depths: Sequence = DEFAULT_DEPTHS,
    width_alignment: int = 128, ffn_alignment: int = 64, head_dim: int = 128,
    tolerance=0.02, ffn_fraction_tolerance=0.05, max_width: int = 8192,
    max_position_embeddings: int = 2048, mlp_type: str = "gelu",
) -> list[dict]:
    """Return bounded candidates, all at or below target and within tolerance.

    ``tolerance`` is the maximum total-parameter shortfall relative to target.
    ``ffn_fraction_tolerance`` is an absolute fraction difference: 0.05 means
    five percentage points. No FFN expansion limit is imposed. Configurations
    are deduplicated and sorted by depth, width, then FFN width. An empty list
    means these constraints have no candidates; the function never returns a
    smaller fallback and presents it as a matched parameter budget.
    """
    target_params = _positive_int(target_params, "target_params")
    vocab_size = _positive_int(vocab_size, "vocab_size")
    max_width = _positive_int(max_width, "max_width")
    head_dim = _positive_int(head_dim, "head_dim")
    _positive_int(max_position_embeddings, "max_position_embeddings")
    _positive_int(width_alignment, "width_alignment")
    _positive_int(ffn_alignment, "ffn_alignment")
    if head_dim % 2:
        raise ValueError("head_dim must be even for Sprint rotary embeddings")
    if width_alignment not in {128, 256} or isinstance(width_alignment, bool):
        raise ValueError("width_alignment must be 128 or 256")
    if ffn_alignment not in {64, 128} or isinstance(ffn_alignment, bool):
        raise ValueError("ffn_alignment must be 64 or 128")
    if mlp_type not in {"gelu", "swiglu"}:
        raise ValueError("mlp_type must be gelu or swiglu")
    total_tolerance = _fraction(tolerance, "tolerance", zero=True)
    fraction_tolerance = _fraction(ffn_fraction_tolerance, "ffn_fraction_tolerance", zero=True)
    shares = sorted({_fraction(value, "ffn_fractions")
                     for value in _nonempty_sequence(ffn_fractions, "ffn_fractions")})
    layer_counts = sorted({_positive_int(value, "depths") for value in _nonempty_sequence(depths, "depths")})
    width_step = math.lcm(width_alignment, head_dim)
    width_limit = min(max_width, target_params // vocab_size,
                      math.isqrt(target_params // (4 * min(layer_counts))))
    if (width_limit // width_step) * len(layer_counts) * len(shares) > MAX_SEARCH_COMBINATIONS:
        raise ValueError("shape search is too broad; reduce max_width, depths, or ffn_fractions")
    minimum_total = _ceil(target_params * (1 - total_tolerance))
    found = {}
    for layers in layer_counts:
        for width in range(width_step, width_limit + 1, width_step):
            fixed = vocab_size * width + layers * 4 * width**2 + (2 * layers + 1) * width
            per_ffn_unit = layers * (3 if mlp_type == "swiglu" else 2) * width
            minimum_ffn = max(ffn_alignment, _ceil(Fraction(minimum_total - fixed, per_ffn_unit * ffn_alignment)) * ffn_alignment)
            maximum_ffn = (target_params - fixed) // (per_ffn_unit * ffn_alignment) * ffn_alignment
            if maximum_ffn < minimum_ffn:
                continue
            for share in shares:
                ideal_units = share * fixed / (1 - share) / (per_ffn_unit * ffn_alignment)
                aligned = {ideal_units.numerator // ideal_units.denominator * ffn_alignment,
                           _ceil(ideal_units) * ffn_alignment, maximum_ffn}
                for ffn_width in sorted({min(maximum_ffn, max(minimum_ffn, value)) for value in aligned}):
                    counts = sprint_parameter_breakdown(vocab_size, width, ffn_width, layers, mlp_type=mlp_type)
                    achieved = Fraction(counts["ffn"], counts["total"])
                    if abs(achieved - share) > fraction_tolerance:
                        continue
                    key = (layers, width, ffn_width)
                    if key not in found:
                        error = target_params - counts["total"]
                        found[key] = {
                            "model_config": {"model_type": "sprint", "vocab_size": vocab_size,
                                             "hidden_size": width, "intermediate_size": ffn_width,
                                             "num_hidden_layers": layers, "num_attention_heads": width // head_dim,
                                             "max_position_embeddings": max_position_embeddings,
                                             "tie_word_embeddings": True, "mlp_type": mlp_type},
                            "parameters": counts["total"], "parameter_breakdown": counts,
                            "target_parameters": target_params, "parameter_shortfall": error,
                            "parameter_shortfall_fraction": error / target_params,
                            "ffn_fraction": float(achieved), "ffn_expansion": ffn_width / width,
                            "matched_ffn_fractions": [],
                        }
                    found[key]["matched_ffn_fractions"].append({
                        "requested": float(share), "absolute_error": float(abs(achieved - share)),
                    })
    return [found[key] for key in sorted(found)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-params", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--ffn-fractions", default=",".join(map(str, DEFAULT_FFN_FRACTIONS)))
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--width-alignment", type=int, choices=(128, 256), default=128)
    parser.add_argument("--ffn-alignment", type=int, choices=(64, 128), default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=0.02, help="Maximum relative parameter shortfall")
    parser.add_argument("--ffn-fraction-tolerance", type=float, default=0.05, help="Absolute FFN fraction tolerance")
    parser.add_argument("--max-width", type=int, default=8192)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--mlp-type", choices=("gelu", "swiglu"), default="gelu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint-shapes.json"))
    args = parser.parse_args()
    try:
        options = {"ffn_fractions": [float(value) for value in args.ffn_fractions.split(",")],
                   "depths": [int(value) for value in args.depths.split(",")],
                   **{key: getattr(args, key) for key in (
                       "width_alignment", "ffn_alignment", "head_dim", "tolerance", "ffn_fraction_tolerance",
                       "max_width", "max_position_embeddings", "mlp_type")}}
        candidates = enumerate_sprint_shapes(args.target_params, args.vocab_size, **options)
    except ValueError as exc:
        parser.error(str(exc))
    report = {
        "schema": "yylab-sprint-shape-candidates-v1", "status": "ok" if candidates else "no_matching_shapes",
        "request": {"target_params": args.target_params, "vocab_size": args.vocab_size, **options},
        "candidate_count": len(candidates), "candidates": candidates,
        "method": "Arithmetic candidates with tied embeddings. Per width/depth/fraction, adjacent fraction-matching FFN widths and the largest budget-fitting FFN are checked. No training or throughput measurement; ordering is depth/width/FFN, not a speed ranking.",
        "limits": "Every candidate is at or below target and within both requested tolerances. FFN fraction includes the complete unique parameter count as its denominator. This is a bounded candidate search, not every possible aligned FFN width or a quality assessment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "candidates": len(candidates), "output": str(args.output)}))
    if not candidates:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
