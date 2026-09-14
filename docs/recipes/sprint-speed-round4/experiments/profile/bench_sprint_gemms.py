"""Isolated GEMM and quantization timings at Sprint's exact decoder shapes.

For every decoder projection (packed QKV, attention output, MLP up, MLP down)
and the tied head, this times the forward, input-gradient and weight-gradient
matrix multiplications in BF16, native MXFP8 and native NVFP4, both as bare
GEMMs on pre-quantized operands and as the complete cast + GEMM paths that
`yylab.train.sprint_precision` / `yylab.train.mxfp8_fast` execute inside a
training step. It also times the individual casts. The result is a measured
GEMM floor per microbatch and a direct measurement of how much of each
projection's time is quantization/padding rather than tensor-core work.

Synthetic operands, one process, no optimizer, no model: this establishes
kernel throughput on this GPU, not training speed or quality. Compare the
roll-up against the forward/backward device time from profile_sprint_step.py.

    .venv/Scripts/python.exe docs/recipes/sprint-speed-round4/experiments/profile/bench_sprint_gemms.py \
        --config docs/recipes/sprint-speed-round3/baseline-b32.json --tokens 2048,4096,8192 --output artifacts/gemm-bench-1b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from importlib import metadata, util
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "yylab").is_dir():
            return candidate
    raise SystemExit("run this script from inside the yyLab repository (pyproject.toml + yylab/ not found)")


ROOT = _find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

# Dense whitepaper tensor rates for the GeForce RTX 5070 (TFLOP/s). FP8 has two
# entries because GeForce Blackwell halves FP8 with FP32 accumulation; which
# rate block-scaled MXFP8 achieves on SM120 is exactly what this measures.
PEAK_TFLOPS = {"bf16": 61.7, "mxfp8": 246.9, "fp8_tensorwise": 246.9, "mxfp8_fp32acc_reference": 123.5, "nvfp4": 493.9}


def _version(*names):
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Timer:
    def __init__(self, iters: int, warmup: int):
        self.iters, self.warmup = iters, warmup

    def time(self, fn, arg_sets: list[tuple]) -> dict:
        rotate = len(arg_sets)
        for index in range(self.warmup):
            fn(*arg_sets[index % rotate])
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.iters)]
        wall = time.perf_counter()
        for index in range(self.iters):
            starts[index].record()
            fn(*arg_sets[index % rotate])
            ends[index].record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - wall
        times = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
        return {"median_ms": statistics.median(times), "min_ms": times[0], "max_ms": times[-1],
                "mean_ms": statistics.fmean(times), "iters": self.iters, "wall_ms_per_iter": wall / self.iters * 1e3}


def _tflops(flop: float, ms: float) -> float:
    return flop / (ms / 1e3) / 1e12 if ms else 0.0


def _row(name, role, precision, shape, flop, timing, *, path="gemm only", note=None) -> dict:
    achieved = _tflops(flop, timing["median_ms"])
    peak = PEAK_TFLOPS.get(precision)
    row = {"linear": name, "role": role, "precision": precision, "path": path, "shape": shape,
           "gflop": round(flop / 1e9, 3), **{k: round(v, 4) for k, v in timing.items()},
           "achieved_tflops": round(achieved, 1), "fraction_of_peak": round(achieved / peak, 3) if peak and flop else None}
    if precision in {"mxfp8", "fp8_tensorwise"} and flop:
        row["fraction_of_fp32acc_reference"] = round(achieved / PEAK_TFLOPS["mxfp8_fp32acc_reference"], 3)
    if note:
        row["note"] = note
    return row


def _fail(name, role, precision, shape, exc, path="gemm only") -> dict:
    return {"linear": name, "role": role, "precision": precision, "path": path, "shape": shape,
            "error": f"{type(exc).__name__}: {exc}"}


def _pad(value: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    if value.shape == (rows, columns):
        return value.contiguous()
    return F.pad(value, (0, columns - value.shape[1], 0, rows - value.shape[0]))


def bench_linear(name: str, n: int, k: int, m: int, timer: Timer, rotate: int, device, *, skip_nvfp4=False) -> list[dict]:
    """Rows for one projection with weight [n, k] and m tokens."""
    from yylab.train import mxfp8 as native
    from yylab.train.mxfp8_fast import _as_mx, mxfp8_linear_backward_fast, mxfp8_mm_nt_fast

    rows = []
    shape = {"tokens": m, "out_features": n, "in_features": k}
    flop = 2.0 * m * n * k
    generator = torch.Generator(device=device).manual_seed(1208)
    xs = [torch.randn((m, k), device=device, dtype=torch.bfloat16, generator=generator) for _ in range(rotate)]
    ws = [torch.randn((n, k), device=device, dtype=torch.bfloat16, generator=generator) * 0.02 for _ in range(rotate)]
    dys = [torch.randn((m, n), device=device, dtype=torch.bfloat16, generator=generator) for _ in range(rotate)]

    # BF16 reference: cuBLAS with FP32 accumulation.
    try:
        rows.append(_row(name, "forward", "bf16", shape, flop, timer.time(lambda x, w: F.linear(x, w), list(zip(xs, ws)))))
        rows.append(_row(name, "dgrad", "bf16", shape, flop, timer.time(lambda dy, w: dy @ w, list(zip(dys, ws)))))
        rows.append(_row(name, "wgrad", "bf16", shape, flop, timer.time(lambda dy, x: dy.t() @ x, list(zip(dys, xs)))))
    except Exception as exc:
        rows.append(_fail(name, "forward/dgrad/wgrad", "bf16", shape, exc))

    # Plain tensorwise FP8 (no block scales) through cuBLAS: separates the SM120
    # FP32-accumulate question from the block-scaled MX kernel selection.
    try:
        a8 = [x.to(torch.float8_e4m3fn) for x in xs]
        b8 = [w.to(torch.float8_e4m3fn) for w in ws]
        one = torch.ones((), device=device, dtype=torch.float32)
        rows.append(_row(name, "forward", "fp8_tensorwise", shape, flop,
                         timer.time(lambda a, b: torch._scaled_mm(a, b.t(), one, one, out_dtype=torch.bfloat16), list(zip(a8, b8))),
                         note="per-tensor scales, reference only"))
        del a8, b8
    except Exception as exc:
        rows.append(_fail(name, "forward", "fp8_tensorwise", shape, exc))

    # MXFP8 forward: Sprint's mxfp8 mode (fast path when aligned) and the bare GEMM.
    try:
        rows.append(_row(name, "forward", "mxfp8", shape, flop, timer.time(lambda x, w: mxfp8_mm_nt_fast(x, w, backend="native"), list(zip(xs, ws))),
                         path="cast + gemm (mxfp8_mm_nt_fast)", note="falls back to padded native path when M/N/K are not multiples of 128"))
        if all(dim % 128 == 0 for dim in (m, n, k)):
            lx, lw = [_as_mx(x) for x in xs], [_as_mx(w) for w in ws]
            rows.append(_row(name, "forward", "mxfp8", shape, flop, timer.time(lambda a, b: torch.mm(a, b.t()), list(zip(lx, lw)))))
            del lx, lw
    except Exception as exc:
        rows.append(_fail(name, "forward", "mxfp8", shape, exc))

    # MXFP8 backward as Sprint runs it, then the bare dgrad/wgrad GEMMs on pre-cast operands.
    try:
        rows.append(_row(name, "dgrad+wgrad", "mxfp8", shape, 2 * flop,
                         timer.time(lambda x, w, dy: mxfp8_linear_backward_fast(x, w, dy, backend="native"), list(zip(xs, ws, dys))),
                         path="pad + casts + 2 gemms (mxfp8_linear_backward_fast)"))
        pm, pn, pk = (native._ceil_multiple(m, 128), native._ceil_multiple(n, 128), native._ceil_multiple(k, 128))
        x_p = [_pad(x, pm, pk) for x in xs]
        w_p = [_pad(w, pn, pk) for w in ws]
        dy_p = [_pad(dy, pm, pn) for dy in dys]
        grad_rows = [_as_mx(dy) for dy in dy_p]
        weight_columns = [_as_mx(w, transposed=True) for w in w_p]
        rows.append(_row(name, "dgrad", "mxfp8", shape, flop, timer.time(lambda a, b: torch.mm(a, b.t()), list(zip(grad_rows, weight_columns)))))
        del grad_rows, weight_columns
        grad_columns = [_as_mx(dy, transposed=True) for dy in dy_p]
        input_columns = [_as_mx(x, transposed=True) for x in x_p]
        rows.append(_row(name, "wgrad", "mxfp8", shape, flop, timer.time(lambda a, b: torch.mm(a, b.t()), list(zip(grad_columns, input_columns)))))
        del grad_columns, input_columns
        # Individual casts at the shapes the backward uses.
        cast_flop = 0.0
        rows.append(_row(name, "cast dY rows (dim0)", "mxfp8", shape, cast_flop, timer.time(lambda dy: _as_mx(dy), [(dy,) for dy in dy_p]), path="triton_to_mxfp8_dim0"))
        rows.append(_row(name, "cast dY columns (dim1)", "mxfp8", shape, cast_flop, timer.time(lambda dy: _as_mx(dy, transposed=True), [(dy,) for dy in dy_p]), path="triton_to_mxfp8_dim1"))
        rows.append(_row(name, "cast X columns (dim1)", "mxfp8", shape, cast_flop, timer.time(lambda x: _as_mx(x, transposed=True), [(x,) for x in x_p]), path="triton_to_mxfp8_dim1"))
        rows.append(_row(name, "cast W columns (dim1)", "mxfp8", shape, cast_flop, timer.time(lambda w: _as_mx(w, transposed=True), [(w,) for w in w_p]), path="triton_to_mxfp8_dim1"))
        if (pm, pn, pk) != (m, n, k):
            rows.append(_row(name, "pad X/W/dY to 128", "bf16", shape, cast_flop,
                             timer.time(lambda x, w, dy: (_pad(x, pm, pk), _pad(w, pn, pk), _pad(dy, pm, pn)), list(zip(xs, ws, dys))),
                             path="F.pad copies", note=f"padded to {pm}x{pn}x{pk}"))
        del x_p, w_p, dy_p
    except Exception as exc:
        rows.append(_fail(name, "backward", "mxfp8", shape, exc, path="mxfp8_linear_backward_fast"))

    # NVFP4 forward exactly as _FastLowPrecisionFunction.forward runs it, then the bare GEMM.
    if not skip_nvfp4:
        try:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, _addmm_nvfp4_dispatch, per_tensor_amax_to_scale

            def nvfp4_forward(x, w):
                input_scale = per_tensor_amax_to_scale(x.detach().abs().amax())
                weight_scale = per_tensor_amax_to_scale(w.detach().abs().amax())
                packed_input = NVFP4Tensor.to_nvfp4(x, per_tensor_scale=input_scale)
                packed_weight = NVFP4Tensor.to_nvfp4(w, per_tensor_scale=weight_scale)
                return _addmm_nvfp4_dispatch(packed_input, packed_weight.t(), None, None)

            rows.append(_row(name, "forward", "nvfp4", shape, flop, timer.time(nvfp4_forward, list(zip(xs, ws))),
                             path="amax + torch-op quantize (x and w) + gemm", note="the path sprint_precision.py executes per microbatch"))
            packed_x = [NVFP4Tensor.to_nvfp4(x, per_tensor_scale=per_tensor_amax_to_scale(x.abs().amax())) for x in xs]
            packed_w = [NVFP4Tensor.to_nvfp4(w, per_tensor_scale=per_tensor_amax_to_scale(w.abs().amax())) for w in ws]
            rows.append(_row(name, "forward", "nvfp4", shape, flop, timer.time(lambda a, b: _addmm_nvfp4_dispatch(a, b.t(), None, None), list(zip(packed_x, packed_w)))))
            del packed_x, packed_w
            rows.append(_row(name, "quantize X (torch ops)", "nvfp4", shape, 0.0,
                             timer.time(lambda x: NVFP4Tensor.to_nvfp4(x, per_tensor_scale=per_tensor_amax_to_scale(x.abs().amax())), [(x,) for x in xs]),
                             path="amax + nvfp4_quantize"))
            rows.append(_row(name, "quantize W (torch ops)", "nvfp4", shape, 0.0,
                             timer.time(lambda w: NVFP4Tensor.to_nvfp4(w, per_tensor_scale=per_tensor_amax_to_scale(w.abs().amax())), [(w,) for w in ws]),
                             path="amax + nvfp4_quantize"))
            rows.append(_row(name, "amax W only", "nvfp4", shape, 0.0, timer.time(lambda w: w.abs().amax(), [(w,) for w in ws]), path="abs().amax()"))
            if util.find_spec("mslk") is not None:
                try:
                    rows.append(_row(name, "quantize X (MSLK Triton)", "nvfp4", shape, 0.0,
                                     timer.time(lambda x: NVFP4Tensor.to_nvfp4(x, per_tensor_scale=per_tensor_amax_to_scale(x.abs().amax()), use_triton_kernel=True), [(x,) for x in xs]),
                                     path="amax + mslk_quantize_nvfp4"))
                except Exception as exc:
                    rows.append(_fail(name, "quantize X (MSLK Triton)", "nvfp4", shape, exc, path="mslk"))
            else:
                rows.append({"linear": name, "role": "quantize X (MSLK Triton)", "precision": "nvfp4", "path": "mslk",
                             "shape": shape, "skipped": "mslk package not installed; torchao's use_triton_kernel=True needs it"})
        except Exception as exc:
            rows.append(_fail(name, "forward", "nvfp4", shape, exc, path="nvfp4 forward"))
    del xs, ws, dys
    torch.cuda.empty_cache()
    return rows


def bench_head(vocab: int, d: int, m: int, timer: Timer, rotate: int, device) -> list[dict]:
    rows = []
    shape = {"tokens": m, "vocab": vocab, "hidden": d}
    flop = 2.0 * m * vocab * d
    generator = torch.Generator(device=device).manual_seed(1208)
    hs = [torch.randn((m, d), device=device, dtype=torch.bfloat16, generator=generator) for _ in range(rotate)]
    es = [torch.randn((vocab, d), device=device, dtype=torch.bfloat16, generator=generator) * 0.02 for _ in range(rotate)]
    dls = [torch.randn((m, vocab), device=device, dtype=torch.bfloat16, generator=generator) for _ in range(rotate)]
    try:
        rows.append(_row("head", "logits", "bf16", shape, flop, timer.time(lambda h, e: F.linear(h, e), list(zip(hs, es)))))
        rows.append(_row("head", "dgrad (dhidden)", "bf16", shape, flop, timer.time(lambda dl, e: dl @ e, list(zip(dls, es)))))
        rows.append(_row("head", "wgrad (dembedding)", "bf16", shape, flop, timer.time(lambda dl, h: dl.t() @ h, list(zip(dls, hs)))))
    except Exception as exc:
        rows.append(_fail("head", "bf16 gemms", "bf16", shape, exc))
    try:
        from yylab.train.mxfp8_fast import mxfp8_linear_backward_fast, mxfp8_mm_nt_fast

        rows.append(_row("head", "logits", "mxfp8", shape, flop, timer.time(lambda h, e: mxfp8_mm_nt_fast(h, e, backend="native"), list(zip(hs, es))),
                         path="cast + gemm", note="what an MXFP8 head would cost; no quality claim"))
        rows.append(_row("head", "dgrad+wgrad", "mxfp8", shape, 2 * flop,
                         timer.time(lambda h, e, dl: mxfp8_linear_backward_fast(h, e, dl, backend="native"), list(zip(hs, es, dls))),
                         path="pad + casts + 2 gemms"))
    except Exception as exc:
        rows.append(_fail("head", "mxfp8 gemms", "mxfp8", shape, exc))
    del hs, es, dls
    torch.cuda.empty_cache()
    return rows


def _rollup(rows: list[dict], layers: int, linears: dict[str, tuple[int, int]], m: int) -> dict:
    """Per-microbatch GEMM floor from the isolated timings, for one token count."""
    def pick(linear, role, precision, path_prefix):
        for row in rows:
            if (row.get("linear") == linear and row.get("role") == role and row.get("precision") == precision
                    and row.get("shape", {}).get("tokens") == m and str(row.get("path", "")).startswith(path_prefix) and "median_ms" in row):
                return row["median_ms"]
        return None

    def total(role_paths):
        acc, missing = 0.0, []
        for linear in linears:
            for role, precision, prefix in role_paths:
                value = pick(linear, role, precision, prefix)
                if value is None:
                    missing.append(f"{linear}:{role}:{precision}")
                else:
                    acc += value
        return acc * layers, missing

    sprint_full, missing_full = total([("forward", "nvfp4", "amax"), ("dgrad+wgrad", "mxfp8", "pad")])
    sprint_gemm, missing_gemm = total([("forward", "nvfp4", "gemm only"), ("dgrad", "mxfp8", "gemm only"), ("wgrad", "mxfp8", "gemm only")])
    bf16, missing_bf16 = total([("forward", "bf16", "gemm only"), ("dgrad", "bf16", "gemm only"), ("wgrad", "bf16", "gemm only")])
    head = sum(v for v in (pick("head", r, "bf16", "gemm only") for r in ("logits", "dgrad (dhidden)", "wgrad (dembedding)")) if v is not None)
    return {"tokens": m, "layers": layers,
            "decoder_sprint_cast_plus_gemm_ms": round(sprint_full, 3), "decoder_sprint_gemm_only_ms": round(sprint_gemm, 3),
            "decoder_cast_and_padding_overhead_ms": round(sprint_full - sprint_gemm, 3),
            "decoder_bf16_gemm_only_ms": round(bf16, 3), "head_bf16_gemms_ms": round(head, 3),
            "sprint_floor_per_microbatch_ms": round(sprint_full + head, 3),
            "missing": missing_full + missing_gemm + missing_bf16,
            "note": "Sum over layers x projections of isolated medians; attention, norms, GELU, RoPE, Liger CE, residuals, "
                    "gradient accumulation and launch gaps are not included. Compare with the per-microbatch forward+backward device time from the step profile."}


def _markdown(report: dict) -> str:
    lines = [f"# Sprint GEMM microbenchmarks", "",
             f"{report['hardware']}; torch {report['runtime_versions'].get('torch')}, torchao {report['runtime_versions'].get('torchao')}; "
             f"{report['iters']} timed iterations after {report['warmup']} warmup calls, {report['rotate']} rotating operand copies; medians.", "",
             "Fraction of peak uses the RTX 5070 whitepaper dense rates (BF16 61.7, FP8 246.9, FP4 493.9 TFLOP/s). "
             "MXFP8 rows also show the fraction of the 123.5 TFLOP/s FP8-with-FP32-accumulate reference.", ""]
    for m, rollup in report["rollup"].items():
        lines += [f"## {m} tokens per microbatch", "",
                  f"Decoder floor as Sprint runs it (NVFP4 forward cast+GEMM, MXFP8 backward pad+casts+GEMMs): **{rollup['decoder_sprint_cast_plus_gemm_ms']:.1f} ms**; "
                  f"bare GEMMs only {rollup['decoder_sprint_gemm_only_ms']:.1f} ms, so casts/padding cost {rollup['decoder_cast_and_padding_overhead_ms']:.1f} ms; "
                  f"BF16 GEMMs would be {rollup['decoder_bf16_gemm_only_ms']:.1f} ms; tied head in BF16 {rollup['head_bf16_gemms_ms']:.1f} ms; "
                  f"Sprint GEMM+cast floor per microbatch **{rollup['sprint_floor_per_microbatch_ms']:.1f} ms**." + (f" Missing rows: {rollup['missing']}" if rollup["missing"] else ""), "",
                  "| Linear | Role | Precision | Path | Median ms | Min ms | GFLOP | TFLOP/s | Of peak | Note |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in report["rows"]:
            if row.get("shape", {}).get("tokens") != m:
                continue
            if "error" in row:
                lines.append(f"| {row['linear']} | {row['role']} | {row['precision']} | {row['path']} | error | | | | | `{row['error'][:80]}` |")
                continue
            if "skipped" in row:
                lines.append(f"| {row['linear']} | {row['role']} | {row['precision']} | {row['path']} | skipped | | | | | {row['skipped']} |")
                continue
            peak = f"{row['fraction_of_peak']:.0%}" if row.get("fraction_of_peak") is not None else ""
            if row.get("fraction_of_fp32acc_reference") is not None:
                peak += f" ({row['fraction_of_fp32acc_reference']:.0%} of 123.5)"
            lines.append(f"| {row['linear']} | {row['role']} | {row['precision']} | {row['path']} | {row['median_ms']:.3f} | {row['min_ms']:.3f} | "
                         f"{row['gflop']:.1f} | {row['achieved_tflops']:.0f} | {peak} | {row.get('note', '')} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="official Sprint recipe; supplies hidden/intermediate/vocab/seq_len/microbatch")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--tokens", help="comma-separated tokens per microbatch, e.g. 2048,4096,8192 (default: recipe microbatch x seq_len)")
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rotate", type=int, default=3, help="operand copies rotated between iterations to reduce L2 residency")
    parser.add_argument("--skip-nvfp4", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint-gemm-bench"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    cfg = json.loads(args.config.read_text("utf-8")) if args.config else {}
    model_config = cfg.get("model_config", {})
    d = args.hidden or model_config.get("hidden_size")
    f = args.intermediate or model_config.get("intermediate_size")
    vocab = model_config.get("vocab_size", args.vocab) if not args.hidden else args.vocab
    layers = args.layers or model_config.get("num_hidden_layers", 1)
    if not d or not f:
        parser.error("--config or --hidden/--intermediate are required")
    seq_len = int(cfg.get("seq_len", 512))
    micro = int(cfg.get("batch_sizes", [8])[0])
    tokens = [int(v) for v in args.tokens.split(",")] if args.tokens else [micro * seq_len]
    linears = {"qkv": (3 * d, d), "o": (d, d), "up": (f, d), "down": (d, f)}
    rd = Path(args.output)
    rd.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "yylab-sprint-gemm-bench-v1", "experimental": True, "diagnostic_only": True,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "hardware": torch.cuda.get_device_name(device), "capability": list(torch.cuda.get_device_capability(device)),
        "host_processor": platform.processor(),
        "runtime_versions": {"torch": str(torch.__version__), "torchao": _version("torchao"), "triton": _version("triton", "triton-windows")},
        "source_sha256": {"bench_sprint_gemms.py": _sha256(Path(__file__).resolve()),
                          **{p: _sha256(ROOT / p) for p in ("yylab/train/mxfp8.py", "yylab/train/mxfp8_fast.py", "yylab/train/sprint_precision.py") if (ROOT / p).is_file()}},
        "config_path": str(args.config) if args.config else None, "config_sha256": _sha256(args.config) if args.config else None,
        "shapes": {"hidden": d, "intermediate": f, "vocab": vocab, "layers": layers, "linears": linears, "tokens": tokens},
        "iters": args.iters, "warmup": args.warmup, "rotate": args.rotate, "peak_tflops_reference": PEAK_TFLOPS,
        "rows": [], "rollup": {}, "status": "running",
    }
    timer = Timer(args.iters, args.warmup)
    write = lambda: (rd / "gemm_bench.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # noqa: E731
    write()
    for m in tokens:
        for name, (n, k) in linears.items():
            print(f"{m} tokens: {name} [{n} x {k}]", flush=True)
            report["rows"].extend(bench_linear(name, n, k, m, timer, args.rotate, device, skip_nvfp4=args.skip_nvfp4))
            write()
        print(f"{m} tokens: head [{vocab} x {d}]", flush=True)
        report["rows"].extend(bench_head(vocab, d, m, timer, args.rotate, device))
        report["rollup"][str(m)] = _rollup(report["rows"], layers, linears, m)
        write()
    report["status"] = "ok"
    write()
    (rd / "gemm_bench.md").write_text(_markdown(report), encoding="utf-8")
    for m, rollup in report["rollup"].items():
        print(f"{m} tokens/microbatch: Sprint GEMM+cast floor {rollup['sprint_floor_per_microbatch_ms']:.1f} ms "
              f"(bare GEMMs {rollup['decoder_sprint_gemm_only_ms'] + rollup['head_bf16_gemms_ms']:.1f} ms, casts/padding {rollup['decoder_cast_and_padding_overhead_ms']:.1f} ms)")
    print(f"wrote {rd / 'gemm_bench.md'}")


if __name__ == "__main__":
    main()
