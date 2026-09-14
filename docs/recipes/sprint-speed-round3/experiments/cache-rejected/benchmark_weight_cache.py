"""Run the real precision benchmark with an explicitly labeled update cache.

No production code or checkpoint semantics change. Refresh/host validation are
part of the official timed train_step; initialization and compilation are not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEY = "experimental_weight_cache"
REQUEST = "nvfp4_forward_per_update"


def run(cfg, output):
    if cfg.get(KEY) != REQUEST:
        raise ValueError(f"require explicit {KEY}='{REQUEST}'")
    if cfg.get("precisions") != ["nvfp4_mxfp8_fast"]:
        raise ValueError("this cache candidate supports only the nvfp4_mxfp8_fast numerical recipe")
    if any(mode not in {"eager", "default"} for mode in cfg.get("compile_modes", ["eager"])):
        raise ValueError("initial cache experiment supports eager/default model compile modes only")
    if cfg.get("gradient_checkpointing", False):
        raise ValueError("initial cache experiment excludes activation checkpointing")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from yylab.train import benchmark_sprint_precision as benchmark, sprint
    from weight_cache import cached_engine_class, VARIANT

    original_engine, original_measure = sprint.SprintStepEngine, benchmark._measure_precision_case
    original_paths, original_write = benchmark.PROVENANCE_SOURCE_PATHS, benchmark.write_json
    captured = {}

    def measured(*args, **kwargs):
        captured.clear()
        row = original_measure(*args, **kwargs)
        row.update(benchmark_variant=VARIANT, numerical_precision_recipe="nvfp4_mxfp8_fast")
        row.setdefault("runtime", {})["experimental_weight_cache"] = dict(captured)
        row.setdefault("precision_metadata", {}).update(
            benchmark_variant=VARIANT, numerical_precision_recipe="nvfp4_mxfp8_fast",
            forward_weight_quantization="once per optimizer update, compiled refresh included in step")
        return row

    def write(path, value):
        if value.get("schema") == "yylab-sprint-precision-benchmark-v1":
            value["benchmark_variant"] = VARIANT
            value["cache_measurement"] = "Every timed update includes forward-weight cache refresh plus host validity checks; inputs/dY are freshly quantized each microbatch. Not a baseline receipt."
        return original_write(path, value)

    sprint.SprintStepEngine = cached_engine_class(original_engine, captured)
    benchmark._measure_precision_case = measured
    benchmark.write_json = write
    benchmark.PROVENANCE_SOURCE_PATHS = (*original_paths, *(
        (Path(__file__).resolve().parent / name).relative_to(root).as_posix()
        for name in ("benchmark_weight_cache.py", "weight_cache.py")))
    try:
        return benchmark.run_sprint_precision_benchmark(cfg, Path(output))
    finally:
        sprint.SprintStepEngine, benchmark._measure_precision_case = original_engine, original_measure
        benchmark.PROVENANCE_SOURCE_PATHS, benchmark.write_json = original_paths, original_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(json.loads(args.config.read_text(encoding="utf-8")), args.output)
    print(json.dumps({key: report.get(key) for key in ("status", "benchmark_variant", "fastest_measured")}))
    raise SystemExit(0 if report["status"] == "ok" else 1)
