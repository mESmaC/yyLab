"""Tiny CUDA first-difference diagnostic; not a speed benchmark or acceptance test.

Run only when the root agent allocates a GPU slot. No candidate code is changed.
Compares identical initial models, eager vs compiled weight refresh, payloads,
layer activations, and 1D/2D blocked-scale reconstruction on identical inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("diagnostic output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "Tiny diagnostic; no optimizer update or throughput claim.",
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("diagnose_weight_cache.py", "weight_cache.py")}, "probes": []}

    def write():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    write()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from weight_cache import ForwardWeightCache, CachedForwardLinear, provider, provider_utils
        from yylab.models.sprint import SprintConfig, SprintForCausalLM
        from yylab.train.sprint_precision import FastLowPrecisionLinear, apply_sprint_precision

        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        report["torch_version"] = str(torch.__version__)

        def make_model():
            torch.manual_seed(3802)
            model = SprintForCausalLM(SprintConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                num_hidden_layers=1, num_attention_heads=1, max_position_embeddings=128, loss_backend="torch"))
            model.to(device="cuda", dtype=torch.bfloat16)
            apply_sprint_precision(model, "nvfp4_mxfp8_fast", "cuda")
            return model

        def difference(actual, expected):
            a, b = actual.detach().double(), expected.detach().double()
            diff = a - b
            return {"shape": list(a.shape), "different": int((actual != expected).sum().item()),
                    "elements": a.numel(), "max_abs": float(diff.abs().max().item()),
                    "max_ref": float(b.abs().max().item()),
                    "relative_l2": float((diff.norm() / b.norm().clamp_min(1e-30)).item())}

        def payload(weight):
            scale = provider.per_tensor_amax_to_scale(weight.detach().abs().amax())
            packed = provider.NVFP4Tensor.to_nvfp4(weight, per_tensor_scale=scale)
            return packed.qdata, provider_utils.to_blocked(packed.scale), scale

        def hooks(model):
            captured, handles = {}, []
            for name, module in model.named_modules():
                if isinstance(module, (FastLowPrecisionLinear, CachedForwardLinear)) or name.endswith("layernorm") or name == "norm":
                    def hook(_module, inputs, output, key=name):
                        captured[key] = (inputs[0].detach().clone(), output.detach().clone())
                    handles.append(module.register_forward_hook(hook))
            return captured, handles

        for compile_refresh in (False, True):
            reference, candidate = make_model(), make_model()
            probe = {"compile_refresh": compile_refresh, "initial_parameter_differences": [],
                     "payloads": {}, "microbatches": [], "identical_input_linear_dispatch": {}}
            report["probes"].append(probe)
            for (name, left), (other, right) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
                if name != other or not torch.equal(left, right):
                    probe["initial_parameter_differences"].append(name)
            cache = ForwardWeightCache(candidate, compile_refresh=compile_refresh)
            ref_capture, ref_hooks = hooks(reference)
            actual_capture, actual_hooks = hooks(candidate)
            generator = torch.Generator(device="cpu").manual_seed(3900)
            batches = [torch.randint(256, (2, 128), generator=generator).to("cuda") for _ in range(2)]
            compiled_payload = torch.compile(payload, fullgraph=True, dynamic=False,
                                              options={"triton.cudagraphs": False})
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                with cache.update_scope():
                    for name, module in cache.targets:
                        eager = payload(module.weight)
                        compiled = compiled_payload(module.weight)
                        cached = (module._update_nvfp4_codes, module._update_nvfp4_scales.flatten(),
                                  module._update_nvfp4_tensor_scale)
                        probe["payloads"][name] = {}
                        for index, label in enumerate(("codes", "blocked_scales", "global_scale")):
                            actual, expect, comp = cached[index], eager[index], compiled[index]
                            if index == 1:
                                actual, expect, comp = (value.view(torch.uint8) for value in (actual, expect, comp))
                            probe["payloads"][name][label] = {
                                "cache_vs_eager": difference(actual, expect),
                                "cache_vs_compiled": difference(actual, comp)}
                    for index, ids in enumerate(batches):
                        expected = reference(input_ids=ids, labels=ids, use_cache=False).loss
                        actual = candidate(input_ids=ids, labels=ids, use_cache=False).loss
                        row = {"index": index, "loss": difference(actual, expected), "stages": {}}
                        for name, (ref_input, ref_output) in ref_capture.items():
                            actual_input, actual_output = actual_capture[name]
                            row["stages"][name] = {"input": difference(actual_input, ref_input),
                                                   "output": difference(actual_output, ref_output)}
                        row["first_output_difference"] = next((name for name, value in row["stages"].items()
                                                               if value["output"]["different"]), None)
                        probe["microbatches"].append(row)
                    # Both reconstructions consume exactly the same fresh
                    # input payload and cached weight bytes; only scale rank
                    # and the provider's preblocked flag differ from baseline.
                    for name, module in cache.targets:
                        flat = ref_capture[name][0].reshape(-1, module.in_features)
                        scale = provider.per_tensor_amax_to_scale(flat.abs().amax())
                        inputs = provider.NVFP4Tensor.to_nvfp4(flat, per_tensor_scale=scale)
                        weight_scale = provider.per_tensor_amax_to_scale(module.weight.abs().amax())
                        fresh = provider.NVFP4Tensor.to_nvfp4(module.weight, per_tensor_scale=weight_scale)
                        expected = provider._addmm_nvfp4_dispatch(inputs, fresh.t(), None, module.bias)
                        comparisons = {}
                        for label, scales in (("cached_2d", module._update_nvfp4_scales),
                                              ("cached_flat", module._update_nvfp4_scales.flatten())):
                            packed = provider.NVFP4Tensor(module._update_nvfp4_codes, scales, 16, torch.bfloat16,
                                per_tensor_scale=module._update_nvfp4_tensor_scale, is_swizzled_scales=True)
                            result = provider._addmm_nvfp4_dispatch(inputs, packed.t(), None, module.bias)
                            comparisons[label] = difference(result, expected)
                        probe["identical_input_linear_dispatch"][name] = comparisons
            for handle in [*ref_hooks, *actual_hooks]:
                handle.remove()
            del cache, reference, candidate, ref_capture, actual_capture
            torch.cuda.synchronize()
            write()
        report["status"] = "ok"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        write()
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
