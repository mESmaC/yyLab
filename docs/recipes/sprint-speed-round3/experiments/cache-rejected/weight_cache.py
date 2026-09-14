"""Disposable per-update NVFP4 weight cache; never a checkpoint format.

Only forward weight preparation is cached. The BF16 parameters, per-microbatch
activation quantization and existing fast MXFP8 backward recipe are unchanged.
All cache validity/version checks run in the outer Python training-step scope.
"""
from __future__ import annotations

import hashlib
import types
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torchao.prototype.mx_formats import nvfp4_tensor as provider
from torchao.prototype.mx_formats import utils as provider_utils

from yylab.train.mxfp8_fast import mxfp8_linear_backward_fast
from yylab.train.sprint_precision import FastLowPrecisionLinear

VARIANT = "nvfp4_mxfp8_fast_weight_cache_per_update"
BUFFER_PREFIX = "_update_nvfp4_"


def _refresh_weight(weight, codes, scales, tensor_scale):
    # Identical provider operations to the baseline, including the BF16 amax
    # before FP32 global scaling. Block reordering is also invariant per update.
    maximum_scale = provider.per_tensor_amax_to_scale(weight.detach().abs().amax())
    block_scales, packed = provider.nvfp4_quantize(weight.detach(), 16, maximum_scale)
    blocked = provider_utils.to_blocked(block_scales.view(weight.shape[0], weight.shape[1] // 16))
    codes.copy_(packed)
    scales.copy_(blocked.view_as(scales))
    tensor_scale.copy_(maximum_scale)


class _CachedForward(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, inputs, weight, bias, codes, scales, tensor_scale):
        ctx.save_for_backward(inputs, weight)
        ctx.has_bias = bias is not None
        flat = inputs.reshape(-1, inputs.shape[-1])
        input_scale = provider.per_tensor_amax_to_scale(flat.detach().abs().amax())
        packed_input = provider.NVFP4Tensor.to_nvfp4(flat, per_tensor_scale=input_scale)
        packed_weight = provider.NVFP4Tensor(
            codes, scales, 16, weight.dtype, per_tensor_scale=tensor_scale, is_swizzled_scales=True)
        output = provider._addmm_nvfp4_dispatch(packed_input, packed_weight.t(), None, bias)
        return output.reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        grad_input, grad_weight = mxfp8_linear_backward_fast(inputs, weight, grad_output, backend="native")
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None, None


class CachedForwardLinear(nn.Linear):
    def forward(self, inputs):
        # This Boolean is always True for a compiled train-step invocation;
        # no tensor version or changing generation counter enters Dynamo.
        if not self._update_nvfp4_active:
            raise RuntimeError("cached NVFP4 forward requires an active update scope")
        return _CachedForward.apply(inputs, self.weight, self.bias, self._update_nvfp4_codes,
                                    self._update_nvfp4_scales, self._update_nvfp4_tensor_scale)


class ForwardWeightCache:
    """Fixed buffers refreshed on the optimizer's caller stream once per update.

    Install AFTER SprintStepEngine places/casts the model, but BEFORE its lazy
    compiled model executes. Load with assign=False outside the update scope;
    the next scope always requantizes. Parameter replacement/device moves and
    concurrent use are rejected. External .data writes bypass PyTorch version
    tracking and are outside this narrowly scoped benchmark contract.
    """
    def __init__(self, model, *, compile_refresh=True):
        self.targets = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, FastLowPrecisionLinear)]
        if not self.targets or any(module.forward_format != "nvfp4" for _, module in self.targets):
            raise ValueError("cache requires the already converted nvfp4_mxfp8_fast decoder")
        devices = {module.weight.device for _, module in self.targets}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("NVFP4 update cache requires one CUDA device")
        self.device = next(iter(devices))
        if any(module.weight.dtype != torch.bfloat16 or not module.weight.is_contiguous()
               or module.in_features % 64 or module.out_features % 64 for _, module in self.targets):
            raise ValueError("cache requires contiguous BF16 projection weights with dimensions divisible by64")
        self.parameter_ids = tuple(id(module.weight) for _, module in self.targets)
        self.active, self.refresh_scopes = False, 0
        self._versions = ()
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self._refresh = {}
        for name, module in self.targets:
            n, k = module.weight.shape
            shapes = {"codes": ((n, k // 2), torch.uint8),
                      "scales": (provider_utils.hp_data_dims_to_swizzled_scale_dims_nvfp4(n, k), torch.float8_e4m3fn),
                      "tensor_scale": ((), torch.float32)}
            for suffix, (shape, dtype) in shapes.items():
                module.register_buffer(BUFFER_PREFIX + suffix,
                                       torch.empty(shape, dtype=dtype, device=self.device), persistent=False)
            module._update_nvfp4_active = False
            module.__class__ = CachedForwardLinear
            key = (n, k)
            if key not in self._refresh:
                # Separate code objects keep Dynamo's guard/recompile limit
                # from conflating the handful of distinct matrix shapes.
                function = types.FunctionType(
                    _refresh_weight.__code__.replace(co_name=f"refresh_nvfp4_{n}_{k}"),
                    _refresh_weight.__globals__, f"refresh_nvfp4_{n}_{k}")
                self._refresh[key] = (torch.compile(function, fullgraph=True, dynamic=False,
                    options={"triton.cudagraphs": False}) if compile_refresh else function)
        self.compile_refresh = bool(compile_refresh)
        self.load_hook = model.register_load_state_dict_pre_hook(self._before_load)

    def _before_load(self, *args):
        was_active = self.active
        self.invalidate()
        if was_active:
            raise RuntimeError("load_state_dict is forbidden during a cached optimizer update")

    def _check_storage(self):
        if tuple(id(module.weight) for _, module in self.targets) != self.parameter_ids:
            raise RuntimeError("cached parameter identity changed; rebuild the disposable model/optimizer")
        for _, module in self.targets:
            if (module.weight.device != self.device or module.weight.dtype != torch.bfloat16
                    or module._update_nvfp4_codes.dtype != torch.uint8
                    or module._update_nvfp4_scales.dtype != torch.float8_e4m3fn
                    or module._update_nvfp4_tensor_scale.dtype != torch.float32
                    or any(buffer.device != self.device for buffer in (
                        module._update_nvfp4_codes, module._update_nvfp4_scales, module._update_nvfp4_tensor_scale))):
                raise RuntimeError("cached model storage moved or changed dtype; rebuild the cache")
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("cached training must use its original optimizer caller stream")

    def validate(self):
        """Call outside compiled code immediately before/after each microbatch."""
        self._check_storage()
        if not self.active or not all(module._update_nvfp4_active for _, module in self.targets):
            raise RuntimeError("cached update scope is inactive or invalidated")
        if tuple(module.weight._version for _, module in self.targets) != self._versions:
            self.invalidate()
            raise RuntimeError("weight mutation invalidated the active NVFP4 update cache")

    def invalidate(self):
        self.active = False
        self._versions = ()
        for _, module in self.targets:
            module._update_nvfp4_active = False

    @contextmanager
    def update_scope(self):
        if self.active:
            raise RuntimeError("nested cached optimizer updates are not supported")
        self._check_storage()
        try:
            # Sprint CPU Adam ends each update with current_stream.wait_stream
            # on its H2D writeback stream. Refresh on this same caller stream
            # therefore observes the complete latest BF16 weights without a
            # host synchronize. Each refresh is inside train_step/timing.
            with torch.no_grad():
                for _, module in self.targets:
                    self._refresh[tuple(module.weight.shape)](
                        module.weight, module._update_nvfp4_codes, module._update_nvfp4_scales,
                        module._update_nvfp4_tensor_scale)
            self._versions = tuple(module.weight._version for _, module in self.targets)
            self.active = True
            for _, module in self.targets:
                module._update_nvfp4_active = True
            self.refresh_scopes += 1
            yield self
        finally:
            self.invalidate()

    def metadata(self):
        buffers = [buffer for _, module in self.targets for name, buffer in module.named_buffers()
                   if name.startswith(BUFFER_PREFIX)]
        return {"variant": VARIANT, "numerical_recipe": "nvfp4_mxfp8_fast",
                "cached": "forward weight FP4 codes, blocked FP8 scales and FP32 global scale",
                "cache_bytes": sum(buffer.numel() * buffer.element_size() for buffer in buffers),
                "converted_modules": [name for name, _ in self.targets],
                "refresh_scopes": self.refresh_scopes, "compile_refresh": self.compile_refresh,
                "refresh_in_measured_train_step": True, "serialized_cache": False,
                "weight_column_cache": False, "parameters_and_ste_unchanged": True,
                "validity": "Host checks between microbatches; same-stream writeback ordering; unconditional refresh each step; finally invalidation. No external .data writes or concurrent use.",
                "provider_sha256": {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                    for module in (provider, provider_utils)}}


def cached_engine_class(base_engine, captured=None):
    """Share the exact benchmark scope implementation with the parity tests."""
    captured = {} if captured is None else captured

    class CacheEngine(base_engine):
        def __init__(self, model, step_cfg, device="cuda"):
            super().__init__(model, step_cfg, device=device)
            self.weight_cache = ForwardWeightCache(self.model)
            captured.update(self.weight_cache.metadata())

        def _device_batches(self, batches):
            for batch in super()._device_batches(batches):
                self.weight_cache.validate()
                yield batch
                self.weight_cache.validate()

        def train_step(self, cpu_batches, *, step=0):
            try:
                with self.weight_cache.update_scope():
                    return super().train_step(cpu_batches, step=step)
            finally:
                captured["refresh_scopes"] = self.weight_cache.refresh_scopes

    return CacheEngine
