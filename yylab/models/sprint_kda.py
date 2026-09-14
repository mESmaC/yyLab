"""Kimi-inspired KDA for Sprint, with independent causal sequence segments.

CUDA uses FLA's chunked training kernels. CPU uses a differentiable reference.
There is no persistent recurrent cache: generation recomputes the full prefix.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


def validate_kda_options(attention_type, backend, disable_recompute):
    if not isinstance(attention_type, str) or attention_type not in {"sdpa", "kda"}:
        raise ValueError("attention_type must be sdpa or kda")
    if not isinstance(backend, str) or backend not in {"auto", "fla", "reference"}:
        raise ValueError("kda_backend must be auto, fla or reference")
    if not isinstance(disable_recompute, bool):
        raise ValueError("kda_disable_recompute must be a boolean")


def kda_parameter_count(config):
    if config.attention_type != "kda":
        return 0
    d, h, r = config.hidden_size, config.num_attention_heads, config.head_dim
    return (config.num_hidden_layers // 4 * 3) * (4 * d * r + d * h + 12 * d + 2 * d + h + r)


class KDASegmentLayout(NamedTuple):
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    lengths: tuple[int, ...]


@torch.compiler.disable
def prepare_kda_layout(attention_mask, position_ids, batch, length, heads, device):
    """Validate masks once and share segment boundaries across all KDA layers.

2D 0/1 masks support left/right padding and reset at an internal padding gap.
4D masks must describe independent contiguous causal segments. Finite attention
biases, sliding windows, and head-dependent visibility cannot represent a KDA
recurrence and are rejected. Explicit non-increasing positions also reset both
the convolution and recurrent state, including the hybrid's SDPA layers.
"""
    valid = torch.ones(batch, length, dtype=torch.bool)
    allowed = None
    if attention_mask is not None:
        if attention_mask.ndim not in (2, 4) or attention_mask.shape[0] not in (1, batch):
            raise ValueError("KDA attention_mask must be a 2D padding or 4D segment-causal mask with matching batch")
        source = attention_mask.detach().cpu()
        if source.ndim == 2:
            if source.shape[-1] != length or not ((source == 0) | (source == 1)).all():
                raise ValueError("KDA 2D attention_mask must contain only 0/1 and cover the full sequence")
            valid = source.bool().expand(batch, length)
        else:
            if source.shape[1] not in (1, heads) or source.shape[2] not in (1, length) or source.shape[3] != length:
                raise ValueError("KDA 4D attention_mask must match heads, query length and full sequence")
            if source.dtype != torch.bool:
                if not source.is_floating_point() or not ((source == 0) | (source <= -10000)).all():
                    raise ValueError("KDA supports boolean masks or 0/negative-infinity masks, not additive attention biases")
                source = source == 0
            source = source.expand(batch, source.shape[1], length, length)
            if source.shape[1] > 1 and not (source == source[:, :1]).all():
                raise ValueError("KDA cannot represent head-dependent attention masks")
            allowed = source[:, 0] & torch.ones(length, length, dtype=torch.bool).tril()
            valid = allowed.diagonal(dim1=-2, dim2=-1)
    positions = None
    if position_ids is not None:
        if position_ids.ndim != 2 or position_ids.shape[0] not in (1, batch) or position_ids.shape[1] != length:
            raise ValueError("KDA position_ids must match batch and full sequence length")
        positions = position_ids.detach().cpu().expand(batch, length)
        if positions.is_floating_point() or positions.dtype == torch.bool or (positions < 0).any():
            raise ValueError("KDA position_ids must be nonnegative integers")
    starts = valid & ~F.pad(valid[:, :-1], (1, 0), value=False)
    if positions is not None:
        starts[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (positions[:, 1:] <= positions[:, :-1])
    if allowed is not None:
        starts[:, 1:] |= valid[:, 1:] & valid[:, :-1] & ~allowed.diagonal(offset=-1, dim1=-2, dim2=-1)
    uninterrupted = bool(valid.all() and starts.sum() == batch)
    if allowed is None and uninterrupted:
        return None, None  # Common all-one training masks need no packing or dense mask.
    segments = starts.flatten().long().cumsum(0).reshape(batch, length).sub(1).masked_fill(~valid, -1)
    mask = ((segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] >= 0)
            & torch.ones(length, length, dtype=torch.bool).tril())
    if allowed is not None and not torch.equal(allowed[valid], mask[valid]):
        raise ValueError("KDA 4D attention_mask must describe independent contiguous causal segments")
    if uninterrupted:
        return None, None
    indices = valid.flatten().nonzero().flatten()
    cumulative = torch.cat([starts.flatten()[indices].nonzero().flatten(), torch.tensor([indices.numel()])])
    lengths = tuple(cumulative.diff().tolist())
    layout = KDASegmentLayout(indices.to(device), cumulative.to(device), cumulative, lengths)
    return layout, mask[:, None].to(device)


def reference_kda(q, k, v, gate, beta, A_log, dt_bias):
    """Fresh-state FP32/64 recurrence, including under a caller's autocast.

    Casting operands to FP32 alone is insufficient: autocast can downcast
    einsum's contractions again. Keep those contractions explicitly outside
    autocast while preserving FLA's input-dtype q/k normalization boundary.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _reference_kda_full_precision(q, k, v, gate, beta, A_log, dt_bias)


def _reference_kda_full_precision(q, k, v, gate, beta, A_log, dt_bias):
    output_dtype = q.dtype
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = [value.to(dtype) for value in (q, k, v)]
    q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).to(output_dtype).to(dtype)
    k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).to(output_dtype).to(dtype)
    batch, length, heads, head_dim = q.shape
    decay = torch.exp(-A_log.to(dtype).exp().view(1, 1, heads, 1)
                      * F.softplus(gate.to(dtype) + dt_bias.to(dtype).view(1, 1, heads, head_dim)))
    beta = beta.to(dtype).sigmoid()
    state = q.new_zeros(batch, heads, head_dim, v.shape[-1])
    outputs = []
    for index in range(length):
        state = state * decay[:, index, :, :, None]
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, index], state)
        residual = (v[:, index] - prediction) * beta[:, index, :, None]
        state = state + k[:, index, :, :, None] * residual[:, :, None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, index], state) * head_dim ** -0.5)
    return torch.stack(outputs, dim=1).to(output_dtype)


class SprintKDAShortConvolution(nn.Conv1d):
    def __init__(self, channels):
        super().__init__(channels, channels, 4, groups=channels, bias=False)

    def forward(self, x, *, reference=False, layout=None):
        if x.is_cuda and not reference:
            from fla.modules.conv.causal_conv1d import causal_conv1d
            output, _ = causal_conv1d(
                x=x, weight=self.weight[:, 0, :], bias=None,
                initial_state=None, output_final_state=False, activation="silu", backend="triton",
                cu_seqlens=layout.cu_seqlens if layout else None,
                cu_seqlens_cpu=layout.cu_seqlens_cpu if layout else None,
            )
            return output
        def convolve(value):
            # FLA accumulates taps and applies SiLU before its single output
            # cast. Autocast conv1d would round weights and the preactivation.
            dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
            with torch.autocast(device_type=value.device.type, enabled=False):
                padded = F.pad(value.to(dtype).transpose(1, 2), (3, 0))
                output = F.silu(F.conv1d(padded, self.weight.to(dtype), groups=self.groups))
            return output.transpose(1, 2).to(value.dtype)
        return torch.cat([convolve(value) for value in x.split(layout.lengths, dim=1)], dim=1) if layout else convolve(x)


class SprintKDAGatedRMSNorm(nn.Module):
    def __init__(self, head_dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x, gate, *, reference=False):
        if x.is_cuda and not reference:
            from fla.modules.fused_norm_gate import rms_norm_gated
            return rms_norm_gated(x, gate, self.weight, None, "sigmoid", eps=self.eps)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        value = x.to(dtype)
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.to(dtype) * gate.to(dtype).sigmoid()).to(x.dtype)


class SprintKDAAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.backend = config.kda_backend
        self.disable_recompute = config.kda_disable_recompute
        self.heads, self.head_dim = config.num_attention_heads, config.head_dim
        d, r = config.hidden_size, config.head_dim
        self.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        self.qkv_conv = SprintKDAShortConvolution(3 * d)
        self.f_proj = nn.Sequential(nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False))
        self.b_proj = nn.Linear(d, self.heads, bias=False)
        self.g_proj = nn.Sequential(nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=True))
        self.o_norm = SprintKDAGatedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(d, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.reset_gate_parameters()

    def reset_gate_parameters(self):
        with torch.no_grad():
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())
            dt = torch.exp(torch.rand_like(self.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _apply(self, fn, recurse=True):
        # Exclude stable gates before conversion so their original FP32 values
        # are never first rounded to BF16 and subsequently widened.
        stable = {name: self._parameters.pop(name) for name in ("A_log", "dt_bias")}
        try:
            super()._apply(fn, recurse=recurse)
            for parameter in stable.values():
                destination = fn(parameter)
                with torch.no_grad():
                    parameter.data = parameter.data.to(device=destination.device, dtype=torch.float32)
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.data.to(device=destination.device, dtype=torch.float32)
        finally:
            self._parameters.update(stable)
        return self

    def forward(self, x, cos=None, sin=None, mask=None, is_causal=True, past_key_values=None, *, layout=None):
        if past_key_values is not None:
            raise ValueError("Sprint KDA does not accept external caches; use full-prefix generation")
        if mask is not None and layout is None:
            raise ValueError("Sprint KDA masks require a validated shared segment layout")
        reference = self.backend == "reference" or (self.backend == "auto" and not x.is_cuda)
        if self.backend == "fla" and not x.is_cuda:
            raise ValueError("FLA KDA requires CUDA; use kda_backend='auto' or 'reference' on CPU")
        batch, length, width = x.shape
        if layout is not None:
            x = x.reshape(-1, width).index_select(0, layout.indices).unsqueeze(0)
            if not layout.lengths:
                return (x.sum() + sum(parameter.sum() * 0 for parameter in self.parameters())).to(x.dtype) * torch.zeros(
                    batch, length, width, device=x.device, dtype=x.dtype)
        packed_batch, packed_length, _ = x.shape
        qkv = self.qkv_conv(self.qkv_proj(x), reference=reference, layout=layout)
        q, k, v = qkv.reshape(packed_batch, packed_length, 3, self.heads, self.head_dim).unbind(2)
        forget = self.f_proj(x).reshape(packed_batch, packed_length, self.heads, self.head_dim)
        beta = self.b_proj(x)
        if reference:
            if layout:
                values = [value.split(layout.lengths, dim=1) for value in (q, k, v, forget, beta)]
                output = torch.cat([reference_kda(*part, self.A_log, self.dt_bias) for part in zip(*values)], dim=1)
            else:
                output = reference_kda(q, k, v, forget, beta, self.A_log, self.dt_bias)
        else:
            from fla.ops.kda import chunk_kda
            output, _ = chunk_kda(
                q=q, k=k, v=v, g=forget, beta=beta, A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
                safe_gate=False, lower_bound=None, disable_recompute=self.disable_recompute,
                cu_seqlens=layout.cu_seqlens if layout else None,
                cu_seqlens_cpu=layout.cu_seqlens_cpu if layout else None,
            )
        gate = self.g_proj(x).reshape(packed_batch, packed_length, self.heads, self.head_dim)
        output = self.o_proj(self.o_norm(output, gate, reference=reference).reshape(packed_batch, packed_length, width))
        if layout is not None:
            output = output.new_zeros(batch * length, width).index_copy(0, layout.indices, output[0]).reshape(batch, length, width)
        return output
