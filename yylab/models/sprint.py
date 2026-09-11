"""Sprint: a fixed-depth dense decoder built for high-throughput training.

No recurrent passes, expert dispatch, quantization simulation, or auxiliary heads.
Packed QKV, two-projection GELU, tied embeddings, SDPA and a loss-only forward
keep the training graph small. Shape/batch choices still require measurement.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


@torch.compiler.disable
def _liger_linear_loss(hidden, weight, targets):
    # Liger 0.8's custom backward uses out_dtype addmm and a host target count.
    # Keep its fused kernel opaque to Dynamo; the decoder remains compiled.
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

    # Normalize before BF16 gradient GEMMs, matching ordinary mean CE's
    # rounding order. Scaling a sum-loss afterward rounds different gradients.
    return liger_fused_linear_cross_entropy(hidden, weight, targets, reduction="mean")


class SprintConfig(PretrainedConfig):
    model_type = "sprint"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, vocab_size=8192, hidden_size=512, intermediate_size=None,
        num_hidden_layers=8, num_attention_heads=8, max_position_embeddings=2048,
        tie_word_embeddings=True, mlp_type="gelu", rope_theta=10000.0,
        rms_norm_eps=1e-6, initializer_range=0.02, loss_backend="auto",
        loss_chunk_size=1024, use_cache=True, **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads if num_attention_heads else 0
        self.max_position_embeddings = max_position_embeddings
        self.mlp_type = mlp_type
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.loss_backend = loss_backend
        self.loss_chunk_size = loss_chunk_size
        self.use_cache = use_cache
        self.architectures = ["SprintForCausalLM"]
        self.validate()

    def validate(self):
        for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "max_position_embeddings", "loss_chunk_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.hidden_size % self.num_attention_heads or self.head_dim % 2:
            raise ValueError("hidden_size must divide evenly into heads with an even head dimension")
        if self.mlp_type not in {"gelu", "swiglu"}:
            raise ValueError("mlp_type must be gelu or swiglu")
        if self.loss_backend not in {"auto", "torch", "chunked", "liger"}:
            raise ValueError("loss_backend must be auto, torch, chunked or liger")
        for key in ("rope_theta", "rms_norm_eps", "initializer_range"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if getattr(self, "rope_scaling", None) or getattr(self, "rope_parameters", None):
            raise ValueError("Sprint currently supports its trained RoPE context only")
        return self

    def to_dict(self):
        values = super().to_dict()
        # yyLab's artifact writer deliberately requires JSON object string keys.
        if values.get("id2label"):
            values["id2label"] = {str(key): value for key, value in values["id2label"].items()}
        return values


def estimate_parameter_count(config: SprintConfig) -> int:
    """Exact unique trainable parameter count, including tied vocabulary floor."""
    d, f, n = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    embeddings = config.vocab_size * d * (1 if config.tie_word_embeddings else 2)
    mlp = (3 if config.mlp_type == "swiglu" else 2) * d * f
    return embeddings + n * (4 * d * d + mlp + 2 * d) + d


def plan_sprint_model(target_params: int, vocab_size: int, *, max_position_embeddings=2048,
                      mlp_type="gelu", tie_word_embeddings=True, profile="throughput") -> SprintConfig:
    """Fit continuous budgets to aligned dense shapes, without allocating tensors.

    Throughput prefers wider, shallower decoders at 10M parameters and above.
    Balanced preserves the original sizing rule, also used for smaller targets
    and vocabulary constraints that prevent an aligned fit within 2%. These
    are sizing heuristics; measure throughput and quality for the intended run.
    """
    if profile not in {"throughput", "balanced"}:
        raise ValueError("profile must be throughput or balanced")
    if isinstance(target_params, bool) or int(target_params) != target_params or target_params <= 0:
        raise ValueError("target_params must be a positive integer")
    if isinstance(vocab_size, bool) or int(vocab_size) != vocab_size or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    target_params, vocab_size = int(target_params), int(vocab_size)
    minimum = SprintConfig(vocab_size=vocab_size, hidden_size=16, intermediate_size=16,
                           num_hidden_layers=1, num_attention_heads=1, mlp_type=mlp_type,
                           max_position_embeddings=max_position_embeddings,
                           tie_word_embeddings=tie_word_embeddings)
    floor = estimate_parameter_count(minimum)
    if target_params < floor:
        raise ValueError(f"Target is below the {floor:,}-parameter minimum for this vocabulary; use a smaller tokenizer")
    candidates = []
    factor = 3 if mlp_type == "swiglu" else 2
    max_width = min(8192, max(32, int(math.sqrt(target_params / 4))))
    throughput = profile == "throughput" and target_params >= 10_000_000
    throughput_depth_floor = 2 if target_params < 50_000_000 else 4
    widths = (range(256, max_width + 1, 256) if throughput else
              sorted(set(range(16, min(256, max_width) + 1, 16)) |
                     set(range(256, max_width + 1, 64))))
    for width in widths:
        heads = width // 128 if throughput else (max(1, width // 64) if width % 64 == 0 else 1)
        ideal_depth = max(throughput_depth_floor, width / 256) if throughput else max(2, round(width / 64))
        for depth in range(throughput_depth_floor if throughput else 1, 65):
            fixed = vocab_size * width * (1 if tie_word_embeddings else 2) + depth * (4 * width**2 + 2 * width) + width
            remaining = target_params - fixed
            alignment = 64 if throughput else 16
            f = int(remaining / (depth * factor * width) // alignment) * alignment
            if f < max(16, width) or f > 6 * width:
                continue
            cfg = SprintConfig(vocab_size=vocab_size, hidden_size=width, intermediate_size=f,
                               num_hidden_layers=depth, num_attention_heads=heads,
                               max_position_embeddings=max_position_embeddings, mlp_type=mlp_type,
                               tie_word_embeddings=tie_word_embeddings)
            count = estimate_parameter_count(cfg)
            error = (target_params - count) / target_params
            if throughput and error > 0.02:
                continue
            expansion = 6 / factor if throughput else (8 / 3 if factor == 3 else 4)
            shape_penalty = abs(math.log2(depth / ideal_depth)) + 0.25 * abs(f / width - expansion)
            alignment_penalty = 0 if throughput else 0.15 * (width % 64 != 0) + 0.1 * (f % 64 != 0)
            candidates.append(((error > 0.02, shape_penalty + alignment_penalty + error * 10, error), cfg))
    if throughput and not candidates:
        return plan_sprint_model(target_params, vocab_size, max_position_embeddings=max_position_embeddings,
                                 mlp_type=mlp_type, tie_word_embeddings=tie_word_embeddings, profile="balanced")
    return min(candidates, key=lambda item: item[0])[1] if candidates else minimum


class SprintAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.heads, self.head_dim, self.layer_idx = config.num_attention_heads, config.head_dim, layer_idx
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask, is_causal, past_key_values):
        b, t, d = x.shape
        q, k, v = self.qkv_proj(x).view(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        def rotate(a):
            left, right = a.chunk(2, dim=-1)
            return torch.cat((-right, left), dim=-1)
        q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, d))


class SprintRMSNorm(nn.RMSNorm):
    def forward(self, x):
        # Match the compute dtype even with the optional FP32-weight recipe,
        # allowing native fused RMSNorm instead of mixed-dtype decomposition.
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


class SprintBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = SprintRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SprintAttention(config, layer_idx)
        self.post_attention_layernorm = SprintRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gated = config.mlp_type == "swiglu"
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size * (2 if self.gated else 1), bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask, is_causal, past_key_values=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, is_causal, past_key_values)
        h = self.up_proj(self.post_attention_layernorm(x))
        if self.gated:
            gate, value = h.chunk(2, dim=-1)
            h = F.silu(gate) * value
        else:
            h = F.gelu(h, approximate="tanh")
        return x + self.down_proj(h)


class SprintForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SprintConfig
    base_model_prefix = "sprint"
    main_input_name = "input_ids"
    _supports_sdpa = True
    _no_split_modules = ["SprintBlock"]
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}
    supports_gradient_checkpointing = True

    def __init__(self, config):
        config.validate()
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SprintBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = SprintRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _causal_loss(self, hidden, labels):
        hidden = hidden[:, :-1].contiguous().view(-1, self.config.hidden_size)
        targets = labels[:, 1:].contiguous().view(-1)
        backend = self.config.loss_backend
        if backend == "auto":
            backend = "liger" if hidden.is_cuda and self.config.vocab_size >= 8192 else "torch"
        if backend == "liger":
            return _liger_linear_loss(hidden, self.lm_head.weight, targets)
        denom = (targets != -100).sum().clamp_min(1)
        if backend == "chunked":
            # Checkpoint each head chunk: ordinary chunking retains every logit
            # tensor until backward and does not actually bound training memory.
            total = hidden.sum() * 0.0
            def chunk_loss(h, w, y):
                return F.cross_entropy(F.linear(h, w).float(), y, ignore_index=-100, reduction="sum")
            for start in range(0, targets.numel(), self.config.loss_chunk_size):
                stop = start + self.config.loss_chunk_size
                total = total + checkpoint(chunk_loss, hidden[start:stop], self.lm_head.weight,
                                           targets[start:stop], use_reentrant=False)
            return total / denom
        return F.cross_entropy(self.lm_head(hidden).float(), targets, ignore_index=-100, reduction="sum") / denom

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values: Cache | None = None, inputs_embeds=None, labels=None,
                use_cache=None, logits_to_keep=0, return_logits=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        batch_size, length, _ = x.shape
        use_cache = (self.config.use_cache and not self.training and labels is None) if use_cache is None else use_cache
        if self.training or labels is not None:
            use_cache = False
            if past_key_values is not None:
                raise ValueError("training with a mutable KV cache is unsupported")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True; the supplied cache would be mutated")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        offset = past_key_values.get_seq_length() if past_key_values is not None else 0
        if offset + length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds Sprint's trained context length")
        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)[:, -length:]
            else:
                position_ids = (torch.arange(length, device=x.device) + offset).unsqueeze(0)
        frequencies = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.head_dim, 2, device=x.device, dtype=torch.float32) / self.config.head_dim))
        angles = position_ids.float().unsqueeze(-1) * frequencies
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        mask = None
        key_length = offset + length
        static_cache = past_key_values is not None and past_key_values.is_compileable
        if past_key_values is not None:
            key_length, key_offset = past_key_values.get_mask_sizes(length, 0)
            if key_offset != 0:
                raise ValueError("Sprint requires a full-context KV cache; sliding-window caches are unsupported")
            if static_cache and key_length < offset + length:
                raise ValueError("Sprint KV cache capacity is smaller than the requested sequence")
        is_causal = length > 1 and offset == 0 and not static_cache
        if attention_mask is not None or static_cache or (offset and length > 1):
            # Static caches return their full allocation, including unused slots.
            # Explicit absolute positions keep those slots and future tokens masked.
            allowed = torch.arange(key_length, device=x.device)[None, :] <= (torch.arange(length, device=x.device) + offset)[:, None]
            mask = allowed[None, None]
            if attention_mask is not None:
                if attention_mask.shape[0] not in (1, batch_size):
                    raise ValueError("attention_mask batch size must match input_ids")
                if attention_mask.ndim == 2:
                    if attention_mask.shape[-1] != offset + length:
                        raise ValueError("attention_mask must cover the cached + current sequence")
                    padding_mask = F.pad(attention_mask.bool(), (0, key_length - attention_mask.shape[-1]), value=False)
                    mask = mask & padding_mask[:, None, None, :]
                elif attention_mask.ndim == 4:
                    if (attention_mask.shape[1] not in (1, self.config.num_attention_heads)
                            or attention_mask.shape[2] not in (1, length)
                            or attention_mask.shape[-1] != key_length):
                        raise ValueError("4D attention_mask must match batch, heads, query length and cache length")
                    if attention_mask.dtype == torch.bool:
                        mask = mask & attention_mask
                    elif attention_mask.is_floating_point():
                        mask = attention_mask.to(x.dtype).masked_fill(~mask, float("-inf"))
                    else:
                        raise ValueError("4D attention_mask must contain booleans or additive floating-point values")
                else:
                    raise ValueError("attention_mask must be a 2D padding mask or a 4D SDPA mask")
            is_causal = False
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, mask, is_causal, use_reentrant=False)
            else:
                x = layer(x, cos, sin, mask, is_causal, past_key_values)
        x = self.norm(x)
        loss = self._causal_loss(x, labels) if labels is not None else None
        return_logits = labels is None if return_logits is None else return_logits
        logits = None
        if return_logits:
            indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(x[:, indices, :])
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                     past_key_values=past_key_values if use_cache else None)


AutoConfig.register("sprint", SprintConfig, exist_ok=True)
AutoModelForCausalLM.register(SprintConfig, SprintForCausalLM, exist_ok=True)
