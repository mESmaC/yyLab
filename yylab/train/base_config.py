from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

BASE_RECIPE_VERSION = 4

FAMILY_LABELS = {
    "qwen3": "Qwen3 decoder",
    "llama": "Llama decoder",
    "bitnet": "BitNet b1.58 decoder",
}

QAT_LABELS = {
    "none": "None (dense training)",
    "int4": "INT4 compatibility QAT (fake quant; no native INT4 Tensor Core GEMM)",
    "nvfp4": "NVFP4 W4A4 QAT (native FP4 forward when available)",
    "ternary": "Ternary BitNet b1.58 QAT (W1.58/A8; experimental)",
}

SPARSITY_LABELS = {
    "none": "Dense weights",
    "paired_4:8": "Paired 4:8 (NVFP4's 50% sparse Tensor Core pattern)",
}

MASTER_WEIGHT_LABELS = {
    "fp32": "FP32 master (maximum update fidelity)",
    "fp8_e4m3_adaptive": (
        "Adaptive FP8 master (selective sparse potentiation; recommended experimental)"
    ),
    "fp8_e4m3_potentiated": (
        "Legacy potentiated FP8 master (dense residual for every weight)"
    ),
}

TRAINING_RECIPE_PRESETS: dict[str, dict[str, object]] = {
    "custom": {},
    "nvfp4_paired_4to8": {
        "parameter_dtype": "bf16",
        "qat": "nvfp4",
        "qat_backend": "native",
        "qat_group_size": 16,
        "activation_qat": "none",
        "sparsity": "paired_4:8",
    },
    "nvfp4_paired_4to8_fp8_master": {
        "parameter_dtype": "bf16",
        "qat": "nvfp4",
        "qat_backend": "native",
        "qat_group_size": 16,
        "activation_qat": "none",
        "sparsity": "paired_4:8",
        "optimizer_state": "cpu_offload",
        "optimizer_impl": "cpu_offload_adamw",
        "master_weight_storage": "fp8_e4m3_potentiated",
    },
    "nvfp4_sparse_mxfp8_adaptive": {
        "parameter_dtype": "bf16",
        "qat": "nvfp4",
        "qat_backend": "native",
        "qat_group_size": 16,
        "activation_qat": "none",
        "sparsity": "paired_4:8",
        "sparsity_backend": "compressed",
        "backward_precision": "mxfp8",
        "backward_backend": "native",
        "optimizer_state": "cpu_offload",
        "optimizer_impl": "cpu_offload_adamw",
        "master_weight_storage": "fp8_e4m3_adaptive",
        "adaptive_max_promoted_fraction": 0.10,
        "adaptive_promote_threshold": 0.55,
        "adaptive_demote_threshold": 0.20,
        "adaptive_demote_patience": 2,
    },
}


@dataclass(frozen=True, slots=True)
class ScratchModelSpec:
    """Serializable architecture contract for a randomly initialized causal LM."""

    family: str = "qwen3"
    vocab_size: int = 0
    vocab_padding_multiple: int = 64
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int = 64
    max_position_embeddings: int = 8192
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False

    @classmethod
    def from_dict(cls, values: dict | None) -> ScratchModelSpec:
        raw = dict(values or {})
        aliases = {
            "layers": "num_hidden_layers",
            "heads": "num_attention_heads",
            "kv_heads": "num_key_value_heads",
            "context_length": "max_position_embeddings",
            "native_context": "max_position_embeddings",
        }
        for old, new in aliases.items():
            if old in raw and new not in raw:
                raw[new] = raw.pop(old)
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in raw.items() if key in allowed})

    def to_dict(self) -> dict:
        return asdict(self)

    def resolved_vocab_size(self, tokenizer_vocab_size: int) -> int:
        tokenizer_vocab_size = int(tokenizer_vocab_size)
        if tokenizer_vocab_size < 1:
            raise ValueError("tokenizer vocabulary is empty")
        requested = int(self.vocab_size)
        if requested and requested < tokenizer_vocab_size:
            raise ValueError(
                f"model vocab_size={requested:,} is smaller than tokenizer size {tokenizer_vocab_size:,}"
            )
        size = requested or tokenizer_vocab_size
        multiple = max(1, int(self.vocab_padding_multiple))
        return math.ceil(size / multiple) * multiple

    def validate(
        self,
        *,
        tokenizer_vocab_size: int | None = None,
        train_sequence_length: int | None = None,
        qat_mode: str = "none",
    ) -> ScratchModelSpec:
        if self.family not in FAMILY_LABELS:
            raise ValueError(f"unsupported scratch architecture: {self.family}")
        integer_fields = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_padding_multiple": self.vocab_padding_multiple,
        }
        bad = [name for name, value in integer_fields.items() if int(value) < 1]
        if bad:
            raise ValueError("architecture fields must be positive: " + ", ".join(bad))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden size must equal attention heads × head dimension")
        if self.hidden_size % 8 or self.intermediate_size % 8 or self.head_dim % 8:
            raise ValueError("hidden, intermediate, and head dimensions must be multiples of 8")
        if self.initializer_range <= 0:
            raise ValueError("initializer range must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("RMS norm epsilon must be positive")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention dropout must be in [0, 1)")
        if self.rope_theta <= 0:
            raise ValueError("RoPE theta must be positive")
        if train_sequence_length is not None:
            train_sequence_length = int(train_sequence_length)
            if train_sequence_length < 1:
                raise ValueError("training sequence length must be positive")
            if train_sequence_length > self.max_position_embeddings:
                raise ValueError(
                    f"training sequence length {train_sequence_length:,} exceeds native model "
                    f"context {self.max_position_embeddings:,}"
                )
        if tokenizer_vocab_size is not None:
            self.resolved_vocab_size(tokenizer_vocab_size)
        if qat_mode not in QAT_LABELS:
            raise ValueError(f"unsupported QAT target: {qat_mode}")
        if qat_mode == "ternary" and self.family != "bitnet":
            raise ValueError("ternary QAT requires the BitNet architecture")
        if self.family == "bitnet" and qat_mode not in {"none", "ternary"}:
            raise ValueError("BitNet supports dense or ternary QAT recipes, not INT4/NVFP4 QAT")
        if qat_mode == "nvfp4" and (
            self.hidden_size % 32 or self.intermediate_size % 32 or self.head_dim % 16
        ):
            raise ValueError(
                "NVFP4 requires hidden/intermediate dimensions divisible by 32 and head dimension by 16"
            )
        return self

    def with_resolved_vocab(self, tokenizer_vocab_size: int) -> ScratchModelSpec:
        return replace(self, vocab_size=self.resolved_vocab_size(tokenizer_vocab_size))


def _parameter_parts(spec: ScratchModelSpec, vocab_size: int) -> tuple[int, int]:
    """Return fixed and per-layer parameter counts for supported decoder families."""

    hidden = int(spec.hidden_size)
    intermediate = int(spec.intermediate_size)
    q_width = int(spec.num_attention_heads) * int(spec.head_dim)
    kv_width = int(spec.num_key_value_heads) * int(spec.head_dim)
    fixed = int(vocab_size) * hidden + hidden
    if not spec.tie_word_embeddings:
        fixed += int(vocab_size) * hidden

    attention = hidden * q_width + 2 * hidden * kv_width + q_width * hidden
    if spec.attention_bias:
        attention += q_width + 2 * kv_width + hidden
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    if spec.family == "qwen3":
        norms += 2 * int(spec.head_dim)
    elif spec.family == "bitnet":
        # BitNet's architecture adds post-projection and post-FFN sub norms.
        norms += hidden + intermediate
    return fixed, attention + mlp + norms


def estimate_parameter_count(spec: ScratchModelSpec, tokenizer_vocab_size: int | None = None) -> int:
    vocab_size = (
        spec.resolved_vocab_size(tokenizer_vocab_size)
        if tokenizer_vocab_size is not None
        else int(spec.vocab_size)
    )
    if vocab_size < 1:
        raise ValueError("a tokenizer-derived or explicit vocabulary size is required")
    fixed, per_layer = _parameter_parts(spec, vocab_size)
    return fixed + int(spec.num_hidden_layers) * per_layer


def fit_layers_to_parameter_target(
    spec: ScratchModelSpec,
    target_parameters: int,
    *,
    tokenizer_vocab_size: int | None = None,
    min_layers: int = 1,
    max_layers: int = 128,
) -> ScratchModelSpec:
    """Fit the layer count to a target while keeping every other dimension exact."""

    vocab_size = (
        spec.resolved_vocab_size(tokenizer_vocab_size)
        if tokenizer_vocab_size is not None
        else int(spec.vocab_size)
    )
    if vocab_size < 1:
        raise ValueError("a tokenizer-derived or explicit vocabulary size is required")
    fixed, per_layer = _parameter_parts(spec, vocab_size)
    layers = round((int(target_parameters) - fixed) / max(per_layer, 1))
    layers = min(max(int(layers), int(min_layers)), int(max_layers))
    return replace(spec, num_hidden_layers=layers)


def build_hf_config(spec: ScratchModelSpec, tokenizer) -> object:
    """Create an HF config without loading any pretrained model weights."""

    from transformers import BitNetConfig, LlamaConfig, Qwen3Config

    tokenizer_size = len(tokenizer)
    resolved = spec.with_resolved_vocab(tokenizer_size)
    resolved.validate(tokenizer_vocab_size=tokenizer_size)
    special = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    common = {
        "vocab_size": resolved.vocab_size,
        "hidden_size": resolved.hidden_size,
        "intermediate_size": resolved.intermediate_size,
        "num_hidden_layers": resolved.num_hidden_layers,
        "num_attention_heads": resolved.num_attention_heads,
        "num_key_value_heads": resolved.num_key_value_heads,
        "max_position_embeddings": resolved.max_position_embeddings,
        "initializer_range": resolved.initializer_range,
        "rms_norm_eps": resolved.rms_norm_eps,
        "tie_word_embeddings": resolved.tie_word_embeddings,
        "attention_bias": resolved.attention_bias,
        "attention_dropout": resolved.attention_dropout,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(resolved.rope_theta),
        },
        **special,
    }
    if resolved.family == "qwen3":
        config = Qwen3Config(head_dim=resolved.head_dim, **common)
    elif resolved.family == "llama":
        config = LlamaConfig(head_dim=resolved.head_dim, **common)
    else:
        config = BitNetConfig(**common)
    config.use_cache = False
    config.yylab_base_recipe_version = BASE_RECIPE_VERSION
    config.yylab_scratch_architecture = resolved.to_dict()
    return config


def projected_training_memory_gb(
    parameters: int,
    *,
    parameter_dtype: str = "bf16",
    optimizer_state: str = "gpu",
    master_weight_storage: str = "fp32",
    adaptive_max_promoted_fraction: float = 0.10,
) -> tuple[float, float]:
    """Return coarse GPU/RAM parameter-state projections, excluding activations."""

    parameter_bytes = 4 if parameter_dtype == "fp32" else 2
    weights_and_grads = int(parameters) * parameter_bytes * 2
    # AdamW moments are FP32; fused/8-bit implementations may use less, so this
    # intentionally reports a conservative upper bound.
    adam_bytes = int(parameters) * 8
    if master_weight_storage == "fp8_e4m3_adaptive":
        blocks = (int(parameters) + 31) // 32
        promoted_blocks = int(blocks * max(0.0, min(1.0, adaptive_max_promoted_fraction)))
        # E4M3 byte/value, FP32 block scale, uint8 hysteresis counter, then
        # only the selected int32 indices and packed 4-bit residual rows.
        master_bytes = (
            int(parameters)
            + blocks * 5
            + promoted_blocks * (4 + 16)
        )
    elif master_weight_storage == "fp8_e4m3_potentiated":
        # One E4M3 byte, one packed residual nibble, and one FP32 scale per
        # 32 values. Adam moments remain FP32 and are counted separately.
        master_bytes = (
            int(parameters)
            + (int(parameters) + 1) // 2
            + ((int(parameters) + 31) // 32) * 4
        )
    else:
        master_bytes = int(parameters) * 4 if parameter_dtype != "fp32" else 0
    if optimizer_state == "cpu_offload":
        gpu = weights_and_grads
        # CPUOffloadAdamW always owns an offloaded master, including when the
        # accelerator parameter itself is FP32.
        ram = adam_bytes + (master_bytes or int(parameters) * 4)
    elif optimizer_state == "cpu_paged":
        gpu = weights_and_grads + int(parameters) * 2
        ram = adam_bytes
    else:
        gpu = weights_and_grads + adam_bytes + master_bytes
        ram = 0
    return gpu / 2**30, ram / 2**30
