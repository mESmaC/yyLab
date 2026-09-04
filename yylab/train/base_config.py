from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

BASE_RECIPE_VERSION = 4

FAMILY_LABELS = {
    "qwen3": "Qwen3 decoder",
    "llama": "Llama decoder",
    "bitnet": "BitNet topology (dense precision only)",
}

QAT_LABELS = {
    "none": "None (dense training)",
    "nvfp4": "NVFP4 W4A4 (native Tensor Core training)",
}

SPARSITY_LABELS = {
    "none": "Dense weights",
    "paired_4:8": "Paired 4:8 (NVFP4's 50% sparse Tensor Core pattern)",
}

TRAINING_RECIPE_PRESETS: dict[str, dict[str, object]] = {
    "custom": {},
    "nvfp4_paired_4to8": {
        "parameter_dtype": "bf16",
        "qat": "nvfp4",
        "qat_backend": "native",
        "qat_group_size": 16,
        "activation_qat": "nvfp4",
        "sparsity": "paired_4:8",
        "sparsity_backend": "compressed",
    },
    "nvfp4_sparse_mxfp8_cpu_offload": {
        "parameter_dtype": "bf16",
        "qat": "nvfp4",
        "qat_backend": "native",
        "qat_group_size": 16,
        "activation_qat": "nvfp4",
        "sparsity": "paired_4:8",
        "sparsity_backend": "compressed",
        "backward_precision": "mxfp8",
        "backward_backend": "native",
        "optimizer_state": "cpu_offload",
        "optimizer_impl": "cpu_offload_adamw",
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
        if self.family == "bitnet" and qat_mode != "none":
            raise ValueError(
                "BitNet currently supports dense training only; no native low-precision "
                "training backend is integrated"
            )
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


def tokenizer_embedding_parameter_floor(
    spec: ScratchModelSpec,
    tokenizer_vocab_size: int,
    *,
    qat_mode: str = "none",
) -> int:
    """Return the irreducible embedding count for a valid yyLab decoder.

    A target-size solver may shrink transformer width and depth, but it may not
    shrink the tokenizer's vocabulary.  Eight is the smallest generally valid
    hidden/head width; native NVFP4 raises the aligned hidden floor to 32.
    """

    vocab_size = spec.resolved_vocab_size(tokenizer_vocab_size)
    minimum_hidden = 32 if qat_mode == "nvfp4" else 8
    embedding_tables = 1 if spec.tie_word_embeddings else 2
    return vocab_size * minimum_hidden * embedding_tables


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


def fit_scratch_to_parameter_target(
    spec: ScratchModelSpec,
    target_parameters: int,
    *,
    tokenizer_vocab_size: int,
    qat_mode: str = "none",
    min_layers: int = 1,
    max_layers: int = 128,
) -> ScratchModelSpec:
    """Fit width, MLP width, attention layout, and depth to a target.

    The search is deterministic and keeps vocabulary, context, embedding
    tying, initialization, and normalization choices exact.  Candidate widths
    remain kernel aligned and the resulting spec is fully validated before it
    is returned.
    """

    target = int(target_parameters)
    if target < 1:
        raise ValueError("target parameter count must be positive")
    if int(min_layers) < 1 or int(max_layers) < int(min_layers):
        raise ValueError("layer bounds must satisfy 1 <= min_layers <= max_layers")
    if qat_mode not in QAT_LABELS:
        raise ValueError(f"unsupported QAT target: {qat_mode}")

    vocab_size = spec.resolved_vocab_size(tokenizer_vocab_size)
    embedding_floor = tokenizer_embedding_parameter_floor(
        spec,
        tokenizer_vocab_size,
        qat_mode=qat_mode,
    )
    if target < embedding_floor:
        table_word = "table" if spec.tie_word_embeddings else "tables"
        minimum_hidden = 32 if qat_mode == "nvfp4" else 8
        raise ValueError(
            f"target {target:,} is below the tokenizer embedding floor "
            f"{embedding_floor:,} ({vocab_size:,} vocabulary x {minimum_hidden} minimum "
            f"hidden x {1 if spec.tie_word_embeddings else 2} {table_word}); choose a "
            "smaller tokenizer vocabulary or a larger target"
        )

    embedding_tables = 1 if spec.tie_word_embeddings else 2
    maximum_hidden = max(
        8,
        min(65_536, target // max(vocab_size * embedding_tables, 1)),
    )
    minimum_head_dim = 16 if qat_mode == "nvfp4" else 8
    head_dims = tuple(
        value
        for value in (8, 16, 32, 64, 128, 256, int(spec.head_dim))
        if value >= minimum_head_dim
        and value <= 512
        and value % (16 if qat_mode == "nvfp4" else 8) == 0
    )
    head_dims = tuple(dict.fromkeys(head_dims))
    kv_ratio = int(spec.num_key_value_heads) / max(int(spec.num_attention_heads), 1)
    best: tuple[tuple[float, ...], ScratchModelSpec] | None = None

    for head_dim in head_dims:
        max_heads = min(512, maximum_hidden // head_dim)
        for attention_heads in range(1, max_heads + 1):
            hidden = attention_heads * head_dim
            if hidden % (32 if qat_mode == "nvfp4" else 8):
                continue
            divisors = [value for value in range(1, attention_heads + 1) if attention_heads % value == 0]
            desired_kv = max(1, round(attention_heads * kv_ratio))
            kv_heads = min(divisors, key=lambda value: (abs(value - desired_kv), value))
            shell = replace(
                spec,
                vocab_size=vocab_size,
                hidden_size=hidden,
                intermediate_size=8,
                num_attention_heads=attention_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
            )
            fixed, per_layer_at_eight = _parameter_parts(shell, vocab_size)
            intermediate_coefficient = 3 * hidden + (1 if spec.family == "bitnet" else 0)
            layer_constant = per_layer_at_eight - intermediate_coefficient * 8
            intermediate_alignment = 32 if qat_mode == "nvfp4" else 8

            for layers in range(int(min_layers), int(max_layers) + 1):
                remaining = target - fixed - layers * layer_constant
                ideal_intermediate = remaining / max(layers * intermediate_coefficient, 1)
                rounded = (
                    round(ideal_intermediate / intermediate_alignment)
                    * intermediate_alignment
                )
                intermediate_candidates = {
                    max(
                        intermediate_alignment,
                        min(262_144, rounded + offset),
                    )
                    for offset in (
                        -intermediate_alignment,
                        0,
                        intermediate_alignment,
                    )
                }
                for intermediate in intermediate_candidates:
                    candidate = replace(
                        shell,
                        intermediate_size=intermediate,
                        num_hidden_layers=layers,
                    )
                    count = estimate_parameter_count(candidate)
                    relative_error = abs(count - target) / target
                    # Within one basis point, prefer a conventional MLP ratio
                    # and a shape near the user's current depth/head layout.
                    shape_penalty = (
                        abs(intermediate / hidden - 3.0)
                        + abs(layers - int(spec.num_hidden_layers)) / max(int(spec.num_hidden_layers), 1)
                        + abs(head_dim - int(spec.head_dim)) / max(int(spec.head_dim), 1)
                    )
                    score = (
                        round(relative_error, 4),
                        shape_penalty,
                        relative_error,
                        float(count),
                    )
                    if best is None or score < best[0]:
                        best = (score, candidate)

    if best is None:
        raise ValueError("no valid scratch architecture fits the requested target and constraints")
    fitted = best[1]
    fitted.validate(tokenizer_vocab_size=tokenizer_vocab_size, qat_mode=qat_mode)
    return fitted


def resolve_training_token_budget(
    *,
    requested_tokens: int = 0,
    seq_len: int,
    micro_batch: int,
    grad_accum: int,
    max_steps: int = 0,
    estimated_steps: int = 0,
) -> dict[str, int | str]:
    """Resolve token, explicit-step, and epoch-derived caps deterministically."""

    sequence = int(seq_len)
    micro = int(micro_batch)
    accumulation = int(grad_accum)
    if min(sequence, micro, accumulation) < 1:
        raise ValueError("sequence length, microbatch, and accumulation must be positive")
    tokens_per_step = sequence * micro * accumulation
    requested = max(0, int(requested_tokens or 0))
    explicit_steps = max(0, int(max_steps or 0))
    data_steps = max(0, int(estimated_steps or 0))
    if requested:
        resolved_steps = max(1, math.ceil(requested / tokens_per_step))
        source = "token_budget"
    elif explicit_steps:
        resolved_steps = explicit_steps
        source = "max_steps"
    elif data_steps:
        resolved_steps = data_steps
        source = "epochs"
    else:
        raise ValueError("a token budget, explicit step cap, or epoch-derived step count is required")
    scheduled = resolved_steps * tokens_per_step
    return {
        "requested_training_tokens": requested,
        "tokens_per_optimizer_step": tokens_per_step,
        "max_steps": resolved_steps,
        "scheduled_training_tokens": scheduled,
        "token_budget_rounding": scheduled - requested if requested else 0,
        "step_cap_source": source,
    }


def normalize_training_token_budget(config: dict) -> dict:
    """Apply an explicit token budget to a replayed config, if present."""

    resolved = dict(config)
    requested = int(resolved.get("requested_training_tokens") or 0)
    if requested < 1:
        return resolved
    budget = resolve_training_token_budget(
        requested_tokens=requested,
        seq_len=int(resolved["seq_len"]),
        micro_batch=int(resolved["micro_batch"]),
        grad_accum=int(resolved["grad_accum"]),
        max_steps=int(resolved.get("max_steps") or 0),
    )
    resolved.update(budget)
    return resolved


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
) -> tuple[float, float]:
    """Return coarse GPU/RAM parameter-state projections, excluding activations."""

    parameter_bytes = 4 if parameter_dtype == "fp32" else 2
    weights_and_grads = int(parameters) * parameter_bytes * 2
    # AdamW moments are FP32; fused/8-bit implementations may use less, so this
    # intentionally reports a conservative upper bound.
    adam_bytes = int(parameters) * 8
    if optimizer_state == "cpu_offload":
        gpu = weights_and_grads
        # CPUOffloadAdamW always owns an FP32 CPU master plus FP32 moments.
        ram = adam_bytes + int(parameters) * 4
    elif optimizer_state == "cpu_paged":
        gpu = weights_and_grads + int(parameters) * 2
        ram = adam_bytes
    else:
        gpu = weights_and_grads + adam_bytes
        ram = 0
    return gpu / 2**30, ram / 2**30
