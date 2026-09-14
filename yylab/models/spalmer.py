"""Compatibility boundary between native SPALMER bundles and yyLab runtimes.

SPALMER checkpoints are trusted local ``.pt`` bundles, not Hugging Face
directories.  This module deliberately keeps their detection and execution
separate from ``AutoModelForCausalLM`` while presenting the small HF-shaped
surface used by yyLab's existing chat and benchmark paths.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from ..config import PATHS

SPALMER_FORMAT = "spalmer.prototype.checkpoint"
SPALMER_FILENAME = "spalmer.pt"
SPALMER_QR_PLE_FORMAT_VERSION = 7
SPALMER_CURRENT_FORMAT_VERSION = 8
# Scanning stays deliberately broader than execution.  Old bundles remain
# identifiable in the registry so yyLab can explain why they are blocked.
SPALMER_SUPPORTED_FORMAT_VERSIONS = frozenset(
    range(1, SPALMER_CURRENT_FORMAT_VERSION + 1)
)
SPALMER_PLE_BACKENDS = frozenset({"fake_qat", "qr"})
SPALMER_NATIVE_EXPERT_PRECISION_PAIRS = frozenset(
    {
        ("bfloat16", "bfloat16"),
        ("float32", "float32"),
        ("mxfp4", "mxfp8"),
        ("nvfp4", "nvfp4"),
        ("nvfp4", "mxfp8"),
        ("mxfp6", "mxfp8"),
        ("mxfp8", "mxfp8"),
    }
)
DEFAULT_CONTEXT_LIMIT = 8_192
ADAPTIVE_EXIT_CRITERIA = ("none", "latent_diff", "kl")
EXIT_STATE_POLICIES = ("fill", "skip")
DEFAULT_RECURRENCE_STEPS = 8
_PER_TOKEN_STEP_LIMIT = 4_096
# Exact operand pairs with an integrated SPALMER executor in format v8.  The
# wider config vocabulary reserves names for future providers, but a checkpoint
# must not be presented as executable merely because one of those names appears
# in otherwise well-formed metadata.
_INTEGRATED_EXPERT_PRECISION_PAIRS = frozenset(
    {
        ("nvfp4", "nvfp4"),
        ("mxfp8", "mxfp8"),
        ("bfloat16", "bfloat16"),
        ("float32", "float32"),
    }
)
# Chat, the benchmark UI and the checkpoints page all read a bundle's recurrent
# core through the same three metadata shapes: a run receipt written by the
# training bridge, a registry row produced by ``scan_spalmer_checkpoint``, and a
# scanned bundle whose native metadata is nested one level down.
_RECURRENCE_METADATA_KEYS = ("recurrence", "recurrence_config")


def recurrence_metadata(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a checkpoint's recurrence block from any recorded metadata shape."""

    if not isinstance(meta, Mapping):
        return None
    for key in _RECURRENCE_METADATA_KEYS:
        value = meta.get(key)
        if isinstance(value, Mapping) and value:
            return dict(value)
    nested = meta.get("metadata")
    if isinstance(nested, Mapping):
        for key in _RECURRENCE_METADATA_KEYS:
            value = nested.get(key)
            if isinstance(value, Mapping) and value:
                return dict(value)
    return None


def recurrence_effective_depth(
    num_layers: int, recurrence: Mapping[str, Any] | None, steps: int | None = None
) -> int:
    """Return P + steps*C + K, or the physical depth when there is no core."""

    if not isinstance(recurrence, Mapping) or not recurrence:
        return int(num_layers)
    try:
        prelude = int(recurrence["prelude_layers"])
        core = int(recurrence["core_layers"])
        coda = int(recurrence["coda_layers"])
        depth = int(steps if steps is not None else recurrence.get("default_steps", DEFAULT_RECURRENCE_STEPS))
    except (KeyError, TypeError, ValueError):
        # A malformed block must never make a registry scan fail; the physical
        # depth stays the honest fallback.
        return int(num_layers)
    if min(prelude, core, coda, depth) < 1:
        return int(num_layers)
    return prelude + depth * core + coda


def _recurrence_config_dict(value: object) -> dict[str, Any] | None:
    """Normalize a native ``RecurrenceConfig`` (or mapping) into plain data."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    fields = (
        "prelude_layers",
        "core_layers",
        "coda_layers",
        "default_steps",
        "latent_init_std",
        "adapter",
        "adapter_init",
    )
    collected = {
        name: getattr(value, name) for name in fields if getattr(value, name, None) is not None
    }
    return collected or None


def _validate_recurrence_policy(
    *,
    recurrent: bool,
    recurrence_steps: int | None,
    adaptive_exit: str | None,
    exit_threshold: float | None,
    min_steps: int | None,
    warm_start: bool | None,
    exit_state_policy: str | None,
) -> None:
    """Reject recurrence policies a checkpoint cannot honour, before placement."""

    requested = (
        recurrence_steps is not None
        or (adaptive_exit is not None and adaptive_exit != "none")
        or exit_threshold is not None
        or bool(warm_start)
    )
    if requested and not recurrent:
        raise ValueError(
            "SPALMER checkpoint has no recurrent core; clear the recurrence settings"
        )
    if adaptive_exit is not None and adaptive_exit not in ADAPTIVE_EXIT_CRITERIA:
        raise ValueError(
            "unknown SPALMER adaptive exit criterion: "
            f"{adaptive_exit!r} (expected one of {', '.join(ADAPTIVE_EXIT_CRITERIA)})"
        )
    if exit_state_policy is not None and exit_state_policy not in EXIT_STATE_POLICIES:
        raise ValueError(
            "unknown SPALMER exit state policy: "
            f"{exit_state_policy!r} (expected one of {', '.join(EXIT_STATE_POLICIES)})"
        )
    if recurrence_steps is not None and int(recurrence_steps) < 1:
        raise ValueError("recurrence_steps must be at least 1")
    if exit_threshold is not None and float(exit_threshold) <= 0:
        raise ValueError("exit_threshold must be positive")
    if min_steps is not None and int(min_steps) < 1:
        raise ValueError("min_steps must be at least 1")


def resolve_spalmer_checkpoint(ref: str | Path) -> Path:
    """Resolve either a direct ``.pt`` bundle or ``directory/spalmer.pt``."""

    path = Path(ref).expanduser()
    if path.is_dir():
        path = path / SPALMER_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"SPALMER checkpoint not found: {path}")
    if path.suffix.casefold() != ".pt":
        raise ValueError("a SPALMER checkpoint must be a .pt bundle")
    return path.resolve()


def is_spalmer_checkpoint(ref: str | Path) -> bool:
    """Return whether ``ref`` resolves to a valid, safely inspectable bundle."""

    try:
        scan_spalmer_checkpoint(ref)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def load_spalmer_checkpoint_payload(ref: str | Path) -> dict[str, Any]:
    """Load a trusted bundle's payload without constructing its model.

    ``weights_only=True`` constrains unpickling to PyTorch's safe allowlist and
    ``mmap=True`` avoids materializing tensors that the caller never touches.
    """

    path = resolve_spalmer_checkpoint(ref)
    payload = torch.load(
        str(path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("SPALMER checkpoint payload must be a plain dictionary")
    if payload.get("format") != SPALMER_FORMAT:
        raise ValueError("not a SPALMER checkpoint")
    return payload


def scan_spalmer_checkpoint(ref: str | Path) -> dict[str, Any]:
    """Inspect a trusted local SPALMER bundle without constructing its model.

    ``weights_only=True`` constrains unpickling to PyTorch's safe allowlist and
    ``mmap=True`` prevents registration from materializing a second in-memory
    copy of all checkpoint tensors.  SPALMER's checkpoint payload is required
    to remain a plain mapping of primitive metadata and tensors.
    """

    path = resolve_spalmer_checkpoint(ref)
    payload = load_spalmer_checkpoint_payload(path)

    model_config = _plain_dict(payload.get("model_config"), "model_config")
    model_state = _tensor_mapping(payload.get("model_state"), "model_state")
    metadata = _plain_dict(payload.get("metadata") or {}, "metadata")
    experts_config = _optional_plain_dict(payload.get("experts_config"), "experts_config")
    kda_config = _optional_plain_dict(payload.get("kda_config"), "kda_config")
    mla_config = _optional_plain_dict(payload.get("mla_config"), "mla_config")
    directional_config = _optional_plain_dict(
        payload.get("directional_config"), "directional_config"
    )
    atxy_config = _optional_plain_dict(payload.get("atxy_config"), "atxy_config")

    hidden_size = _positive_int(model_config.get("d_model"), "model_config.d_model")
    num_layers = _positive_int(model_config.get("n_layers"), "model_config.n_layers")
    vocab_size = _positive_int(model_config.get("vocab_size"), "model_config.vocab_size")
    recurrence_config = _optional_plain_dict(
        model_config.get("recurrence"), "model_config.recurrence"
    )
    effective_depth = recurrence_effective_depth(num_layers, recurrence_config)
    # A pre-v7 bundle can omit the key because fake-QAT was then the sole
    # topology.  Version 7 introduced QR-PLE and made the discriminator
    # mandatory; failing closed here prevents the registry from laundering a
    # malformed QR bundle into an apparently loadable legacy checkpoint.
    format_version = int(payload.get("format_version") or 0)
    if format_version not in SPALMER_SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"unsupported SPALMER checkpoint format version: {format_version}")
    raw_ple_backend = model_config.get("ple_backend")
    if format_version >= SPALMER_QR_PLE_FORMAT_VERSION and raw_ple_backend is None:
        raise ValueError("SPALMER format-version 7+ checkpoint is missing model_config.ple_backend")
    ple_backend = str(raw_ple_backend or "fake_qat")
    if ple_backend not in SPALMER_PLE_BACKENDS:
        raise ValueError(f"unsupported SPALMER PLE backend: {ple_backend!r}")
    if format_version < SPALMER_QR_PLE_FORMAT_VERSION and ple_backend != "fake_qat":
        raise ValueError(
            f"SPALMER format-version {format_version} predates QR-PLE but declares "
            f"backend {ple_backend!r}"
        )
    expert_precision = _scanned_expert_precision(ple_backend, experts_config)
    state_tensor_count = _unique_tensor_numel(model_state.values())
    planned_parameter_count = _metadata_parameter_count(metadata)
    parameter_count = planned_parameter_count or state_tensor_count
    trained_context = _metadata_context_limit(metadata)
    base_cfg = {
        **model_config,
        "model_type": "spalmer",
        "architectures": ["SPALMERCausalLM"],
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "vocab_size": vocab_size,
        "ple_backend": ple_backend,
        # This is a yyLab safety budget, not a RoPE limit: SPALMER is NoPE.
        "max_position_embeddings": trained_context or DEFAULT_CONTEXT_LIMIT,
        "spalmer_format_version": payload.get("format_version"),
        "spalmer_parameter_dtype": payload.get("model_parameter_dtype"),
        "spalmer_experts_config": experts_config,
        "spalmer_kda_config": kda_config,
        "spalmer_mla_config": mla_config,
        "spalmer_directional_config": directional_config,
        "spalmer_atxy_config": atxy_config,
        # ``num_hidden_layers`` stays physical; recurrence multiplies the core.
        "spalmer_recurrence_config": recurrence_config,
        "spalmer_ple_backend": ple_backend,
        "spalmer_expert_precision": expert_precision,
        "spalmer_precision_executable": expert_precision["executable"],
        "effective_depth": effective_depth,
    }
    return {
        "family": "spalmer",
        "model_family": "spalmer",
        "format": SPALMER_FORMAT,
        "format_version": payload.get("format_version"),
        "arch": "SPALMERCausalLM",
        "params_b": parameter_count / 1_000_000_000,
        "parameter_count": parameter_count,
        "parameter_count_source": (
            "metadata.plan.parameters.total"
            if planned_parameter_count is not None
            else "model_state tensors (parameters and persistent buffers)"
        ),
        "state_tensor_count": state_tensor_count,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "recurrence_config": recurrence_config,
        "ple_backend": ple_backend,
        "expert_precision": expert_precision,
        "precision_executable": expert_precision["executable"],
        "precision_rejection_reason": expert_precision["rejection_reason"],
        "effective_depth": effective_depth,
        "parameter_dtype": payload.get("model_parameter_dtype"),
        "trained_context_length": trained_context,
        "base_cfg": base_cfg,
        "metadata": metadata,
        "experts_config": experts_config,
        "kda_config": kda_config,
        "mla_config": mla_config,
        "directional_config": directional_config,
        "atxy_config": atxy_config,
        "tokenizer_binding": _optional_plain_dict(
            payload.get("tokenizer_binding"), "tokenizer_binding"
        ),
        "checkpoint_ref": str(path),
        "path": str(path.parent),
    }


def _scanned_expert_precision(
    ple_backend: str,
    experts_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Classify checkpoint numerics without importing SPALMER or running kernels."""

    experts = dict(experts_config or {})
    weight = str(experts.get("expert_weight_format") or "").strip().casefold()
    activation = str(experts.get("expert_activation_format") or "").strip().casefold()
    backend = str(experts.get("expert_qat_backend") or "").strip().casefold()
    master = str(experts.get("expert_master_dtype") or "").strip().casefold()
    execution = str(experts.get("expert_execution") or "").strip().casefold()
    retired: list[str] = []
    if ple_backend != "qr":
        retired.append("retired fake-QAT PLE")
    if weight == "legacy_int":
        retired.append("legacy expert weights")
    if backend == "reference":
        retired.append("reference expert backend")
    if bool(experts.get("expert_fake_quantization", False)):
        retired.append("expert fake quantization")
    if bool(experts.get("expert_stochastic_rounding", False)):
        retired.append("emulated expert stochastic quantization")
    missing: list[str] = []
    if not weight:
        missing.append("weight format")
    if not activation:
        missing.append("activation format")
    if not backend:
        missing.append("native backend policy")
    if not master:
        missing.append("master dtype")
    if not execution:
        missing.append("execution mode")

    reason = ""
    if retired:
        reason = (
            "checkpoint requires retired emulated precision ("
            + ", ".join(retired)
            + "); yyLab has no fake or reference fallback"
        )
    elif missing:
        reason = (
            "checkpoint does not prove an executable native precision contract; missing "
            + ", ".join(missing)
        )
    elif (weight, activation) not in SPALMER_NATIVE_EXPERT_PRECISION_PAIRS:
        reason = f"checkpoint declares unsupported expert precision pair {weight!r}/{activation!r}"
    elif backend not in {"auto", "native"}:
        reason = f"checkpoint declares unsupported expert backend {backend!r}"
    elif master != ("float32" if weight == "float32" else "bfloat16") and not (
        master == "float32"
        and (weight, activation) in {("bfloat16", "bfloat16"), ("nvfp4", "nvfp4")}
    ):
        reason = (
            "checkpoint expert master dtype does not match its execution pair: "
            f"{master!r} for {weight!r}/{activation!r}"
        )
    elif execution not in {"grouped", "loop"}:
        reason = f"checkpoint declares unsupported expert execution mode {execution!r}"
    elif (weight, activation) not in _INTEGRATED_EXPERT_PRECISION_PAIRS:
        reason = (
            "checkpoint declares an expert precision pair with no integrated native "
            f"SPALMER executor: {weight or 'missing'}/{activation or 'missing'}"
        )

    return {
        "weight_format": weight or None,
        "activation_format": activation or None,
        "master_dtype": master or None,
        "backend": backend or None,
        "execution": execution or None,
        "ple_backend": ple_backend,
        "executable": not reason,
        "rejection_reason": reason or None,
        "training_recipe": (
            "torchao_nvfp4_training_sm120_fp32_master"
            if (master, weight, activation) == ("float32", "nvfp4", "nvfp4")
            else "fp32_master_bf16_amp"
            if (master, weight, activation) == ("float32", "bfloat16", "bfloat16")
            else None
        ),
    }


def require_executable_spalmer_checkpoint(ref: str | Path) -> dict[str, Any]:
    """Return scan metadata or reject a bundle that requires emulation."""

    scanned = scan_spalmer_checkpoint(ref)
    reason = str(scanned.get("precision_rejection_reason") or "")
    if reason:
        raise ValueError(f"SPALMER checkpoint is metadata-only and cannot execute: {reason}")
    return scanned


def _require_native_precision_capability(
    scanned: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Require the checkpoint's exact pair and dispatch mode on this device."""

    precision = _plain_dict(scanned.get("expert_precision"), "expert_precision")
    if (
        precision.get("master_dtype"),
        precision.get("weight_format"),
        precision.get("activation_format"),
    ) == ("float32", "nvfp4", "nvfp4"):
        if precision.get("execution") != "loop":
            raise RuntimeError("standard NVFP4 training requires dense per-expert execution")
        from spalmer.experts.standard_nvfp4 import require_standard_nvfp4

        return {
            "provider_id": require_standard_nvfp4(device),
            "weight_format": "nvfp4",
            "activation_format": "nvfp4",
            "forward_available": True,
            "backward_available": True,
            "grouped_available": False,
            "verified": True,
            "detail": "FP32 masters; upstream NVFP4 training with SM120 stochastic rounding",
        }
    from spalmer.precision import detect_precision_capabilities

    report = detect_precision_capabilities(device)
    capability = report.require(
        str(precision["weight_format"]),
        str(precision["activation_format"]),
    )
    if precision.get("execution") == "grouped" and not capability.grouped_available:
        raise RuntimeError(
            f"native provider {capability.provider_id!r} is dense per-expert, but the "
            "checkpoint requires grouped execution; yyLab will not substitute BF16 bmm"
        )
    return capability.to_dict()


def require_spalmer_precision_capability(
    ref: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Validate a bundle's exact declared precision against the current device."""

    scanned = require_executable_spalmer_checkpoint(ref)
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return _require_native_precision_capability(scanned, resolved)


class SpalmerTokenizerFacade:
    """Expose a generic SPALMER tokenizer through yyLab's HF-like subset."""

    def __init__(
        self,
        backend: object,
        *,
        model_max_length: int,
        default_kind: str = "mixed",
    ) -> None:
        self.backend = backend
        self.model_max_length = model_max_length
        self.default_kind = default_kind if default_kind in {"mixed", "prose", "code"} else "mixed"
        specials = getattr(backend, "special_tokens", None)
        self.bos_token_id = getattr(specials, "bos_id", None)
        self.eos_token_id = getattr(specials, "eos_id", None)
        self.pad_token_id = getattr(specials, "pad_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        self.bos_token = "<|bos|>" if self.bos_token_id is not None else None
        self.eos_token = "<|eos|>" if self.eos_token_id is not None else None
        self.pad_token = "<|pad|>" if self.pad_token_id is not None else self.eos_token
        self.vocab_size = int(getattr(backend, "vocab_size", 0) or len(backend))

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, *, add_special_tokens: bool = False, **_kwargs: Any) -> list[int]:
        encode = getattr(self.backend, "encode", None)
        if not callable(encode):
            raise TypeError("SPALMER tokenizer backend does not implement encode")
        try:
            values = encode(
                text,
                add_special_tokens=add_special_tokens,
                kind=self.default_kind,
            )
        except TypeError:
            values = encode(text, add_special_tokens=add_special_tokens)
        return [int(value) for value in values]

    def decode(
        self,
        token_ids: Iterable[int] | Tensor,
        *,
        skip_special_tokens: bool = False,
        **_kwargs: Any,
    ) -> str:
        values = _flat_token_ids(token_ids)
        decode = getattr(self.backend, "decode", None)
        if not callable(decode):
            raise TypeError("SPALMER tokenizer backend does not implement decode")
        try:
            return str(
                decode(
                    values,
                    skip_special_tokens=skip_special_tokens,
                    errors="replace",
                )
            )
        except TypeError:
            return str(decode(values, skip_special_tokens=skip_special_tokens))

    def batch_decode(
        self,
        sequences: Iterable[Iterable[int]] | Tensor,
        *,
        skip_special_tokens: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        if isinstance(sequences, Tensor):
            sequences = sequences.detach().cpu().tolist()
        return [
            self.decode(row, skip_special_tokens=skip_special_tokens, **kwargs)
            for row in sequences
        ]

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
        padding: bool | str = False,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in inputs]
        if not encoded:
            raise ValueError("tokenizer input batch cannot be empty")

        needs_padding = bool(padding) or return_tensors == "pt"
        width = max((len(row) for row in encoded), default=0)
        if needs_padding and any(len(row) != width for row in encoded) and self.pad_token_id is None:
            raise ValueError("batched tokenization needs a pad or EOS token")
        padded: list[list[int]] = []
        masks: list[list[int]] = []
        for row in encoded:
            missing = width - len(row) if needs_padding else 0
            padded.append([*row, *([int(self.pad_token_id)] * missing)])
            masks.append([1] * len(row) + [0] * missing)

        if return_tensors is None:
            ids: Any = padded[0] if single else padded
            mask: Any = masks[0] if single else masks
        elif return_tensors == "pt":
            ids = torch.tensor(padded, dtype=torch.long)
            mask = torch.tensor(masks, dtype=torch.long)
        else:
            raise ValueError("SPALMER tokenizer facade supports only return_tensors='pt'")
        result = {"input_ids": ids}
        if return_attention_mask:
            result["attention_mask"] = mask
        return result

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        **_kwargs: Any,
    ) -> str | list[int]:
        if enable_thinking:
            raise TypeError("generic SPALMER tokenizers do not define a thinking protocol")
        parts = [
            f"### {str(message.get('role') or 'user').capitalize()}\n"
            f"{str(message.get('content') or '')}"
            for message in messages
        ]
        if add_generation_prompt:
            parts.append("### Assistant\n")
        prompt = "\n\n".join(parts)
        return self.encode(prompt) if tokenize else prompt


class _ForwardOutputFacade:
    """Attach HF's ``hidden_states`` field without copying SPALMER outputs."""

    __slots__ = ("_output", "hidden_states")

    def __init__(self, output: object, hidden_states: Tensor) -> None:
        self._output = output
        self.hidden_states = (hidden_states,)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._output, name)


def _native_compute_context(model: nn.Module):
    """Keep SPALMER optional until a model declares a standard AMP recipe."""

    experts = getattr(getattr(model, "residency", None), "config", None)
    if getattr(experts, "expert_master_dtype", None) != "float32":
        return nullcontext()
    from spalmer.runtime import model_compute_context

    return model_compute_context(model)


class SpalmerCausalLMFacade(nn.Module):
    """HF-shaped inference facade over one native :class:`SPALMERCausalLM`."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        *,
        checkpoint_path: Path,
        metadata: Mapping[str, Any],
        dynamic_residency: bool,
        expert_paging: bool,
        full_pool_routing: bool,
        max_context_tokens: int,
        cuda_available: bool,
        notes: Sequence[str] = (),
        recurrence_steps: int | None = None,
        adaptive_exit: str = "none",
        exit_threshold: float | None = None,
        min_steps: int = 2,
        warm_start: bool = False,
        exit_state_policy: str = "fill",
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.spalmer_config = model.config
        self.checkpoint_path = checkpoint_path
        self.metadata = dict(metadata)
        self.dynamic_residency = bool(dynamic_residency)
        # ``expert_paging`` describes an actually enabled physical backend;
        # ``full_pool_routing`` is the quality policy and also applies when all
        # expert masters already reside on CPU or GPU.
        self.expert_paging = bool(expert_paging)
        self.full_pool_routing = bool(full_pool_routing)
        if self.expert_paging and not self.full_pool_routing:
            raise ValueError("physical expert paging requires full-pool logical routing")
        if self.full_pool_routing and self.dynamic_residency:
            raise ValueError(
                "full-pool expert inference cannot use legacy request-global dynamic residency"
            )
        self.cuda_available = bool(cuda_available)
        self.cuda = self.cuda_available
        self.notes = list(notes)
        self.last_routing_telemetry: dict[str, object] | None = None

        recurrence = _recurrence_config_dict(getattr(model.config, "recurrence", None))
        self.recurrence_config = recurrence
        self.recurrent = recurrence is not None
        _validate_recurrence_policy(
            recurrent=self.recurrent,
            recurrence_steps=recurrence_steps,
            adaptive_exit=adaptive_exit,
            exit_threshold=exit_threshold,
            min_steps=min_steps,
            warm_start=warm_start,
            exit_state_policy=exit_state_policy,
        )
        default_steps = (
            int((recurrence or {}).get("default_steps") or DEFAULT_RECURRENCE_STEPS)
            if self.recurrent
            else None
        )
        self.recurrence_steps: int | None = (
            None
            if not self.recurrent
            else int(recurrence_steps) if recurrence_steps is not None else default_steps
        )
        self.adaptive_exit = str(adaptive_exit or "none") if self.recurrent else "none"
        self.exit_threshold = (
            float(exit_threshold) if self.recurrent and exit_threshold is not None else None
        )
        self.min_steps = int(min_steps)
        self.warm_start = bool(warm_start) and self.recurrent
        self.exit_state_policy = str(exit_state_policy or "fill")
        self.last_recurrence_telemetry: dict[str, object] | None = None
        physical_layers = int(model.config.n_layers)
        self.effective_depth = recurrence_effective_depth(
            physical_layers, recurrence, self.recurrence_steps
        )
        if self.recurrent:
            # Honour the loaded depth even for bare ``model(ids)`` calls made by
            # native helpers yyLab does not thread kwargs through.
            set_defaults = getattr(model, "set_recurrence_defaults", None)
            if callable(set_defaults):
                set_defaults(self.recurrence_steps)

        self.config = SimpleNamespace(
            model_type="spalmer",
            architectures=["SPALMERCausalLM"],
            vocab_size=int(model.config.vocab_size),
            hidden_size=int(model.config.d_model),
            # Physical blocks; ``effective_depth`` carries P + r*C + K.
            num_hidden_layers=physical_layers,
            max_position_embeddings=int(max_context_tokens),
            use_cache=True,
            is_encoder_decoder=False,
            spalmer_recurrence=recurrence,
            effective_depth=self.effective_depth,
            recurrence_steps=self.recurrence_steps,
        )

    def _native_recurrence_kwargs(
        self,
        *,
        decode: bool,
        steps: int | None = None,
        adaptive_exit: str | None = None,
        exit_threshold: float | None = None,
        min_steps: int | None = None,
        exit_state_policy: str | None = None,
    ) -> dict[str, Any]:
        """Build the native depth kwargs; ``{}`` for a flat checkpoint."""

        if not self.recurrent:
            return {}
        kwargs: dict[str, Any] = {
            "recurrence_steps": self.recurrence_steps if steps is None else int(steps)
        }
        criterion = self.adaptive_exit if adaptive_exit is None else str(adaptive_exit)
        if not decode or criterion == "none":
            return kwargs
        threshold = self.exit_threshold if exit_threshold is None else exit_threshold
        from spalmer.modeling import AdaptiveExit

        kwargs["adaptive_exit"] = AdaptiveExit(
            criterion=criterion,
            threshold=None if threshold is None else float(threshold),
            min_steps=int(self.min_steps if min_steps is None else min_steps),
            state_policy=str(
                self.exit_state_policy if exit_state_policy is None else exit_state_policy
            ),
        )
        return kwargs

    def _recurrence_receipt(
        self,
        *,
        steps: int | None,
        criterion: str,
        threshold: float | None,
        min_steps: int,
        warm_start: bool,
        prefill_steps: int | None,
        decode_steps: Sequence[int],
    ) -> dict[str, object]:
        """Summarize one request's realized depth for chat and benchmarks."""

        observed = [int(value) for value in decode_steps]
        mean_steps = (sum(observed) / len(observed)) if observed else None
        recurrence = self.recurrence_config or {}
        prelude = int(recurrence.get("prelude_layers") or 0)
        core = int(recurrence.get("core_layers") or 0)
        coda = int(recurrence.get("coda_layers") or 0)
        return {
            "recurrence_steps": steps,
            "adaptive_exit": criterion,
            "exit_threshold": threshold,
            "min_steps": int(min_steps),
            "warm_start": bool(warm_start),
            "exit_state_policy": self.exit_state_policy,
            "prefill_steps": prefill_steps,
            "decode_tokens": len(observed),
            "mean_decode_steps": mean_steps,
            "min_decode_steps": min(observed) if observed else None,
            "max_decode_steps": max(observed) if observed else None,
            "early_exits": sum(
                1 for value in observed if steps is not None and value < int(steps)
            ),
            "effective_depth_mean": (
                None if mean_steps is None else prelude + mean_steps * core + coda
            ),
            # Bounded so a long generation never grows an unbounded receipt.
            "per_token_steps": observed[:_PER_TOKEN_STEP_LIMIT],
        }

    @property
    def device(self) -> torch.device:
        execution_device = getattr(self.model, "execution_device", None)
        if execution_device is not None:
            return torch.device(execution_device)
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        return buffer.device if buffer is not None else torch.device("cpu")

    def set_dynamic_residency(self, enabled: bool) -> None:
        if self.full_pool_routing and enabled:
            raise ValueError(
                "full-pool expert inference cannot enable legacy request-global residency"
            )
        self.dynamic_residency = bool(enabled)

    def routing_telemetry(self) -> dict[str, object] | None:
        if self.full_pool_routing:
            # Paging counters and layer-local identities change during decode,
            # so always read a fresh native snapshot instead of returning the
            # prefill receipt captured below.
            value = _full_pool_routing_telemetry(
                self.model,
                expert_paging=self.expert_paging,
            )
        elif self.last_routing_telemetry is not None:
            value = dict(self.last_routing_telemetry)
        else:
            value = _fixed_residency_telemetry(self.model)
        if self.last_recurrence_telemetry is None:
            return value
        # A recurrent request always publishes its realized depth, even when a
        # CPU/no-offload load has no expert-placement snapshot to report.
        receipt = dict(value) if isinstance(value, dict) else {}
        receipt["recurrence"] = dict(self.last_recurrence_telemetry)
        return receipt

    def expert_offload_telemetry(self) -> dict[str, object] | None:
        """Return a serialization-safe physical placement snapshot."""

        return _expert_offload_telemetry(self.model)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool = False,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> object:
        del return_dict
        if use_cache is not None:
            kwargs["use_cache"] = use_cache
        if self.recurrent:
            # NLL scoring and context compaction call the facade directly; they
            # must run at the loaded depth rather than the checkpoint default.
            kwargs.setdefault("recurrence_steps", self.recurrence_steps)
        with _native_compute_context(self.model):
            if not output_hidden_states:
                return self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )

            lm_head = getattr(self.model, "lm_head", None)
            if not isinstance(lm_head, nn.Module):
                raise TypeError("SPALMER model does not expose its language-model head")
            captured: list[Tensor] = []

            def capture_hidden(_module: nn.Module, args: tuple[object, ...]) -> None:
                if args and isinstance(args[0], Tensor):
                    captured.append(args[0])

            handle = lm_head.register_forward_pre_hook(capture_hidden)
            try:
                output = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs,
                )
            finally:
                handle.remove()
            if not captured:
                raise RuntimeError("SPALMER language-model head did not receive hidden states")
            return _ForwardOutputFacade(output, captured[-1])

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: int | Sequence[int] | Tensor | None = None,
        pad_token_id: int | None = None,
        streamer: object | None = None,
        stopping_criteria: object | None = None,
        logits_processor: object | None = None,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        dynamic_residency: bool | None = None,
        recurrence_steps: int | None = None,
        adaptive_exit: str | None = None,
        exit_threshold: float | None = None,
        min_steps: int | None = None,
        warm_start: bool | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Generate one sequence with SPALMER's explicit recurrent state."""

        del pad_token_id, use_cache
        _validate_ignored_generation_kwargs(kwargs)
        _validate_generation_inputs(input_ids, attention_mask, max_new_tokens)
        if do_sample and temperature <= 0:
            raise ValueError("sampling temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not repetition_penalty > 0:
            raise ValueError("repetition_penalty must be positive")
        if not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be a non-negative integer")
        _validate_recurrence_policy(
            recurrent=self.recurrent,
            recurrence_steps=recurrence_steps,
            adaptive_exit=adaptive_exit,
            exit_threshold=exit_threshold,
            min_steps=min_steps,
            warm_start=warm_start,
            exit_state_policy=None,
        )
        steps = self.recurrence_steps if recurrence_steps is None else int(recurrence_steps)
        criterion = self.adaptive_exit if adaptive_exit is None else str(adaptive_exit)
        threshold = self.exit_threshold if exit_threshold is None else float(exit_threshold)
        exit_min_steps = int(self.min_steps if min_steps is None else min_steps)
        use_warm_start = bool(self.warm_start if warm_start is None else warm_start)
        prefill_kwargs = self._native_recurrence_kwargs(
            decode=False,
            steps=steps,
            adaptive_exit=criterion,
            exit_threshold=threshold,
            min_steps=exit_min_steps,
        )
        decode_kwargs = self._native_recurrence_kwargs(
            decode=True,
            steps=steps,
            adaptive_exit=criterion,
            exit_threshold=threshold,
            min_steps=exit_min_steps,
        )

        generated = input_ids.to(device=self.device, dtype=torch.long)
        mask = None if attention_mask is None else attention_mask.to(device=self.device)
        eos_ids = _eos_ids(
            self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        )
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(torch.initial_seed() if seed is None else seed))
        # A seeded request must also reproduce the latent initialization; with
        # no seed the native forward keeps using the global torch RNG.
        if self.recurrent and seed is not None:
            latent_generator = torch.Generator(device=self.device)
            latent_generator.manual_seed(int(seed))
            prefill_kwargs["latent_generator"] = latent_generator
            decode_kwargs["latent_generator"] = latent_generator

        if self.full_pool_routing and dynamic_residency:
            raise ValueError(
                "dynamic_residency is the legacy request-global policy and cannot be "
                "combined with paged expert inference"
            )
        use_dynamic = (
            False
            if self.full_pool_routing
            else self.dynamic_residency if dynamic_residency is None else dynamic_residency
        )
        request_open = False
        was_training = self.model.training
        decision = None
        self.last_routing_telemetry = None
        self.last_recurrence_telemetry = None
        decode_steps: list[int] = []
        prefill_steps_run: int | None = None
        ran_prefill = False
        self.model.eval()
        try:
            with _native_compute_context(self.model):
                if streamer is not None:
                    streamer.put(generated.detach().cpu())
                if max_new_tokens == 0:
                    return generated

                if self.full_pool_routing:
                    # Native paging is a physical placement policy only.  The
                    # router must see the full trained pool, and the selected rows
                    # are materialized by the layer-local cache on demand.
                    output = self.model(generated, attention_mask=mask, **prefill_kwargs)
                    self.last_routing_telemetry = _full_pool_routing_telemetry(
                        self.model,
                        expert_paging=self.expert_paging,
                    )
                elif use_dynamic:
                    from spalmer.experts import choose_inference_residency

                    # The kwarg is omitted for flat checkpoints so residency
                    # controllers that predate recurrence keep their signature.
                    decision = choose_inference_residency(
                        self.model,
                        generated,
                        **prefill_kwargs,
                    )
                    output = decision.output
                    request_open = True
                    # Publish the request receipt as soon as residency selection
                    # finishes so chat can display it while recurrent decoding is
                    # still streaming, rather than only after the final token.
                    self.last_routing_telemetry = _residency_telemetry(self.model, decision)
                else:
                    output = self.model(generated, attention_mask=mask, **prefill_kwargs)
                    self.last_routing_telemetry = _fixed_residency_telemetry(self.model)
                ran_prefill = True
                prefill_steps_run = _output_recurrence_steps(output)
                token_states = output.token_mixer_states
                channel_states = output.channel_mixer_states

                for index in range(max_new_tokens):
                    scores = output.logits[:, -1]
                    scores = _apply_repetition_guards(
                        generated,
                        scores,
                        repetition_penalty=float(repetition_penalty),
                        no_repeat_ngram_size=int(no_repeat_ngram_size),
                    )
                    scores = _apply_logits_processors(logits_processor, generated, scores)
                    next_token = _select_token(
                        scores,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        generator=generator,
                    )
                    generated = torch.cat((generated, next_token), dim=-1)
                    if streamer is not None:
                        streamer.put(next_token.detach().cpu())
                    reached_eos = bool(eos_ids and int(next_token.item()) in eos_ids)
                    externally_stopped = _stopping_requested(
                        stopping_criteria,
                        generated,
                        scores,
                    )
                    if reached_eos or externally_stopped or index + 1 == max_new_tokens:
                        break
                    step_kwargs = dict(decode_kwargs)
                    if use_warm_start:
                        latent = getattr(output, "latent_states", None)
                        if latent is not None:
                            step_kwargs["latent_init"] = latent[:, -1:, :]
                    output = self.model(
                        next_token,
                        token_mixer_states=token_states,
                        channel_mixer_states=channel_states,
                        execution_mode="decode",
                        **step_kwargs,
                    )
                    steps_run = _output_recurrence_steps(output)
                    if steps_run is not None:
                        decode_steps.append(steps_run)
                    token_states = output.token_mixer_states
                    channel_states = output.channel_mixer_states
                return generated
        finally:
            if self.recurrent and ran_prefill:
                self.last_recurrence_telemetry = self._recurrence_receipt(
                    steps=steps,
                    criterion=criterion,
                    threshold=threshold,
                    min_steps=exit_min_steps,
                    warm_start=use_warm_start,
                    prefill_steps=(
                        steps if prefill_steps_run is None else prefill_steps_run
                    ),
                    decode_steps=decode_steps,
                )
            if request_open:
                self.model.end_residency_request()
            self.model.train(was_training)
            if streamer is not None:
                streamer.end()


def _output_recurrence_steps(output: object) -> int | None:
    """Read a native output's realized iteration count when it reports one."""

    value = getattr(output, "recurrence_steps", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_tokenizer_binding(path: Path) -> dict[str, Any] | None:
    """Read only the tokenizer binding from a bundle, without building tensors."""

    payload = load_spalmer_checkpoint_payload(path)
    return _optional_plain_dict(payload.get("tokenizer_binding"), "tokenizer_binding")


def _local_tokenizer_artifacts(preferred_name: str) -> list[Path]:
    """Return local tokenizer artifact directories, same-named candidate first."""

    root = Path(PATHS.tokenizers)
    if not root.is_dir():
        return []
    candidates = sorted(
        candidate.resolve()
        for candidate in root.iterdir()
        if candidate.is_dir() and (candidate / "tokenizer.json").is_file()
    )
    return sorted(candidates, key=lambda candidate: candidate.name != preferred_name)


def relocated_spalmer_tokenizer(binding: Mapping[str, Any] | None) -> Any | None:
    """Rebuild a checkpoint's Hugging Face tokenizer when its recorded path moved.

    SPALMER records a local tokenizer artifact by absolute path.  Checkpoints
    copied from another machine (or another artifact root) therefore point at
    a directory that no longer exists, and Hugging Face would misread that
    path as a Hub repository id.  The tokenizer identity is content-addressed,
    so a local artifact with the same fingerprint is the same tokenizer; it is
    returned as a ``tokenizer_override`` that SPALMER re-verifies on load.
    Returns ``None`` whenever the recorded reference should be resolved by
    SPALMER itself.
    """

    if not binding or binding.get("kind") != "huggingface":
        return None
    identity = _plain_dict(binding.get("identity"), "tokenizer_binding.identity")
    source = str(identity.get("source") or "")
    source_path = Path(source).expanduser()
    if not source_path.is_absolute() or source_path.exists():
        return None
    expected = str(identity.get("artifact_fingerprint") or "").strip()
    if not expected:
        raise ValueError("SPALMER tokenizer binding is missing its artifact_fingerprint")

    from spalmer.tokenizer import HFTokenizerAdapter, SpecialTokenIds, fingerprint_local_artifact

    candidates = _local_tokenizer_artifacts(source_path.name)
    for candidate in candidates:
        if fingerprint_local_artifact(candidate) != expected:
            continue
        special_tokens = _plain_dict(
            identity.get("special_tokens") or {}, "tokenizer_binding.identity.special_tokens"
        )
        return HFTokenizerAdapter.from_pretrained(
            str(candidate),
            revision=identity.get("revision"),
            local_files_only=True,
            trust_remote_code=False,
            special_tokens=SpecialTokenIds(**special_tokens),
        )
    raise FileNotFoundError(
        "SPALMER checkpoint tokenizer was recorded at "
        f"{source}, which does not exist on this machine, and none of the "
        f"{len(candidates)} local tokenizer artifacts under {PATHS.tokenizers} match "
        f"artifact fingerprint {expected}; import the original tokenizer artifact first"
    )


def load_spalmer_runtime(
    ref: str | Path,
    *,
    dynamic_residency: bool = False,
    max_context_tokens: int | None = None,
    expert_offload: bool = True,
    expert_cache_size: int | None = None,
    expert_paging: bool = True,
    recurrence_steps: int | None = None,
    adaptive_exit: str = "none",
    exit_threshold: float | None = None,
    min_steps: int = 2,
    warm_start: bool = False,
    exit_state_policy: str = "fill",
) -> SpalmerCausalLMFacade:
    """Load a native bundle and return an HF-shaped inference facade."""

    checkpoint_path = resolve_spalmer_checkpoint(ref)
    scanned = require_executable_spalmer_checkpoint(checkpoint_path)
    if max_context_tokens is not None and max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    if expert_cache_size is not None and expert_cache_size <= 0:
        raise ValueError("expert_cache_size must be positive")
    if expert_paging and dynamic_residency:
        raise ValueError(
            "paged expert inference cannot use legacy request-global dynamic residency"
        )

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    precision_capability = _require_native_precision_capability(scanned, device)

    # Keep SPALMER optional for yyLab installations that only use HF models.
    from spalmer import load_checkpoint

    binding = _checkpoint_tokenizer_binding(checkpoint_path)
    tokenizer_override = relocated_spalmer_tokenizer(binding)
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if tokenizer_override is not None:
        load_kwargs["tokenizer_override"] = tokenizer_override
    model, loaded_tokenizer, metadata = load_checkpoint(checkpoint_path, **load_kwargs)
    # Validate the depth policy against the loaded architecture before any
    # accelerator placement so a mismatch never costs a GPU transfer.
    recurrence_config = _recurrence_config_dict(getattr(model.config, "recurrence", None))
    _validate_recurrence_policy(
        recurrent=recurrence_config is not None,
        recurrence_steps=recurrence_steps,
        adaptive_exit=adaptive_exit,
        exit_threshold=exit_threshold,
        min_steps=min_steps,
        warm_start=warm_start,
        exit_state_policy=exit_state_policy,
    )
    offload: dict[str, object] | None = None
    if cuda:
        # Checkpoint loading is deliberately CPU-first.  Casting on CPU is
        # unnecessary: SPALMER has already validated and restored the exact
        # checkpoint master dtype.  The selective placement call below is the
        # only operation that may move weights onto the accelerator.
        model.to(device="cpu")
        model.eval()
        if expert_offload:
            enable_offload = getattr(model, "enable_expert_offload", None)
            if not callable(enable_offload):
                raise RuntimeError(
                    "this SPALMER runtime does not implement physical expert offload"
                )
            native_cap = (
                _native_num_experts(model) if expert_paging else _native_resident_cap(model)
            )
            if expert_cache_size is not None and expert_cache_size > native_cap:
                raise ValueError(
                    "expert_cache_size cannot exceed the checkpoint's "
                    f"{'expert count' if expert_paging else 'bounded resident cap'} "
                    f"({native_cap})"
                )
            supports_paging = _accepts_keyword(enable_offload, "paging")
            if expert_paging and not supports_paging:
                raise RuntimeError(
                    "this SPALMER runtime predates quality-preserving expert paging; "
                    "upgrade SPALMER or explicitly select legacy request-global mode"
                )
            offload_kwargs: dict[str, object] = {
                "cache_size": expert_cache_size,
                "non_blocking": True,
                "pin_memory": True,
            }
            if supports_paging:
                offload_kwargs["paging"] = expert_paging
            snapshot = enable_offload(device, **offload_kwargs)
            try:
                offload = _validated_expert_offload_telemetry(
                    model,
                    snapshot,
                    expected_device=device,
                    expected_capacity=expert_cache_size,
                    expected_paging=expert_paging,
                )
            except BaseException:
                disable_offload = getattr(model, "disable_expert_offload", None)
                if callable(disable_offload):
                    try:
                        disable_offload(device="cpu")
                    except Exception:  # noqa: BLE001 - preserve the placement-proof failure
                        pass
                raise
        else:
            model.to(device=device)
    else:
        # Preserve the checkpoint dtype that the exact CPU capability probe
        # just approved.  This matters for native FP32 CPU development bundles
        # and prevents a validated pair from being silently changed afterward.
        model.to(device=device)
        model.eval()

    context_limit = int(
        max_context_tokens or _metadata_context_limit(metadata) or DEFAULT_CONTEXT_LIMIT
    )
    identity = getattr(loaded_tokenizer, "identity", None)
    if getattr(identity, "backend", None) == "huggingface" and hasattr(
        loaded_tokenizer, "tokenizer"
    ):
        tokenizer = loaded_tokenizer.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and getattr(
            tokenizer, "eos_token", None
        ) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer_backend = _runtime_tokenizer_backend(loaded_tokenizer)
        tokenizer = SpalmerTokenizerFacade(
            tokenizer_backend,
            model_max_length=context_limit,
            default_kind=str(metadata.get("kind") or "mixed"),
        )

    notes = [
        "Loaded capability-verified native SPALMER checkpoint.",
        "Expert precision: "
        f"{precision_capability['weight_format'].upper()}/"
        f"{precision_capability['activation_format'].upper()} via "
        f"{precision_capability['provider_id']} "
        f"({'grouped' if precision_capability['grouped_available'] else 'dense per expert'}).",
        "SPALMER uses NoPE and native recurrent KDA/MLA state; HF RoPE and KV-cache modes do not apply.",
    ]
    if tokenizer_override is not None:
        recorded_source = str((binding or {}).get("identity", {}).get("source") or "")
        notes.append(
            f"SPALMER tokenizer relocated: the checkpoint recorded {recorded_source}, "
            "which does not exist here; using the fingerprint-matched local artifact "
            f"{tokenizer_override.identity.source}."
        )
    if offload is not None:
        pinned = "pinned " if offload["expert_masters_pinned"] else ""
        if expert_paging:
            notes.extend(
                (
                    "Paged SPALMER expert offload enabled: the trained router can select "
                    "from the full expert pool while selected rows are loaded from "
                    f"{pinned}CPU RAM into layer-local {offload['expert_cache_device']} caches.",
                    "SPALMER's per-layer expert cache is hard-bounded at "
                    f"{offload['expert_cache_capacity']}/{offload['total_experts']} identities; "
                    "physical residency does not mask logical routing.",
                )
            )
        else:
            notes.extend(
                (
                    "Legacy request-global SPALMER expert offload is enabled (experimental): "
                    f"complete expert masters remain in {pinned}CPU RAM and only one shared "
                    f"resident set executes on {offload['expert_cache_device']}.",
                    "This legacy policy can change the trained router's output by masking "
                    "experts outside the request-global resident set.",
                )
            )
    elif cuda:
        notes.append("SPALMER parameters run as BF16 on CUDA; physical expert offload is disabled.")
    else:
        notes.append(
            "SPALMER is using native BF16 CPU execution; all experts remain CPU-resident and "
            "an accelerator expert cache is not applicable."
        )
    if recurrence_config is not None:
        resolved_steps = int(
            recurrence_steps
            if recurrence_steps is not None
            else (recurrence_config.get("default_steps") or DEFAULT_RECURRENCE_STEPS)
        )
        depth = recurrence_effective_depth(
            int(model.config.n_layers), recurrence_config, resolved_steps
        )
        exit_text = (
            "fixed depth"
            if adaptive_exit == "none"
            else (
                f"adaptive exit {adaptive_exit}"
                + (
                    ""
                    if exit_threshold is None
                    else f" (threshold {float(exit_threshold):g})"
                )
                + f", minimum {int(min_steps)} steps, {exit_state_policy} state policy"
            )
        )
        notes.extend(
            (
                "Recurrent depth: prelude "
                f"{int(recurrence_config.get('prelude_layers') or 0)} · core "
                f"{int(recurrence_config.get('core_layers') or 0)} × {resolved_steps} · coda "
                f"{int(recurrence_config.get('coda_layers') or 0)} "
                f"(effective depth {depth}); {exit_text}; warm start "
                f"{'on' if warm_start else 'off'}.",
                "The paged expert cache is re-staged on every core iteration; size it to "
                "the union of the routes taken across iterations.",
            )
        )
    return SpalmerCausalLMFacade(
        model,
        tokenizer,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        dynamic_residency=dynamic_residency,
        expert_paging=bool(offload is not None and offload["expert_paging"]),
        full_pool_routing=expert_paging,
        max_context_tokens=context_limit,
        cuda_available=cuda,
        notes=notes,
        recurrence_steps=recurrence_steps,
        adaptive_exit=adaptive_exit,
        exit_threshold=exit_threshold,
        min_steps=min_steps,
        warm_start=warm_start,
        exit_state_policy=exit_state_policy,
    )


def _runtime_tokenizer_backend(tokenizer: object) -> object:
    if callable(getattr(tokenizer, "encode", None)) and callable(
        getattr(tokenizer, "decode", None)
    ):
        return tokenizer
    # Legacy/current embedded Vocab bundles expose entries rather than the
    # runtime protocol. Wrap them without changing checkpoint identity.
    from spalmer.tokenizer import RPDTokenizerAdapter, Vocab

    if isinstance(tokenizer, Vocab):
        return RPDTokenizerAdapter(tokenizer)
    raise TypeError(f"unsupported SPALMER checkpoint tokenizer: {type(tokenizer).__name__}")


def _plain_dict(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"SPALMER checkpoint {name} must be a plain dictionary")
    return dict(value)


def _optional_plain_dict(value: object, name: str) -> dict[str, Any] | None:
    return None if value is None else _plain_dict(value, name)


def _tensor_mapping(value: object, name: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"SPALMER checkpoint {name} must be a tensor mapping")
    if any(not isinstance(key, str) or not isinstance(tensor, Tensor) for key, tensor in value.items()):
        raise TypeError(f"SPALMER checkpoint {name} must map names to tensors")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"SPALMER checkpoint {name} must be a positive integer")
    return value


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _metadata_context_limit(metadata: Mapping[str, Any]) -> int | None:
    """Read context safety metadata from legacy and yyLab-native bundles."""

    for value in (
        metadata.get("max_context_tokens"),
        metadata.get("sequence_length"),
        metadata.get("training_sequence_length"),
    ):
        resolved = _optional_positive_int(value)
        if resolved is not None:
            return resolved
    training_config = metadata.get("training_config")
    if isinstance(training_config, Mapping):
        return _optional_positive_int(training_config.get("sequence_length"))
    return None


def _metadata_parameter_count(metadata: Mapping[str, Any]) -> int | None:
    """Return SPALMER's planned trainable count when the bundle records it."""

    plan = metadata.get("plan")
    if not isinstance(plan, Mapping):
        return None
    parameters = plan.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    return _optional_positive_int(parameters.get("total"))


def _unique_tensor_numel(tensors: Iterable[Tensor]) -> int:
    seen: set[tuple[int, int, int, torch.dtype]] = set()
    total = 0
    for tensor in tensors:
        key = (tensor.data_ptr(), tensor.storage_offset(), tensor.numel(), tensor.dtype)
        if key in seen:
            continue
        seen.add(key)
        total += tensor.numel()
    return total


def _flat_token_ids(values: Iterable[int] | Tensor) -> list[int]:
    if isinstance(values, Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    return [int(value) for value in values]


def _validate_generation_inputs(
    input_ids: Tensor,
    attention_mask: Tensor | None,
    max_new_tokens: int,
) -> None:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("SPALMER generation requires one non-empty input sequence")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if not bool(attention_mask.bool().all().item()):
            raise ValueError("SPALMER generation does not accept padded input")


def _eos_ids(value: int | Sequence[int] | Tensor | None) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, int):
        values = [value]
    else:
        values = list(value)
    return frozenset(int(item) for item in values)


def _validate_ignored_generation_kwargs(kwargs: Mapping[str, Any]) -> None:
    """Accept inert HF defaults but never silently discard active features."""

    inert_defaults = {
        "assistant_model": None,
        "cache_config": None,
        "cache_implementation": None,
        "compile_config": None,
        "decoder_start_token_id": None,
        "generation_config": None,
        "max_length": None,
        "min_length": 0,
        "min_new_tokens": None,
        "negative_prompt_attention_mask": None,
        "negative_prompt_ids": None,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_logits": False,
        "output_scores": False,
        "past_key_values": None,
        "prefill_chunk_size": None,
        "prefix_allowed_tokens_fn": None,
        "return_dict_in_generate": False,
        "synced_gpus": False,
    }
    unsupported = [
        key
        for key, value in kwargs.items()
        if key not in inert_defaults or value != inert_defaults[key]
    ]
    if not unsupported:
        return

    cache_options = sorted(
        key
        for key in unsupported
        if key
        in {
            "cache_config",
            "cache_implementation",
            "past_key_values",
            "prefill_chunk_size",
        }
    )
    if cache_options:
        raise ValueError(
            "SPALMER owns recurrent KDA/MLA state; unsupported HF cache options: "
            + ", ".join(cache_options)
        )
    raise ValueError(
        "unsupported non-default SPALMER generation options: "
        + ", ".join(sorted(unsupported))
    )


def _apply_repetition_guards(
    input_ids: Tensor,
    scores: Tensor,
    *,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> Tensor:
    """Apply HF-compatible repetition guards before caller processors."""

    if repetition_penalty != 1.0:
        selected = scores.gather(1, input_ids)
        selected = torch.where(
            selected < 0,
            selected * repetition_penalty,
            selected / repetition_penalty,
        )
        scores = scores.scatter(1, input_ids, selected)

    if no_repeat_ngram_size < 1:
        return scores
    current_length = input_ids.shape[-1]
    if current_length < no_repeat_ngram_size:
        return scores
    prefix = input_ids[:, current_length + 1 - no_repeat_ngram_size :]
    windows = input_ids.unfold(dimension=1, size=no_repeat_ngram_size, step=1)
    matches = (windows[..., :-1] == prefix.unsqueeze(1)).all(dim=-1)
    vocab_size = scores.shape[-1]
    banned_mask = scores.new_zeros((scores.shape[0], vocab_size + 1), dtype=torch.bool)
    banned_mask.scatter_(
        1,
        torch.where(matches, windows[..., -1], vocab_size),
        True,
    )
    return scores.masked_fill(banned_mask[:, :vocab_size], -float("inf"))


def _apply_logits_processors(processors: object, input_ids: Tensor, scores: Tensor) -> Tensor:
    if processors is None:
        return scores
    if callable(processors):
        return processors(input_ids, scores)
    for processor in processors:  # type: ignore[union-attr]
        scores = processor(input_ids, scores)
    return scores


def _select_token(
    logits: Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int | None,
    generator: torch.Generator,
) -> Tensor:
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits.float() / temperature
    if top_k:
        count = min(int(top_k), scores.shape[-1])
        cutoff = scores.topk(count, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    if top_p < 1:
        ordered, order = scores.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        ordered = ordered.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, order, ordered)
    probabilities = scores.softmax(dim=-1)
    if not bool(torch.isfinite(probabilities).all().item()) or bool(
        (probabilities.sum(dim=-1) <= 0).any().item()
    ):
        raise RuntimeError("SPALMER sampling produced invalid probabilities")
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def _stopping_requested(criteria: object, input_ids: Tensor, scores: Tensor) -> bool:
    if criteria is None:
        return False
    if not callable(criteria):
        if isinstance(criteria, (str, bytes)):
            raise TypeError("stopping_criteria must be callable or an iterable of callables")
        try:
            iterator = iter(criteria)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("stopping_criteria must be callable or an iterable of callables") from exc
        return any(
            _stopping_requested(criterion, input_ids, scores) for criterion in iterator
        )
    result = criteria(input_ids, scores)
    if isinstance(result, Tensor):
        return bool(result.bool().all().item())
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return bool(result) and all(bool(value) for value in result)
    return bool(result)


def _native_resident_cap(model: nn.Module) -> int:
    residency = getattr(model, "residency", None)
    config = getattr(residency, "config", None)
    cap = int(getattr(config, "resident_cap", 0) or 0)
    if cap < 1:
        raise RuntimeError("SPALMER checkpoint does not declare a bounded expert resident cap")
    return cap


def _native_num_experts(model: nn.Module) -> int:
    residency = getattr(model, "residency", None)
    config = getattr(residency, "config", None)
    count = int(getattr(config, "num_experts", 0) or 0)
    if count < 1:
        raise RuntimeError("SPALMER checkpoint does not declare its expert count")
    return count


def _accepts_keyword(function: object, keyword: str) -> bool:
    """Return whether a native API explicitly supports a compatibility keyword."""

    try:
        signature = inspect.signature(function)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(keyword)
    if parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
        return True
    return any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    )


def _expert_offload_telemetry(
    model: nn.Module,
    snapshot: object | None = None,
) -> dict[str, object] | None:
    """Normalize SPALMER's dataclass telemetry for UI and JSON artifacts."""

    if snapshot is None:
        getter = getattr(model, "expert_offload_telemetry", None)
        if not callable(getter):
            return None
        snapshot = getter()
    if snapshot is None:
        return None

    resident_ids = tuple(int(value) for value in getattr(snapshot, "resident_expert_ids", ()))
    cached_by_layer = tuple(
        tuple(int(value) for value in layer)
        for layer in getattr(snapshot, "cached_expert_ids_by_layer", ())
    )
    capacity = int(getattr(snapshot, "capacity", 0) or 0)
    cache_bytes = int(getattr(snapshot, "cache_bytes", 0) or 0)
    config = getattr(getattr(model, "residency", None), "config", None)
    total_experts = int(getattr(config, "num_experts", capacity) or capacity)
    cache_devices = tuple(str(value) for value in getattr(snapshot, "cache_devices", ()))
    target_device = str(getattr(snapshot, "target_device", "") or "")
    paging = bool(getattr(snapshot, "paging", False))
    raw_mode = str(getattr(snapshot, "mode", "") or "").strip().casefold()
    mode = raw_mode or ("paged" if paging else "request_global")
    cached_rows = sum(len(layer) for layer in cached_by_layer)
    cached_identities = sorted({identity for layer in cached_by_layer for identity in layer})
    return {
        "physical_expert_offload": bool(getattr(snapshot, "enabled", False)),
        "expert_offload_mode": mode,
        "expert_paging": paging,
        "expert_masters_on_cpu": bool(getattr(snapshot, "masters_on_cpu", False)),
        "expert_masters_pinned": bool(getattr(snapshot, "masters_pinned", False)),
        "expert_master_devices": [
            str(value) for value in getattr(snapshot, "master_devices", ())
        ],
        "expert_master_bytes": int(getattr(snapshot, "master_bytes", 0) or 0),
        "expert_cache_on_target": bool(getattr(snapshot, "caches_on_target", False)),
        "expert_cache_device": target_device,
        "expert_cache_devices": list(cache_devices),
        "expert_cache_capacity": capacity,
        "expert_cache_occupancy": float(getattr(snapshot, "occupancy", 0.0) or 0.0),
        "expert_cache_bytes": cache_bytes,
        "expert_cache_index_bytes": int(getattr(snapshot, "cache_index_bytes", 0) or 0),
        "expert_cache_stage_operations": int(
            getattr(snapshot, "stage_operations", 0) or 0
        ),
        "expert_cache_transferred_rows": int(
            getattr(snapshot, "transferred_expert_rows", 0) or 0
        ),
        "expert_cache_evicted_rows": int(
            getattr(snapshot, "evicted_expert_rows", 0) or 0
        ),
        "expert_cache_transfer_bytes": int(getattr(snapshot, "transfer_bytes", 0) or 0),
        "expert_cache_layer_count": int(getattr(snapshot, "layer_count", 0) or 0),
        "expert_cache_ids_by_layer": [list(layer) for layer in cached_by_layer],
        "expert_cache_cached_rows": cached_rows,
        "expert_cache_cached_identities": cached_identities,
        "resident_expert_count": len(resident_ids),
        "resident_experts": list(resident_ids),
        "resident_capacity": capacity,
        "resident_bytes": cache_bytes,
        "total_experts": total_experts,
    }


def _validated_expert_offload_telemetry(
    model: nn.Module,
    snapshot: object,
    *,
    expected_device: torch.device,
    expected_capacity: int | None,
    expected_paging: bool,
) -> dict[str, object]:
    """Fail closed unless native telemetry proves bounded selective placement."""

    value = _expert_offload_telemetry(model, snapshot)
    assert value is not None
    problems: list[str] = []
    if not value["physical_expert_offload"]:
        problems.append("offload backend is not enabled")
    if not value["expert_masters_on_cpu"]:
        problems.append("complete expert masters are not all on CPU")
    if not value["expert_cache_on_target"]:
        problems.append("layer caches are not all on the execution device")
    if bool(value["expert_paging"]) != expected_paging:
        problems.append(
            "native offload mode does not match the requested "
            f"{'paged' if expected_paging else 'legacy request-global'} policy"
        )

    target_text = str(value["expert_cache_device"])
    try:
        target = torch.device(target_text)
    except (RuntimeError, ValueError):
        problems.append(f"invalid cache target {target_text!r}")
    else:
        if target.type != expected_device.type or (
            expected_device.index is not None and target.index != expected_device.index
        ):
            problems.append(f"cache target {target} does not match {expected_device}")

    capacity = int(value["expert_cache_capacity"])
    native_cap = _native_num_experts(model) if expected_paging else _native_resident_cap(model)
    resident_ids = tuple(int(item) for item in value["resident_experts"])
    if not 1 <= capacity <= native_cap:
        problems.append("cache capacity is outside the checkpoint's bounded policy")
    if expected_capacity is not None and capacity != expected_capacity:
        problems.append(f"cache capacity {capacity} does not match requested {expected_capacity}")
    layer_count = int(value["expert_cache_layer_count"])
    ids_by_layer = tuple(tuple(layer) for layer in value["expert_cache_ids_by_layer"])
    if layer_count < 1 or len(ids_by_layer) != layer_count:
        problems.append("cache telemetry does not cover every expert layer")
    elif expected_paging:
        if resident_ids:
            problems.append("paged offload reports an unexpected request-global resident set")
        if not _valid_layer_cache_ids(ids_by_layer, capacity, int(value["total_experts"])):
            problems.append("paged layer cache identities exceed their bounded capacity")
    else:
        if not 1 <= len(resident_ids) <= capacity:
            problems.append("legacy offload has no bounded request-global resident set")
        if any(layer != resident_ids for layer in ids_by_layer):
            problems.append("layer caches do not share one coherent resident identity set")
    if int(value["expert_master_bytes"]) <= 0:
        problems.append("physical expert master storage reports no allocated bytes")
    if not expected_paging and int(value["expert_cache_bytes"]) <= 0:
        problems.append("legacy physical expert cache reports no allocated bytes")
    if problems:
        raise RuntimeError("SPALMER physical expert placement proof failed: " + "; ".join(problems))
    return value


def _assert_live_expert_placement(
    value: Mapping[str, object],
    *,
    expected_ids: tuple[int, ...] | None,
) -> None:
    """Reject stale or escaped caches instead of publishing misleading telemetry."""

    resident_ids = tuple(int(item) for item in value["resident_experts"])
    capacity = int(value["expert_cache_capacity"])
    ids_by_layer = tuple(
        tuple(int(item) for item in layer)
        for layer in value["expert_cache_ids_by_layer"]
    )
    paging = bool(value.get("expert_paging"))
    total_experts = int(value.get("total_experts") or 0)
    layer_count = int(value.get("expert_cache_layer_count") or 0)
    valid_cache = (
        layer_count >= 1
        and len(ids_by_layer) == layer_count
        and _valid_layer_cache_ids(ids_by_layer, capacity, total_experts)
    )
    coherent = all(layer == resident_ids for layer in ids_by_layer)
    identity_policy_valid = (
        not resident_ids and expected_ids is None and valid_cache
        if paging
        else (
            1 <= len(resident_ids) <= capacity
            and coherent
            and valid_cache
            and (expected_ids is None or resident_ids == expected_ids)
        )
    )
    if (
        not value["physical_expert_offload"]
        or not value["expert_masters_on_cpu"]
        or not value["expert_cache_on_target"]
        or capacity < 1
        or not identity_policy_valid
    ):
        raise RuntimeError("SPALMER physical expert cache lost placement or identity coherence")


def _valid_layer_cache_ids(
    ids_by_layer: Sequence[Sequence[int]],
    capacity: int,
    total_experts: int,
) -> bool:
    if capacity < 1 or total_experts < 1:
        return False
    for layer in ids_by_layer:
        identities = tuple(int(item) for item in layer)
        if len(identities) > capacity or len(set(identities)) != len(identities):
            return False
        if any(identity < 0 or identity >= total_experts for identity in identities):
            return False
    return True


def _full_pool_routing_telemetry(
    model: nn.Module,
    *,
    expert_paging: bool,
) -> dict[str, object]:
    """Describe full-pool trained routing with optional physical expert paging."""

    residency = getattr(model, "residency", None)
    config = getattr(residency, "config", None)
    active = int(
        getattr(config, "active_experts", 0)
        or getattr(config, "min_active_experts", 0)
        or 0
    )
    num_experts = int(getattr(config, "num_experts", active) or active or 1)
    physical = _expert_offload_telemetry(model)
    if expert_paging and physical is None:
        raise RuntimeError("SPALMER physical paging backend is no longer enabled")
    if physical is not None:
        _assert_live_expert_placement(physical, expected_ids=None)
        if bool(physical["expert_paging"]) != expert_paging:
            raise RuntimeError("SPALMER physical paging state changed after load")
    value: dict[str, object] = {
        "routing_mode": "paged_top_k" if expert_paging else "unrestricted_top_k",
        "active_experts": active,
        "minimum_experts": active,
        "maximum_experts": active,
        "active_fraction": active / max(1, num_experts),
        "mean_active_experts": float(active),
        "peak_active_experts": active,
        "expansions": 0,
        "contractions": 0,
        "recomputations": 0,
        "total_experts": num_experts,
        "logical_eligible_experts": num_experts,
        "stop_reason": (
            "full-pool trained top-k with layer-local on-demand paging"
            if expert_paging
            else "full-pool trained top-k"
        ),
    }
    if physical is not None:
        value.update(physical)
    return value


def _fixed_residency_telemetry(model: nn.Module) -> dict[str, object] | None:
    physical = _expert_offload_telemetry(model)
    if physical is None:
        return None
    residency = getattr(model, "residency", None)
    config = getattr(residency, "config", None)
    logical_ids = tuple(int(value) for value in getattr(residency, "ids", ()))
    _assert_live_expert_placement(physical, expected_ids=logical_ids or None)
    active = int(
        getattr(config, "active_experts", 0)
        or getattr(config, "min_active_experts", 0)
        or 0
    )
    total = int(physical["total_experts"] or 1)
    return {
        "routing_mode": "fixed_resident",
        "active_experts": active,
        "minimum_experts": active,
        "maximum_experts": active,
        "active_fraction": active / max(1, total),
        "mean_active_experts": float(active),
        "peak_active_experts": active,
        "expansions": 0,
        "contractions": 0,
        "recomputations": 0,
        "stop_reason": "fixed request-resident expert cache",
        **physical,
    }


def _residency_telemetry(model: nn.Module, decision: object) -> dict[str, object]:
    resident_ids = tuple(int(value) for value in decision.resident_ids)
    active = int(decision.active_experts)
    residency = getattr(model, "residency", None)
    config = getattr(residency, "config", None)
    num_experts = int(getattr(config, "num_experts", len(resident_ids)) or len(resident_ids))
    configured_minimum = int(getattr(config, "min_active_experts", active) or active)
    configured_maximum = int(getattr(config, "max_active_experts", active) or active)
    resident_cap = int(getattr(config, "resident_cap", configured_maximum) or configured_maximum)
    physical = _expert_offload_telemetry(model)
    if physical is not None:
        _assert_live_expert_placement(physical, expected_ids=resident_ids)
        physical_cap = int(physical["expert_cache_capacity"])
        resident_cap = min(resident_cap, physical_cap)
    trace = tuple((int(count), float(signal)) for count, signal in decision.trace)
    expansions = tuple(tuple(int(value) for value in group) for group in decision.expansions)
    effective_nll = getattr(decision, "effective_nll", None)
    signal = float(decision.predictive_entropy if effective_nll is None else effective_nll)
    value = {
        "routing_mode": "adaptive_surprise",
        "active_experts": active,
        "minimum_experts": active,
        "maximum_experts": active,
        "configured_minimum_active_experts": configured_minimum,
        "configured_maximum_active_experts": configured_maximum,
        "active_fraction": active / max(1, num_experts),
        "mean_active_experts": float(active),
        "peak_active_experts": active,
        "surprise": signal,
        "signal_kind": "predictive entropy" if effective_nll is None else "prompt NLL",
        "surprise_ema": float(decision.average_surprise),
        "resident_expert_count": len(resident_ids),
        "resident_experts": list(resident_ids),
        "resident_capacity": resident_cap,
        "resident_fraction": len(resident_ids) / max(1, num_experts),
        "expansions": len(expansions),
        "contractions": 0,
        "recomputations": max(0, len(trace) - 1),
        "trace": [
            {
                "resident_experts": count,
                "active_experts": active,
                "signal": signal,
            }
            for count, signal in trace
        ],
        "expansion_groups": [list(group) for group in expansions],
        "stop_reason": "request capacity selected at prefill",
    }
    if physical is not None:
        value.update(physical)
    return value


__all__ = [
    "SpalmerCausalLMFacade",
    "SpalmerTokenizerFacade",
    "is_spalmer_checkpoint",
    "load_spalmer_runtime",
    "require_executable_spalmer_checkpoint",
    "resolve_spalmer_checkpoint",
    "scan_spalmer_checkpoint",
    "require_spalmer_precision_capability",
]
