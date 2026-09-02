"""Compatibility boundary between native SPALMER bundles and yyLab runtimes.

SPALMER checkpoints are trusted local ``.pt`` bundles, not Hugging Face
directories.  This module deliberately keeps their detection and execution
separate from ``AutoModelForCausalLM`` while presenting the small HF-shaped
surface used by yyLab's existing chat and benchmark paths.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

SPALMER_FORMAT = "spalmer.prototype.checkpoint"
SPALMER_FILENAME = "spalmer.pt"
DEFAULT_CONTEXT_LIMIT = 8_192


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


def scan_spalmer_checkpoint(ref: str | Path) -> dict[str, Any]:
    """Inspect a trusted local SPALMER bundle without constructing its model.

    ``weights_only=True`` constrains unpickling to PyTorch's safe allowlist and
    ``mmap=True`` prevents registration from materializing a second in-memory
    copy of all checkpoint tensors.  SPALMER's checkpoint payload is required
    to remain a plain mapping of primitive metadata and tensors.
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
        # This is a yyLab safety budget, not a RoPE limit: SPALMER is NoPE.
        "max_position_embeddings": trained_context or DEFAULT_CONTEXT_LIMIT,
        "spalmer_format_version": payload.get("format_version"),
        "spalmer_parameter_dtype": payload.get("model_parameter_dtype"),
        "spalmer_experts_config": experts_config,
        "spalmer_kda_config": kda_config,
        "spalmer_mla_config": mla_config,
        "spalmer_directional_config": directional_config,
        "spalmer_atxy_config": atxy_config,
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
        max_context_tokens: int,
        cuda_available: bool,
        notes: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.spalmer_config = model.config
        self.checkpoint_path = checkpoint_path
        self.metadata = dict(metadata)
        self.dynamic_residency = bool(dynamic_residency)
        self.cuda_available = bool(cuda_available)
        self.cuda = self.cuda_available
        self.notes = list(notes)
        self.last_routing_telemetry: dict[str, object] | None = None
        self.config = SimpleNamespace(
            model_type="spalmer",
            architectures=["SPALMERCausalLM"],
            vocab_size=int(model.config.vocab_size),
            hidden_size=int(model.config.d_model),
            num_hidden_layers=int(model.config.n_layers),
            max_position_embeddings=int(max_context_tokens),
            use_cache=True,
            is_encoder_decoder=False,
        )

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
        self.dynamic_residency = bool(enabled)

    def routing_telemetry(self) -> dict[str, object] | None:
        if self.last_routing_telemetry is not None:
            return dict(self.last_routing_telemetry)
        return _fixed_residency_telemetry(self.model)

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
        del use_cache, return_dict
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

        generated = input_ids.to(device=self.device, dtype=torch.long)
        mask = None if attention_mask is None else attention_mask.to(device=self.device)
        eos_ids = _eos_ids(
            self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        )
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(torch.initial_seed() if seed is None else seed))

        use_dynamic = self.dynamic_residency if dynamic_residency is None else dynamic_residency
        request_open = False
        was_training = self.model.training
        decision = None
        self.last_routing_telemetry = None
        self.model.eval()
        try:
            if streamer is not None:
                streamer.put(generated.detach().cpu())
            if max_new_tokens == 0:
                return generated

            if use_dynamic:
                from spalmer.experts import choose_inference_residency

                decision = choose_inference_residency(self.model, generated)
                output = decision.output
                request_open = True
                # Publish the request receipt as soon as residency selection
                # finishes so chat can display it while recurrent decoding is
                # still streaming, rather than only after the final token.
                self.last_routing_telemetry = _residency_telemetry(self.model, decision)
            else:
                output = self.model(generated, attention_mask=mask)
                self.last_routing_telemetry = _fixed_residency_telemetry(self.model)
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
                output = self.model(
                    next_token,
                    token_mixer_states=token_states,
                    channel_mixer_states=channel_states,
                    execution_mode="decode",
                )
                token_states = output.token_mixer_states
                channel_states = output.channel_mixer_states
            return generated
        finally:
            if request_open:
                self.model.end_residency_request()
            self.model.train(was_training)
            if streamer is not None:
                streamer.end()


def load_spalmer_runtime(
    ref: str | Path,
    *,
    dynamic_residency: bool = True,
    max_context_tokens: int | None = None,
    expert_offload: bool = True,
    expert_cache_size: int | None = None,
) -> SpalmerCausalLMFacade:
    """Load a native bundle and return an HF-shaped inference facade."""

    checkpoint_path = resolve_spalmer_checkpoint(ref)
    if max_context_tokens is not None and max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    if expert_cache_size is not None and expert_cache_size <= 0:
        raise ValueError("expert_cache_size must be positive")

    # Keep SPALMER optional for yyLab installations that only use HF models.
    from spalmer import load_checkpoint

    model, loaded_tokenizer, metadata = load_checkpoint(checkpoint_path, map_location="cpu")
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    offload: dict[str, object] | None = None
    if cuda:
        # Checkpoint loading is deliberately CPU-first.  Casting on CPU is
        # safe; the selective placement call below is the only operation that
        # may move SPALMER weights onto the accelerator.
        model.to(device="cpu", dtype=torch.bfloat16)
        model.eval()
        if expert_offload:
            enable_offload = getattr(model, "enable_expert_offload", None)
            if not callable(enable_offload):
                raise RuntimeError(
                    "this SPALMER runtime does not implement physical expert offload"
                )
            native_cap = _native_resident_cap(model)
            if expert_cache_size is not None and expert_cache_size > native_cap:
                raise ValueError(
                    "expert_cache_size cannot exceed the checkpoint's bounded "
                    f"resident cap ({native_cap})"
                )
            snapshot = enable_offload(
                device,
                cache_size=expert_cache_size,
                non_blocking=True,
                pin_memory=True,
            )
            try:
                offload = _validated_expert_offload_telemetry(
                    model,
                    snapshot,
                    expected_device=device,
                    expected_capacity=expert_cache_size,
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
        model.to(device=device, dtype=torch.float32)
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
        "Loaded native SPALMER checkpoint.",
        "SPALMER uses NoPE and native recurrent KDA/MLA state; HF RoPE and KV-cache modes do not apply.",
    ]
    if offload is not None:
        pinned = "pinned " if offload["expert_masters_pinned"] else ""
        notes.extend(
            (
                "Physical SPALMER expert offload enabled: complete expert master banks "
                f"remain in {pinned}CPU RAM and only request-resident rows execute on "
                f"{offload['expert_cache_device']}.",
                "SPALMER expert cache is hard-bounded at "
                f"{offload['expert_cache_capacity']}/{offload['total_experts']} identities "
                "for fixed and adaptive routing.",
            )
        )
    elif cuda:
        notes.append("SPALMER parameters run as BF16 on CUDA; physical expert offload is disabled.")
    else:
        notes.append(
            "SPALMER is using the FP32 CPU fallback; all experts remain CPU-resident and "
            "an accelerator expert cache is not applicable."
        )
    return SpalmerCausalLMFacade(
        model,
        tokenizer,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        dynamic_residency=dynamic_residency,
        max_context_tokens=context_limit,
        cuda_available=cuda,
        notes=notes,
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
    return {
        "physical_expert_offload": bool(getattr(snapshot, "enabled", False)),
        "expert_offload_mode": "cpu_master_gpu_cache",
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
    native_cap = _native_resident_cap(model)
    resident_ids = tuple(int(item) for item in value["resident_experts"])
    if not 1 <= len(resident_ids) <= capacity <= native_cap:
        problems.append(
            "resident identity count/capacity is outside the checkpoint's bounded policy"
        )
    if expected_capacity is not None and capacity != expected_capacity:
        problems.append(f"cache capacity {capacity} does not match requested {expected_capacity}")
    layer_count = int(value["expert_cache_layer_count"])
    ids_by_layer = tuple(tuple(layer) for layer in value["expert_cache_ids_by_layer"])
    if layer_count < 1 or len(ids_by_layer) != layer_count:
        problems.append("cache telemetry does not cover every expert layer")
    elif any(layer != resident_ids for layer in ids_by_layer):
        problems.append("layer caches do not share one coherent resident identity set")
    if int(value["expert_master_bytes"]) <= 0 or int(value["expert_cache_bytes"]) <= 0:
        problems.append("physical expert storage reports no allocated bytes")
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
    coherent = all(
        tuple(int(item) for item in layer) == resident_ids
        for layer in value["expert_cache_ids_by_layer"]
    )
    if (
        not value["physical_expert_offload"]
        or not value["expert_masters_on_cpu"]
        or not value["expert_cache_on_target"]
        or not 1 <= len(resident_ids) <= capacity
        or not coherent
        or (expected_ids is not None and resident_ids != expected_ids)
    ):
        raise RuntimeError("SPALMER physical expert cache lost placement or identity coherence")


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
    "resolve_spalmer_checkpoint",
    "scan_spalmer_checkpoint",
]
