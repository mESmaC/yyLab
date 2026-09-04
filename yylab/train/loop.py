from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from torch.optim import Optimizer


@dataclass
class LoopOpts:
    micro_batch: int = 1
    grad_accum: int = 16
    max_steps: int = 100
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int | None = None
    device: str = "cuda"
    run_dir: Path | None = None
    stage: str = "train"
    step_offset: int = 0
    total_steps: int | None = None
    prefer_cudnn_sdpa: bool = True
    notes: list[str] = field(default_factory=list)


FUSED_ADAMW_MAX_PARAMS = 50_000_000

OPTIMIZER_IMPLEMENTATIONS = (
    "torch_adamw",
    "fused_adamw",
    "adamw_8bit",
    "paged_adamw_8bit",
    "cpu_offload_adamw",
)
OPTIMIZER_STATE_MODES = ("auto", "gpu", "cpu_paged", "cpu_offload")

_OPTIMIZER_PARAMETER_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class OptimizerResolution:
    implementation: str
    state: str
    state_precision: str
    master_weight_precision: str
    capability_detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "implementation": self.implementation,
            "state": self.state,
            "state_precision": self.state_precision,
            "master_weight_precision": self.master_weight_precision,
            "capability_detail": self.capability_detail,
        }


def _parameter_dtype_name(params: list[torch.Tensor]) -> str:
    names = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in params})
    return names[0] if len(names) == 1 else "mixed[" + ",".join(names) + "]"


def _probe_optimizer_implementation(
    implementation: str,
    device: torch.device,
    parameter_dtype: str,
) -> tuple[bool, str]:
    """Execute one real update and verify the requested state representation."""

    try:
        if implementation == "paged_adamw_8bit":
            # bitsandbytes only allocates paged storage above its 100k-element
            # threshold. A toy scalar probe would therefore be a false positive.
            elements = 131_072
        elif implementation == "adamw_8bit":
            elements = 8_192
        else:
            elements = 8
        dtype = _OPTIMIZER_PARAMETER_DTYPES.get(parameter_dtype)
        if dtype is None:
            raise RuntimeError(
                f"optimizer capability probing requires one exact parameter dtype, "
                f"got {parameter_dtype!r}"
            )
        if device.type == "cpu" and dtype != torch.float32:
            raise RuntimeError(
                f"CPU optimizer execution is exposed only for FP32 parameters, got {dtype}"
            )
        parameter = torch.nn.Parameter(
            torch.zeros(elements, device=device, dtype=dtype)
        )
        parameter.grad = torch.ones_like(parameter)
        if implementation == "torch_adamw":
            optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        elif implementation == "fused_adamw":
            if device.type != "cuda":
                raise RuntimeError("fused AdamW requires CUDA")
            optimizer = torch.optim.AdamW([parameter], lr=1e-3, fused=True)
        elif implementation in {"adamw_8bit", "paged_adamw_8bit"}:
            if device.type != "cuda":
                raise RuntimeError("bitsandbytes optimizers require CUDA")
            import bitsandbytes as bnb

            optimizer_type = (
                bnb.optim.PagedAdamW8bit
                if implementation == "paged_adamw_8bit"
                else bnb.optim.AdamW8bit
            )
            optimizer = optimizer_type([parameter], lr=1e-3, min_8bit_size=0)
        elif implementation == "cpu_offload_adamw":
            optimizer = CPUOffloadAdamW([parameter], lr=1e-3)
        else:  # pragma: no cover - callers use the public constant
            raise ValueError(f"unknown optimizer implementation: {implementation}")

        optimizer.step()
        state = optimizer.state[parameter]
        if not bool(torch.isfinite(parameter).all()) or bool(
            torch.equal(parameter, torch.zeros_like(parameter))
        ):
            raise RuntimeError("optimizer update was non-finite or left the parameter unchanged")
        if implementation in {"torch_adamw", "fused_adamw"}:
            moments = [state.get("exp_avg"), state.get("exp_avg_sq")]
            if not all(
                isinstance(value, torch.Tensor)
                and value.device == device
                and value.dtype == parameter.dtype
                for value in moments
            ):
                raise RuntimeError(
                    "Adam moments did not match the parameter device and precision"
                )
            detail = (
                f"real update passed with {parameter.dtype} moments on {device}"
            )
        elif implementation == "adamw_8bit":
            moments = [state.get("state1"), state.get("state2")]
            if not all(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.device == device
                for value in moments
            ):
                raise RuntimeError("8-bit Adam moments were not uint8 CUDA tensors")
            detail = "real update passed with blockwise uint8 CUDA moments"
        elif implementation == "paged_adamw_8bit":
            moments = [state.get("state1"), state.get("state2")]
            if not all(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.device.type == "cpu"
                and bool(getattr(value, "is_paged", False))
                for value in moments
            ):
                raise RuntimeError("paged 8-bit moments were not CPU-backed paged tensors")
            detail = (
                "real update passed with CPU-backed paged uint8 large moments; "
                "small buffers remain CUDA-resident"
            )
        else:
            tensors = [state.get("master_param"), state.get("exp_avg"), state.get("exp_avg_sq")]
            if not all(
                isinstance(value, torch.Tensor)
                and value.device.type == "cpu"
                and value.dtype == torch.float32
                for value in tensors
            ):
                raise RuntimeError("offloaded master and moments were not FP32 CPU tensors")
            detail = "real update passed with FP32 CPU master and moments"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return True, detail
    except Exception as exc:  # capability detection must fail closed
        return False, f"real optimizer update failed: {exc}"


@lru_cache(maxsize=8)
def _probe_optimizer_capabilities(
    use_cuda: bool,
    device_index: int,
    parameter_dtype: str,
) -> tuple[tuple[str, bool, str], ...]:
    device = torch.device("cuda", device_index) if use_cuda else torch.device("cpu")
    return tuple(
        (
            implementation,
            *_probe_optimizer_implementation(
                implementation,
                device,
                parameter_dtype,
            ),
        )
        for implementation in OPTIMIZER_IMPLEMENTATIONS
    )


def optimizer_capability_status(
    *,
    cuda: bool | None = None,
    device_index: int | None = None,
    parameter_dtype: str | None = None,
    force_refresh: bool = False,
) -> dict[str, tuple[bool, str]]:
    """Return optimizer paths proven executable on the selected local device."""

    requested_cuda = torch.cuda.is_available() if cuda is None else bool(cuda)
    if requested_cuda and not torch.cuda.is_available():
        return {
            implementation: (False, "CUDA was requested but is unavailable")
            for implementation in OPTIMIZER_IMPLEMENTATIONS
        }
    index = 0
    if requested_cuda:
        index = torch.cuda.current_device() if device_index is None else int(device_index)
        if index < 0 or index >= torch.cuda.device_count():
            return {
                implementation: (False, f"CUDA device index {index} is unavailable")
                for implementation in OPTIMIZER_IMPLEMENTATIONS
            }
    if force_refresh:
        _probe_optimizer_capabilities.cache_clear()
    resolved_dtype = str(
        parameter_dtype or ("bfloat16" if requested_cuda else "float32")
    ).strip().casefold()
    return {
        implementation: (available, detail)
        for implementation, available, detail in _probe_optimizer_capabilities(
            requested_cuda,
            index,
            resolved_dtype,
        )
    }


def _implementation_state(implementation: str, *, cuda: bool) -> str:
    if implementation == "paged_adamw_8bit":
        return "cpu_paged"
    if implementation == "cpu_offload_adamw":
        return "cpu_offload"
    return "gpu" if cuda else "cpu"


def resolve_optimizer_choice(
    *,
    trainable_parameters: int,
    cuda: bool,
    optimizer_state: str,
    implementation: str,
    parameter_dtype: str,
    device_index: int | None = None,
) -> OptimizerResolution:
    """Resolve one verified optimizer without cross-implementation fallback."""

    requested_state = str(optimizer_state or "auto")
    requested_implementation = str(implementation or "auto")
    if requested_state not in OPTIMIZER_STATE_MODES:
        raise ValueError(f"unknown optimizer-state mode: {requested_state}")
    if requested_implementation not in {"auto", *OPTIMIZER_IMPLEMENTATIONS}:
        raise ValueError(f"unknown optimizer implementation: {requested_implementation}")
    if requested_state == "gpu" and not cuda:
        raise ValueError("GPU optimizer state was requested, but training is not using CUDA")

    inferred_state = (
        _implementation_state(requested_implementation, cuda=cuda)
        if requested_implementation != "auto"
        else None
    )
    if (
        requested_state != "auto"
        and inferred_state is not None
        and requested_state != inferred_state
    ):
        raise ValueError(
            f"optimizer implementation {requested_implementation!r} stores state as "
            f"{inferred_state!r}, not requested {requested_state!r}"
        )

    status = optimizer_capability_status(
        cuda=cuda,
        device_index=device_index,
        parameter_dtype=parameter_dtype,
    )
    if requested_implementation != "auto":
        selected = requested_implementation
    elif requested_state == "cpu_offload":
        selected = "cpu_offload_adamw"
    elif requested_state == "cpu_paged":
        selected = "paged_adamw_8bit"
    elif requested_state == "gpu":
        candidates = (
            ("fused_adamw", "adamw_8bit", "torch_adamw")
            if trainable_parameters <= FUSED_ADAMW_MAX_PARAMS
            else ("adamw_8bit", "fused_adamw", "torch_adamw")
        )
        selected = next((candidate for candidate in candidates if status[candidate][0]), "")
    elif cuda and trainable_parameters > FUSED_ADAMW_MAX_PARAMS:
        candidates = (
            "paged_adamw_8bit",
            "adamw_8bit",
            "fused_adamw",
            "torch_adamw",
        )
        selected = next((candidate for candidate in candidates if status[candidate][0]), "")
    else:
        candidates = (
            ("fused_adamw", "adamw_8bit", "torch_adamw")
            if cuda
            else ("torch_adamw", "cpu_offload_adamw")
        )
        selected = next((candidate for candidate in candidates if status[candidate][0]), "")
    if not selected:
        raise RuntimeError(
            f"no execution-verified optimizer is available for state mode {requested_state!r}"
        )
    available, detail = status[selected]
    if not available:
        raise RuntimeError(f"{selected} is unavailable: {detail}")

    resolved_state = _implementation_state(selected, cuda=cuda)
    if requested_state != "auto" and requested_state != resolved_state:
        raise ValueError(
            f"optimizer state {requested_state!r} cannot be provided by {selected!r}"
        )
    if selected in {"adamw_8bit", "paged_adamw_8bit"}:
        state_precision = "int8_blockwise"
        master_precision = "none"
    elif selected == "cpu_offload_adamw":
        state_precision = "fp32"
        master_precision = "fp32"
    else:
        state_precision = parameter_dtype
        master_precision = "none"
    return OptimizerResolution(
        implementation=selected,
        state=resolved_state,
        state_precision=state_precision,
        master_weight_precision=master_precision,
        capability_detail=detail,
    )


def optimizer_runtime_metadata(optimizer: Optimizer) -> dict[str, str]:
    metadata = getattr(optimizer, "_yylab_optimizer_resolution", None)
    if not isinstance(metadata, dict):
        raise RuntimeError("optimizer has no yyLab resolution receipt")
    return dict(metadata)


class CPUOffloadAdamW(Optimizer):
    """AdamW with FP32 master weights and moments in system RAM.

    Trainable parameters and gradients remain on the accelerator. Each step
    transfers gradients to CPU, updates the offloaded state, then copies the
    new parameter values back. This trades PCIe traffic for predictable VRAM
    use and is the useful part of ZeRO-Offload on a single-GPU workstation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if eps < 0:
            raise ValueError("epsilon must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Adam betas must be within [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )

    def master_state_summary(self) -> dict[str, float | int | str]:
        values = list(self.state.values())
        moment_storage = sum(
            tensor.numel() * tensor.element_size()
            for value in values
            for name in ("exp_avg", "exp_avg_sq")
            if isinstance((tensor := value.get(name)), torch.Tensor)
        )
        storage = sum(
            value["master_param"].numel() * value["master_param"].element_size()
            for value in values
            if "master_param" in value
        )
        return {
            "format": "fp32",
            "bytes": storage,
            "promoted_fraction": 0.0,
            "moment_format": "fp32",
            "moment_bytes": moment_storage,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore offloaded state without a transient copy on the accelerator.

        ``Optimizer.load_state_dict`` casts every state tensor to its owning
        parameter's dtype and device. That is appropriate for ordinary GPU
        optimizers, but can briefly materialize the entire offloaded state on
        the accelerator before we move it back to CPU. Map saved parameter IDs
        directly so large resumes retain the same memory behavior as fresh
        runs.
        """

        if not isinstance(state_dict, dict):
            raise TypeError("optimizer state_dict must be a dictionary")
        saved_groups = state_dict.get("param_groups", [])
        saved_states = state_dict.get("state", {})
        for saved_group in saved_groups:
            master_format = str(saved_group.get("master_weight_storage", "fp32") or "fp32")
            if master_format != "fp32":
                raise ValueError(
                    "legacy FP8 optimizer-master checkpoints are not executable; "
                    "restart with an FP32 CPU master"
                )
        if any("master_fp8" in saved for saved in saved_states.values()):
            raise ValueError(
                "legacy FP8 optimizer-master checkpoints are not executable; "
                "restart with an FP32 CPU master"
            )
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "loaded optimizer state has a different number of parameter groups"
            )
        self.state.clear()
        for saved_group, live_group in zip(saved_groups, self.param_groups, strict=True):
            saved_params = list(saved_group.get("params", []))
            live_params = list(live_group.get("params", []))
            if len(saved_params) != len(live_params):
                raise ValueError(
                    "loaded optimizer state contains a parameter group that doesn't "
                    "match the size of optimizer's group"
                )
            group_values = {
                name: copy.deepcopy(value)
                for name, value in saved_group.items()
                if name
                not in {
                    "params",
                    "master_weight_storage",
                    "adaptive_max_promoted_fraction",
                    "adaptive_promote_threshold",
                    "adaptive_demote_threshold",
                    "adaptive_demote_patience",
                }
            }
            live_group.clear()
            live_group.update(group_values)
            live_group["params"] = live_params
            for saved_id, parameter in zip(
                saved_params, live_params, strict=True
            ):
                saved = saved_states.get(saved_id, {})
                live: dict = {}
                tensor_dtypes = {
                    "master_param": torch.float32,
                    "exp_avg": torch.float32,
                    "exp_avg_sq": torch.float32,
                }
                for name, value in saved.items():
                    if isinstance(value, torch.Tensor):
                        live[name] = value.detach().to(
                            device="cpu",
                            dtype=tensor_dtypes.get(name, value.dtype),
                            copy=True,
                        )
                    else:
                        live[name] = copy.deepcopy(value)
                self.state[parameter] = live

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("CPUOffloadAdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    master = parameter.detach().to(
                        device="cpu", dtype=torch.float32, copy=True
                    )
                    state["exp_avg"] = torch.zeros_like(master)
                    state["exp_avg_sq"] = torch.zeros_like(master)
                    state["master_param"] = master
                state["step"] += 1
                master = state["master_param"]
                gradient = parameter.grad.detach().to(
                    device="cpu", dtype=torch.float32, copy=True
                )
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group["weight_decay"]:
                    master.mul_(1.0 - lr * group["weight_decay"])
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(group["eps"])
                master.addcdiv_(exp_avg, denominator, value=-(lr / bias_correction1))
                parameter.copy_(master.to(device=parameter.device, dtype=parameter.dtype))
        return loss


def _model_config(model: torch.nn.Module):
    config = getattr(model, "config", None)
    if config is not None:
        return config
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return getattr(get_base_model(), "config", None)
    return None


def cudnn_sdpa_eligible(model: torch.nn.Module, device: torch.device | str) -> bool:
    """Return whether cuDNN SDPA is a safe first choice for this model.

    yyLab currently enables this path only for the Qwen3 grouped-query shape we
    benchmark. Other architectures and unsupported Qwen shapes retain PyTorch's
    normal SDPA selection.
    """

    device = torch.device(device)
    config = _model_config(model)
    if (
        device.type != "cuda"
        or config is None
        or str(getattr(config, "model_type", "")).lower() != "qwen3"
        or getattr(config, "_attn_implementation", "sdpa") != "sdpa"
    ):
        return False
    heads = int(getattr(config, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    head_dim = int(getattr(config, "head_dim", 0) or 0)
    if head_dim <= 0 and heads > 0:
        head_dim = int(getattr(config, "hidden_size", 0) or 0) // heads
    if not (heads > kv_heads > 0 and heads % kv_heads == 0 and 0 < head_dim <= 256 and head_dim % 8 == 0):
        return False
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(index) >= (8, 0)
            and torch.backends.cudnn.is_available()
            and torch.backends.cuda.cudnn_sdp_enabled()
        )
    except (AttributeError, RuntimeError):
        return False


def attention_backend_context(
    model: torch.nn.Module,
    device: torch.device | str,
    *,
    enabled: bool = True,
) -> AbstractContextManager:
    """Prefer cuDNN for compatible Qwen3 GQA and retain all SDPA fallbacks.

    This scoped context restores PyTorch's backend state on exit. Callers keep
    it active through backward so checkpoint recomputation uses the same kernel
    policy as the original forward.
    """

    if not enabled or not cudnn_sdpa_eligible(model, device):
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        try:
            return sdpa_kernel(backends, set_priority=True)
        except TypeError:
            return sdpa_kernel(backends)
    except (AttributeError, ImportError):
        return nullcontext()


def make_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    cuda: bool = True,
    optimizer_state: str = "auto",
    implementation: str = "auto",
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("cannot build an optimizer without trainable parameters")
    trainable = sum(p.numel() for p in params)
    parameter_devices = {parameter.device for parameter in params}
    if len(parameter_devices) != 1:
        devices = ", ".join(sorted(str(device) for device in parameter_devices))
        raise ValueError(
            "optimizer selection requires all trainable parameters on one exact device; "
            f"found {devices}"
        )
    parameter_device = next(iter(parameter_devices))
    if cuda and parameter_device.type != "cuda":
        raise ValueError("CUDA optimizer requested for parameters that are not all on CUDA")
    if not cuda and parameter_device.type != "cpu":
        raise ValueError(
            "CPU optimizer selection requires CPU parameters; "
            f"found {parameter_device}"
        )
    device_index = parameter_device.index if parameter_device.type == "cuda" else None
    resolution = resolve_optimizer_choice(
        trainable_parameters=trainable,
        cuda=cuda,
        optimizer_state=optimizer_state,
        implementation=implementation,
        parameter_dtype=_parameter_dtype_name(params),
        device_index=device_index,
    )
    common = {"lr": lr, "weight_decay": weight_decay}
    # Keep the old default call shape for compatibility with lightweight test
    # doubles while forwarding every user-selected numerical control.
    if betas != (0.9, 0.999):
        common["betas"] = betas
    if eps != 1e-8:
        common["eps"] = eps
    try:
        if resolution.implementation == "cpu_offload_adamw":
            optimizer: Optimizer = CPUOffloadAdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
        elif resolution.implementation == "paged_adamw_8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdamW8bit(params, **common, min_8bit_size=0)
        elif resolution.implementation == "adamw_8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, **common, min_8bit_size=0)
        elif resolution.implementation == "fused_adamw":
            optimizer = torch.optim.AdamW(params, **common, fused=True)
        elif resolution.implementation == "torch_adamw":
            optimizer = torch.optim.AdamW(params, **common)
        else:  # pragma: no cover - resolution is closed over the public enum
            raise AssertionError(f"unhandled optimizer: {resolution.implementation}")
    except Exception as exc:  # no implementation substitution after resolution
        raise RuntimeError(
            f"verified optimizer {resolution.implementation} failed to construct"
        ) from exc
    optimizer._yylab_optimizer_resolution = resolution.to_dict()
    return optimizer


def train_loop(
    model: torch.nn.Module,
    batches: Iterator[dict],
    optimizer: Optimizer,
    scheduler,
    opts: LoopOpts,
    aux_fn: Callable[[], torch.Tensor] | None = None,
    save_fn: Callable[[int], None] | None = None,
    metrics_fn: Callable[[], dict] | None = None,
    loss_aux_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    after_step_fn: Callable[[int, torch.Tensor], None] | None = None,
) -> dict:
    metrics_path = opts.run_dir / "metrics.jsonl" if opts.run_dir else None
    progress_path = opts.run_dir / "progress.json" if opts.run_dir else None
    if opts.run_dir:
        opts.run_dir.mkdir(parents=True, exist_ok=True)

    def write_progress(stage: str, step: int, loss: float, msg: str = "") -> None:
        if progress_path is None:
            return
        global_step = opts.step_offset + step
        progress_path.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "step": global_step,
                    "stage_step": step,
                    "stage_steps": opts.max_steps,
                    "max_steps": opts.total_steps or opts.max_steps,
                    "loss": loss,
                    "msg": msg,
                }
            ),
            "utf-8",
        )

    def log_metrics(rec: dict) -> None:
        if metrics_path is not None:
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    device = torch.device(opts.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    micro_step = 0
    opt_step = 0
    window_loss = torch.zeros((), dtype=torch.float32, device=device)
    window_aux = torch.zeros((), dtype=torch.float32, device=device)
    window_micro = 0
    last_loss = torch.zeros((), dtype=torch.float32, device=device)
    tokens = 0
    supervised_tokens = 0
    sequences_forwarded = 0
    forwarded_real_tokens = 0
    padding_tokens = 0
    model_target_slots = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    use_cudnn_sdpa = opts.prefer_cudnn_sdpa and cudnn_sdpa_eligible(model, device)

    def flush_pending(accumulated_micro: int) -> None:
        nonlocal opt_step
        if 0 < accumulated_micro < opts.grad_accum:
            # Every micro loss is divided by the configured accumulation
            # target. If a finite data slice ends mid-window, restore the
            # intended mean instead of silently shrinking its final update.
            scale = opts.grad_accum / accumulated_micro
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    parameter.grad.mul_(scale)
        if opts.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], opts.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1

    for batch in batches:
        ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        labels = ids
        batch_target_slots = ids.shape[0] * max(ids.shape[1] - 1, 0)
        model_target_slots += int(batch_target_slots)
        if batch.get("labels") is not None:
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
            if labels.shape != ids.shape:
                raise ValueError("batch labels must have the same shape as input_ids")
            # Hugging Face causal-LM heads shift internally and optimize
            # labels[:, 1:]. The first column is context, never a target.
            supervised_tokens += int(labels[:, 1:].ne(-100).sum().item())
        else:
            supervised_tokens += int(batch_target_slots)
        model_inputs = {"input_ids": ids, "labels": labels}
        if batch.get("attention_mask") is not None:
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            if attention_mask.shape != ids.shape:
                raise ValueError("batch attention_mask must have the same shape as input_ids")
            model_inputs["attention_mask"] = attention_mask
            real_tokens = int(attention_mask.sum().item())
        else:
            real_tokens = int(ids.numel())
        # Backward stays inside the context because gradient checkpointing
        # recomputes model forwards during this call.
        with attention_backend_context(model, device, enabled=opts.prefer_cudnn_sdpa):
            out = model(**model_inputs)
            loss_metric = out.loss.detach().float()
            loss = out.loss / opts.grad_accum
            aux_metric = None
            if aux_fn is not None:
                aux = aux_fn() / opts.grad_accum
                aux_metric = aux.detach().float()
                loss = loss + aux
            if loss_aux_fn is not None:
                task_aux = loss_aux_fn(out.loss) / opts.grad_accum
                task_aux_metric = task_aux.detach().float()
                aux_metric = (
                    task_aux_metric
                    if aux_metric is None
                    else aux_metric + task_aux_metric
                )
                loss = loss + task_aux
            loss.backward()
        window_loss.add_(loss_metric)
        if aux_metric is not None:
            window_aux.add_(aux_metric)
        window_micro += 1
        last_loss.copy_(loss_metric)
        tokens += ids.numel()
        sequences_forwarded += int(ids.shape[0])
        forwarded_real_tokens += real_tokens
        padding_tokens += int(ids.numel()) - real_tokens
        micro_step += 1

        if micro_step % opts.grad_accum != 0:
            continue
        flush_pending(opts.grad_accum)

        if opt_step % opts.log_every == 0 or opt_step == opts.max_steps:
            elapsed = time.time() - t0
            rec = {
                "step": opts.step_offset + opt_step,
                "stage": opts.stage,
                "stage_step": opt_step,
                # Synchronize metrics only at the configured logging cadence,
                # not between every forward and backward launch.
                "loss": round((window_loss / max(window_micro, 1)).item(), 4),
                "aux": round((window_aux / max(window_micro, 1)).item(), 4),
                "tokens": tokens,
                "tok_per_s": round(tokens / max(elapsed, 1e-6), 1),
                "elapsed_s": round(elapsed, 1),
            }
            if device.type == "cuda":
                rec["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                rec["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
            if metrics_fn is not None:
                rec.update(metrics_fn())
            log_metrics(rec)
            route_status = (
                f" · route H={rec['router_entropy']:.3f}, dead={rec['dead_experts']}"
                if "router_entropy" in rec
                else ""
            )
            write_progress(
                opts.stage,
                opt_step,
                rec["loss"],
                f"{rec['tok_per_s']} tok/s{route_status}",
            )
            terminal_fields = [
                f"{opts.stage}: step {rec['step']}/{opts.total_steps or opts.max_steps}",
                f"loss={rec['loss']}",
                f"aux={rec['aux']}",
                f"{rec['tok_per_s']} tok/s",
            ]
            if "router_entropy" in rec:
                terminal_fields.extend(
                    (
                        f"route H={rec['router_entropy']:.3f}",
                        f"dead={rec['dead_experts']}",
                    )
                )
            if "cuda_peak_allocated_gb" in rec:
                terminal_fields.append(f"CUDA peak={rec['cuda_peak_allocated_gb']} GB")
            print(" | ".join(terminal_fields), flush=True)
            window_loss.zero_()
            window_aux.zero_()
            window_micro = 0
        if after_step_fn is not None:
            after_step_fn(opt_step, loss_metric)
        if opts.save_every and save_fn and opt_step % opts.save_every == 0:
            save_fn(opt_step)
        if opt_step >= opts.max_steps:
            break

    if micro_step % opts.grad_accum != 0 and opt_step < opts.max_steps:
        flush_pending(micro_step % opts.grad_accum)
        if after_step_fn is not None:
            after_step_fn(opt_step, last_loss)

    data_exhausted = opt_step < opts.max_steps
    final_message = "data exhausted before step cap; saving" if data_exhausted else "step cap reached; saving"
    write_progress(opts.stage, opt_step, last_loss.item(), final_message)
    result = {
        "steps": opt_step,
        "step_cap": opts.max_steps,
        "data_exhausted": data_exhausted,
        "tokens": tokens,
        "sequences_forwarded": sequences_forwarded,
        "forwarded_model_tokens": tokens,
        "forwarded_real_tokens": forwarded_real_tokens,
        "padding_tokens": padding_tokens,
        "model_target_slots": model_target_slots,
        "supervised_tokens": supervised_tokens,
        "ignored_target_slots": model_target_slots - supervised_tokens,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        result["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        result["attention_backend"] = "cudnn-priority-sdpa" if use_cudnn_sdpa else "default-sdpa"
    try:
        import psutil

        result["process_rss_gb"] = round(psutil.Process().memory_info().rss / 2**30, 3)
    except Exception:  # noqa: BLE001
        pass
    return result
