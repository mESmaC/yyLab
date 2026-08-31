from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import Optimizer

from .potentiated import (
    DEFAULT_ADAPTIVE_DEMOTE_PATIENCE,
    DEFAULT_ADAPTIVE_DEMOTE_THRESHOLD,
    DEFAULT_ADAPTIVE_MAX_PROMOTED_FRACTION,
    DEFAULT_ADAPTIVE_PROMOTE_THRESHOLD,
    DEFAULT_POTENTIATION_BLOCK_SIZE,
    FP8_ADAPTIVE_MASTER,
    FP8_POTENTIATED_MASTER,
    MASTER_WEIGHT_FORMATS,
    AdaptivePotentiatedFP8State,
    PotentiatedFP8State,
    encode_adaptive_potentiated_fp8,
    encode_potentiated_fp8,
)

POTENTIATED_MASTER_CHUNK_SIZE = 1_048_576


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


class CPUOffloadAdamW(Optimizer):
    """AdamW with offloaded master weights and FP32 moments in system RAM.

    Trainable parameters and gradients remain on the accelerator. Each step
    transfers gradients to CPU, updates the offloaded state, then copies the
    new parameter values back. This trades PCIe traffic for predictable VRAM
    use and is the useful part of ZeRO-Offload on a single-GPU workstation.

    The potentiated-FP8 modes persist each master as a block-scaled E4M3 base.
    The legacy format stores a packed four-bit sub-ULP residual everywhere;
    the adaptive format stores that residual only for troublesome blocks and
    demotes them with hysteresis after they stabilize. Adam moments and all
    optimizer arithmetic remain FP32. These modes are therefore not equivalent
    to casting a normal AdamW parameter to float8, which would discard most
    small updates.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        master_weight_storage: str = "fp32",
        master_chunk_size: int = POTENTIATED_MASTER_CHUNK_SIZE,
        adaptive_max_promoted_fraction: float = DEFAULT_ADAPTIVE_MAX_PROMOTED_FRACTION,
        adaptive_promote_threshold: float = DEFAULT_ADAPTIVE_PROMOTE_THRESHOLD,
        adaptive_demote_threshold: float = DEFAULT_ADAPTIVE_DEMOTE_THRESHOLD,
        adaptive_demote_patience: int = DEFAULT_ADAPTIVE_DEMOTE_PATIENCE,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if eps < 0:
            raise ValueError("epsilon must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Adam betas must be within [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        if master_weight_storage not in MASTER_WEIGHT_FORMATS:
            raise ValueError(f"unknown master-weight storage: {master_weight_storage}")
        if master_chunk_size < 1 or master_chunk_size % DEFAULT_POTENTIATION_BLOCK_SIZE:
            raise ValueError(
                "potentiated-master chunk size must be a positive multiple of the block size"
            )
        if not 0.0 <= adaptive_max_promoted_fraction <= 1.0:
            raise ValueError("adaptive max promoted fraction must be within [0, 1]")
        if adaptive_demote_threshold > adaptive_promote_threshold:
            raise ValueError("adaptive demote threshold cannot exceed promote threshold")
        if not 1 <= adaptive_demote_patience <= 255:
            raise ValueError("adaptive demote patience must be within [1, 255]")
        self.master_weight_storage = master_weight_storage
        self.master_chunk_size = int(master_chunk_size)
        self.adaptive_max_promoted_fraction = float(adaptive_max_promoted_fraction)
        self.adaptive_promote_threshold = float(adaptive_promote_threshold)
        self.adaptive_demote_threshold = float(adaptive_demote_threshold)
        self.adaptive_demote_patience = int(adaptive_demote_patience)
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "master_weight_storage": master_weight_storage,
                "adaptive_max_promoted_fraction": self.adaptive_max_promoted_fraction,
                "adaptive_promote_threshold": self.adaptive_promote_threshold,
                "adaptive_demote_threshold": self.adaptive_demote_threshold,
                "adaptive_demote_patience": self.adaptive_demote_patience,
            },
        )

    @staticmethod
    def _restore_potentiated_master(state: dict) -> PotentiatedFP8State:
        return PotentiatedFP8State(
            base=state["master_fp8"],
            potentiation=state["master_potentiation"],
            scale=state["master_scale"],
            shape=tuple(state["master_shape"]),
            block_size=int(state["master_block_size"]),
        )

    @staticmethod
    def _store_potentiated_master(state: dict, master: PotentiatedFP8State) -> None:
        state["master_fp8"] = master.base
        state["master_potentiation"] = master.potentiation
        state["master_scale"] = master.scale
        state["master_shape"] = master.shape
        state["master_block_size"] = master.block_size

    @staticmethod
    def _store_potentiated_range(
        state: dict,
        master: PotentiatedFP8State,
        start: int,
        end: int,
    ) -> None:
        if start % 2 or start % master.block_size:
            raise ValueError("potentiated-master chunks must start on block boundaries")
        state["master_fp8"][start:end].copy_(master.base.reshape(-1))
        first_byte = start // 2
        last_byte = (end + 1) // 2
        state["master_potentiation"][first_byte:last_byte].copy_(master.potentiation)
        first_block = start // master.block_size
        last_block = (end + master.block_size - 1) // master.block_size
        state["master_scale"][first_block:last_block].copy_(master.scale)

    @staticmethod
    def _restore_adaptive_master(state: dict) -> AdaptivePotentiatedFP8State:
        return AdaptivePotentiatedFP8State(
            base=state["master_fp8"],
            promoted_blocks=state["master_promoted_blocks"],
            potentiation=state["master_promoted_potentiation"],
            scale=state["master_scale"],
            stability=state["master_promotion_stability"],
            shape=tuple(state["master_shape"]),
            block_size=int(state["master_block_size"]),
            max_promoted_fraction=float(state["master_max_promoted_fraction"]),
            promote_threshold=float(state["master_promote_threshold"]),
            demote_threshold=float(state["master_demote_threshold"]),
            demote_patience=int(state["master_demote_patience"]),
        )

    @staticmethod
    def _store_adaptive_master(
        state: dict, master: AdaptivePotentiatedFP8State
    ) -> None:
        state["master_fp8"] = master.base
        state["master_promoted_blocks"] = master.promoted_blocks
        state["master_promoted_potentiation"] = master.potentiation
        state["master_scale"] = master.scale
        state["master_promotion_stability"] = master.stability
        state["master_shape"] = master.shape
        state["master_block_size"] = master.block_size
        state["master_max_promoted_fraction"] = master.max_promoted_fraction
        state["master_promote_threshold"] = master.promote_threshold
        state["master_demote_threshold"] = master.demote_threshold
        state["master_demote_patience"] = master.demote_patience

    @staticmethod
    def _store_adaptive_dense_range(
        state: dict,
        master: AdaptivePotentiatedFP8State,
        start: int,
        end: int,
    ) -> None:
        if start % master.block_size:
            raise ValueError("adaptive-master chunks must start on block boundaries")
        state["master_fp8"][start:end].copy_(master.base.reshape(-1))
        first_block = start // master.block_size
        last_block = (end + master.block_size - 1) // master.block_size
        state["master_scale"][first_block:last_block].copy_(master.scale)
        state["master_promotion_stability"][first_block:last_block].copy_(
            master.stability
        )

    @staticmethod
    def _adaptive_chunk_quota(
        first_block: int,
        last_block: int,
        total_blocks: int,
        max_promoted_fraction: float,
    ) -> int:
        """Distribute a hard parameter-wide promotion cap across chunks."""

        total_limit = int(total_blocks * max_promoted_fraction)
        before = first_block * total_limit // max(1, total_blocks)
        after = last_block * total_limit // max(1, total_blocks)
        return after - before

    @staticmethod
    def _potentiation_generator(state: dict, start: int) -> torch.Generator:
        # Stateless chunk seeding makes an optimizer state_dict resume exact
        # without persisting a large generator blob for every parameter.
        seed = (
            int(state["master_seed"])
            + int(state["step"]) * 6_364_136_223_846_793_005
            + int(start)
        ) % (2**63 - 1)
        return torch.Generator(device="cpu").manual_seed(seed)

    def _initialize_potentiated_master(self, state: dict, parameter: torch.Tensor) -> None:
        count = parameter.numel()
        block_size = DEFAULT_POTENTIATION_BLOCK_SIZE
        state["master_fp8"] = torch.empty(count, dtype=torch.float8_e4m3fn, device="cpu")
        state["master_potentiation"] = torch.zeros((count + 1) // 2, dtype=torch.uint8)
        state["master_scale"] = torch.empty((count + block_size - 1) // block_size)
        state["master_shape"] = tuple(parameter.shape)
        state["master_block_size"] = block_size
        state["master_seed"] = int(torch.randint(0, 2**31 - 1, ()).item())
        flat = parameter.detach().reshape(-1)
        for start in range(0, count, self.master_chunk_size):
            end = min(start + self.master_chunk_size, count)
            source = flat[start:end].to(device="cpu", dtype=torch.float32, copy=True)
            encoded = encode_potentiated_fp8(
                source,
                block_size=block_size,
                generator=self._potentiation_generator(state, start),
            )
            self._store_potentiated_range(state, encoded, start, end)

    def _initialize_adaptive_master(self, state: dict, parameter: torch.Tensor) -> None:
        count = parameter.numel()
        block_size = DEFAULT_POTENTIATION_BLOCK_SIZE
        total_blocks = (count + block_size - 1) // block_size
        state["master_fp8"] = torch.empty(count, dtype=torch.float8_e4m3fn, device="cpu")
        state["master_scale"] = torch.empty(total_blocks, dtype=torch.float32)
        state["master_promotion_stability"] = torch.zeros(
            total_blocks, dtype=torch.uint8
        )
        state["master_shape"] = tuple(parameter.shape)
        state["master_block_size"] = block_size
        state["master_max_promoted_fraction"] = self.adaptive_max_promoted_fraction
        state["master_promote_threshold"] = self.adaptive_promote_threshold
        state["master_demote_threshold"] = self.adaptive_demote_threshold
        state["master_demote_patience"] = self.adaptive_demote_patience
        state["master_seed"] = int(torch.randint(0, 2**31 - 1, ()).item())
        promoted_blocks: list[torch.Tensor] = []
        potentiation: list[torch.Tensor] = []
        flat = parameter.detach().reshape(-1)
        for start in range(0, count, self.master_chunk_size):
            end = min(start + self.master_chunk_size, count)
            first_block = start // block_size
            last_block = (end + block_size - 1) // block_size
            source = flat[start:end].to(device="cpu", dtype=torch.float32, copy=True)
            encoded = encode_adaptive_potentiated_fp8(
                source,
                block_size=block_size,
                generator=self._potentiation_generator(state, start),
                max_promoted_fraction=self.adaptive_max_promoted_fraction,
                max_promoted_blocks=self._adaptive_chunk_quota(
                    first_block,
                    last_block,
                    total_blocks,
                    self.adaptive_max_promoted_fraction,
                ),
                promote_threshold=self.adaptive_promote_threshold,
                demote_threshold=self.adaptive_demote_threshold,
                demote_patience=self.adaptive_demote_patience,
            )
            self._store_adaptive_dense_range(state, encoded, start, end)
            if encoded.promoted_blocks.numel():
                promoted_blocks.append(encoded.promoted_blocks + first_block)
                potentiation.append(encoded.potentiation)
        state["master_promoted_blocks"] = (
            torch.cat(promoted_blocks).to(torch.int32)
            if promoted_blocks
            else torch.empty(0, dtype=torch.int32)
        )
        state["master_promoted_potentiation"] = (
            torch.cat(potentiation)
            if potentiation
            else torch.empty((0, block_size // 2), dtype=torch.uint8)
        )

    def master_state_summary(self) -> dict[str, float | int | str]:
        values = list(self.state.values())
        moment_storage = sum(
            tensor.numel() * tensor.element_size()
            for value in values
            for name in ("exp_avg", "exp_avg_sq")
            if isinstance((tensor := value.get(name)), torch.Tensor)
        )
        if self.master_weight_storage == "fp32":
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
        storage = 0
        promoted = 0
        count = 0
        for value in values:
            if "master_fp8" not in value:
                continue
            if self.master_weight_storage == FP8_ADAPTIVE_MASTER:
                master = self._restore_adaptive_master(value)
            else:
                master = self._restore_potentiated_master(value)
            storage += master.storage_bytes
            promoted += round(master.promoted_fraction * master.numel)
            count += master.numel
        return {
            "format": self.master_weight_storage,
            "bytes": storage,
            "promoted_fraction": promoted / count if count else 0.0,
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
                if name != "params"
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
                    "master_fp8": torch.float8_e4m3fn,
                    "master_potentiation": torch.uint8,
                    "master_promoted_blocks": torch.int32,
                    "master_promoted_potentiation": torch.uint8,
                    "master_promotion_stability": torch.uint8,
                    "master_scale": torch.float32,
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
        if self.param_groups:
            group = self.param_groups[0]
            self.master_weight_storage = str(
                group.get("master_weight_storage", self.master_weight_storage)
            )
            self.adaptive_max_promoted_fraction = float(
                group.get(
                    "adaptive_max_promoted_fraction",
                    self.adaptive_max_promoted_fraction,
                )
            )
            self.adaptive_promote_threshold = float(
                group.get("adaptive_promote_threshold", self.adaptive_promote_threshold)
            )
            self.adaptive_demote_threshold = float(
                group.get("adaptive_demote_threshold", self.adaptive_demote_threshold)
            )
            self.adaptive_demote_patience = int(
                group.get("adaptive_demote_patience", self.adaptive_demote_patience)
            )

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
                    if self.master_weight_storage in {
                        FP8_POTENTIATED_MASTER,
                        FP8_ADAPTIVE_MASTER,
                    }:
                        state["exp_avg"] = torch.zeros(parameter.numel(), dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros(parameter.numel(), dtype=torch.float32)
                        if self.master_weight_storage == FP8_ADAPTIVE_MASTER:
                            self._initialize_adaptive_master(state, parameter)
                        else:
                            self._initialize_potentiated_master(state, parameter)
                    else:
                        master = parameter.detach().to(
                            device="cpu", dtype=torch.float32, copy=True
                        )
                        state["exp_avg"] = torch.zeros_like(master)
                        state["exp_avg_sq"] = torch.zeros_like(master)
                        state["master_param"] = master
                state["step"] += 1
                if self.master_weight_storage == FP8_ADAPTIVE_MASTER:
                    persisted = self._restore_adaptive_master(state)
                    flat_parameter = parameter.reshape(-1)
                    flat_gradient = parameter.grad.detach().reshape(-1)
                    count = parameter.numel()
                    block_size = persisted.block_size
                    total_blocks = persisted.num_blocks
                    promoted_blocks: list[torch.Tensor] = []
                    potentiation: list[torch.Tensor] = []
                    for start in range(0, count, self.master_chunk_size):
                        end = min(start + self.master_chunk_size, count)
                        first_block = start // block_size
                        last_block = (end + block_size - 1) // block_size
                        previous = persisted.slice_range(start, end)
                        master = previous.dequantize()
                        reference = master.clone()
                        gradient = flat_gradient[start:end].to(
                            device="cpu", dtype=torch.float32, copy=True
                        )
                        exp_avg = state["exp_avg"][start:end]
                        exp_avg_sq = state["exp_avg_sq"][start:end]
                        if group["weight_decay"]:
                            master.mul_(1.0 - lr * group["weight_decay"])
                        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(
                            gradient, gradient, value=1.0 - beta2
                        )
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        denominator = (
                            exp_avg_sq.sqrt()
                            .div_(bias_correction2**0.5)
                            .add_(group["eps"])
                        )
                        master.addcdiv_(
                            exp_avg,
                            denominator,
                            value=-(lr / bias_correction1),
                        )
                        encoded = encode_adaptive_potentiated_fp8(
                            master,
                            block_size=block_size,
                            generator=self._potentiation_generator(state, start),
                            previous=previous,
                            update_reference=reference,
                            max_promoted_fraction=persisted.max_promoted_fraction,
                            max_promoted_blocks=self._adaptive_chunk_quota(
                                first_block,
                                last_block,
                                total_blocks,
                                persisted.max_promoted_fraction,
                            ),
                            promote_threshold=persisted.promote_threshold,
                            demote_threshold=persisted.demote_threshold,
                            demote_patience=persisted.demote_patience,
                        )
                        self._store_adaptive_dense_range(state, encoded, start, end)
                        if encoded.promoted_blocks.numel():
                            promoted_blocks.append(encoded.promoted_blocks + first_block)
                            potentiation.append(encoded.potentiation)
                        flat_parameter[start:end].copy_(
                            encoded.dequantize().to(
                                device=parameter.device,
                                dtype=parameter.dtype,
                            )
                        )
                    state["master_promoted_blocks"] = (
                        torch.cat(promoted_blocks).to(torch.int32)
                        if promoted_blocks
                        else torch.empty(0, dtype=torch.int32)
                    )
                    state["master_promoted_potentiation"] = (
                        torch.cat(potentiation)
                        if potentiation
                        else torch.empty((0, block_size // 2), dtype=torch.uint8)
                    )
                    continue
                if self.master_weight_storage == FP8_POTENTIATED_MASTER:
                    persisted = self._restore_potentiated_master(state)
                    flat_parameter = parameter.reshape(-1)
                    flat_gradient = parameter.grad.detach().reshape(-1)
                    count = parameter.numel()
                    for start in range(0, count, self.master_chunk_size):
                        end = min(start + self.master_chunk_size, count)
                        master = persisted.dequantize_range(start, end)
                        gradient = flat_gradient[start:end].to(
                            device="cpu", dtype=torch.float32, copy=True
                        )
                        exp_avg = state["exp_avg"][start:end]
                        exp_avg_sq = state["exp_avg_sq"][start:end]
                        if group["weight_decay"]:
                            master.mul_(1.0 - lr * group["weight_decay"])
                        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(
                            gradient, gradient, value=1.0 - beta2
                        )
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        denominator = (
                            exp_avg_sq.sqrt()
                            .div_(bias_correction2**0.5)
                            .add_(group["eps"])
                        )
                        master.addcdiv_(
                            exp_avg,
                            denominator,
                            value=-(lr / bias_correction1),
                        )
                        encoded = encode_potentiated_fp8(
                            master,
                            block_size=persisted.block_size,
                            generator=self._potentiation_generator(state, start),
                            previous_scale=persisted.scale[
                                start // persisted.block_size : (
                                    end + persisted.block_size - 1
                                )
                                // persisted.block_size
                            ],
                        )
                        self._store_potentiated_range(state, encoded, start, end)
                        flat_parameter[start:end].copy_(
                            encoded.dequantize().to(
                                device=parameter.device,
                                dtype=parameter.dtype,
                            )
                        )
                    continue

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
    optimizer_state: str = "gpu",
    implementation: str = "auto",
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    master_weight_storage: str = "fp32",
    adaptive_max_promoted_fraction: float = DEFAULT_ADAPTIVE_MAX_PROMOTED_FRACTION,
    adaptive_promote_threshold: float = DEFAULT_ADAPTIVE_PROMOTE_THRESHOLD,
    adaptive_demote_threshold: float = DEFAULT_ADAPTIVE_DEMOTE_THRESHOLD,
    adaptive_demote_patience: int = DEFAULT_ADAPTIVE_DEMOTE_PATIENCE,
) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    if optimizer_state not in {"gpu", "auto", "cpu_paged", "cpu_offload"}:
        raise ValueError(f"unknown optimizer-state mode: {optimizer_state}")
    implementations = {
        "auto",
        "torch_adamw",
        "fused_adamw",
        "adamw_8bit",
        "paged_adamw_8bit",
        "cpu_offload_adamw",
    }
    if implementation not in implementations:
        raise ValueError(f"unknown optimizer implementation: {implementation}")
    if master_weight_storage not in MASTER_WEIGHT_FORMATS:
        raise ValueError(f"unknown master-weight storage: {master_weight_storage}")
    cpu_offload = optimizer_state == "cpu_offload" or implementation == "cpu_offload_adamw"
    if master_weight_storage != "fp32" and not cpu_offload:
        raise ValueError("potentiated FP8 master weights require CPU-offload AdamW")
    common = {"lr": lr, "weight_decay": weight_decay}
    # Keep the old default call shape for compatibility with lightweight test
    # doubles while forwarding every user-selected numerical control.
    if betas != (0.9, 0.999):
        common["betas"] = betas
    if eps != 1e-8:
        common["eps"] = eps
    if cpu_offload:
        return CPUOffloadAdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            master_weight_storage=master_weight_storage,
            adaptive_max_promoted_fraction=adaptive_max_promoted_fraction,
            adaptive_promote_threshold=adaptive_promote_threshold,
            adaptive_demote_threshold=adaptive_demote_threshold,
            adaptive_demote_patience=adaptive_demote_patience,
        )
    use_paged = implementation == "paged_adamw_8bit" or (
        implementation == "auto"
        and (
            optimizer_state == "cpu_paged"
            or (optimizer_state == "auto" and trainable > FUSED_ADAMW_MAX_PARAMS)
        )
    )
    if cuda and use_paged:
        try:
            import bitsandbytes as bnb

            # On a single GPU there is no data-parallel state for ZeRO to
            # partition. PagedAdamW8bit provides the useful equivalent here:
            # compact states backed by unified memory that can page to CPU RAM.
            return bnb.optim.PagedAdamW8bit(
                params,
                **common,
            )
        except Exception as exc:  # noqa: BLE001
            if implementation == "paged_adamw_8bit":
                raise RuntimeError("PagedAdamW8bit was requested but is unavailable") from exc
    # Fused AdamW was substantially faster for the small active expert/router
    # parameter sets and adds negligible memory. Large/full fine-tunes retain
    # the memory-oriented 8-bit optimizer path.
    use_fused = implementation == "fused_adamw" or (
        implementation == "auto" and trainable <= FUSED_ADAMW_MAX_PARAMS
    )
    if cuda and use_fused and params and all(p.is_cuda for p in params):
        try:
            return torch.optim.AdamW(params, **common, fused=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            if implementation == "fused_adamw":
                raise RuntimeError("fused AdamW was requested but is unavailable") from exc
    use_8bit = implementation == "adamw_8bit" or implementation == "auto"
    if cuda and use_8bit:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, **common)
        except Exception as exc:  # noqa: BLE001
            if implementation == "adamw_8bit":
                raise RuntimeError("AdamW8bit was requested but is unavailable") from exc
    return torch.optim.AdamW(params, **common)


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
        # Backward stays inside the context because gradient checkpointing
        # recomputes model forwards during this call.
        with attention_backend_context(model, device, enabled=opts.prefer_cudnn_sdpa):
            out = model(input_ids=ids, labels=ids)
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
