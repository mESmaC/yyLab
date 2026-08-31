from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

FP8_POTENTIATED_MASTER = "fp8_e4m3_potentiated"
FP8_ADAPTIVE_MASTER = "fp8_e4m3_adaptive"
MASTER_WEIGHT_FORMATS = {"fp32", FP8_POTENTIATED_MASTER, FP8_ADAPTIVE_MASTER}
DEFAULT_POTENTIATION_BLOCK_SIZE = 32
POTENTIATION_LEVELS = 15
DEFAULT_ADAPTIVE_MAX_PROMOTED_FRACTION = 0.10
DEFAULT_ADAPTIVE_PROMOTE_THRESHOLD = 0.55
DEFAULT_ADAPTIVE_DEMOTE_THRESHOLD = 0.20
DEFAULT_ADAPTIVE_DEMOTE_PATIENCE = 2

_E4M3_LEVELS: torch.Tensor | None = None


def _e4m3_levels() -> torch.Tensor:
    """Return the finite E4M3FN values in increasing order on CPU."""

    global _E4M3_LEVELS
    if _E4M3_LEVELS is None:
        raw = torch.arange(256, dtype=torch.uint8)
        values = raw.view(torch.float8_e4m3fn).float()
        _E4M3_LEVELS = torch.unique(values[torch.isfinite(values)], sorted=True)
    return _E4M3_LEVELS


def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    values = values.to(device="cpu", dtype=torch.uint8).reshape(-1)
    if values.numel() % 2:
        values = F.pad(values, (0, 1))
    return values[0::2] | (values[1::2] << 4)


def _unpack_nibbles(values: torch.Tensor, count: int) -> torch.Tensor:
    packed = values.to(device="cpu", dtype=torch.uint8).reshape(-1)
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    unpacked[0::2] = packed & 0x0F
    unpacked[1::2] = packed >> 4
    return unpacked[:count]


def _unpack_nibble_range(values: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if not 0 <= start <= end:
        raise ValueError("invalid potentiation range")
    first_byte = start // 2
    last_byte = (end + 1) // 2
    unpacked = _unpack_nibbles(
        values.reshape(-1)[first_byte:last_byte],
        (last_byte - first_byte) * 2,
    )
    offset = start % 2
    return unpacked[offset : offset + (end - start)]


@dataclass(frozen=True, slots=True)
class PotentiatedFP8State:
    """Persistent E4M3 base plus a packed four-bit sub-ULP accumulator.

    The base uses one byte per value. A four-bit interpolation coordinate is
    stored only logically when a value lies between adjacent E4M3 numbers
    (zero means no promotion), while remaining physically packed at two
    values per byte. Power-of-two scales are shared by blocks of 32 values.
    This is an optimizer-state format, not a Tensor Core compute dtype.
    """

    base: torch.Tensor
    potentiation: torch.Tensor
    scale: torch.Tensor
    shape: tuple[int, ...]
    block_size: int = DEFAULT_POTENTIATION_BLOCK_SIZE

    @property
    def numel(self) -> int:
        return self.base.numel()

    @property
    def storage_bytes(self) -> int:
        return (
            self.base.numel() * self.base.element_size()
            + self.potentiation.numel() * self.potentiation.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    @property
    def promoted_fraction(self) -> float:
        if not self.numel:
            return 0.0
        packed = self.potentiation.reshape(-1)
        promoted = (packed & 0x0F).count_nonzero() + (packed >> 4).count_nonzero()
        if self.numel % 2 and int((packed[-1] >> 4).item()) != 0:
            promoted -= 1
        return float(promoted.item()) / self.numel

    def dequantize(self) -> torch.Tensor:
        return self.dequantize_range(0, self.numel).reshape(self.shape)

    def dequantize_range(self, start: int, end: int) -> torch.Tensor:
        if not 0 <= start <= end <= self.numel:
            raise ValueError("invalid potentiated-master range")
        levels = _e4m3_levels()
        base = self.base.reshape(-1)[start:end].float()
        fraction = _unpack_nibble_range(self.potentiation, start, end).float()
        upper_index = torch.searchsorted(levels, base, right=True).clamp_max(
            levels.numel() - 1
        )
        upper = levels[upper_index]
        normalized = base + (upper - base) * (fraction / POTENTIATION_LEVELS)
        first_block = start // self.block_size
        last_block = (end + self.block_size - 1) // self.block_size
        scale_offset = start - first_block * self.block_size
        scales = self.scale[first_block:last_block].repeat_interleave(self.block_size)
        scales = scales[scale_offset : scale_offset + (end - start)]
        return normalized * scales


@dataclass(frozen=True, slots=True)
class AdaptivePotentiatedFP8State:
    """Block-scaled E4M3 with a sparse, selectively promoted residual.

    Every value has a real one-byte E4M3 representation. Only blocks whose
    quantization error or lost update is troublesome carry the four-bit
    sub-ULP interpolation payload. ``stability`` is one byte per block and
    implements promotion hysteresis without allocating residual storage for
    ordinary blocks.
    """

    base: torch.Tensor
    promoted_blocks: torch.Tensor
    potentiation: torch.Tensor
    scale: torch.Tensor
    stability: torch.Tensor
    shape: tuple[int, ...]
    block_size: int = DEFAULT_POTENTIATION_BLOCK_SIZE
    max_promoted_fraction: float = DEFAULT_ADAPTIVE_MAX_PROMOTED_FRACTION
    promote_threshold: float = DEFAULT_ADAPTIVE_PROMOTE_THRESHOLD
    demote_threshold: float = DEFAULT_ADAPTIVE_DEMOTE_THRESHOLD
    demote_patience: int = DEFAULT_ADAPTIVE_DEMOTE_PATIENCE

    @property
    def numel(self) -> int:
        return self.base.numel()

    @property
    def num_blocks(self) -> int:
        return (self.numel + self.block_size - 1) // self.block_size

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.base,
                self.promoted_blocks,
                self.potentiation,
                self.scale,
                self.stability,
            )
        )

    @property
    def nbytes(self) -> int:
        """Actual persistent tensor bytes, including sparse indices."""

        return self.storage_bytes

    @property
    def promoted_fraction(self) -> float:
        if not self.numel:
            return 0.0
        promoted_values = 0
        for block in self.promoted_blocks.tolist():
            promoted_values += min(
                self.block_size,
                max(0, self.numel - int(block) * self.block_size),
            )
        return promoted_values / self.numel

    def dequantize(self) -> torch.Tensor:
        return self.dequantize_range(0, self.numel).reshape(self.shape)

    def dequantize_range(self, start: int, end: int) -> torch.Tensor:
        if not 0 <= start <= end <= self.numel:
            raise ValueError("invalid adaptive-master range")
        if start == end:
            return torch.empty(0, dtype=torch.float32)

        first_block = start // self.block_size
        last_block = (end + self.block_size - 1) // self.block_size
        scale_offset = start - first_block * self.block_size
        scales = self.scale[first_block:last_block].repeat_interleave(self.block_size)
        scales = scales[scale_offset : scale_offset + (end - start)]
        base = self.base.reshape(-1)[start:end].float()
        restored = base * scales

        promoted = self.promoted_blocks.reshape(-1)
        selected = (promoted >= first_block) & (promoted < last_block)
        if not bool(selected.any()):
            return restored
        rows = torch.nonzero(selected, as_tuple=False).reshape(-1)
        block_indices = promoted[rows].to(torch.long)
        residual_rows = self.potentiation[rows]
        residual = _unpack_nibbles(
            residual_rows.reshape(-1),
            residual_rows.shape[0] * self.block_size,
        ).reshape(-1, self.block_size)
        positions = (
            block_indices.unsqueeze(1) * self.block_size
            + torch.arange(self.block_size, dtype=torch.long).unsqueeze(0)
        )
        valid = (positions >= start) & (positions < end) & (positions < self.numel)
        positions = positions[valid]
        fractions = residual[valid].float()
        local_positions = positions - start
        promoted_base = self.base.reshape(-1)[positions].float()
        levels = _e4m3_levels()
        upper_index = torch.searchsorted(levels, promoted_base, right=True).clamp_max(
            levels.numel() - 1
        )
        upper = levels[upper_index]
        normalized = promoted_base + (upper - promoted_base) * (
            fractions / POTENTIATION_LEVELS
        )
        promoted_scales = self.scale[(positions // self.block_size).to(torch.long)]
        restored[local_positions] = normalized * promoted_scales
        return restored

    def slice_range(self, start: int, end: int) -> AdaptivePotentiatedFP8State:
        """Return a block-aligned view encoded with local block indices."""

        if start % self.block_size or not 0 <= start <= end <= self.numel:
            raise ValueError("adaptive-master slices must start on block boundaries")
        first_block = start // self.block_size
        last_block = (end + self.block_size - 1) // self.block_size
        selected = (self.promoted_blocks >= first_block) & (
            self.promoted_blocks < last_block
        )
        return AdaptivePotentiatedFP8State(
            base=self.base.reshape(-1)[start:end],
            promoted_blocks=(self.promoted_blocks[selected] - first_block).to(torch.int32),
            potentiation=self.potentiation[selected],
            scale=self.scale[first_block:last_block],
            stability=self.stability[first_block:last_block],
            shape=(end - start,),
            block_size=self.block_size,
            max_promoted_fraction=self.max_promoted_fraction,
            promote_threshold=self.promote_threshold,
            demote_threshold=self.demote_threshold,
            demote_patience=self.demote_patience,
        )


def encode_potentiated_fp8(
    value: torch.Tensor,
    *,
    block_size: int = DEFAULT_POTENTIATION_BLOCK_SIZE,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
    previous_scale: torch.Tensor | None = None,
) -> PotentiatedFP8State:
    """Encode a tensor as block-scaled E4M3 plus a four-bit residual.

    Values are first bracketed by adjacent E4M3 numbers. The E4M3 lower
    endpoint is persisted as the base and the packed residual interpolates
    between endpoints. Stochastic rounding is therefore deferred to a grid
    roughly sixteen times finer than bare E4M3 instead of discarding every
    sub-ULP update at the FP8 boundary.
    """

    if block_size < 1 or block_size % 2:
        raise ValueError("potentiation block size must be a positive even integer")
    work = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not torch.isfinite(work).all():
        raise FloatingPointError("potentiated FP8 master cannot encode NaN or infinity")
    count = work.numel()
    padding = (-count) % block_size
    padded = F.pad(work, (0, padding)) if padding else work
    blocks = padded.reshape(-1, block_size)
    amax = blocks.abs().amax(dim=-1)
    raw_scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    target_scales = torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))
    target_scales = torch.where(amax > 0, target_scales, torch.ones_like(target_scales))
    if previous_scale is not None:
        previous_scale = previous_scale.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if previous_scale.shape != target_scales.shape:
            raise ValueError("previous potentiated-FP8 scale shape does not match the value blocks")
        # Keep scales stable across ordinary updates. Grow immediately before
        # saturation, but shrink only after 4x headroom to avoid power-of-two
        # boundary chatter injecting noise into an entire block.
        rescale = (amax > previous_scale * 448.0) | (
            (amax > 0) & (amax < previous_scale * 112.0)
        )
        scales = torch.where(rescale, target_scales, previous_scale)
    else:
        scales = target_scales
    normalized = (blocks / scales.unsqueeze(-1)).clamp(-448.0, 448.0)

    levels = _e4m3_levels()
    upper_index = torch.searchsorted(levels, normalized, right=False).clamp(
        1, levels.numel() - 1
    )
    lower = levels[upper_index - 1]
    upper = levels[upper_index]
    interval = (upper - lower).clamp_min(torch.finfo(torch.float32).tiny)
    coordinate = ((normalized - lower) / interval).clamp(0.0, 1.0)
    fine_coordinate = coordinate * POTENTIATION_LEVELS
    if stochastic:
        floor = fine_coordinate.floor()
        probability = fine_coordinate - floor
        random = torch.rand(
            probability.shape,
            dtype=probability.dtype,
            device=probability.device,
            generator=generator,
        )
        residual = floor + (random < probability)
    else:
        residual = fine_coordinate.round()
    residual = residual.to(torch.uint8)

    # Canonicalize the upper endpoint to an exact E4M3 base with no residual.
    carry = residual >= POTENTIATION_LEVELS
    base = torch.where(carry, upper, lower)
    residual = torch.where(carry, torch.zeros_like(residual), residual)

    base = base.reshape(-1)[:count].to(torch.float8_e4m3fn)
    residual = residual.reshape(-1)[:count]
    return PotentiatedFP8State(
        base=base,
        potentiation=_pack_nibbles(residual),
        scale=scales,
        shape=tuple(value.shape),
        block_size=block_size,
    )


def _validate_adaptive_configuration(
    *,
    block_size: int,
    max_promoted_fraction: float,
    promote_threshold: float,
    demote_threshold: float,
    demote_patience: int,
) -> None:
    if block_size < 2 or block_size % 2:
        raise ValueError("adaptive FP8 block size must be a positive even integer")
    if not 0.0 <= max_promoted_fraction <= 1.0:
        raise ValueError("max promoted fraction must be within [0, 1]")
    if demote_threshold > promote_threshold:
        raise ValueError("demote threshold cannot exceed promote threshold")
    if demote_threshold < 0 or promote_threshold < 0:
        raise ValueError("adaptive FP8 thresholds must be non-negative")
    if demote_patience < 1 or demote_patience > 255:
        raise ValueError("adaptive FP8 demotion patience must be within [1, 255]")


def encode_adaptive_potentiated_fp8(
    value: torch.Tensor,
    *,
    block_size: int = DEFAULT_POTENTIATION_BLOCK_SIZE,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
    previous: AdaptivePotentiatedFP8State | None = None,
    update_reference: torch.Tensor | None = None,
    max_promoted_fraction: float = DEFAULT_ADAPTIVE_MAX_PROMOTED_FRACTION,
    max_promoted_blocks: int | None = None,
    promote_threshold: float = DEFAULT_ADAPTIVE_PROMOTE_THRESHOLD,
    demote_threshold: float = DEFAULT_ADAPTIVE_DEMOTE_THRESHOLD,
    demote_patience: int = DEFAULT_ADAPTIVE_DEMOTE_PATIENCE,
) -> AdaptivePotentiatedFP8State:
    """Encode E4M3 state and selectively retain sub-ULP detail.

    Promotion is decided per block. On first encode, the score is normalized
    nearest-E4M3 reconstruction loss. On subsequent encodes it also measures
    how much of the requested update bare E4M3 would discard. Existing blocks
    use a lower demotion threshold and a patience counter, avoiding promotion
    chatter near a threshold. ``max_promoted_blocks`` is primarily useful to
    enforce a parameter-wide cap while encoding it in bounded-size chunks.
    """

    _validate_adaptive_configuration(
        block_size=block_size,
        max_promoted_fraction=max_promoted_fraction,
        promote_threshold=promote_threshold,
        demote_threshold=demote_threshold,
        demote_patience=demote_patience,
    )
    work = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not torch.isfinite(work).all():
        raise FloatingPointError("adaptive FP8 master cannot encode NaN or infinity")
    count = work.numel()
    padding = (-count) % block_size
    padded = F.pad(work, (0, padding)) if padding else work
    blocks = padded.reshape(-1, block_size)
    block_count = blocks.shape[0]
    if previous is not None:
        if previous.numel != count or previous.block_size != block_size:
            raise ValueError("previous adaptive FP8 state does not match the value")
        previous_scale = previous.scale
    else:
        previous_scale = None

    amax = blocks.abs().amax(dim=-1)
    raw_scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    target_scales = torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))
    target_scales = torch.where(amax > 0, target_scales, torch.ones_like(target_scales))
    if previous_scale is not None:
        previous_scale = previous_scale.detach().to(
            device="cpu", dtype=torch.float32
        ).reshape(-1)
        if previous_scale.shape != target_scales.shape:
            raise ValueError("previous adaptive-FP8 scale shape does not match")
        rescale = (amax > previous_scale * 448.0) | (
            (amax > 0) & (amax < previous_scale * 112.0)
        )
        scales = torch.where(rescale, target_scales, previous_scale)
    else:
        scales = target_scales
    normalized = (blocks / scales.unsqueeze(-1)).clamp(-448.0, 448.0)

    levels = _e4m3_levels()
    upper_index = torch.searchsorted(levels, normalized, right=False).clamp(
        1, levels.numel() - 1
    )
    lower = levels[upper_index - 1]
    upper = levels[upper_index]
    interval = (upper - lower).clamp_min(torch.finfo(torch.float32).tiny)
    coordinate = ((normalized - lower) / interval).clamp(0.0, 1.0)
    nearest = torch.where(coordinate >= 0.5, upper, lower)
    valid = torch.arange(padded.numel()).reshape_as(blocks) < count
    valid_count = valid.sum(dim=-1).clamp_min(1)
    reconstruction_loss = (
        (normalized - nearest).abs().div(interval * 0.5).clamp_max(1.0) * valid
    ).sum(dim=-1) / valid_count

    if previous is None and update_reference is not None:
        raise ValueError("an update reference requires previous adaptive FP8 state")
    if previous is not None:
        if update_reference is None:
            reference = previous.dequantize().reshape(-1)
        else:
            reference = update_reference.detach().to(
                device="cpu", dtype=torch.float32
            ).reshape(-1)
            if reference.numel() != count:
                raise ValueError("adaptive FP8 update reference shape does not match")
        reference_padded = F.pad(reference, (0, padding)) if padding else reference
        reference_blocks = reference_padded.reshape_as(blocks)
        bare = nearest * scales.unsqueeze(-1)
        requested_update = blocks - reference_blocks
        quantization_loss = (bare - blocks).abs()
        # Ignore exact no-op values. For real updates, a sixteenth of the local
        # ULP is the finest detail the potentiation nibble can preserve.
        update_floor = interval * scales.unsqueeze(-1) / POTENTIATION_LEVELS
        active = requested_update.abs() > torch.finfo(torch.float32).eps
        normalized_update_loss = quantization_loss.div(
            torch.maximum(requested_update.abs(), update_floor)
        ).clamp_max(1.0)
        active_valid = active & valid
        active_count = active_valid.sum(dim=-1)
        update_loss = (normalized_update_loss * active_valid).sum(dim=-1).div(
            active_count.clamp_min(1)
        )
        update_loss = torch.where(active_count > 0, update_loss, torch.zeros_like(update_loss))
        scores = reconstruction_loss.mul(0.5).add(update_loss, alpha=0.5)
    else:
        scores = reconstruction_loss

    previous_mask = torch.zeros(block_count, dtype=torch.bool)
    previous_stability = torch.zeros(block_count, dtype=torch.uint8)
    if previous is not None:
        previous_mask[previous.promoted_blocks.to(torch.long)] = True
        previous_stability.copy_(previous.stability)
    low_score = scores <= demote_threshold
    stability = torch.where(
        previous_mask & low_score,
        (previous_stability.to(torch.int16) + 1).clamp_max(255).to(torch.uint8),
        torch.zeros_like(previous_stability),
    )
    retained = previous_mask & (stability < demote_patience)
    newly_troublesome = (~previous_mask) & (scores >= promote_threshold)
    candidates = retained | newly_troublesome

    if max_promoted_blocks is None:
        promotion_limit = int(block_count * max_promoted_fraction)
    else:
        promotion_limit = int(max_promoted_blocks)
        if not 0 <= promotion_limit <= block_count:
            raise ValueError("max promoted blocks must be within the encoded block count")
    candidate_indices = torch.nonzero(candidates, as_tuple=False).reshape(-1)
    if candidate_indices.numel() > promotion_limit:
        candidate_scores = scores[candidate_indices]
        order = torch.argsort(candidate_scores, descending=True, stable=True)
        promoted_blocks = candidate_indices[order[:promotion_limit]].sort().values
    else:
        promoted_blocks = candidate_indices.sort().values
    promoted_mask = torch.zeros(block_count, dtype=torch.bool)
    promoted_mask[promoted_blocks] = True
    stability = torch.where(promoted_mask, stability, torch.zeros_like(stability))

    if stochastic:
        random = torch.rand(
            coordinate.shape,
            dtype=coordinate.dtype,
            generator=generator,
        )
        base_values = torch.where(random < coordinate, upper, lower)
    else:
        base_values = nearest

    packed_width = block_size // 2
    if promoted_blocks.numel():
        promoted_coordinate = coordinate[promoted_blocks]
        fine_coordinate = promoted_coordinate * POTENTIATION_LEVELS
        if stochastic:
            floor = fine_coordinate.floor()
            probability = fine_coordinate - floor
            random = torch.rand(
                probability.shape,
                dtype=probability.dtype,
                generator=generator,
            )
            residual = floor + (random < probability)
        else:
            residual = fine_coordinate.round()
        residual = residual.to(torch.uint8)
        promoted_lower = lower[promoted_blocks]
        promoted_upper = upper[promoted_blocks]
        carry = residual >= POTENTIATION_LEVELS
        base_values[promoted_blocks] = torch.where(carry, promoted_upper, promoted_lower)
        residual = torch.where(carry, torch.zeros_like(residual), residual)
        potentiation = _pack_nibbles(residual).reshape(-1, packed_width)
    else:
        potentiation = torch.empty((0, packed_width), dtype=torch.uint8)

    base = base_values.reshape(-1)[:count].to(torch.float8_e4m3fn)
    return AdaptivePotentiatedFP8State(
        base=base,
        promoted_blocks=promoted_blocks.to(torch.int32),
        potentiation=potentiation,
        scale=scales,
        stability=stability,
        shape=tuple(value.shape),
        block_size=block_size,
        max_promoted_fraction=max_promoted_fraction,
        promote_threshold=promote_threshold,
        demote_threshold=demote_threshold,
        demote_patience=demote_patience,
    )


def potentiated_master_bytes(parameters: int, block_size: int = 32) -> int:
    """Persistent bytes for E4M3 base, packed residual, and FP32 scales."""

    parameters = max(0, int(parameters))
    return parameters + (parameters + 1) // 2 + ((parameters + block_size - 1) // block_size) * 4


def adaptive_master_bytes(
    parameters: int,
    promoted_blocks: int,
    block_size: int = DEFAULT_POTENTIATION_BLOCK_SIZE,
) -> int:
    """Actual bytes for adaptive E4M3 state at a given promotion occupancy."""

    parameters = max(0, int(parameters))
    blocks = (parameters + block_size - 1) // block_size
    promoted_blocks = int(promoted_blocks)
    if not 0 <= promoted_blocks <= blocks:
        raise ValueError("promoted block count must be within the parameter block count")
    return (
        parameters
        + blocks * 4  # FP32 block scales
        + blocks  # uint8 hysteresis counters
        + promoted_blocks * 4  # int32 sparse block indices
        + promoted_blocks * (block_size // 2)  # packed four-bit residuals
    )
