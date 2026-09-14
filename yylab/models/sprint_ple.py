"""Per-layer token embedding refreshes for Sprint.

Layer zero keeps Sprint's exact token embedding. Full PLE uses one independent
vocabulary table per later physical layer. QR-PLE uses distinct prime quotient/
remainder lanes, learned scalar lane mixing and a scalar gate. Codebooks use
dense gradients and ordinary parameter storage; this is not fake quantization.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def validate_ple_options(ple_type, ple_lanes):
    if not isinstance(ple_type, str) or ple_type not in {"none", "full", "qr"}:
        raise ValueError("ple_type must be none, full or qr")
    if isinstance(ple_lanes, bool) or not isinstance(ple_lanes, int) or ple_lanes <= 0:
        raise ValueError("ple_lanes must be a positive integer")


def qr_lane_moduli(vocab_size: int, lanes: int = 4) -> tuple[int, ...]:
    """Distinct ascending primes at/above ceil(sqrt(vocabulary)), as in SPALMER."""
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    validate_ple_options("qr", lanes)
    candidate = max(2, math.isqrt(vocab_size))
    if candidate * candidate < vocab_size:
        candidate += 1
    values = []
    while len(values) < lanes:
        if all(candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1)):
            values.append(candidate)
        candidate += 1
    return tuple(values)


def qr_codebook_rows(vocab_size: int, lanes: int = 4) -> tuple[int, int]:
    values = qr_lane_moduli(vocab_size, lanes)
    return sum(values), sum((vocab_size + modulus - 1) // modulus for modulus in values)


def ple_parameter_counts(vocab_size, width, num_layers, ple_type, ple_lanes=4):
    """Return exact (table, control) counts, excluding the layer-zero embedding."""
    validate_ple_options(ple_type, ple_lanes)
    later_layers = num_layers - 1
    if ple_type == "none":
        return 0, 0
    if ple_type == "full":
        return later_layers * vocab_size * width, later_layers
    remainder, quotient = qr_codebook_rows(vocab_size, ple_lanes)
    return later_layers * (remainder + quotient) * width, later_layers * (ple_lanes + 1)


def _offsets(sizes):
    result, current = [], 0
    for size in sizes:
        result.append(current)
        current += size
    return tuple(result)


class SprintQRLayout(nn.Module):
    """Immutable layout shared by all refreshes; never caches per-call IDs."""

    def __init__(self, vocab_size, lanes=4):
        super().__init__()
        self.modulus_values = qr_lane_moduli(vocab_size, lanes)
        quotient_sizes = tuple((vocab_size + modulus - 1) // modulus for modulus in self.modulus_values)
        self.register_buffer("moduli", torch.tensor(self.modulus_values, dtype=torch.int64))
        self.register_buffer("remainder_offsets", torch.tensor(_offsets(self.modulus_values), dtype=torch.int64))
        self.register_buffer("quotient_offsets", torch.tensor(_offsets(quotient_sizes), dtype=torch.int64))

    def prepare_indices(self, input_ids):
        lane_ids = input_ids.unsqueeze(-1)
        quotients = torch.div(lane_ids, self.moduli, rounding_mode="floor")
        remainders = lane_ids - quotients * self.moduli
        return remainders + self.remainder_offsets, quotients + self.quotient_offsets


class SprintFullPLE(nn.Module):
    """One full vocabulary-by-width table and gate for a later physical layer."""

    def __init__(self, config):
        super().__init__()
        self.initializer_range = config.initializer_range
        self.gate_init = 1 / math.sqrt(config.num_hidden_layers)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gate = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.initializer_range)
        with torch.no_grad():
            self.gate.fill_(self.gate_init)

    def forward(self, input_ids, *, dtype=None, qr_indices=None):
        if qr_indices is not None:
            raise ValueError("full PLE does not use QR addresses")
        lexical = self.embedding(input_ids)
        lexical = lexical.to(dtype=dtype or lexical.dtype)
        return lexical * self.gate.to(dtype=lexical.dtype)


class SprintQRPLE(nn.Module):
    """Compositional QR refresh with dense gradients and learned lane weights."""

    def __init__(self, config):
        super().__init__()
        self.lanes = config.ple_lanes
        self.initializer_range = config.initializer_range
        self.gate_init = 1 / math.sqrt(config.num_hidden_layers)
        remainder_rows, quotient_rows = qr_codebook_rows(config.vocab_size, self.lanes)
        self.remainder_embedding = nn.Embedding(remainder_rows, config.hidden_size)
        self.quotient_embedding = nn.Embedding(quotient_rows, config.hidden_size)
        self.lane_logits = nn.Parameter(torch.empty(self.lanes))
        self.gate = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.remainder_embedding.weight, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.quotient_embedding.weight, mean=1.0, std=self.initializer_range)
        nn.init.zeros_(self.lane_logits)
        with torch.no_grad():
            self.gate.fill_(self.gate_init)

    def forward(self, input_ids, *, dtype=None, qr_indices=None):
        if qr_indices is None:
            raise ValueError("QR-PLE requires addresses prepared by the shared SprintQRLayout")
        remainder_ids, quotient_ids = qr_indices
        remainder = self.remainder_embedding(remainder_ids)
        quotient = self.quotient_embedding(quotient_ids)
        dtype = dtype or remainder.dtype
        # Embedding lookup does not follow autocast. Convert before product and
        # reduction so FP32 table storage never widens the BF16 residual stream.
        expanded = remainder.to(dtype=dtype) * quotient.to(dtype=dtype)
        weights = self.lane_logits.softmax(0).to(dtype=dtype)
        weight_shape = (1,) * (expanded.ndim - 2) + (self.lanes, 1)
        lexical = (expanded * weights.view(weight_shape)).sum(-2, dtype=dtype)
        return lexical * self.gate.to(dtype=dtype)
