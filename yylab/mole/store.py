from __future__ import annotations

import datetime as dt
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .. import __version__
from ..artifacts import read_json, write_json
from .inject import MoLEConfig, mole_layers

WEIGHTS_NAME = "mole_weights.safetensors"
CONFIG_NAME = "mole_config.json"


def save_mole(
    directory: Path,
    model,
    cfg: MoLEConfig,
    base_ref: str = "",
    extra_meta: dict | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state: dict[str, torch.Tensor] = {}
    layer_names = []
    for name, layer in mole_layers(model):
        layer_names.append(name)
        key = name.replace(".", "__")
        for e, expert in enumerate(layer.experts):
            state[f"{key}.expert_{e}.lora_A"] = expert.lora_A.detach().contiguous()
            state[f"{key}.expert_{e}.lora_B"] = expert.lora_B.detach().contiguous()
        state[f"{key}.router.gate.weight"] = layer.router.gate.weight.detach().contiguous()
    save_file(state, str(directory / WEIGHTS_NAME))
    write_json(
        directory / CONFIG_NAME,
        {
            "arch": "mole",
            "base_ref": base_ref,
            "yylab_version": __version__,
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            "layers": layer_names,
            "config": cfg.to_dict(),
            "meta": extra_meta or {},
        },
    )
    return directory / WEIGHTS_NAME


def load_mole_config(directory: Path) -> dict:
    return read_json(Path(directory) / CONFIG_NAME)


def load_mole_weights(directory: Path) -> dict[str, torch.Tensor]:
    return load_file(str(Path(directory) / WEIGHTS_NAME))


def apply_mole_weights(model, state: dict[str, torch.Tensor]) -> int:
    applied = 0
    for name, layer in mole_layers(model):
        key = name.replace(".", "__")
        for e, expert in enumerate(layer.experts):
            a = state.get(f"{key}.expert_{e}.lora_A")
            b = state.get(f"{key}.expert_{e}.lora_B")
            if a is not None and b is not None:
                expert.lora_A.data.copy_(a.to(expert.lora_A.device, expert.lora_A.dtype))
                expert.lora_B.data.copy_(b.to(expert.lora_B.device, expert.lora_B.dtype))
                applied += 1
        g = state.get(f"{key}.router.gate.weight")
        if g is not None:
            layer.router.gate.weight.data.copy_(g.to(layer.router.gate.weight.device))
    return applied
