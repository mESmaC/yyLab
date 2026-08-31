from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .. import __version__
from ..artifacts import read_json, write_json
from .inject import MoLEConfig, mole_layers, shared_router, token_routers

WEIGHTS_NAME = "mole_weights.safetensors"
CONFIG_NAME = "mole_config.json"
SHARED_ROUTER_PREFIX = "shared_router."
TOKEN_ROUTER_PREFIX = "token_router."
EXPERTS_DIR = "experts"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def _atomic_save_tensors(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(state, str(temporary))
    temporary.replace(path)


def _write_mole_config(
    directory: Path,
    cfg: MoLEConfig,
    layer_names: list[str],
    base_ref: str,
    extra_meta: dict | None,
    expert_manifest: list[dict] | None = None,
    routing_sha256: str | None = None,
) -> None:
    format_version = 4 if cfg.adaptive.enabled else (3 if cfg.router_type == "per_token" else 2)
    payload = {
        "arch": "mole",
        "format_version": format_version,
        "base_ref": base_ref,
        "yylab_version": __version__,
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "layers": layer_names,
        "config": cfg.to_dict(),
        "meta": extra_meta or {},
    }
    if expert_manifest is not None:
        payload["expert_manifest"] = expert_manifest
    if routing_sha256 is not None:
        payload["routing_sha256"] = routing_sha256
    temporary = directory / f".{CONFIG_NAME}.tmp"
    write_json(temporary, payload)
    temporary.replace(directory / CONFIG_NAME)


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
        if not cfg.adaptive.enabled:
            for e, expert in enumerate(layer.experts):
                state[f"{key}.expert_{e}.lora_A"] = _cpu_tensor(expert.lora_A)
                state[f"{key}.expert_{e}.lora_B"] = _cpu_tensor(expert.lora_B)
        if layer.shared_token_router() is None:
            if layer.router is None:
                raise RuntimeError("non-shared MoLE layer is missing its router")
            state[f"{key}.router.gate.weight"] = _cpu_tensor(layer.router.gate.weight)
    for name, router in token_routers(model):
        for parameter_name, tensor in router.state_dict().items():
            state[f"{TOKEN_ROUTER_PREFIX}{name}.{parameter_name}"] = _cpu_tensor(tensor)
    prompt_router = shared_router(model)
    if prompt_router is not None:
        for name, tensor in prompt_router.state_dict().items():
            state[f"{SHARED_ROUTER_PREFIX}{name}"] = _cpu_tensor(tensor)
    manifest: list[dict] | None = None
    if cfg.adaptive.enabled:
        experts_dir = directory / EXPERTS_DIR
        experts_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        layers = mole_layers(model)
        for expert_index in range(cfg.num_experts):
            lane: dict[str, torch.Tensor] = {}
            parameters = 0
            storage_bytes = 0
            for name, layer in layers:
                key = name.replace(".", "__")
                expert = layer.experts[expert_index]
                for suffix, tensor in (
                    ("lora_A", expert.lora_A),
                    ("lora_B", expert.lora_B),
                ):
                    cpu = _cpu_tensor(tensor)
                    lane[f"{key}.expert_{expert_index}.{suffix}"] = cpu
                    parameters += cpu.numel()
                    storage_bytes += cpu.numel() * cpu.element_size()
            relative = f"{EXPERTS_DIR}/expert_{expert_index:03d}.safetensors"
            shard_path = directory / relative
            _atomic_save_tensors(shard_path, lane)
            expected_parameters = cfg.adaptive.expert_parameter_count
            if expected_parameters and parameters != expected_parameters:
                raise ValueError(
                    "saved adaptive lane does not match the resolved parameter budget"
                )
            manifest.append(
                {
                    "index": expert_index,
                    "file": relative,
                    "parameters": parameters,
                    "bytes": storage_bytes,
                    "sha256": _sha256(shard_path),
                }
            )
    routing_path = directory / WEIGHTS_NAME
    _atomic_save_tensors(routing_path, state)
    _write_mole_config(
        directory,
        cfg,
        layer_names,
        base_ref,
        extra_meta,
        expert_manifest=manifest,
        routing_sha256=_sha256(routing_path) if cfg.adaptive.enabled else None,
    )
    return directory / WEIGHTS_NAME


def save_isolated_expert(path: Path, model, expert_index: int = 0) -> Path:
    """Save one resident expert without serializing the base or router."""

    path = Path(path)
    state: dict[str, torch.Tensor] = {}
    for name, layer in mole_layers(model):
        if not 0 <= expert_index < len(layer.experts):
            raise IndexError(f"expert index {expert_index} is out of range")
        key = name.replace(".", "__")
        expert = layer.experts[expert_index]
        state[f"{key}.lora_A"] = _cpu_tensor(expert.lora_A)
        state[f"{key}.lora_B"] = _cpu_tensor(expert.lora_B)
    if not state:
        raise ValueError("model has no MoLE layers")
    _atomic_save_tensors(path, state)
    return path


def assemble_mole_experts(
    directory: Path,
    model,
    cfg: MoLEConfig,
    expert_paths: list[Path],
    base_ref: str = "",
    extra_meta: dict | None = None,
) -> Path:
    """Assemble isolated stage-1 checkpoints into one inference/router bank."""

    if len(expert_paths) != cfg.num_experts:
        raise ValueError("isolated expert count does not match MoLE config")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layers = mole_layers(model)
    layer_names = [name for name, _ in layers]
    expected = {
        f"{name.replace('.', '__')}.{suffix}" for name, _ in layers for suffix in ("lora_A", "lora_B")
    }
    state: dict[str, torch.Tensor] = {}
    for expert_index, source in enumerate(expert_paths):
        tensors = load_file(str(source), device="cpu")
        if set(tensors) != expected:
            missing = sorted(expected.difference(tensors))
            extra = sorted(set(tensors).difference(expected))
            raise ValueError(
                f"isolated expert {expert_index} tensor set mismatch; missing={missing[:3]}, extra={extra[:3]}"
            )
        for name, layer in layers:
            key = name.replace(".", "__")
            a = tensors[f"{key}.lora_A"]
            b = tensors[f"{key}.lora_B"]
            reference = layer.experts[0]
            if a.shape != reference.lora_A.shape or b.shape != reference.lora_B.shape:
                raise ValueError(f"isolated expert {expert_index} shape mismatch at {name}")
            state[f"{key}.expert_{expert_index}.lora_A"] = a.contiguous()
            state[f"{key}.expert_{expert_index}.lora_B"] = b.contiguous()
    for name, layer in layers:
        key = name.replace(".", "__")
        gate = layer.router.gate.weight
        state[f"{key}.router.gate.weight"] = torch.zeros(
            cfg.num_experts,
            layer.in_features,
            dtype=gate.dtype,
            device="cpu",
        )
    _atomic_save_tensors(directory / WEIGHTS_NAME, state)
    _write_mole_config(directory, cfg, layer_names, base_ref, extra_meta)
    return directory / WEIGHTS_NAME


def _normalized_base_ref(value: str) -> str:
    path = Path(value)
    if path.exists():
        return str(path.resolve()).replace("\\", "/").casefold()
    return value.strip().replace("\\", "/").rstrip("/").casefold()


def load_mole_config(directory: Path, *, expected_base_ref: str | None = None) -> dict:
    payload = read_json(Path(directory) / CONFIG_NAME)
    if not isinstance(payload, dict) or payload.get("arch") != "mole":
        raise ValueError("invalid MoLE artifact config")
    version = int(payload.get("format_version", 1))
    if version not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported MoLE artifact format version: {version}")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("MoLE artifact config is missing its architecture settings")
    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers or any(not isinstance(name, str) for name in layers):
        raise ValueError("MoLE artifact config has no valid layer manifest")
    if len(layers) != len(set(layers)):
        raise ValueError("MoLE artifact layer manifest contains duplicates")
    if expected_base_ref is not None:
        artifact_base = str(payload.get("base_ref") or "")
        if not artifact_base or _normalized_base_ref(artifact_base) != _normalized_base_ref(
            expected_base_ref
        ):
            raise ValueError("MoLE artifact targets a different frozen base")
    if version == 4:
        mole_cfg = MoLEConfig.from_dict(payload["config"])
        adaptive = mole_cfg.adaptive
        if not adaptive.enabled:
            raise ValueError("MoLE v4 artifact must enable adaptive occupancy")
        manifest = payload.get("expert_manifest")
        if not isinstance(manifest, list) or len(manifest) != int(
            payload["config"].get("num_experts", 0)
        ):
            raise ValueError("MoLE v4 artifact has no complete expert manifest")
        indices = [entry.get("index") for entry in manifest if isinstance(entry, dict)]
        if indices != list(range(len(manifest))):
            raise ValueError("MoLE v4 expert manifest indices are not contiguous")
        routing_checksum = payload.get("routing_sha256")
        routing_path = Path(directory) / WEIGHTS_NAME
        if not routing_path.is_file():
            raise ValueError("MoLE v4 routing weights are missing")
        if not isinstance(routing_checksum, str) or len(routing_checksum) != 64:
            raise ValueError("MoLE v4 artifact has no valid routing SHA-256 digest")
        if _sha256(routing_path) != routing_checksum:
            raise ValueError("MoLE v4 routing weights failed their SHA-256 check")
        seen_files: set[str] = set()
        for entry in manifest:
            relative = str(entry.get("file") or "")
            if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("MoLE v4 expert manifest contains an unsafe path")
            shard_path = Path(directory) / relative
            if not shard_path.is_file():
                raise ValueError(f"MoLE v4 expert shard is missing: {relative}")
            if relative in seen_files:
                raise ValueError("MoLE v4 expert manifest contains duplicate shard paths")
            seen_files.add(relative)
            parameters = entry.get("parameters")
            storage_bytes = entry.get("bytes")
            checksum = entry.get("sha256")
            if not isinstance(parameters, int) or parameters < 1:
                raise ValueError("MoLE v4 expert manifest has an invalid parameter count")
            if adaptive.expert_parameter_count and parameters != adaptive.expert_parameter_count:
                raise ValueError("MoLE v4 expert manifest violates its parameter budget")
            if not isinstance(storage_bytes, int) or storage_bytes < 1:
                raise ValueError("MoLE v4 expert manifest has an invalid byte count")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError("MoLE v4 expert manifest has no valid SHA-256 digest")
    return payload


def load_mole_weights(directory: Path) -> dict[str, torch.Tensor]:
    directory = Path(directory)
    payload = load_mole_config(directory)
    if int(payload.get("format_version", 1)) == 4:
        store = LazyExpertStore(directory)
        state = store.routing_weights()
        for expert_index in range(store.num_experts):
            state.update(store.expert_weights(expert_index))
        return state
    return dict(load_file(str(directory / WEIGHTS_NAME)))


class LazyExpertStore:
    """Strict v4 lane store that materializes only requested expert shards."""

    def __init__(self, directory: Path, *, expected_base_ref: str | None = None) -> None:
        self.directory = Path(directory)
        self.payload = load_mole_config(
            self.directory,
            expected_base_ref=expected_base_ref,
        )
        if int(self.payload.get("format_version", 1)) != 4:
            raise ValueError("lazy expert residency requires a MoLE v4 artifact")
        self.manifest = {
            int(entry["index"]): dict(entry) for entry in self.payload["expert_manifest"]
        }
        self._verified: set[int] = set()

    @property
    def num_experts(self) -> int:
        return len(self.manifest)

    @property
    def bytes_per_expert(self) -> int:
        values = [int(entry.get("bytes") or 0) for entry in self.manifest.values()]
        return max(values, default=0)

    def routing_weights(self) -> dict[str, torch.Tensor]:
        return dict(load_file(str(self.directory / WEIGHTS_NAME), device="cpu"))

    def expert_weights(self, expert_index: int) -> dict[str, torch.Tensor]:
        try:
            entry = self.manifest[int(expert_index)]
        except KeyError as exc:
            raise IndexError(f"expert index {expert_index} is outside the v4 bank") from exc
        path = self.directory / entry["file"]
        tensors = dict(load_file(str(path), device="cpu"))
        if expert_index not in self._verified:
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"adaptive expert {expert_index} failed its SHA-256 check")
            self._verified.add(expert_index)
        return tensors

    def validate_model(self, model) -> None:
        layers = mole_layers(model)
        if not layers:
            raise ValueError("model has no injected MoLE layers")
        for expert_index, entry in self.manifest.items():
            expected: dict[str, tuple[int, ...]] = {}
            for name, layer in layers:
                key = name.replace(".", "__")
                expert = layer.experts[expert_index]
                expected[f"{key}.expert_{expert_index}.lora_A"] = tuple(
                    expert.lora_A.shape
                )
                expected[f"{key}.expert_{expert_index}.lora_B"] = tuple(
                    expert.lora_B.shape
                )
            path = self.directory / entry["file"]
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                actual = set(handle.keys())
                if actual != set(expected):
                    missing = sorted(set(expected) - actual)
                    extra = sorted(actual - set(expected))
                    raise ValueError(
                        f"adaptive expert {expert_index} tensor set mismatch; "
                        f"missing={missing[:3]}, extra={extra[:3]}"
                    )
                for key, shape in expected.items():
                    if tuple(handle.get_slice(key).get_shape()) != shape:
                        raise ValueError(
                            f"adaptive expert {expert_index} shape mismatch for {key}"
                        )


def validate_mole_weights(model, state: dict[str, torch.Tensor]) -> int:
    """Require a complete, shape-compatible expert bank before any copy."""

    expected: dict[str, torch.Tensor] = {}
    layers = mole_layers(model)
    if not layers:
        raise ValueError("model has no injected MoLE layers")
    expert_count = 0
    for name, layer in layers:
        key = name.replace(".", "__")
        for expert_index, expert in enumerate(layer.experts):
            expected[f"{key}.expert_{expert_index}.lora_A"] = expert.lora_A
            expected[f"{key}.expert_{expert_index}.lora_B"] = expert.lora_B
            expert_count += 1
        if layer.shared_token_router() is None:
            if layer.router is None:
                raise RuntimeError("non-shared MoLE layer is missing its router")
            expected[f"{key}.router.gate.weight"] = layer.router.gate.weight
    for name, router in token_routers(model):
        for parameter_name, tensor in router.state_dict().items():
            expected[f"{TOKEN_ROUTER_PREFIX}{name}.{parameter_name}"] = tensor
    prompt_router = shared_router(model)
    if prompt_router is not None:
        for name, tensor in prompt_router.state_dict().items():
            expected[f"{SHARED_ROUTER_PREFIX}{name}"] = tensor

    actual_keys = set(state)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"MoLE tensor set mismatch; missing={missing[:3]}, extra={extra[:3]}")
    for key, reference in expected.items():
        if state[key].shape != reference.shape:
            raise ValueError(
                f"MoLE tensor shape mismatch for {key}: got {tuple(state[key].shape)}, "
                f"expected {tuple(reference.shape)}"
            )
    return expert_count


def validate_mole_routing_weights(
    model,
    state: dict[str, torch.Tensor],
) -> int:
    """Validate the router-only portion of a lazily resident v4 artifact."""

    expected: dict[str, torch.Tensor] = {}
    routers = token_routers(model)
    if not routers:
        raise ValueError("adaptive MoLE model has no shared token routers")
    for name, router in routers:
        for parameter_name, tensor in router.state_dict().items():
            expected[f"{TOKEN_ROUTER_PREFIX}{name}.{parameter_name}"] = tensor
    actual = set(state)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(
            f"MoLE routing tensor set mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    for key, reference in expected.items():
        if tuple(state[key].shape) != tuple(reference.shape):
            raise ValueError(
                f"MoLE routing tensor shape mismatch for {key}: got {tuple(state[key].shape)}, "
                f"expected {tuple(reference.shape)}"
            )
    return len(expected)


def apply_mole_routing_weights(
    model,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> int:
    """Load only the cheap routing heads, leaving all v4 lane shards lazy."""

    expected = validate_mole_routing_weights(model, state) if strict else None
    applied = 0
    for name, router in token_routers(model):
        prefix = f"{TOKEN_ROUTER_PREFIX}{name}."
        router_state = {
            key.removeprefix(prefix): tensor
            for key, tensor in state.items()
            if key.startswith(prefix)
        }
        applied += len(router_state)
        router.load_state_dict(router_state, strict=True)
    if expected is not None and applied != expected:
        raise RuntimeError(f"applied {applied} routing tensors, expected {expected}")
    return applied


def apply_mole_weights(model, state: dict[str, torch.Tensor], *, strict: bool = True) -> int:
    if strict:
        expected_count = validate_mole_weights(model, state)
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
            if layer.router is None:
                raise ValueError("artifact contains a legacy gate for an adaptive layer")
            layer.router.gate.weight.data.copy_(g.to(layer.router.gate.weight.device))
    for name, router in token_routers(model):
        prefix = f"{TOKEN_ROUTER_PREFIX}{name}."
        router_state = {
            key.removeprefix(prefix): tensor for key, tensor in state.items() if key.startswith(prefix)
        }
        router.load_state_dict(router_state, strict=True)
    prompt_router = shared_router(model)
    if prompt_router is not None:
        shared_state = {
            key.removeprefix(SHARED_ROUTER_PREFIX): tensor
            for key, tensor in state.items()
            if key.startswith(SHARED_ROUTER_PREFIX)
        }
        prompt_router.load_state_dict(shared_state, strict=True)
    if strict and applied != expected_count:
        raise RuntimeError(f"applied {applied} MoLE experts, expected {expected_count}")
    return applied
