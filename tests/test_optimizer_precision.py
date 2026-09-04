from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yylab.train import loop
from yylab.train.loop import (
    CPUOffloadAdamW,
    make_optimizer,
    optimizer_runtime_metadata,
    resolve_optimizer_choice,
)
from yylab.ui.pages.train_base import (
    _available_optimizer_options,
    _compatible_optimizer_state_options,
    _validate_family_qat_selection,
)


def _status(**overrides: bool) -> dict[str, tuple[bool, str]]:
    return {
        implementation: (
            overrides.get(implementation, True),
            f"{implementation} probe detail",
        )
        for implementation in loop.OPTIMIZER_IMPLEMENTATIONS
    }


@pytest.mark.parametrize(
    ("implementation", "state"),
    [
        ("torch_adamw", "cpu_offload"),
        ("fused_adamw", "cpu_paged"),
        ("adamw_8bit", "cpu_paged"),
        ("paged_adamw_8bit", "gpu"),
        ("cpu_offload_adamw", "gpu"),
    ],
)
def test_contradictory_optimizer_implementation_and_state_reject(
    implementation: str,
    state: str,
) -> None:
    with pytest.raises(ValueError, match="not requested|stores state"):
        resolve_optimizer_choice(
            trainable_parameters=10,
            cuda=True,
            optimizer_state=state,
            implementation=implementation,
            parameter_dtype="bfloat16",
        )


def test_explicit_paged_state_fails_closed_when_real_probe_failed(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "optimizer_capability_status",
        lambda **_kwargs: _status(paged_adamw_8bit=False),
    )
    with pytest.raises(RuntimeError, match="paged_adamw_8bit is unavailable"):
        resolve_optimizer_choice(
            trainable_parameters=100_000_000,
            cuda=True,
            optimizer_state="cpu_paged",
            implementation="auto",
            parameter_dtype="bfloat16",
        )


def test_auto_uses_only_a_verified_choice_and_records_it(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "optimizer_capability_status",
        lambda **_kwargs: _status(
            paged_adamw_8bit=False,
            fused_adamw=False,
            torch_adamw=False,
        ),
    )
    resolved = resolve_optimizer_choice(
        trainable_parameters=100_000_000,
        cuda=True,
        optimizer_state="auto",
        implementation="auto",
        parameter_dtype="bfloat16",
    )
    assert resolved.implementation == "adamw_8bit"
    assert resolved.state == "gpu"
    assert resolved.state_precision == "int8_blockwise"
    assert resolved.capability_detail == "adamw_8bit probe detail"


def test_verified_construction_failure_does_not_fall_through(monkeypatch) -> None:
    class FakeParameter:
        requires_grad = True
        is_cuda = True
        device = torch.device("cuda:0")
        dtype = torch.bfloat16

        def numel(self):
            return 100_000_000

    class FakeModel:
        def parameters(self):
            return [FakeParameter()]

    def fail_paged(*_args, **_kwargs):
        raise RuntimeError("construction changed after probe")

    monkeypatch.setattr(loop, "optimizer_capability_status", lambda **_kwargs: _status())
    monkeypatch.setitem(
        __import__("sys").modules,
        "bitsandbytes",
        SimpleNamespace(optim=SimpleNamespace(PagedAdamW8bit=fail_paged)),
    )
    with pytest.raises(RuntimeError, match="failed to construct"):
        make_optimizer(FakeModel(), 1e-3, cuda=True)


def test_make_optimizer_probes_the_trainable_parameters_exact_cuda_device(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeParameter:
        requires_grad = True
        is_cuda = True
        device = torch.device("cuda:3")
        dtype = torch.bfloat16

        def numel(self):
            return 8

    class FakeModel:
        def parameters(self):
            return [FakeParameter()]

    def resolve(**kwargs):
        observed.update(kwargs)
        return loop.OptimizerResolution(
            implementation="torch_adamw",
            state="gpu",
            state_precision="bfloat16",
            master_weight_precision="none",
            capability_detail="exact cuda:3 probe",
        )

    monkeypatch.setattr(loop, "resolve_optimizer_choice", resolve)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *_args, **_kwargs: SimpleNamespace())

    optimizer = make_optimizer(FakeModel(), 1e-3, cuda=True)

    assert observed["device_index"] == 3
    assert observed["parameter_dtype"] == "bfloat16"
    assert optimizer_runtime_metadata(optimizer)["capability_detail"] == "exact cuda:3 probe"


def test_optimizer_capability_probe_uses_the_requested_parameter_dtype(monkeypatch) -> None:
    observed: list[tuple[bool, int, str]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def probe(use_cuda: bool, device_index: int, parameter_dtype: str):
        observed.append((use_cuda, device_index, parameter_dtype))
        return tuple(
            (implementation, True, "executed")
            for implementation in loop.OPTIMIZER_IMPLEMENTATIONS
        )

    monkeypatch.setattr(loop, "_probe_optimizer_capabilities", probe)

    loop.optimizer_capability_status(
        cuda=True,
        device_index=0,
        parameter_dtype="float16",
    )

    assert observed == [(True, 0, "float16")]


def test_make_optimizer_rejects_mixed_cuda_devices_before_probe(monkeypatch) -> None:
    class FakeParameter:
        requires_grad = True
        is_cuda = True
        dtype = torch.bfloat16

        def __init__(self, index: int):
            self.device = torch.device("cuda", index)

        def numel(self):
            return 8

    class FakeModel:
        def parameters(self):
            return [FakeParameter(0), FakeParameter(1)]

    called = False

    def resolve(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(loop, "resolve_optimizer_choice", resolve)

    with pytest.raises(ValueError, match="one exact device"):
        make_optimizer(FakeModel(), 1e-3, cuda=True)
    assert called is False


def test_cpu_offload_receipt_reports_fp32_master_and_moments() -> None:
    optimizer = make_optimizer(
        torch.nn.Linear(2, 2),
        1e-3,
        cuda=False,
        optimizer_state="cpu_offload",
        implementation="cpu_offload_adamw",
    )
    assert isinstance(optimizer, CPUOffloadAdamW)
    assert optimizer_runtime_metadata(optimizer) == {
        "implementation": "cpu_offload_adamw",
        "state": "cpu_offload",
        "state_precision": "fp32",
        "master_weight_precision": "fp32",
        "capability_detail": "real update passed with FP32 CPU master and moments",
    }


def test_base_worker_persists_the_resolved_optimizer_receipt() -> None:
    from yylab.workers.train import _make_base_optimizer

    config = {
        "optimizer": {
            "implementation": "cpu_offload_adamw",
            "state": "cpu_offload",
            "lr": 2e-4,
        }
    }
    optimizer = _make_base_optimizer(torch.nn.Linear(2, 2), config, cuda=False)
    assert config["optimizer"]["resolved"] == optimizer_runtime_metadata(optimizer)
    assert config["optimizer"]["resolved"]["master_weight_precision"] == "fp32"


def test_base_worker_rejects_a_contradictory_optimizer_recipe() -> None:
    from yylab.workers.train import _make_base_optimizer

    config = {
        "optimizer": {
            "implementation": "torch_adamw",
            "state": "cpu_offload",
            "lr": 2e-4,
        }
    }
    with pytest.raises(ValueError, match="stores state"):
        _make_base_optimizer(torch.nn.Linear(2, 2), config, cuda=False)


def test_base_ui_only_offers_probed_implementations_and_compatible_states() -> None:
    status = _status(adamw_8bit=False, paged_adamw_8bit=False)
    implementations = _available_optimizer_options(status)
    assert "adamw_8bit" not in implementations
    assert "paged_adamw_8bit" not in implementations
    assert "fused_adamw" in implementations

    assert _compatible_optimizer_state_options(
        "fused_adamw", status, cuda_available=True
    ) == {"gpu": "GPU-resident state"}
    assert _compatible_optimizer_state_options(
        "cpu_offload_adamw", status, cuda_available=True
    ) == {"cpu_offload": "Full FP32 CPU offload"}
    auto_states = _compatible_optimizer_state_options(
        "auto", status, cuda_available=True
    )
    assert "cpu_paged" not in auto_states
    assert "gpu" in auto_states
    assert "cpu_offload" in auto_states


def test_bitnet_rejects_nvfp4_instead_of_rewriting_it() -> None:
    with pytest.raises(ValueError, match="BitNet supports dense precision only"):
        _validate_family_qat_selection("bitnet", "nvfp4")

    _validate_family_qat_selection("bitnet", "none")
    _validate_family_qat_selection("qwen3", "nvfp4")
