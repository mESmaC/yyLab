"""Compiled CPU AdamW for large BF16 offload slices, with FP32 master state.

Compilation and validation run on disposable buffers before live optimizer
state is touched. Callers may choose native Adam on preparation failure; errors
from an already-selected live kernel must propagate instead of retrying writes.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import FunctionType

import torch

MIN_COMPILED_ELEMENTS = 1_048_576


def _activate_local_compiler() -> None:
    """Activate an installed Windows toolchain in this worker, without installs."""
    if sys.platform != "win32" or shutil.which(os.environ.get("CXX", "cl")):
        return
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError("no activated MSVC/clang-cl compiler or Visual Studio installation was found")
    result = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        text=True, capture_output=True, timeout=30, check=True,
    )
    installation = result.stdout.strip()
    if not installation:
        raise RuntimeError("Visual Studio has no installed C++ compiler toolchain")
    vcvars = Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
    if not vcvars.is_file():
        raise RuntimeError("the installed Visual Studio toolchain has no vcvars64.bat")
    # cmd requires its own outer quoting; Python's list quoting would insert
    # backslash-escaped quotes which cmd does not interpret as escapes.
    setup = subprocess.run(
        f'cmd.exe /d /s /c ""{vcvars}" >nul && set"',
        text=True, capture_output=True, timeout=60,
    )
    if setup.returncode:
        raise RuntimeError("activating the installed MSVC environment failed")
    # Do not print the environment: it may contain account tokens or secrets.
    for line in setup.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            os.environ[key] = value


def _fused_update(master, moment, variance, staging_bf16, scalars):
    gradient = staging_bf16.float()
    next_moment = moment * scalars[0] + gradient * scalars[1]
    next_variance = variance * scalars[2] + gradient.square() * scalars[3]
    denominator = next_variance.sqrt() * scalars[4] + scalars[5]
    next_master = master * scalars[6] - (next_moment / denominator) * scalars[7]
    moment.copy_(next_moment)
    variance.copy_(next_variance)
    master.copy_(next_master)
    # This is the SAME tensor that supplied gradients. Fusion must preserve the
    # alias and finish all CPU writes before the caller enqueues GPU writeback.
    staging_bf16.copy_(next_master)


def _coefficients(step, lr, beta1, beta2, eps, weight_decay) -> torch.Tensor:
    return torch.tensor([
        beta1, 1 - beta1, beta2, 1 - beta2, 1 / math.sqrt(1 - beta2**step), eps,
        1 - lr * weight_decay, lr / (1 - beta1**step),
    ], dtype=torch.float32)


@torch.no_grad()
def _validate_alias_kernel(kernel) -> None:
    """Validate shared staging and changing scalars before selecting the kernel."""
    from torch.optim import _functional

    master = torch.linspace(-0.8, 0.8, 257)
    moment = torch.linspace(-0.03, 0.03, 257)
    variance = torch.linspace(0.02, 0.2, 257)
    staging = torch.empty(257, dtype=torch.bfloat16)
    expected = [value.clone() for value in (master, moment, variance)]
    step_tensor = torch.tensor(0.0)
    scalars = torch.empty(8)
    for step in range(1, 4):
        lr = 0.0003 * (1 + 0.07 * step)
        staging.copy_(torch.linspace(-1.5, 1.5, 257) * (1 + 0.1 * step))
        gradient = staging.float()
        scalars.copy_(_coefficients(step, lr, 0.9, 0.95, 1e-8, 0.1))
        kernel(master, moment, variance, staging, scalars)
        _functional.adamw(
            [expected[0]], [gradient], [expected[1]], [expected[2]], [], [step_tensor],
            fused=True, foreach=False, capturable=False, differentiable=False, amsgrad=False,
            beta1=0.9, beta2=0.95, lr=lr, weight_decay=0.1, eps=1e-8, maximize=False,
        )
        for actual, reference in zip((master, moment, variance), expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-6)
        if not torch.equal(staging, master.bfloat16()):
            raise RuntimeError("compiled CPU Adam did not preserve shared BF16 staging writeback")


@lru_cache(maxsize=8)
def _prepare_kernel(cpu_threads: int, warmup_elements: int):
    _activate_local_compiler()
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor import config
    from torch._inductor.cpp_builder import get_cpp_compiler

    compiler = get_cpp_compiler()
    inductor = lookup_backend("inductor")

    def backend(graph, inputs):
        # A tiny first shape can schedule a serial CPU loop even for later large
        # dynamic shapes. Warm with a real bounded bucket and isolate these
        # settings from compilation of the model's CUDA graph.
        with config.patch({"cpp.threads": cpu_threads, "compile_threads": 1, "fx_graph_cache": False}):
            return inductor(graph, inputs)

    # Dynamo counts recompilations by Python code object. Candidate sweeps can
    # prepare more than eight combinations of thread count and bucket size;
    # isolate their startup caches without raising any process-wide limit.
    name = f"_sprint_cpu_adam_{cpu_threads}_{warmup_elements}"
    code = _fused_update.__code__.replace(co_name=name, co_qualname=name)
    update = FunctionType(code, _fused_update.__globals__, name=name)
    kernel = torch.compile(update, backend=backend, fullgraph=True, dynamic=True)
    with torch.no_grad():
        master = torch.zeros(warmup_elements, dtype=torch.float32)
        moment = torch.zeros_like(master)
        variance = torch.zeros_like(master)
        staging = torch.ones(warmup_elements, dtype=torch.bfloat16)
        scalars = _coefficients(1, 0.0003, 0.9, 0.95, 1e-8, 0.1)
        kernel(master, moment, variance, staging, scalars)
        del master, moment, variance, staging, scalars
        _validate_alias_kernel(kernel)
    return kernel, str(compiler)


class CompiledCPUAdam:
    """One optimizer's scalar storage around a prepared dynamic CPU kernel."""

    def __init__(self, *, cpu_threads: int, warmup_elements: int):
        if cpu_threads < 1 or warmup_elements < MIN_COMPILED_ELEMENTS:
            raise ValueError("compiled CPU Adam requires positive threads and a representative large slice")
        self.cpu_threads = int(cpu_threads)
        self.warmup_elements = int(warmup_elements)
        self.kernel, self.compiler = _prepare_kernel(self.cpu_threads, self.warmup_elements)
        self.scalars = torch.empty(8, dtype=torch.float32)
        self._last_settings = None

    @torch.no_grad()
    def update(self, master, moment, variance, staging_bf16, *, step, lr, beta1, beta2, eps, weight_decay):
        settings = (int(step), float(lr), float(beta1), float(beta2), float(eps), float(weight_decay))
        if settings != self._last_settings:
            self.scalars.copy_(_coefficients(*settings))
            self._last_settings = settings
        # No exception handler here: a failed live call may have mutated a
        # subset of the state, so retrying with native Adam would be incorrect.
        self.kernel(master, moment, variance, staging_bf16, self.scalars)
