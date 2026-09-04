from __future__ import annotations

import ctypes
import importlib.metadata
import os
import platform
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

import torch


class SparseNVFP4Error(RuntimeError):
    """Raised when the native cuSPARSELt NVFP4 path cannot run safely."""


# Values are part of CUDA 13.3's public library_types.h/cusparse.h ABI.
_CUDA_R_16BF = 14
_CUDA_R_4F_E2M1 = 33
_CUSPARSE_OPERATION_NON_TRANSPOSE = 0
_CUSPARSE_ORDER_COL = 1
_CUSPARSE_ORDER_ROW = 2
_CUSPARSE_COMPUTE_32F = 2
_CUSPARSELT_SPARSITY_50_PERCENT = 0
_CUSPARSELT_MATMUL_ALG_DEFAULT = 0
_CUSPARSELT_PRUNE_SPMMA_STRIP = 1

# cusparseLtMatmulDescAttribute_t values from cuSPARSELt 0.9.1.
_CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING = 5
_CUSPARSELT_MATMUL_A_SCALE_MODE = 10
_CUSPARSELT_MATMUL_B_SCALE_MODE = 11
_CUSPARSELT_MATMUL_A_SCALE_POINTER = 15
_CUSPARSELT_MATMUL_B_SCALE_POINTER = 16
_CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3 = 2

_OPAQUE_BYTES = 512
_FP4_MAX = 6.0
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 0.015625
_MAX_OUTPUT_DIM = 2_097_120

_DLL_LOCK = threading.RLock()
_DLL: ctypes.WinDLL | ctypes.CDLL | None = None
_DLL_DIR_HANDLES: list[object] = []
_HANDLES: dict[int, _DeviceHandle] = {}
_EXECUTION_CACHE: dict[tuple[int, int, int, int, int], _ExecutionResources] = {}
_COMPRESSED_WEIGHT_CACHE: dict[
    tuple[int, int, int, int, int, int], _CompressedWeight
] = {}
_MAX_CACHED_PLANS = 64
_STATUS_CACHE: dict[int, tuple[bool, str]] = {}
_MXFP8_STATUS_CACHE: dict[int, tuple[bool, str]] = {}


def _ceil_multiple(value: int, multiple: int) -> int:
    if value < 1:
        raise ValueError("NVFP4 GEMM dimensions must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def _resolved_cuda_index(
    device: str | torch.device | None = None,
) -> tuple[int | None, str]:
    if not torch.cuda.is_available():
        return None, "CUDA is unavailable"
    resolved = torch.device(device if device is not None else "cuda")
    if resolved.type != "cuda":
        return None, f"sparse NVFP4 requires CUDA, got {resolved}"
    index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
    if index < 0 or index >= torch.cuda.device_count():
        return None, f"CUDA device index {index} is unavailable"
    return index, ""


@dataclass(frozen=True)
class _PaddedShape:
    m: int
    n: int
    k: int
    padded_m: int
    padded_n: int
    padded_k: int


def _padded_shape(m: int, n: int, k: int) -> _PaddedShape:
    padded_m = _ceil_multiple(int(m), 16)
    padded_n = _ceil_multiple(int(n), 32)
    # The 0.9.1 SM120 E2M1 descriptors accept K in multiples of 32, but the
    # selected compressed-matmul kernel returns CUSPARSE_STATUS_INTERNAL_ERROR
    # for K == 32 (mod 64).  Use the stronger, smoke-tested execution contract.
    padded_k = _ceil_multiple(int(k), 64)
    if padded_m > _MAX_OUTPUT_DIM or padded_n > _MAX_OUTPUT_DIM:
        raise ValueError(
            f"cuSPARSELt output dimensions must not exceed {_MAX_OUTPUT_DIM}, "
            f"got padded M={padded_m}, N={padded_n}"
        )
    return _PaddedShape(int(m), int(n), int(k), padded_m, padded_n, padded_k)


class _Opaque:
    """Own a 512-byte cuSPARSELt object with the ABI-required 16-byte alignment."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(_OPAQUE_BYTES + 15)
        base = ctypes.addressof(self._storage)
        self.address = (base + 15) & ~15
        if self.address % 16:
            raise SparseNVFP4Error("failed to allocate aligned cuSPARSELt host state")

    @property
    def ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(self.address)


def _distribution_root() -> tuple[Path, str]:
    try:
        dist = importlib.metadata.distribution("nvidia-cusparselt-cu13")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SparseNVFP4Error("nvidia-cusparselt-cu13 is not installed") from exc
    return Path(dist.locate_file("nvidia/cu13")), dist.version


def _configure_function(
    library: ctypes.WinDLL | ctypes.CDLL,
    name: str,
    argtypes: list[object],
    restype: object = ctypes.c_int,
) -> None:
    function = getattr(library, name)
    function.argtypes = argtypes
    function.restype = restype


def _load_library() -> ctypes.WinDLL | ctypes.CDLL:
    global _DLL
    with _DLL_LOCK:
        if _DLL is not None:
            return _DLL
        if platform.system() != "Windows":
            raise SparseNVFP4Error(
                "this ctypes backend currently targets the cuSPARSELt Windows wheel"
            )

        root, package_version = _distribution_root()
        if tuple(int(part) for part in package_version.split(".")[:3]) < (0, 9, 1):
            raise SparseNVFP4Error(
                f"cuSPARSELt >= 0.9.1 is required, found {package_version}"
            )
        dll_path = root / "bin" / "x64" / "cusparseLt.dll"
        if not dll_path.is_file():
            raise SparseNVFP4Error(f"cuSPARSELt DLL is missing: {dll_path}")

        candidate_dirs = [dll_path.parent]
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            candidate_dirs.extend([Path(cuda_path) / "bin" / "x64", Path(cuda_path) / "bin"])
        for directory in candidate_dirs:
            if directory.is_dir():
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))

        try:
            library = ctypes.WinDLL(str(dll_path))
        except OSError as exc:
            raise SparseNVFP4Error(f"failed to load {dll_path}: {exc}") from exc

        pointer = ctypes.c_void_p
        size_pointer = ctypes.POINTER(ctypes.c_size_t)
        _configure_function(library, "cusparseLtInit", [pointer])
        _configure_function(library, "cusparseLtDestroy", [pointer])
        _configure_function(library, "cusparseLtGetVersion", [pointer, ctypes.POINTER(ctypes.c_int)])
        _configure_function(
            library, "cusparseLtGetErrorName", [ctypes.c_int], ctypes.c_char_p
        )
        _configure_function(
            library, "cusparseLtGetErrorString", [ctypes.c_int], ctypes.c_char_p
        )
        _configure_function(
            library,
            "cusparseLtDenseDescriptorInit",
            [pointer, pointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
             ctypes.c_uint32, ctypes.c_int, ctypes.c_int],
        )
        _configure_function(
            library,
            "cusparseLtStructuredDescriptorInit",
            [pointer, pointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
             ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        )
        _configure_function(library, "cusparseLtMatDescriptorDestroy", [pointer])
        _configure_function(
            library,
            "cusparseLtMatmulDescriptorInit",
            [pointer, pointer, ctypes.c_int, ctypes.c_int, pointer, pointer, pointer,
             pointer, ctypes.c_int],
        )
        _configure_function(
            library,
            "cusparseLtMatmulDescSetAttribute",
            [pointer, pointer, ctypes.c_int, pointer, ctypes.c_size_t],
        )
        _configure_function(
            library,
            "cusparseLtMatmulAlgSelectionInit",
            [pointer, pointer, pointer, ctypes.c_int],
        )
        _configure_function(library, "cusparseLtMatmulAlgSelectionDestroy", [pointer])
        _configure_function(
            library, "cusparseLtMatmulPlanInit", [pointer, pointer, pointer, pointer]
        )
        _configure_function(library, "cusparseLtMatmulPlanDestroy", [pointer])
        _configure_function(
            library, "cusparseLtSpMMACompressedSize", [pointer, pointer, size_pointer, size_pointer]
        )
        _configure_function(
            library,
            "cusparseLtSpMMAPrune",
            [pointer, pointer, pointer, pointer, ctypes.c_int, pointer],
        )
        _configure_function(
            library,
            "cusparseLtSpMMACompress",
            [pointer, pointer, pointer, pointer, pointer, pointer],
        )
        _configure_function(
            library, "cusparseLtMatmulGetWorkspace", [pointer, pointer, size_pointer]
        )
        _configure_function(
            library,
            "cusparseLtMatmul",
            [pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
             pointer, ctypes.POINTER(pointer), ctypes.c_int32],
        )
        _DLL = library
        return library


def _error_text(status: int) -> str:
    library = _load_library()
    try:
        name = library.cusparseLtGetErrorName(status)
        detail = library.cusparseLtGetErrorString(status)
        decoded_name = name.decode("utf-8", "replace") if name else "unknown"
        decoded_detail = detail.decode("utf-8", "replace") if detail else "unknown error"
        return f"{decoded_name}: {decoded_detail}"
    except Exception:
        return "unknown cuSPARSELt error"


def _check(status: int, operation: str) -> None:
    if status != 0:
        raise SparseNVFP4Error(
            f"{operation} failed with cuSPARSELt status {status} ({_error_text(status)})"
        )


@dataclass
class _DeviceHandle:
    state: _Opaque
    version: int


def _device_handle(device_index: int) -> _DeviceHandle:
    with _DLL_LOCK:
        cached = _HANDLES.get(device_index)
        if cached is not None:
            return cached
        library = _load_library()
        with torch.cuda.device(device_index):
            state = _Opaque()
            _check(library.cusparseLtInit(state.ptr), "cusparseLtInit")
            version = ctypes.c_int()
            _check(
                library.cusparseLtGetVersion(state.ptr, ctypes.byref(version)),
                "cusparseLtGetVersion",
            )
        if version.value < 901:
            library.cusparseLtDestroy(state.ptr)
            raise SparseNVFP4Error(
                f"cuSPARSELt runtime 0.9.1 or newer is required, found {version.value}"
            )
        result = _DeviceHandle(state=state, version=version.value)
        _HANDLES[device_index] = result
        return result


def _paired_4to8_mask(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"paired 4:8 weights must be 2-D, got shape {tuple(value.shape)}")
    if value.shape[-1] % 8:
        raise ValueError("paired 4:8 weights require K divisible by 8")
    groups = value.reshape(-1, 4, 2)
    pair_scores = groups.detach().float().abs().sum(dim=-1)
    selected = pair_scores.topk(2, dim=-1, largest=True, sorted=False).indices
    pair_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
    pair_mask.scatter_(1, selected, True)
    return pair_mask.unsqueeze(-1).expand_as(groups).reshape_as(value)


def paired_4to8(value: torch.Tensor) -> torch.Tensor:
    """Apply deterministic two-pair-per-eight magnitude sparsity along K."""

    return value * _paired_4to8_mask(value)


def _padded_paired_weight(value: torch.Tensor, padded_k: int) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"paired 4:8 weights must be 2-D, got shape {tuple(value.shape)}")
    if padded_k < value.shape[1] or padded_k % 32:
        raise ValueError("padded K must cover the weight and be divisible by 32")
    padded = torch.zeros(
        (value.shape[0], padded_k), device=value.device, dtype=value.dtype
    )
    padded[:, : value.shape[1]] = value
    return paired_4to8(padded)


def _effective_weight_and_mask(
    value: torch.Tensor,
    padded_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.zeros(
        (value.shape[0], padded_k), device=value.device, dtype=value.dtype
    )
    padded[:, : value.shape[1]] = value
    mask = _paired_4to8_mask(padded)[:, : value.shape[1]]
    return value * mask, mask


def _quantize_group32(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack E2M1 values and return blocked UE4M3 scales plus a global scale."""

    if value.ndim != 2 or value.shape[-1] % 32:
        raise ValueError("group-32 NVFP4 quantization requires a 2-D tensor with K divisible by 32")
    if value.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"group-32 NVFP4 quantization requires BF16/FP32, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError("group-32 NVFP4 quantization requires a CUDA tensor")

    from torchao.prototype.mx_formats.nvfp4_tensor import (
        f32_to_f4_unpacked,
        pack_uint4,
        to_blocked,
    )

    rows, k = value.shape
    grouped = value.contiguous().float().reshape(rows, k // 32, 32)
    max_abs = grouped.abs().amax(dim=-1)
    global_amax = max_abs.amax()
    raw_global_scale = global_amax / (_FP4_MAX * _E4M3_MAX)
    global_scale = torch.where(
        global_amax > 0,
        raw_global_scale,
        torch.ones_like(raw_global_scale),
    )
    block_scales = max_abs / _FP4_MAX
    scaled_scales = torch.clamp(
        block_scales / global_scale,
        min=_E4M3_MIN_NORMAL,
        max=_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    denominator = scaled_scales.float() * global_scale
    scaled = torch.clamp(grouped / denominator.unsqueeze(-1), -_FP4_MAX, _FP4_MAX)
    packed = pack_uint4(f32_to_f4_unpacked(scaled.reshape(rows, k))).contiguous()
    blocked_scales = to_blocked(scaled_scales).contiguous()
    return packed, blocked_scales, global_scale


class _MatmulPlan:
    def __init__(
        self,
        handle: _DeviceHandle,
        shape: _PaddedShape,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> None:
        self.library = _load_library()
        self.handle = handle
        self.shape = shape
        self.mat_a = _Opaque()
        self.mat_b = _Opaque()
        self.mat_c = _Opaque()
        self.mat_d = _Opaque()
        self.matmul = _Opaque()
        self.algorithm = _Opaque()
        self.plan = _Opaque()
        self._descriptors = [self.mat_a, self.mat_b, self.mat_c, self.mat_d]
        self._algorithm_initialized = False
        self._plan_initialized = False
        self._closed = False

        h = handle.state.ptr
        s = shape
        _check(
            self.library.cusparseLtStructuredDescriptorInit(
                h, self.mat_a.ptr, s.padded_n, s.padded_k, s.padded_k, 16,
                _CUDA_R_4F_E2M1, _CUSPARSE_ORDER_ROW, _CUSPARSELT_SPARSITY_50_PERCENT,
            ),
            "cusparseLtStructuredDescriptorInit(A)",
        )
        _check(
            self.library.cusparseLtDenseDescriptorInit(
                h, self.mat_b.ptr, s.padded_k, s.padded_m, s.padded_k, 16,
                _CUDA_R_4F_E2M1, _CUSPARSE_ORDER_COL,
            ),
            "cusparseLtDenseDescriptorInit(B)",
        )
        for name, descriptor in (("C", self.mat_c), ("D", self.mat_d)):
            _check(
                self.library.cusparseLtDenseDescriptorInit(
                    h, descriptor.ptr, s.padded_n, s.padded_m, s.padded_n, 16,
                    _CUDA_R_16BF, _CUSPARSE_ORDER_COL,
                ),
                f"cusparseLtDenseDescriptorInit({name})",
            )
        _check(
            self.library.cusparseLtMatmulDescriptorInit(
                h, self.matmul.ptr,
                _CUSPARSE_OPERATION_NON_TRANSPOSE,
                _CUSPARSE_OPERATION_NON_TRANSPOSE,
                self.mat_a.ptr, self.mat_b.ptr, self.mat_c.ptr, self.mat_d.ptr,
                _CUSPARSE_COMPUTE_32F,
            ),
            "cusparseLtMatmulDescriptorInit",
        )
        self._set_int_attribute(
            _CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, 1, "alpha vector mode"
        )
        self._set_int_attribute(
            _CUSPARSELT_MATMUL_A_SCALE_MODE,
            _CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3,
            "A scale mode",
        )
        self._set_int_attribute(
            _CUSPARSELT_MATMUL_B_SCALE_MODE,
            _CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3,
            "B scale mode",
        )
        self._set_pointer_attribute(
            _CUSPARSELT_MATMUL_A_SCALE_POINTER, a_scale.data_ptr(), "A scale pointer"
        )
        self._set_pointer_attribute(
            _CUSPARSELT_MATMUL_B_SCALE_POINTER, b_scale.data_ptr(), "B scale pointer"
        )
        _check(
            self.library.cusparseLtMatmulAlgSelectionInit(
                h, self.algorithm.ptr, self.matmul.ptr, _CUSPARSELT_MATMUL_ALG_DEFAULT
            ),
            "cusparseLtMatmulAlgSelectionInit",
        )
        self._algorithm_initialized = True
        _check(
            self.library.cusparseLtMatmulPlanInit(
                h, self.plan.ptr, self.matmul.ptr, self.algorithm.ptr
            ),
            "cusparseLtMatmulPlanInit",
        )
        self._plan_initialized = True

        compressed_size = ctypes.c_size_t()
        compress_buffer_size = ctypes.c_size_t()
        _check(
            self.library.cusparseLtSpMMACompressedSize(
                h, self.plan.ptr,
                ctypes.byref(compressed_size), ctypes.byref(compress_buffer_size),
            ),
            "cusparseLtSpMMACompressedSize",
        )
        workspace_size = ctypes.c_size_t()
        _check(
            self.library.cusparseLtMatmulGetWorkspace(
                h, self.plan.ptr, ctypes.byref(workspace_size)
            ),
            "cusparseLtMatmulGetWorkspace",
        )
        self.compressed_size = int(compressed_size.value)
        self.compress_buffer_size = int(compress_buffer_size.value)
        self.workspace_size = int(workspace_size.value)

    def _set_int_attribute(self, attribute: int, value: int, label: str) -> None:
        data = ctypes.c_int(value)
        _check(
            self.library.cusparseLtMatmulDescSetAttribute(
                self.handle.state.ptr, self.matmul.ptr, attribute,
                ctypes.byref(data), ctypes.sizeof(data),
            ),
            f"cusparseLtMatmulDescSetAttribute({label})",
        )

    def _set_pointer_attribute(self, attribute: int, address: int, label: str) -> None:
        data = ctypes.c_void_p(address)
        _check(
            self.library.cusparseLtMatmulDescSetAttribute(
                self.handle.state.ptr, self.matmul.ptr, attribute,
                ctypes.byref(data), ctypes.sizeof(data),
            ),
            f"cusparseLtMatmulDescSetAttribute({label})",
        )

    def update_scale_pointers(self, a_scale: torch.Tensor, b_scale: torch.Tensor) -> None:
        if self._closed:
            raise SparseNVFP4Error("cannot update a closed cuSPARSELt plan")
        self._set_pointer_attribute(
            _CUSPARSELT_MATMUL_A_SCALE_POINTER, a_scale.data_ptr(), "A scale pointer"
        )
        self._set_pointer_attribute(
            _CUSPARSELT_MATMUL_B_SCALE_POINTER, b_scale.data_ptr(), "B scale pointer"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._plan_initialized:
            self.library.cusparseLtMatmulPlanDestroy(self.plan.ptr)
        if self._algorithm_initialized:
            self.library.cusparseLtMatmulAlgSelectionDestroy(self.algorithm.ptr)
        for descriptor in reversed(self._descriptors):
            self.library.cusparseLtMatDescriptorDestroy(descriptor.ptr)

    def __enter__(self) -> _MatmulPlan:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def _device_buffer(size: int, device: torch.device) -> torch.Tensor:
    return torch.empty(max(0, int(size)), device=device, dtype=torch.uint8)


def _pointer(tensor: torch.Tensor | None) -> ctypes.c_void_p:
    if tensor is None or tensor.numel() == 0:
        return ctypes.c_void_p()
    address = tensor.data_ptr()
    if address % 16:
        raise SparseNVFP4Error(f"device pointer 0x{address:x} is not 16-byte aligned")
    return ctypes.c_void_p(address)


@dataclass
class _ExecutionResources:
    plan: _MatmulPlan
    a_scale: torch.Tensor
    b_scale: torch.Tensor
    compressed: torch.Tensor
    compress_buffer: torch.Tensor
    workspace: torch.Tensor


@dataclass
class _CompressedWeight:
    """Compressed native payload tied to one immutable parameter version."""

    source: weakref.ReferenceType[torch.Tensor]
    source_version: int
    source_data_ptr: int
    source_shape: tuple[int, ...]
    source_stride: tuple[int, ...]
    source_dtype: torch.dtype
    source_device: torch.device
    scales: torch.Tensor
    global_scale: torch.Tensor
    compressed: torch.Tensor


def _compressed_weight_key(
    source: torch.Tensor,
    device_index: int,
    stream_value: int,
    shape: _PaddedShape,
) -> tuple[int, int, int, int, int, int]:
    return (
        id(source),
        device_index,
        stream_value,
        shape.padded_m,
        shape.padded_n,
        shape.padded_k,
    )


def _valid_compressed_weight(
    cached: _CompressedWeight | None,
    source: torch.Tensor,
) -> bool:
    return bool(
        cached is not None
        and cached.source() is source
        and cached.source_version == int(source._version)
        and cached.source_data_ptr == source.data_ptr()
        and cached.source_shape == tuple(source.shape)
        and cached.source_stride == tuple(source.stride())
        and cached.source_dtype == source.dtype
        and cached.source_device == source.device
    )


def _new_execution_resources(
    handle: _DeviceHandle,
    shape: _PaddedShape,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    device: torch.device,
) -> _ExecutionResources:
    # Scale pointers are captured by the matmul plan.  Keep stable, plan-owned
    # buffers rather than pointing it at per-call tensors that the CUDA caching
    # allocator may recycle before the next invocation.
    a_scale_buffer = torch.empty_like(a_scale)
    b_scale_buffer = torch.empty_like(b_scale)
    a_scale_buffer.copy_(a_scale)
    b_scale_buffer.copy_(b_scale)
    plan = _MatmulPlan(handle, shape, a_scale_buffer, b_scale_buffer)
    return _ExecutionResources(
        plan=plan,
        a_scale=a_scale_buffer,
        b_scale=b_scale_buffer,
        compressed=_device_buffer(plan.compressed_size, device),
        compress_buffer=_device_buffer(plan.compress_buffer_size, device),
        workspace=_device_buffer(plan.workspace_size, device),
    )


def _execution_resources(
    handle: _DeviceHandle,
    device_index: int,
    stream_value: int,
    shape: _PaddedShape,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    device: torch.device,
) -> tuple[_ExecutionResources, bool]:
    key = (
        device_index,
        stream_value,
        shape.padded_m,
        shape.padded_n,
        shape.padded_k,
    )
    cached = _EXECUTION_CACHE.get(key)
    if cached is not None:
        if (
            cached.a_scale.shape != a_scale.shape
            or cached.b_scale.shape != b_scale.shape
            or cached.a_scale.dtype != a_scale.dtype
            or cached.b_scale.dtype != b_scale.dtype
        ):
            raise SparseNVFP4Error("cached cuSPARSELt scale buffers do not match this GEMM")
        cached.a_scale.copy_(a_scale)
        cached.b_scale.copy_(b_scale)
        return cached, False
    resources = _new_execution_resources(handle, shape, a_scale, b_scale, device)
    if len(_EXECUTION_CACHE) < _MAX_CACHED_PLANS:
        _EXECUTION_CACHE[key] = resources
        return resources, False
    return resources, True


def clear_sparse_nvfp4_cache() -> None:
    """Synchronize and destroy cached plans and compressed weight payloads."""

    with _DLL_LOCK:
        devices = {key[0] for key in _EXECUTION_CACHE}
        devices.update(key[1] for key in _COMPRESSED_WEIGHT_CACHE)
        for device_index in devices:
            torch.cuda.synchronize(device_index)
        for resources in _EXECUTION_CACHE.values():
            resources.plan.close()
        _EXECUTION_CACHE.clear()
        _COMPRESSED_WEIGHT_CACHE.clear()


def sparse_nvfp4_forward(
    input: torch.Tensor,
    paired_weight: torch.Tensor,
    *,
    _weight_source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a compressed paired-4:8 E2M1 forward GEMM and return BF16 output.

    ``input`` is ``[M, K]`` and ``paired_weight`` is ``[N, K]``. The weight is
    pruned once more by cuSPARSELt's strip algorithm before compression, so the
    library always receives a valid paired-4:8 payload.
    """

    if input.ndim != 2 or paired_weight.ndim != 2:
        raise ValueError("sparse NVFP4 forward requires 2-D input and weight tensors")
    if input.shape[1] != paired_weight.shape[1]:
        raise ValueError(
            f"input K={input.shape[1]} does not match weight K={paired_weight.shape[1]}"
        )
    if input.dtype != torch.bfloat16 or paired_weight.dtype != torch.bfloat16:
        raise TypeError("sparse NVFP4 forward requires BF16 input and weight tensors")
    if not input.is_cuda or not paired_weight.is_cuda:
        raise ValueError("sparse NVFP4 forward requires CUDA tensors")
    if input.device != paired_weight.device:
        raise ValueError("input and weight must be on the same CUDA device")
    if _weight_source is not None:
        if (
            _weight_source.shape != paired_weight.shape
            or _weight_source.dtype != paired_weight.dtype
            or _weight_source.device != paired_weight.device
        ):
            raise ValueError("compressed-weight cache source must match the effective weight")
    m, k = input.shape
    n = paired_weight.shape[0]
    shape = _padded_shape(m, n, k)
    device = input.device
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise SparseNVFP4Error(
            f"this backend is validated only for SM120, found SM{capability[0]}{capability[1]}"
        )

    with torch.cuda.device(device), _DLL_LOCK:
        input_padded = torch.zeros(
            (shape.padded_m, shape.padded_k), device=device, dtype=torch.bfloat16
        )
        input_padded[:m, :k] = input
        packed_input, input_scales, input_global = _quantize_group32(input_padded)
        stream_value = int(torch.cuda.current_stream(device).cuda_stream)
        stream = ctypes.c_void_p(stream_value)
        stream_array = (ctypes.c_void_p * 1)(stream)
        handle = _device_handle(device_index)

        weight_cache_key = None
        cached_weight = None
        source_version = None
        execution_key = (
            device_index,
            stream_value,
            shape.padded_m,
            shape.padded_n,
            shape.padded_k,
        )
        persistent_plan = (
            execution_key in _EXECUTION_CACHE
            or len(_EXECUTION_CACHE) < _MAX_CACHED_PLANS
        )
        if _weight_source is not None and persistent_plan:
            source_version = int(_weight_source._version)
            weight_cache_key = _compressed_weight_key(
                _weight_source,
                device_index,
                stream_value,
                shape,
            )
            candidate = _COMPRESSED_WEIGHT_CACHE.get(weight_cache_key)
            if _valid_compressed_weight(candidate, _weight_source):
                cached_weight = candidate

        packed_weight = None
        if cached_weight is None:
            paired_rows = _padded_paired_weight(paired_weight, shape.padded_k)
            weight_padded = torch.zeros(
                (shape.padded_n, shape.padded_k),
                device=device,
                dtype=torch.bfloat16,
            )
            weight_padded[:n] = paired_rows
            packed_weight, weight_scales, weight_global = _quantize_group32(weight_padded)
        else:
            weight_scales = cached_weight.scales
            weight_global = cached_weight.global_scale

        resources, ephemeral_resources = _execution_resources(
            handle,
            device_index,
            stream_value,
            shape,
            weight_scales,
            input_scales,
            device,
        )
        plan = resources.plan
        compressed_weight = (
            cached_weight.compressed if cached_weight is not None else resources.compressed
        )
        try:
            library = plan.library
            if cached_weight is None:
                if packed_weight is None:
                    raise SparseNVFP4Error("missing packed weight on compressed-cache miss")
                if weight_cache_key is not None:
                    # Allocate before replacing a stale entry. On a given cache
                    # key this is the same CUDA stream, so queued use of the old
                    # payload precedes compression into the new allocation.
                    compressed_weight = _device_buffer(plan.compressed_size, device)
                _check(
                    library.cusparseLtSpMMAPrune(
                        handle.state.ptr,
                        plan.matmul.ptr,
                        _pointer(packed_weight),
                        _pointer(packed_weight),
                        _CUSPARSELT_PRUNE_SPMMA_STRIP,
                        stream,
                    ),
                    "cusparseLtSpMMAPrune",
                )
                _check(
                    library.cusparseLtSpMMACompress(
                        handle.state.ptr,
                        plan.plan.ptr,
                        _pointer(packed_weight),
                        _pointer(compressed_weight),
                        _pointer(resources.compress_buffer),
                        stream,
                    ),
                    "cusparseLtSpMMACompress",
                )
                if weight_cache_key is not None and _weight_source is not None:
                    if source_version is None:
                        raise SparseNVFP4Error("missing compressed-weight source version")
                    _COMPRESSED_WEIGHT_CACHE[weight_cache_key] = _CompressedWeight(
                        source=weakref.ref(_weight_source),
                        source_version=source_version,
                        source_data_ptr=_weight_source.data_ptr(),
                        source_shape=tuple(_weight_source.shape),
                        source_stride=tuple(_weight_source.stride()),
                        source_dtype=_weight_source.dtype,
                        source_device=_weight_source.device,
                        scales=weight_scales,
                        global_scale=weight_global,
                        compressed=compressed_weight,
                    )
                    if int(_weight_source._version) != source_version:
                        _COMPRESSED_WEIGHT_CACHE.pop(weight_cache_key, None)
                        raise SparseNVFP4Error(
                            "weight changed while its compressed NVFP4 payload was being built"
                        )

            alpha = torch.full(
                (max(shape.padded_m, shape.padded_n),),
                1.0,
                device=device,
                dtype=torch.float32,
            ) * (weight_global * input_global)
            output = torch.zeros(
                (shape.padded_m, shape.padded_n),
                device=device,
                dtype=torch.bfloat16,
            )
            _check(
                library.cusparseLtMatmul(
                    handle.state.ptr,
                    plan.plan.ptr,
                    _pointer(alpha),
                    _pointer(compressed_weight),
                    _pointer(packed_input),
                    ctypes.c_void_p(),
                    _pointer(output),
                    _pointer(output),
                    _pointer(resources.workspace),
                    stream_array,
                    1,
                ),
                "cusparseLtMatmul",
            )
        finally:
            if ephemeral_resources:
                plan.close()
        return output[:m, :n].contiguous()


class _SparseNVFP4Linear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: object,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        backward_precision: str,
        backward_backend: str,
    ) -> torch.Tensor:
        original_shape = input.shape
        flat_input = input.reshape(-1, input.shape[-1])
        padded_k = _ceil_multiple(weight.shape[1], 64)
        sparse_weight, sparse_mask = _effective_weight_and_mask(weight, padded_k)
        output = sparse_nvfp4_forward(
            flat_input,
            sparse_weight,
            _weight_source=weight,
        )
        ctx.save_for_backward(flat_input, sparse_weight, sparse_mask)
        ctx.input_shape = original_shape
        ctx.has_bias = bias is not None
        ctx.backward_precision = backward_precision
        ctx.backward_backend = backward_backend
        if bias is not None:
            output = output + bias
        return output.reshape(*original_shape[:-1], weight.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: object, grad_output: torch.Tensor):
        input, sparse_weight, sparse_mask = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        if ctx.backward_precision == "mxfp8":
            from .mxfp8 import mxfp8_linear_backward

            grad_input, grad_weight = mxfp8_linear_backward(
                input,
                sparse_weight,
                grad_flat,
                backend=ctx.backward_backend,
            )
            grad_input = grad_input.reshape(ctx.input_shape)
        else:
            grad_input = grad_flat.mm(sparse_weight).reshape(ctx.input_shape)
            grad_weight = grad_flat.t().mm(input)
        # The hard 4:8 choice is part of the executed operator.  Pruned
        # coefficients therefore receive no gradient; returning the dense
        # wgrad here would silently reintroduce a straight-through mask.
        grad_weight = grad_weight * sparse_mask
        grad_bias = grad_flat.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None


def sparse_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    backward_precision: str = "bf16",
    backward_backend: str = "none",
) -> torch.Tensor:
    """Autograd wrapper around the compressed sparse NVFP4 forward primitive."""

    if input.ndim < 1:
        raise ValueError("sparse NVFP4 linear input must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError("sparse NVFP4 linear weight must be 2-D")
    if input.shape[-1] != weight.shape[-1]:
        raise ValueError("sparse NVFP4 linear input and weight K dimensions must match")
    if backward_precision not in {"bf16", "mxfp8"}:
        raise ValueError(f"unknown sparse NVFP4 backward precision: {backward_precision}")
    if backward_precision == "bf16":
        if backward_backend != "none":
            raise ValueError("BF16 backward does not use a low-precision backend")
    else:
        from .mxfp8 import NATIVE_MXFP8_BACKEND

        if backward_backend not in {"native", NATIVE_MXFP8_BACKEND}:
            raise ValueError(
                f"MXFP8 backward requires {NATIVE_MXFP8_BACKEND!r}, "
                f"got {backward_backend!r}"
            )
    if bias is not None:
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"bias must have shape {(weight.shape[0],)}, got {tuple(bias.shape)}")
        if bias.dtype != torch.bfloat16 or bias.device != input.device:
            raise TypeError("sparse NVFP4 bias must be BF16 on the input device")
    return _SparseNVFP4Linear.apply(
        input,
        weight,
        bias,
        backward_precision,
        backward_backend,
    )


def sparse_nvfp4_status(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[bool, str]:
    """Report availability only after a real compressed sparse FP4 smoke GEMM."""

    device_index, unavailable = _resolved_cuda_index(device)
    if device_index is None:
        return False, unavailable
    resolved = torch.device("cuda", device_index)
    if force_refresh:
        _STATUS_CACHE.pop(device_index, None)
    if device_index in _STATUS_CACHE:
        return _STATUS_CACHE[device_index]
    try:
        capability = torch.cuda.get_device_capability(device_index)
        if capability != (12, 0):
            raise SparseNVFP4Error(
                f"the validated backend requires SM120, found SM{capability[0]}{capability[1]}"
            )
        _, package_version = _distribution_root()
        generator = torch.Generator(device="cpu").manual_seed(120)
        input_cpu = torch.randn((19, 47), generator=generator, dtype=torch.float32) * 0.25
        weight_cpu = torch.randn((35, 47), generator=generator, dtype=torch.float32) * 0.25
        grad_cpu = torch.randn((19, 35), generator=generator, dtype=torch.float32) * 0.25
        with torch.cuda.device(resolved), torch.inference_mode(False), torch.enable_grad():
            input = input_cpu.to(device=resolved, dtype=torch.bfloat16).requires_grad_(True)
            weight = weight_cpu.to(device=resolved, dtype=torch.bfloat16).requires_grad_(True)
            grad_output = grad_cpu.to(device=resolved, dtype=torch.bfloat16)
            sparse_weight, sparse_mask = _effective_weight_and_mask(
                weight.detach(), _ceil_multiple(weight.shape[1], 64)
            )
            output = sparse_nvfp4_linear(
                input,
                weight,
                backward_precision="bf16",
                backward_backend="none",
            )
            grad_input, grad_weight = torch.autograd.grad(
                output,
                (input, weight),
                grad_outputs=grad_output,
            )
        torch.cuda.synchronize(input.device)
        if output.dtype != torch.bfloat16 or output.shape != (19, 35):
            raise SparseNVFP4Error(
                f"smoke GEMM returned shape={tuple(output.shape)}, dtype={output.dtype}"
            )
        if not bool(torch.isfinite(output).all()) or not bool(output.abs().max() > 0):
            raise SparseNVFP4Error("smoke GEMM returned zero or non-finite output")
        for label, gradient in (("dgrad", grad_input), ("wgrad", grad_weight)):
            if not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0):
                raise SparseNVFP4Error(
                    f"smoke autograd returned zero or non-finite {label}"
                )
        reference = input.detach().float().mm(sparse_weight.float().t())
        error = (
            (output.detach().float() - reference).norm()
            / reference.norm().clamp_min(1e-12)
        )
        reference_dgrad = grad_output.float().mm(sparse_weight.float())
        reference_wgrad = grad_output.float().t().mm(input.detach().float())
        reference_wgrad.mul_(sparse_mask)
        dgrad_error = (
            (grad_input.float() - reference_dgrad).norm()
            / reference_dgrad.norm().clamp_min(1e-12)
        )
        wgrad_error = (
            (grad_weight.float() - reference_wgrad).norm()
            / reference_wgrad.norm().clamp_min(1e-12)
        )
        if not bool(torch.isfinite(error)) or float(error) > 0.55:
            raise SparseNVFP4Error(
                f"smoke GEMM normalized error is too large ({float(error):.4f})"
            )
        if (
            not bool(torch.isfinite(dgrad_error))
            or not bool(torch.isfinite(wgrad_error))
            or float(dgrad_error) > 0.02
            or float(wgrad_error) > 0.02
        ):
            raise SparseNVFP4Error(
                "BF16 backward normalized error is too large: "
                f"dgrad={float(dgrad_error):.4f}, wgrad={float(wgrad_error):.4f}"
            )
        runtime_version = _device_handle(device_index).version
        _STATUS_CACHE[device_index] = (
            True,
            f"SM120 compressed paired-4:8 E2M1 via cuSPARSELt {package_version} "
            f"(runtime {runtime_version}); native sparse NVFP4 forward + BF16 "
            f"backward passed (forward/dgrad/wgrad relative error "
            f"{float(error):.4f}/{float(dgrad_error):.4f}/{float(wgrad_error):.4f})",
        )
    except Exception as exc:
        _STATUS_CACHE[device_index] = (
            False,
            f"cuSPARSELt sparse NVFP4 smoke failed: {exc}",
        )
    return _STATUS_CACHE[device_index]


def sparse_nvfp4_mxfp8_status(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[bool, str]:
    """Validate the exact compressed-NVFP4-forward/MXFP8-backward path."""

    device_index, unavailable = _resolved_cuda_index(device)
    if device_index is None:
        return False, unavailable
    resolved = torch.device("cuda", device_index)
    if force_refresh:
        _MXFP8_STATUS_CACHE.pop(device_index, None)
    if device_index in _MXFP8_STATUS_CACHE:
        return _MXFP8_STATUS_CACHE[device_index]
    try:
        from .mxfp8 import native_mxfp8_status

        mxfp8_available, mxfp8_note = native_mxfp8_status(
            resolved,
            force_refresh=force_refresh,
        )
        if not mxfp8_available:
            raise SparseNVFP4Error(mxfp8_note)
        sparse_available, sparse_note = sparse_nvfp4_status(
            resolved,
            force_refresh=force_refresh,
        )
        if not sparse_available:
            raise SparseNVFP4Error(sparse_note)

        generator = torch.Generator(device="cpu").manual_seed(8120)
        input = (
            torch.randn((19, 47), generator=generator, dtype=torch.float32)
            .mul(0.25)
            .to(device=resolved, dtype=torch.bfloat16)
            .requires_grad_(True)
        )
        weight = (
            torch.randn((35, 47), generator=generator, dtype=torch.float32)
            .mul(0.25)
            .to(device=resolved, dtype=torch.bfloat16)
            .requires_grad_(True)
        )
        grad_output = (
            torch.randn((19, 35), generator=generator, dtype=torch.float32)
            .mul(0.25)
            .to(device=resolved, dtype=torch.bfloat16)
        )
        sparse_weight, sparse_mask = _effective_weight_and_mask(weight.detach(), 64)
        output = sparse_nvfp4_linear(
            input,
            weight,
            backward_precision="mxfp8",
            backward_backend="native",
        )
        grad_input, grad_weight = torch.autograd.grad(
            output,
            (input, weight),
            grad_outputs=grad_output,
        )
        torch.cuda.synchronize(input.device)

        reference = input.detach().float().mm(sparse_weight.float().t())
        reference_dgrad = grad_output.float().mm(sparse_weight.float())
        reference_wgrad = grad_output.float().t().mm(input.detach().float())
        reference_wgrad.mul_(sparse_mask)
        forward_error = (
            (output.detach().float() - reference).norm()
            / reference.norm().clamp_min(1e-12)
        )
        dgrad_error = (
            (grad_input.float() - reference_dgrad).norm()
            / reference_dgrad.norm().clamp_min(1e-12)
        )
        wgrad_error = (
            (grad_weight.float() - reference_wgrad).norm()
            / reference_wgrad.norm().clamp_min(1e-12)
        )
        if (
            not bool(torch.isfinite(forward_error))
            or not bool(torch.isfinite(dgrad_error))
            or not bool(torch.isfinite(wgrad_error))
            or float(forward_error) > 0.55
            or float(dgrad_error) > 0.15
            or float(wgrad_error) > 0.15
        ):
            raise SparseNVFP4Error(
                "combined autograd relative error is too large: "
                f"forward={float(forward_error):.4f}, "
                f"dgrad={float(dgrad_error):.4f}, wgrad={float(wgrad_error):.4f}"
            )
        _MXFP8_STATUS_CACHE[device_index] = (
            True,
            "native compressed sparse NVFP4 forward + MXFP8 dgrad/wgrad passed "
            "(forward/dgrad/wgrad relative error "
            f"{float(forward_error):.4f}/{float(dgrad_error):.4f}/"
            f"{float(wgrad_error):.4f})",
        )
    except Exception as exc:
        _MXFP8_STATUS_CACHE[device_index] = (
            False,
            f"combined sparse NVFP4 + MXFP8 autograd smoke failed: {exc}",
        )
    return _MXFP8_STATUS_CACHE[device_index]


def sparse_nvfp4_capability_status(
    device: str | torch.device | None = None,
) -> tuple[bool, str]:
    """Cheap server-side preflight; workers still run the real GEMM smoke."""

    try:
        if platform.system() != "Windows":
            raise SparseNVFP4Error("the current ctypes integration targets Windows")
        device_index, unavailable = _resolved_cuda_index(device)
        if device_index is None:
            raise SparseNVFP4Error(unavailable)
        capability = torch.cuda.get_device_capability(device_index)
        if capability != (12, 0):
            raise SparseNVFP4Error(
                f"the validated backend requires SM120, found SM{capability[0]}{capability[1]}"
            )
        root, package_version = _distribution_root()
        dll_path = root / "bin" / "x64" / "cusparseLt.dll"
        if not dll_path.is_file():
            raise SparseNVFP4Error(f"cuSPARSELt DLL is missing: {dll_path}")
        return (
            True,
            f"SM120 and cuSPARSELt {package_version} are configured; "
            "the training worker requires a real compressed-GEMM smoke",
        )
    except Exception as exc:
        return False, str(exc)


__all__ = [
    "SparseNVFP4Error",
    "clear_sparse_nvfp4_cache",
    "paired_4to8",
    "sparse_nvfp4_forward",
    "sparse_nvfp4_linear",
    "sparse_nvfp4_mxfp8_status",
    "sparse_nvfp4_capability_status",
    "sparse_nvfp4_status",
]
