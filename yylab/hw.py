from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import asdict, dataclass


@dataclass
class GpuInfo:
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: str
    driver: str


def detect_gpus() -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                try:
                    cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                    cc_str = f"{cc[0]}.{cc[1]}"
                except Exception:
                    cc_str = "?"
                gpus.append(
                    GpuInfo(
                        index=i,
                        name=pynvml.nvmlDeviceGetName(h),
                        vram_total_gb=round(mem.total / 2**30, 2),
                        vram_free_gb=round(mem.free / 2**30, 2),
                        compute_capability=cc_str,
                        driver=pynvml.nvmlSystemGetDriverVersion(),
                    )
                )
        finally:
            pynvml.nvmlShutdown()
        return gpus
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                cc = torch.cuda.get_device_capability(i)
                gpus.append(
                    GpuInfo(
                        index=i,
                        name=torch.cuda.get_device_name(i),
                        vram_total_gb=round(total / 2**30, 2),
                        vram_free_gb=round(free / 2**30, 2),
                        compute_capability=f"{cc[0]}.{cc[1]}",
                        driver="",
                    )
                )
    except Exception:
        pass
    return gpus


def best_gpu() -> GpuInfo | None:
    gpus = detect_gpus()
    if not gpus:
        return None
    return max(gpus, key=lambda g: g.vram_free_gb)


def cpu_name() -> str:
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        winreg.CloseKey(key)
        return str(name).strip()
    except Exception:
        return platform.processor() or "unknown"


def ram_total_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        return 0.0


def has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def system_report() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_name(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": ram_total_gb(),
        "gpus": [asdict(g) for g in detect_gpus()],
        "triton": has_triton(),
    }
