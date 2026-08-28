from __future__ import annotations

from dataclasses import dataclass, field

QUANT_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.1, "nf4": 0.6}
OPTIMIZER_BYTES_PER_PARAM = 16.0
SAFETY = 0.88


@dataclass
class PlanInput:
    params_b: float
    vram_gb: float
    ram_gb: float = 32.0
    seq_len: int = 2048
    method: str = "auto"
    quant: str = "nf4"
    lora_rank: int = 16
    num_experts: int = 4
    sm_version: str = ""
    has_triton: bool = False


@dataclass
class TrainPlan:
    method: str
    quant: str
    micro_batch: int
    grad_accum: int
    seq_len: int
    grad_ckpt: bool
    offload: bool
    use_compile: bool
    est_peak_gb: float
    feasible: bool
    notes: list[str] = field(default_factory=list)


def _lora_trainable_b(params_b: float, rank: int, num_experts: int, num_targets: int = 7) -> float:
    hidden = (params_b * 12) ** 0.5 * 320
    per_matrix = 2 * rank * hidden * num_experts
    return per_matrix * num_targets / 1e9


def _activation_gb(micro_batch: int, seq_len: int, params_b: float, grad_ckpt: bool) -> float:
    est = micro_batch * (seq_len / 2048.0) * (0.3 + 0.12 * params_b)
    return est * (0.35 if grad_ckpt else 1.0)


def _fits(vram_gb: float, peak: float) -> bool:
    return peak <= vram_gb * SAFETY


def plan_training(inp: PlanInput) -> TrainPlan:
    notes: list[str] = []
    p = inp.params_b
    seq = inp.seq_len

    def base_gb(quant: str) -> float:
        return p * QUANT_BYTES.get(quant, 2.0) * 1.12

    def adapter_gb() -> float:
        return _lora_trainable_b(p, inp.lora_rank, inp.num_experts) * OPTIMIZER_BYTES_PER_PARAM

    def build(method: str, quant: str, grad_ckpt: bool, offload: bool, micro_batch: int) -> TrainPlan:
        act = _activation_gb(micro_batch, seq, p, grad_ckpt)
        if method == "full":
            peak = base_gb("bf16") + p * OPTIMIZER_BYTES_PER_PARAM + act
        else:
            peak = base_gb(quant) + adapter_gb() + act
        if offload:
            peak *= 0.85
        use_compile = inp.has_triton and inp.sm_version >= "8.0" and method != "full"
        return TrainPlan(
            method=method,
            quant=quant if method != "full" else "bf16",
            micro_batch=micro_batch,
            grad_accum=max(1, 16 // micro_batch),
            seq_len=seq,
            grad_ckpt=grad_ckpt,
            offload=offload,
            use_compile=use_compile,
            est_peak_gb=round(peak, 1),
            feasible=_fits(inp.vram_gb, peak),
            notes=list(notes),
        )

    candidates: list[tuple[str, str, bool, bool, int]] = []
    if inp.method in {"auto", "full"}:
        candidates += [("full", "bf16", True, False, 1), ("full", "bf16", False, False, 1)]
    if inp.method in {"auto", "lora"}:
        candidates += [("lora", "bf16", True, False, 2), ("lora", "bf16", True, False, 1)]
    if inp.method in {"auto", "qlora"}:
        for mb in (2, 1):
            candidates += [
                ("qlora", inp.quant, True, False, mb),
                ("qlora", inp.quant, True, True, mb),
            ]

    best: TrainPlan | None = None
    for method, quant, gc, off, mb in candidates:
        plan = build(method, quant, gc, off, mb)
        if plan.feasible:
            best = plan
            break
        if best is None or plan.est_peak_gb < best.est_peak_gb:
            best = plan

    plan = best or build("qlora", inp.quant, True, True, 1)
    if not plan.feasible:
        if inp.ram_gb >= base_gb("nf4") * 2:
            plan.offload = True
            plan.notes.append("Model exceeds VRAM: CPU offload suggested (slow).")
        else:
            plan.notes.append("Model is too large for this system at the requested sequence length.")
        if seq > 512:
            plan.notes.append("Try reducing sequence length.")
    if plan.method == "full" and p > 3.0:
        plan.notes.append("Full fine-tune of a large base is rarely practical on consumer GPUs.")
    if plan.use_compile:
        plan.notes.append("torch.compile enabled (triton detected).")
    if inp.sm_version and inp.sm_version >= "8.9" and plan.quant == "nf4":
        plan.notes.append("FP8 may be faster than NF4 on this GPU (Ada/Blackwell).")
    return plan
