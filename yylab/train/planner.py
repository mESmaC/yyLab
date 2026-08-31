from __future__ import annotations

from dataclasses import dataclass, field

QUANT_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.1, "nf4": 0.6}
OPTIMIZER_BYTES_PER_PARAM = 16.0
ADAPTER_TRAINING_BYTES_PER_PARAM = 6.0
JOINT_ADAPTER_TRAINING_BYTES_PER_PARAM = 8.0
TOKEN_ROUTER_TRAINING_BYTES_PER_PARAM = 16.0
PAGED_ADAPTER_TRAINING_BYTES_PER_PARAM = 6.0
PAGED_TOKEN_ROUTER_TRAINING_BYTES_PER_PARAM = 10.0
OFFLOADED_ADAPTER_TRAINING_BYTES_PER_PARAM = 4.0
OFFLOADED_TOKEN_ROUTER_TRAINING_BYTES_PER_PARAM = 8.0
PAGED_OPTIMIZER_MIN_PARAMS = 50_000_000
SAFETY = 0.88
TARGET_EFFECTIVE_BATCH = 16
FAST_ADAPTER_BATCHES = (16, 8, 4, 2, 1)
STANDARD_ADAPTER_BATCHES = (2, 1)


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
    hidden_size: int = 0
    intermediate_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    model_type: str = ""
    adapter_only: bool = False
    isolated_expert_training: bool = False
    train_token_routers: bool = False
    optimizer_state: str = "gpu"
    fused_linear_ce: bool = False
    full_liger_kernels: bool = False
    prefer_cudnn_sdpa: bool = False
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


def _lora_trainable_b(inp: PlanInput, active_experts: int) -> float:
    hidden = inp.hidden_size
    layers = inp.num_layers
    if hidden > 0 and layers > 0:
        intermediate = inp.intermediate_size or hidden * 4
        head_dim = inp.head_dim
        if head_dim <= 0 and inp.num_attention_heads > 0:
            head_dim = hidden // inp.num_attention_heads
        query_hidden = hidden
        if head_dim > 0 and inp.num_attention_heads > 0:
            query_hidden = head_dim * inp.num_attention_heads
        kv_hidden = query_hidden
        if head_dim > 0 and inp.num_kv_heads > 0:
            kv_hidden = head_dim * inp.num_kv_heads
        # q, k, v, o, gate, up, down projection (in + out) widths.
        per_layer_widths = (
            (hidden + query_hidden)
            + 2 * (hidden + kv_hidden)
            + (query_hidden + hidden)
            + 2 * (hidden + intermediate)
            + (intermediate + hidden)
        )
        return layers * inp.lora_rank * per_layer_widths * active_experts / 1e9
    # Architecture-free fallback calibrated to seven-projection decoder LoRA.
    return inp.params_b * 0.00105 * inp.lora_rank * active_experts


def _token_router_trainable_b(inp: PlanInput) -> float:
    if not inp.train_token_routers:
        return 0.0
    if inp.hidden_size > 0 and inp.num_layers > 0:
        # One E-way token gate per transformer block, shared by every adapted
        # projection in that block.
        return inp.num_layers * inp.hidden_size * inp.num_experts / 1e9
    return inp.params_b * 0.000048 * inp.num_experts


def estimate_adapter_trainable_params(inp: PlanInput, active_experts: int | None = None) -> int:
    """Estimate LoRA parameters from decoder projection shapes when available."""

    active = (
        (1 if inp.isolated_expert_training else inp.num_experts) if active_experts is None else active_experts
    )
    return round((_lora_trainable_b(inp, active) + _token_router_trainable_b(inp)) * 1e9)


def _fast_qwen_profile(inp: PlanInput) -> bool:
    heads = inp.num_attention_heads
    kv_heads = inp.num_kv_heads
    head_dim = inp.head_dim or (inp.hidden_size // heads if heads > 0 else 0)
    return (
        inp.model_type.lower() == "qwen3"
        and inp.fused_linear_ce
        and inp.full_liger_kernels
        and inp.prefer_cudnn_sdpa
        and _compute_capability(inp.sm_version) >= (8, 0)
        and heads > kv_heads > 0
        and heads % kv_heads == 0
        and 0 < head_dim <= 256
        and head_dim % 8 == 0
    )


def _activation_gb(inp: PlanInput, micro_batch: int, grad_ckpt: bool) -> float:
    # Includes transformer/adaptor intermediates. Qwen's very wide vocabulary
    # gets a separate allowance when fused linear cross entropy is unavailable.
    if _fast_qwen_profile(inp) and grad_ckpt:
        # Conservative fit over the measured Qwen3 cuDNN + full-Liger path.
        # On the local 0.6B model at sequence 2048, batch 4 reserved 2.38 GB
        # total; this model assigns ~3.6 GB after base/adapter allowances.
        per_sequence = 0.35 + 0.45 * inp.params_b
    else:
        per_sequence = 1.8 + inp.params_b
    est = micro_batch * (inp.seq_len / 2048.0) * per_sequence
    if not grad_ckpt:
        est *= 2.4
    if not inp.fused_linear_ce and inp.vocab_size:
        est += micro_batch * inp.seq_len * inp.vocab_size * 4 / 2**30
    return est


def _compute_capability(value: str) -> tuple[int, int]:
    try:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except (AttributeError, TypeError, ValueError):
        return (0, 0)


def _fits(vram_gb: float, peak: float) -> bool:
    return peak <= vram_gb * SAFETY


def plan_training(inp: PlanInput) -> TrainPlan:
    notes: list[str] = []
    p = inp.params_b
    seq = inp.seq_len

    def base_gb(quant: str) -> float:
        return p * QUANT_BYTES.get(quant, 2.0) * 1.12

    def adapter_gb() -> float:
        active = 1 if inp.isolated_expert_training else inp.num_experts
        lora_b = _lora_trainable_b(inp, active)
        router_b = _token_router_trainable_b(inp)
        optimizer_mode = inp.optimizer_state
        if optimizer_mode == "auto":
            optimizer_mode = "cpu_paged" if (lora_b + router_b) * 1e9 > PAGED_OPTIMIZER_MIN_PARAMS else "gpu"
        if optimizer_mode == "cpu_offload":
            lora_bytes = OFFLOADED_ADAPTER_TRAINING_BYTES_PER_PARAM
            router_bytes = OFFLOADED_TOKEN_ROUTER_TRAINING_BYTES_PER_PARAM
        elif optimizer_mode == "cpu_paged":
            lora_bytes = PAGED_ADAPTER_TRAINING_BYTES_PER_PARAM
            router_bytes = PAGED_TOKEN_ROUTER_TRAINING_BYTES_PER_PARAM
        else:
            lora_bytes = (
                JOINT_ADAPTER_TRAINING_BYTES_PER_PARAM
                if inp.train_token_routers
                else ADAPTER_TRAINING_BYTES_PER_PARAM
            )
            router_bytes = TOKEN_ROUTER_TRAINING_BYTES_PER_PARAM
        return lora_b * lora_bytes + router_b * router_bytes

    def build(method: str, quant: str, grad_ckpt: bool, offload: bool, micro_batch: int) -> TrainPlan:
        act = _activation_gb(inp, micro_batch, grad_ckpt)
        if method == "full":
            peak = base_gb("bf16") + p * OPTIMIZER_BYTES_PER_PARAM + act
        else:
            peak = base_gb(quant) + adapter_gb() + act
        if offload:
            peak *= 0.85
        # Compile is deliberately not advertised until the worker owns a
        # validated compile path. Capability parsing remains numeric for notes.
        use_compile = False
        return TrainPlan(
            method=method,
            quant=quant if method != "full" else "bf16",
            micro_batch=micro_batch,
            # Auto batch candidates are divisors of the target, so increasing
            # the resident microbatch reduces accumulation rather than changing
            # the optimization batch or epoch step count.
            grad_accum=max(1, TARGET_EFFECTIVE_BATCH // micro_batch),
            seq_len=seq,
            grad_ckpt=grad_ckpt,
            offload=offload or inp.optimizer_state == "cpu_offload",
            use_compile=use_compile,
            est_peak_gb=round(peak, 1),
            feasible=_fits(inp.vram_gb, peak),
            notes=list(notes),
        )

    # The fused Qwen path used to stop at microbatch 4 even when its memory
    # projection showed several GB idle. Try the largest optimization-equivalent
    # power-of-two batch first and let the peak estimator plus safety reserve
    # select the first candidate that fits.
    adapter_batches = FAST_ADAPTER_BATCHES if _fast_qwen_profile(inp) else STANDARD_ADAPTER_BATCHES
    candidates: list[tuple[str, str, bool, bool, int]] = []
    if not inp.adapter_only and inp.method in {"auto", "full"}:
        candidates += [("full", "bf16", True, False, 1), ("full", "bf16", False, False, 1)]
    if inp.method in {"auto", "lora"}:
        candidates += [("lora", "bf16", True, False, mb) for mb in adapter_batches]
    if inp.method in {"auto", "qlora"}:
        for mb in adapter_batches:
            candidates.append(("qlora", inp.quant, True, False, mb))

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
        plan.notes.append("Model is too large for the implemented training path at this sequence length.")
        if seq > 512:
            plan.notes.append("Try reducing sequence length.")
    if plan.method == "full" and p > 3.0:
        plan.notes.append("Full fine-tune of a large base is rarely practical on consumer GPUs.")
    if plan.use_compile:
        plan.notes.append("torch.compile enabled (triton detected).")
    if inp.isolated_expert_training and inp.num_experts > 1:
        plan.notes.append(
            "Experts train sequentially; optimizer and activation memory stay O(1) in expert count."
        )
    if inp.train_token_routers:
        plan.notes.append(
            "All experts train jointly; one learned top-K token router is shared within each transformer block."
        )
        estimated_trainable = estimate_adapter_trainable_params(inp)
        if inp.optimizer_state == "cpu_offload":
            cpu_state_gb = estimated_trainable * 12 / 2**30
            plan.notes.append(
                f"Full optimizer offload keeps about {cpu_state_gb:.2f} GB of FP32 master/moment state in RAM."
            )
        elif inp.optimizer_state == "cpu_paged" or (
            inp.optimizer_state == "auto" and estimated_trainable > PAGED_OPTIMIZER_MIN_PARAMS
        ):
            plan.notes.append("Paged 8-bit optimizer state can spill from VRAM into system RAM.")
    if inp.fused_linear_ce and inp.vocab_size:
        plan.notes.append(
            "Fused linear cross entropy avoids the full sequence × vocabulary logits allocation."
        )
    if _fast_qwen_profile(inp):
        plan.notes.append(
            "Qwen3 fast path uses full Liger kernels and cuDNN-priority SDPA with safe fallbacks."
        )
        plan.notes.append(
            f"Microbatch {plan.micro_batch} is the largest projected fit with a {round((1 - SAFETY) * 100)}% "
            f"free-VRAM reserve; accumulation keeps {plan.micro_batch * plan.grad_accum} sequences per update."
        )
    if _compute_capability(inp.sm_version) >= (8, 9) and plan.quant == "nf4":
        plan.notes.append(
            "This GPU supports fast low-precision compute; NF4 remains the implemented base format."
        )
    return plan
