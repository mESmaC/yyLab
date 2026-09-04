from __future__ import annotations

from collections.abc import Mapping


def reject_unverified_checkpoint_quantization(base_cfg) -> None:
    """Reject checkpoint-owned quantizers that bypass yyLab capability probes.

    Native yyLab QAT is recorded separately in ``yylab_low_precision``. Any
    Transformers ``quantization_config`` would otherwise be activated by
    ``from_pretrained`` without going through yyLab's exact operator checks;
    that includes BitNet's software STE/dense-linear integration.
    """

    raw = getattr(base_cfg, "quantization_config", None)
    if raw is None:
        return
    if isinstance(raw, Mapping):
        values = dict(raw)
    else:
        converter = getattr(raw, "to_dict", None)
        values = dict(converter()) if callable(converter) else {}
    method = str(values.get("quant_method") or type(raw).__name__).strip()
    raise ValueError(
        "checkpoint-embedded quantization is not executable through yyLab's "
        f"native capability contract ({method or 'unknown'}); load an unquantized "
        "checkpoint and choose one of the verified runtime precision options"
    )


def trained_context_length(base_cfg) -> int:
    """Return the checkpoint's trained RoPE length rather than a padded runtime limit."""

    native_context = int(getattr(base_cfg, "max_position_embeddings", 0) or 0)
    if getattr(base_cfg, "model_type", "") == "qwen3" and native_context >= 32_768:
        # Qwen3 reserves an additional 8,192 runtime positions around its
        # 32,768-token trained sequence. YaRN's original length must remain the
        # trained length or the scale factor is applied to the wrong base.
        return 32_768
    return native_context


def _merged_rope_parameters(base_cfg, rope_scaling: dict) -> dict:
    merged = dict(getattr(base_cfg, "rope_scaling", None) or {})
    merged.update(rope_scaling)
    if merged.get("rope_type") != "yarn" or merged.get("original_max_position_embeddings"):
        return merged
    original_context = trained_context_length(base_cfg)
    if original_context:
        merged["original_max_position_embeddings"] = original_context
    return merged


def scan_checkpoint(path) -> dict:
    import json
    import math
    from pathlib import Path

    from safetensors import safe_open

    path = Path(path)
    from .spalmer import scan_spalmer_checkpoint

    if (path.is_file() and path.suffix.casefold() == ".pt") or (
        path.is_dir() and (path / "spalmer.pt").is_file()
    ):
        return scan_spalmer_checkpoint(path)
    cfg_file = path / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError("config.json not found — not a HuggingFace checkpoint folder")
    cfg = json.loads(cfg_file.read_text("utf-8"))
    arch = (cfg.get("architectures") or ["unknown"])[0]
    params = 0
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as sf:
            for key in sf.keys():
                shape = sf.get_slice(key).get_shape()
                params += math.prod(shape) if shape else 0
    return {
        "arch": arch,
        "params_b": round(params / 1e9, 3),
        "hidden_size": cfg.get("hidden_size", 0) or cfg.get("n_embd", 0),
        "num_layers": cfg.get("num_hidden_layers", 0) or cfg.get("n_layer", 0),
        "base_cfg": cfg,
    }


def load_base(
    ref: str,
    quant: str,
    attn_implementation: str = "sdpa",
    rope_scaling: dict | None = None,
    max_position_embeddings: int | None = None,
    restore_saved_qat: bool = True,
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from ..train.low_precision import apply_saved_qat, preflight_saved_qat
    from .bitsandbytes_capabilities import bitsandbytes_compute_precision
    from .families import SPALMER_MODEL_FAMILY, family_from_reference
    from .precision_capabilities import require_model_precision

    notes: list[str] = []
    cuda = torch.cuda.is_available()
    quant = str(quant).strip().casefold()
    if family_from_reference(ref) == SPALMER_MODEL_FAMILY:
        raise ValueError(
            "SPALMER checkpoints are not Hugging Face base models; use the native "
            "SPALMER training or inference path"
        )
    device = "cuda:0" if cuda else "cpu"
    require_model_precision(quant, device)

    tokenizer = AutoTokenizer.from_pretrained(ref)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_cfg = AutoConfig.from_pretrained(ref)
    reject_unverified_checkpoint_quantization(base_cfg)
    saved_qat = getattr(base_cfg, "yylab_low_precision", None)
    if saved_qat and quant in {"nf4", "int8"}:
        raise ValueError(
            "A QAT-trained base cannot also be loaded through bitsandbytes NF4/INT8. "
            "Load its saved BF16/FP16/FP32 shadow checkpoint instead."
        )
    if saved_qat and restore_saved_qat:
        if quant != "bf16":
            raise ValueError(
                "Restoring native low-precision checkpoint numerics requires BF16 loading"
            )
        preflight_saved_qat(dict(saved_qat), device)

    kwargs: dict = {
        # Reuse the exact config object inspected above. Letting
        # ``from_pretrained`` resolve the reference a second time would give a
        # checkpoint-embedded quantizer another path around the fail-closed
        # capability boundary.
        "config": base_cfg,
        "attn_implementation": attn_implementation,
        # Stream checkpoint shards directly into their destination tensors.
        # This avoids a second full-sized CPU state dict during worker startup.
        "low_cpu_mem_usage": True,
    }
    if rope_scaling:
        # Qwen3's default 40,960 runtime allocation is 8,192 prompt
        # positions plus its native 32,768-token trained sequence.
        merged_rope = _merged_rope_parameters(base_cfg, rope_scaling)
        # A supplied config makes Transformers pass unknown kwargs to the
        # model constructor instead of applying them as config overrides.
        # Mutate the already-validated copy explicitly so there is still only
        # one config resolution between validation and model construction.
        base_cfg.rope_scaling = merged_rope
        if max_position_embeddings is not None:
            target_context = int(max_position_embeddings)
            if target_context < 1:
                raise ValueError("max_position_embeddings must be positive")
            base_cfg.max_position_embeddings = target_context
            notes.append(
                f"{str(merged_rope.get('rope_type') or 'RoPE').upper()} context extension: "
                f"{target_context:,} positions."
            )
    if cuda and quant == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["dtype"] = torch.bfloat16
    elif cuda and quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        compute_precision = bitsandbytes_compute_precision(quant)
        if compute_precision != "fp16":
            raise RuntimeError(
                "the verified bitsandbytes INT8 provider no longer declares FP16 compute"
            )
        kwargs["dtype"] = torch.float16
    else:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        kwargs["dtype"] = dtype_map[quant]
    if cuda:
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(ref, **kwargs)
    if restore_saved_qat:
        qat_count, qat_meta = apply_saved_qat(model)
    else:
        qat_count, qat_meta = 0, None
        if saved_qat is not None:
            # A saved QAT checkpoint contains ordinary high-precision shadow
            # weights. Continued training that explicitly requests dense
            # numerics must not silently re-enable the inherited forward mask.
            # Remove the metadata as well, otherwise save_pretrained() would
            # publish a dense-trained checkpoint that reactivates QAT at load.
            if hasattr(model.config, "yylab_low_precision"):
                delattr(model.config, "yylab_low_precision")
            notes.append(
                "Loaded high-precision shadow weights without restoring inherited QAT numerics."
            )
    if qat_meta is not None:
        if not qat_meta.get("native_low_bit_gemm"):
            raise RuntimeError("saved low-precision recipe is not backed by a native GEMM")
        execution = "native low-bit forward"
        sparsity = str(qat_meta.get("sparsity") or "none")
        if qat_meta.get("native_sparse_gemm"):
            sparsity_note = f"; {sparsity} compressed sparse forward"
        elif sparsity != "none":
            raise RuntimeError("saved sparse precision recipe is not backed by a native compressed GEMM")
        else:
            sparsity_note = ""
        backward = str(qat_meta.get("backward_precision") or "bf16")
        if backward == "mxfp8" and not qat_meta.get("native_low_bit_backward"):
            raise RuntimeError("saved MXFP8 backward recipe is not backed by a native GEMM")
        backward_note = (
            "; native MXFP8 backward"
            if qat_meta.get("native_low_bit_backward")
            else "; BF16 backward"
        )
        notes.append(
            f"restored {qat_meta.get('format')} QAT numerics on {qat_count} Linear modules "
            f"({execution}{sparsity_note}{backward_note}; high-precision shadow checkpoint)"
        )
    model.config.use_cache = True
    return model, tokenizer, cuda, quant, notes
