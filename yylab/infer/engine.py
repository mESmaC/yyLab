from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace


@dataclass
class LoadSpec:
    ref: str
    quant: str
    model_family: str = ""
    adapter_path: str | None = None
    adapter_arch: str = ""
    router_path: str | None = None
    settings: dict = field(default_factory=dict)

    def key(self) -> str:
        return "|".join(
            [
                self.ref,
                self.quant,
                self.model_family,
                self.adapter_path or "",
                self.router_path or "",
                json.dumps(self.settings, sort_keys=True),
            ]
        )


@dataclass(frozen=True, slots=True)
class ChatContextPlan:
    """Exact prompt-budget plan for one chat generation."""

    messages: tuple[dict, ...]
    prompt_tokens: int
    prompt_budget: int
    context_limit: int
    answer_budget: int
    prompt_mode: str = "chat"
    dropped_messages: int = 0
    dropped_turns: int = 0
    strategy: str = "recency"
    selection_budget: int = 0
    evaluated_turns: int = 0
    retrieved_turns: int = 0
    recent_turns: int = 0
    atxy_candidates: int = 0
    baseline_nll: float | None = None
    final_nll: float | None = None
    nll_gain_nats: float | None = None
    fallback_reason: str = ""

    @property
    def compacted(self) -> bool:
        return self.dropped_messages > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_budget": self.prompt_budget,
            "context_limit": self.context_limit,
            "answer_budget": self.answer_budget,
            "prompt_mode": self.prompt_mode,
            "dropped_messages": self.dropped_messages,
            "dropped_turns": self.dropped_turns,
            "compacted": self.compacted,
            "strategy": self.strategy,
            "selection_budget": self.selection_budget or self.prompt_budget,
            "evaluated_turns": self.evaluated_turns,
            "retrieved_turns": self.retrieved_turns,
            "recent_turns": self.recent_turns,
            "atxy_candidates": self.atxy_candidates,
            "baseline_nll": self.baseline_nll,
            "final_nll": self.final_nll,
            "nll_gain_nats": self.nll_gain_nats,
            "fallback_reason": self.fallback_reason,
            "position_policy": "contiguous_rebase" if self.compacted else "native_order",
        }


def format_messages(tokenizer, messages: list[dict], *, prompt_mode: str = "chat") -> str:
    """Render one generation prompt in chat-template or raw-completion mode.

    Raw completion deliberately accepts exactly one payload. Treating a list
    of prior user/assistant messages as plain text would invent separators and
    multi-turn semantics that a base pretrained checkpoint has not learned.
    """

    if prompt_mode == "raw":
        if len(messages) != 1:
            raise ValueError("Raw completion requires exactly one independent prompt.")
        content = messages[0].get("content", "")
        if not isinstance(content, str):
            raise TypeError("Raw completion prompt content must be text.")
        return content
    if prompt_mode != "chat":
        raise ValueError(f"unknown prompt mode: {prompt_mode}")
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        parts = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            parts.append(f"### {role}\n{m.get('content', '')}")
        parts.append("### Assistant\n")
        return "\n\n".join(parts)


def _rope_scaling(settings: dict) -> dict | None:
    kind = settings.get("rope_scaling", "none")
    factor = float(settings.get("rope_factor", 2.0) or 2.0)
    if kind == "linear":
        return {"rope_type": "linear", "factor": factor}
    if kind in {"yarn", "dynamic"}:
        return {"rope_type": "yarn", "factor": factor}
    return None


def _encoded_length(encoded) -> int:
    input_ids = encoded["input_ids"]
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if input_ids and isinstance(input_ids[0], list | tuple):
        return len(input_ids[0])
    return len(input_ids)


def _conversation_groups(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """Split protected instructions and user-led turns.

    System/developer messages are never eligible for retrieval eviction, even
    when a malformed/imported transcript places one after ordinary turns.
    """

    protected = [
        dict(message)
        for message in messages
        if str(message.get("role") or "") in {"system", "developer"}
    ]
    ordinary = [
        message
        for message in messages
        if str(message.get("role") or "") not in {"system", "developer"}
    ]

    groups: list[list[dict]] = []
    for raw in ordinary:
        message = dict(raw)
        if str(message.get("role") or "") == "user" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return protected, groups


def _context_mode_settings(ref: str, settings: dict) -> tuple[dict, int | None]:
    """Resolve chat-native and YaRN presets before loading model weights."""

    mode = str(settings.get("context_mode") or "")
    if mode not in {"native", "yarn_2x", "yarn_4x"}:
        return settings, None

    from transformers import AutoConfig

    from ..models.loading import trained_context_length

    base_cfg = AutoConfig.from_pretrained(ref)
    native_context = int(getattr(base_cfg, "max_position_embeddings", 0) or 0)
    if native_context < 1:
        raise ValueError("checkpoint config does not declare max_position_embeddings")

    resolved = dict(settings)
    if mode == "native":
        resolved.update(
            rope_scaling="none",
            rope_factor=1.0,
            max_context_tokens=native_context,
        )
        return resolved, native_context

    factor = 2.0 if mode == "yarn_2x" else 4.0
    trained_context = trained_context_length(base_cfg)
    if trained_context < 1:
        raise ValueError("checkpoint config does not declare a trained RoPE context length")
    target_context = int(round(trained_context * factor))
    resolved.update(
        rope_scaling="yarn",
        rope_factor=factor,
        max_context_tokens=target_context,
    )
    return resolved, target_context


def _cache_kwargs(settings: dict, cuda: bool) -> tuple[dict, list[str]]:
    notes: list[str] = []
    impl = settings.get("cache_implementation", "dynamic")
    max_ctx = int(settings.get("max_context_tokens", 8192) or 8192)
    chunk = int(settings.get("prefill_chunk_size", 8192) or 8192)
    if impl == "dynamic" or not cuda:
        if impl != "dynamic" and not cuda:
            notes.append("CPU inference: using dynamic cache.")
        return {}, notes
    if impl == "offloaded":
        notes.append("KV cache offloaded to CPU RAM (dynamic).")
        return {"cache_implementation": "offloaded"}, notes
    if impl == "offloaded_static":
        notes.append("KV cache offloaded to CPU RAM (static, chunked prefill).")
        return {
            "cache_implementation": "offloaded_static",
            "max_cache_len": max_ctx,
            "prefill_chunk_size": chunk,
        }, notes
    if impl == "quantized_offloaded":
        kvq = str(settings.get("kv_quant", "int4"))
        if kvq not in {"int4", "int8"}:
            raise ValueError("quantized KV offload requires int4 or int8 KV quantization")
        notes.append(f"KV history uses packed {kvq} CPU offload with active-layer dequantization.")
        return {}, notes
    if impl == "static":
        return {
            "cache_implementation": "static",
            "max_cache_len": max_ctx,
            "prefill_chunk_size": chunk,
        }, notes
    if impl == "quantized":
        kvq = str(settings.get("kv_quant", "int4"))
        if kvq != "int4":
            raise ValueError(
                "the installed Transformers Quanto resident cache supports int4, not int8; "
                "select int4 or use quantized KV offload for int4/int8 CPU residency"
            )
        notes.append("KV cache quantized to int4 in VRAM (Optimum Quanto).")
        return {
            "cache_implementation": "quantized",
            "cache_config": {
                "backend": "quanto",
                "nbits": 4,
                "q_group_size": 64,
                "residual_length": 128,
            },
        }, notes
    if impl == "sliding_window":
        return {
            "cache_implementation": "sliding_window",
            "cache_config": {"window_size": int(settings.get("sliding_window", 4096))},
        }, notes
    return {}, notes


def _adaptive_kv_reserve_bytes(
    model_config,
    settings: dict,
    *,
    prompt_tokens: int,
    answer_tokens: int,
    cuda: bool,
) -> int:
    """Estimate request KV headroom before adaptive lanes claim free VRAM."""

    if not cuda:
        return 0
    from .context_math import kv_bytes_per_token

    implementation = str(settings.get("cache_implementation", "dynamic"))
    total_tokens = max(0, int(prompt_tokens)) + max(0, int(answer_tokens))
    if implementation == "quantized_offloaded":
        layers = max(1, int(getattr(model_config, "num_hidden_layers", 1) or 1))
        all_layer_bytes = kv_bytes_per_token(model_config, "bf16")
        active_layer_bytes = all_layer_bytes / layers * total_tokens
        residual_tokens = min(
            total_tokens,
            max(0, int(settings.get("kv_offload_residual_length", 32) or 0)),
        )
        resident_residual_bytes = all_layer_bytes * residual_tokens
        return int(math.ceil(active_layer_bytes + resident_residual_bytes))
    if implementation in {"offloaded", "offloaded_static"}:
        working_tokens = int(settings.get("adaptive_offloaded_kv_working_tokens", 256) or 256)
        reserve_tokens = min(total_tokens, max(1, working_tokens))
        kv_quant = "bf16"
    elif implementation == "sliding_window":
        reserve_tokens = min(
            total_tokens,
            max(1, int(settings.get("sliding_window", 4096) or 4096)),
        )
        kv_quant = "bf16"
    elif implementation == "quantized" and settings.get("kv_quant", "int8") != "none":
        reserve_tokens = total_tokens
        requested_quant = str(settings.get("kv_quant", "int8"))
        # Transformers currently resolves the UI's fp8 request to int8.
        kv_quant = "int8" if requested_quant == "fp8" else requested_quant
    else:
        reserve_tokens = total_tokens
        kv_quant = "bf16"
    return int(math.ceil(kv_bytes_per_token(model_config, kv_quant) * reserve_tokens))


class Engine:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.spec: LoadSpec | None = None
        self.notes: list[str] = []
        self.device: str = "cpu"
        self.cuda = False
        self._lock = threading.Lock()
        self._generation_lock = threading.RLock()
        self._cancel = threading.Event()
        self.adaptive_runtime = None

    def cancel(self) -> None:
        self._cancel.set()

    def nll_ids(self, prefix_ids, target_ids) -> float:
        import torch

        with self._generation_lock:
            if not self.is_loaded():
                raise RuntimeError("no model loaded")
            t = list(target_ids)
            if not t:
                return 0.0
            p = list(prefix_ids)
            dev = torch.device(self.device)
            input_ids = torch.tensor([p + t], dtype=torch.long, device=dev)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor([[-100] * len(p) + t], dtype=torch.long, device=dev)
            runtime = self.adaptive_runtime
            if runtime is not None:
                runtime.prepare_request(input_ids, attention_mask, kv_reserve_bytes=0)
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
            loss = float(out.loss.item())
            if not math.isfinite(loss):
                raise RuntimeError("NLL probe returned a non-finite loss")
            return loss

    def embed_ids(self, ids):
        import torch

        if not self.is_loaded():
            raise RuntimeError("no model loaded")
        ids = list(ids)
        if not ids:
            return None
        dev = torch.device(self.device)
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][0]
        return h.mean(dim=0).float().cpu()

    def is_loaded(self) -> bool:
        return self.model is not None

    def status(self) -> str:
        if not self.is_loaded():
            return "no model loaded"
        s = self.spec
        bits = [s.ref.split("/")[-1].split("\\")[-1]]
        if s.adapter_path:
            bits.append(f"{s.adapter_arch}:{s.adapter_path.split('/')[-1].split(chr(92))[-1]}")
        if s.router_path:
            bits.append("router")
        return " + ".join(bits)

    def context_limit(self) -> int:
        """Return the active model's configured generation context limit."""

        settings = self.spec.settings if self.spec else {}
        configured = settings.get("max_context_tokens")
        if isinstance(configured, int | float) and int(configured) > 0:
            return int(configured)
        model_context = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(model_context, int) and model_context > 0:
            return model_context
        tokenizer_context = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_context, int) and 0 < tokenizer_context < 10_000_000:
            return tokenizer_context
        return 8192

    def _context_extension_action(self) -> str:
        """Return valid overflow guidance for the active architecture."""

        from ..models.families import SPALMER_MODEL_FAMILY

        if self.spec and self.spec.model_family == SPALMER_MODEL_FAMILY:
            return "increase the configured native SPALMER context limit"
        return "use YaRN"

    def _prompt_tokens(self, messages: list[dict], *, prompt_mode: str = "chat") -> int:
        prompt = format_messages(self.tokenizer, messages, prompt_mode=prompt_mode)
        return _encoded_length(self.tokenizer(prompt))

    def plan_context(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        prompt_mode: str = "chat",
        compact: bool = True,
        strategy: str | None = None,
        recent_turns: int | None = None,
        max_nll_probes: int | None = None,
        nll_probe_tokens: int | None = None,
        min_nll_gain_nats: float | None = None,
        nll_token_cost_nats: float | None = None,
        trigger_ratio: float | None = None,
    ) -> ChatContextPlan:
        """Plan either conversational context or one literal completion prompt."""

        if prompt_mode == "chat":
            return self.plan_chat_context(
                messages,
                max_new_tokens=max_new_tokens,
                compact=compact,
                strategy=strategy,
                recent_turns=recent_turns,
                max_nll_probes=max_nll_probes,
                nll_probe_tokens=nll_probe_tokens,
                min_nll_gain_nats=min_nll_gain_nats,
                nll_token_cost_nats=nll_token_cost_nats,
                trigger_ratio=trigger_ratio,
            )
        if prompt_mode != "raw":
            raise ValueError(f"unknown prompt mode: {prompt_mode}")
        if not self.is_loaded():
            raise RuntimeError("no model loaded")

        answer_budget = int(max_new_tokens)
        if answer_budget < 1:
            raise ValueError("max_new_tokens must be positive")
        context_limit = self.context_limit()
        prompt_budget = context_limit - answer_budget
        if prompt_budget < 1:
            raise ValueError(
                f"Answer budget ({answer_budget}) leaves no prompt space in the "
                f"{context_limit}-token context."
            )

        copied = [dict(message) for message in messages]
        prompt_tokens = self._prompt_tokens(copied, prompt_mode="raw")
        if prompt_tokens > prompt_budget:
            raise ValueError(
                f"Raw prompt needs {prompt_tokens} tokens but only {prompt_budget} remain after "
                f"reserving {answer_budget} completion tokens. Raw prompts are never compacted or "
                "truncated; shorten the prompt, lower the completion budget, or "
                f"{self._context_extension_action()}."
            )
        return ChatContextPlan(
            messages=tuple(copied),
            prompt_tokens=prompt_tokens,
            prompt_budget=prompt_budget,
            context_limit=context_limit,
            answer_budget=answer_budget,
            prompt_mode="raw",
            strategy="none",
            selection_budget=prompt_budget,
        )

    def _token_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(token) for token in encoded]

    def _context_target_nll(
        self,
        prefix_messages: list[dict],
        query_group: list[dict],
        *,
        probe_tokens: int,
    ) -> float:
        """Score how well selected context predicts the current user request.

        True answer-side ``R_y`` remains the offline teacher because the answer
        is not known yet.  This bounded query-side probe is an inference-time
        accept/reject signal for candidates proposed by ATXY/lexical routing.
        """

        query = next(
            (
                str(message.get("content") or "")
                for message in reversed(query_group)
                if str(message.get("role") or "") == "user"
            ),
            str(query_group[-1].get("content") or "") if query_group else "",
        )
        # Keep the same explicit role boundary and leading-space tokenization
        # for the baseline and every candidate.  Otherwise BPE token zero and
        # HF's causal-label shift are not directly comparable.
        target_ids = self._token_ids(f" {query}")
        if not target_ids:
            return 0.0
        probe_tokens = max(1, int(probe_tokens))
        if len(target_ids) > probe_tokens:
            head = (probe_tokens + 1) // 2
            tail = probe_tokens - head
            target_ids = target_ids[:head] + (target_ids[-tail:] if tail else [])

        prefix_text = "\n".join(
            f"{str(message.get('role') or 'context')}: {str(message.get('content') or '')}"
            for message in prefix_messages
        )
        prefix_text = f"{prefix_text}\nuser:" if prefix_text else "user:"
        prefix_ids = self._token_ids(prefix_text)
        max_prefix = max(1, self.context_limit() - len(target_ids) - 1)
        prefix_ids = prefix_ids[-max_prefix:]
        if not prefix_ids:
            sentinel = getattr(self.tokenizer, "bos_token_id", None)
            if sentinel is None:
                sentinel = getattr(self.tokenizer, "eos_token_id", None)
            if sentinel is not None:
                prefix_ids = [int(sentinel)]
        return self.nll_ids(prefix_ids, target_ids)

    @staticmethod
    def _flatten_context(protected: list[dict], groups: list[list[dict]]) -> list[dict]:
        return [*protected, *(message for group in groups for message in group)]

    def _recency_context_plan(
        self,
        copied: list[dict],
        protected: list[dict],
        groups: list[list[dict]],
        *,
        prompt_budget: int,
        context_limit: int,
        answer_budget: int,
        strategy: str = "recency",
        fallback_reason: str = "",
    ) -> ChatContextPlan:
        selected = [groups[-1]]
        candidate = self._flatten_context(protected, selected)
        candidate_tokens = self._prompt_tokens(candidate)
        if candidate_tokens > prompt_budget:
            raise ValueError(
                f"The current turn plus protected instructions needs {candidate_tokens} tokens, but only "
                f"{prompt_budget} remain after reserving the answer. Reduce the message or "
                f"{self._context_extension_action()}."
            )

        for group in reversed(groups[:-1]):
            trial = [group, *selected]
            trial_messages = self._flatten_context(protected, trial)
            trial_tokens = self._prompt_tokens(trial_messages)
            if trial_tokens > prompt_budget:
                break
            selected = trial
            candidate = trial_messages
            candidate_tokens = trial_tokens

        return ChatContextPlan(
            messages=tuple(candidate),
            prompt_tokens=candidate_tokens,
            prompt_budget=prompt_budget,
            context_limit=context_limit,
            answer_budget=answer_budget,
            dropped_messages=len(copied) - len(candidate),
            dropped_turns=len(groups) - len(selected),
            strategy=strategy,
            selection_budget=prompt_budget,
            recent_turns=max(0, len(selected) - 1),
            fallback_reason=fallback_reason,
        )

    def _nll_context_plan(
        self,
        copied: list[dict],
        protected: list[dict],
        groups: list[list[dict]],
        *,
        prompt_budget: int,
        selection_budget: int,
        context_limit: int,
        answer_budget: int,
        recent_turns: int,
        max_probes: int,
        probe_tokens: int,
        min_gain_nats: float,
        token_cost_nats: float,
    ) -> ChatContextPlan:
        from .context_retrieval import rank_context_candidates

        latest_index = len(groups) - 1
        selected_indices = {latest_index}

        # A short contiguous tail is an invariant for conversational continuity;
        # older turns compete for the rest of the budget by marginal NLL.
        for index in range(latest_index - 1, max(-1, latest_index - recent_turns - 1), -1):
            trial_indices = selected_indices | {index}
            trial_groups = [groups[i] for i in sorted(trial_indices)]
            if self._prompt_tokens(self._flatten_context(protected, trial_groups)) <= selection_budget:
                selected_indices.add(index)

        mandatory = self._flatten_context(protected, [groups[i] for i in sorted(selected_indices)])
        mandatory_tokens = self._prompt_tokens(mandatory)
        if mandatory_tokens > prompt_budget:
            raise ValueError(
                f"The current turn plus protected instructions needs {mandatory_tokens} tokens, but only "
                f"{prompt_budget} remain after reserving the answer. Reduce the message or "
                f"{self._context_extension_action()}."
            )
        selection_budget = max(selection_budget, mandatory_tokens)

        prefix_indices = sorted(selected_indices - {latest_index})
        prefix_messages = self._flatten_context(protected, [groups[i] for i in prefix_indices])
        baseline_nll = self._context_target_nll(
            prefix_messages,
            groups[latest_index],
            probe_tokens=probe_tokens,
        )
        accepted_nll = baseline_nll
        accepted_prompt_tokens = mandatory_tokens
        evaluated = 0
        retrieved = 0

        ranked = [
            candidate
            for candidate in rank_context_candidates(groups[:-1], groups[latest_index])
            if candidate.index not in selected_indices
        ]
        atxy_candidates = sum(candidate.atxy_score > 0.0 for candidate in ranked)
        for candidate in ranked:
            if evaluated >= max(0, max_probes):
                break
            if candidate.query_echo:
                continue
            trial_indices = selected_indices | {candidate.index}
            trial_groups = [groups[i] for i in sorted(trial_indices)]
            trial_messages = self._flatten_context(protected, trial_groups)
            if self._prompt_tokens(trial_messages) > selection_budget:
                continue

            trial_prefix_indices = sorted(trial_indices - {latest_index})
            trial_prefix = self._flatten_context(
                protected,
                [groups[i] for i in trial_prefix_indices],
            )
            candidate_nll = self._context_target_nll(
                trial_prefix,
                groups[latest_index],
                probe_tokens=probe_tokens,
            )
            evaluated += 1
            marginal_gain = accepted_nll - candidate_nll
            incremental_tokens = max(0, self._prompt_tokens(trial_messages) - accepted_prompt_tokens)
            net_gain = marginal_gain - token_cost_nats * incremental_tokens
            if net_gain + 1e-9 < min_gain_nats:
                continue
            selected_indices.add(candidate.index)
            accepted_nll = candidate_nll
            accepted_prompt_tokens += incremental_tokens
            retrieved += 1

        selected_groups = [groups[i] for i in sorted(selected_indices)]
        selected_messages = self._flatten_context(protected, selected_groups)
        selected_tokens = self._prompt_tokens(selected_messages)
        return ChatContextPlan(
            messages=tuple(selected_messages),
            prompt_tokens=selected_tokens,
            prompt_budget=prompt_budget,
            context_limit=context_limit,
            answer_budget=answer_budget,
            dropped_messages=len(copied) - len(selected_messages),
            dropped_turns=len(groups) - len(selected_groups),
            strategy="nll_atxy",
            selection_budget=selection_budget,
            evaluated_turns=evaluated,
            retrieved_turns=retrieved,
            recent_turns=len(prefix_indices),
            atxy_candidates=atxy_candidates,
            baseline_nll=baseline_nll,
            final_nll=accepted_nll,
            nll_gain_nats=baseline_nll - accepted_nll,
        )

    def plan_chat_context(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        compact: bool = True,
        strategy: str | None = None,
        recent_turns: int | None = None,
        max_nll_probes: int | None = None,
        nll_probe_tokens: int | None = None,
        min_nll_gain_nats: float | None = None,
        nll_token_cost_nats: float | None = None,
        trigger_ratio: float | None = None,
    ) -> ChatContextPlan:
        """Build a cache-consistent prompt while reserving the answer budget.

        ``nll_atxy`` uses A=(T,X,Y), lexical overlap, and recency only to
        propose a bounded shortlist.  Each payload must then reduce query NLL
        before it is packed.  The final prompt is rebuilt once with contiguous
        logical positions, so ordinary RoPE/YaRN and KV caching remain valid.
        """

        if not self.is_loaded():
            raise RuntimeError("no model loaded")
        answer_budget = int(max_new_tokens)
        if answer_budget < 1:
            raise ValueError("max_new_tokens must be positive")
        context_limit = self.context_limit()
        prompt_budget = context_limit - answer_budget
        if prompt_budget < 1:
            raise ValueError(
                f"Answer budget ({answer_budget}) leaves no prompt space in the "
                f"{context_limit}-token context."
            )

        settings = self.spec.settings if self.spec else {}
        strategy = str(strategy or settings.get("context_compaction_strategy") or "recency")
        if strategy not in {"recency", "nll_atxy"}:
            raise ValueError(f"unknown context compaction strategy: {strategy}")
        recent_turns = int(
            settings.get("context_recent_turns", 1) if recent_turns is None else recent_turns
        )
        max_nll_probes = int(
            settings.get("context_nll_max_probes", 8)
            if max_nll_probes is None
            else max_nll_probes
        )
        nll_probe_tokens = int(
            settings.get("context_nll_probe_tokens", 128)
            if nll_probe_tokens is None
            else nll_probe_tokens
        )
        min_nll_gain_nats = float(
            settings.get("context_min_nll_gain_nats", 0.01)
            if min_nll_gain_nats is None
            else min_nll_gain_nats
        )
        nll_token_cost_nats = float(
            settings.get("context_nll_token_cost_nats", 0.00002)
            if nll_token_cost_nats is None
            else nll_token_cost_nats
        )
        trigger_ratio = float(
            settings.get("context_nll_trigger_ratio", 0.9)
            if trigger_ratio is None
            else trigger_ratio
        )
        trigger_ratio = min(1.0, max(0.1, trigger_ratio))
        if self.adaptive_runtime is not None:
            # Each probe must prepare a coherent adaptive expert pool. Keep
            # that experimental path bounded until a fixed-pool scorer exists.
            max_nll_probes = min(max_nll_probes, 2)

        copied = [dict(message) for message in messages]
        prompt_tokens = self._prompt_tokens(copied)
        selection_budget = (
            max(1, int(prompt_budget * trigger_ratio)) if strategy == "nll_atxy" else prompt_budget
        )
        if prompt_tokens <= selection_budget:
            return ChatContextPlan(
                messages=tuple(copied),
                prompt_tokens=prompt_tokens,
                prompt_budget=prompt_budget,
                context_limit=context_limit,
                answer_budget=answer_budget,
                strategy=strategy,
                selection_budget=selection_budget,
            )
        if not compact:
            if prompt_tokens <= prompt_budget:
                return ChatContextPlan(
                    messages=tuple(copied),
                    prompt_tokens=prompt_tokens,
                    prompt_budget=prompt_budget,
                    context_limit=context_limit,
                    answer_budget=answer_budget,
                    strategy=strategy,
                    selection_budget=selection_budget,
                )
            raise ValueError(
                f"Chat prompt needs {prompt_tokens} tokens but only {prompt_budget} remain after "
                f"reserving {answer_budget} answer tokens. Enable context compaction or "
                f"{self._context_extension_action()}."
            )

        protected, groups = _conversation_groups(copied)
        if not groups:
            raise ValueError(
                f"Protected chat instructions need {prompt_tokens} tokens but the prompt budget is "
                f"{prompt_budget}. Reduce them, lower the answer budget, or "
                f"{self._context_extension_action()}."
            )

        if strategy == "recency":
            return self._recency_context_plan(
                copied,
                protected,
                groups,
                prompt_budget=prompt_budget,
                context_limit=context_limit,
                answer_budget=answer_budget,
            )

        try:
            with self._generation_lock:
                return self._nll_context_plan(
                    copied,
                    protected,
                    groups,
                    prompt_budget=prompt_budget,
                    selection_budget=selection_budget,
                    context_limit=context_limit,
                    answer_budget=answer_budget,
                    recent_turns=max(0, recent_turns),
                    max_probes=max(0, max_nll_probes),
                    probe_tokens=max(1, nll_probe_tokens),
                    min_gain_nats=max(0.0, min_nll_gain_nats),
                    token_cost_nats=max(0.0, nll_token_cost_nats),
                )
        except (RuntimeError, TypeError, ValueError) as exc:
            if self.cuda:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            return self._recency_context_plan(
                copied,
                protected,
                groups,
                prompt_budget=prompt_budget,
                context_limit=context_limit,
                answer_budget=answer_budget,
                strategy="recency_fallback",
                fallback_reason=str(exc).splitlines()[0][:240],
            )

    def unload(self) -> None:
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            self.tokenizer = None
            self.spec = None
            self.notes = []
            self.adaptive_runtime = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def load(self, spec: LoadSpec) -> list[str]:
        from ..models.families import SPALMER_MODEL_FAMILY, family_from_reference
        from ..models.loading import load_base
        from ..mole.inject import MoLEConfig, inject_mole, set_stage
        from ..mole.runtime import AdaptiveExpertRuntime
        from ..mole.store import (
            LazyExpertStore,
            apply_mole_weights,
            load_mole_config,
            load_mole_weights,
        )
        model_family = spec.model_family or family_from_reference(spec.ref)
        if model_family == SPALMER_MODEL_FAMILY:
            return self._load_spalmer(replace(spec, model_family=model_family))

        resolved_settings, preset_context = _context_mode_settings(spec.ref, dict(spec.settings))
        spec = replace(spec, settings=resolved_settings)
        adaptive_runtime = None

        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            self.adaptive_runtime = None
            import torch

            rope_scaling = _rope_scaling(spec.settings)
            load_options = {
                "attn_implementation": spec.settings.get("attn_implementation", "sdpa"),
                "rope_scaling": rope_scaling,
            }
            if rope_scaling is not None:
                load_options["max_position_embeddings"] = int(
                    spec.settings.get("max_context_tokens", 8192) or 8192
                )
            model, tokenizer, cuda, quant, notes = load_base(
                spec.ref,
                spec.quant,
                **load_options,
            )
            if preset_context and str(spec.settings.get("context_mode")) == "native":
                notes.append(f"Native checkpoint context: {preset_context:,} positions.")
            model.config.use_cache = True
            if spec.adapter_path and spec.adapter_arch == "mole":
                adapter_cfg = load_mole_config(spec.adapter_path, expected_base_ref=spec.ref)
                adapter_router_type = adapter_cfg.get("config", {}).get("router_type", "per_layer")
                if adapter_router_type == "per_token" and spec.router_path:
                    raise ValueError(
                        "per-token MoLE adapters already contain learned block routers; "
                        "a legacy prompt-router snapshot cannot override them"
                    )
                cfg = (
                    load_mole_config(spec.router_path, expected_base_ref=spec.ref)
                    if spec.router_path
                    else adapter_cfg
                )
                if cfg.get("layers") != adapter_cfg.get("layers"):
                    raise ValueError("router snapshot targets a different MoLE layer bank")
                mole_cfg = MoLEConfig.from_dict(cfg["config"])
                src = spec.router_path or spec.adapter_path
                expert_dtype = torch.bfloat16 if cuda else torch.float32
                if mole_cfg.adaptive.enabled:
                    inject_mole(
                        model,
                        mole_cfg,
                        expert_dtype=expert_dtype,
                        expert_device="meta",
                    )
                    lazy_store = LazyExpertStore(src, expected_base_ref=spec.ref)
                    adaptive_runtime = AdaptiveExpertRuntime(
                        model,
                        mole_cfg,
                        lazy_store,
                        device=torch.device("cuda" if cuda else "cpu"),
                        prompt_probe_tokens=int(
                            spec.settings.get("adaptive_prompt_probe_tokens", 128) or 128
                        ),
                    )
                    notes.append("Adaptive MoLE v4 routing loaded; expert lanes materialize per request.")
                else:
                    inject_mole(model, mole_cfg, expert_dtype=expert_dtype)
                    apply_mole_weights(model, load_mole_weights(src))
                set_stage(model, "none")
                notes.append(f"MoLE adapter applied ({len(cfg.get('layers', []))} layers) from {src}")
                if spec.router_path:
                    notes.append("Router snapshot overrides adapter router weights.")
            elif spec.adapter_path and spec.adapter_arch == "lora":
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, spec.adapter_path)
                notes.append("PEFT LoRA adapter applied.")
            model.eval()
            self.model = model
            self.tokenizer = tokenizer
            self.spec = spec
            self.notes = notes
            self.cuda = cuda
            self.device = "cuda" if cuda else "cpu"
            self.adaptive_runtime = adaptive_runtime
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            return list(notes)

    def _load_spalmer(self, spec: LoadSpec) -> list[str]:
        """Load a native SPALMER bundle behind yyLab's inference contract."""

        from ..models.spalmer import load_spalmer_runtime

        if spec.adapter_path or spec.router_path:
            raise ValueError(
                "SPALMER checkpoints contain their native micro-expert bank; "
                "yyLab LoRA/MoLE adapters and router snapshots are not compatible"
            )
        mode = str(spec.settings.get("context_mode") or "native")
        if mode not in {"", "native"} or str(spec.settings.get("rope_scaling") or "none") != "none":
            raise ValueError(
                "SPALMER uses recurrent KDA/NoPE MLA state and does not use RoPE or YaRN; "
                "select native context"
            )
        if spec.quant not in {"", "auto", "bf16", "checkpoint", "native"}:
            raise ValueError(
                "SPALMER precision is checkpoint-native (FP4 expert forward, MXFP8 "
                "activations, BF16 parameters); select Auto or BF16"
            )

        settings = dict(spec.settings)
        settings.update(context_mode="native", rope_scaling="none", rope_factor=1.0)
        spec = replace(spec, quant="checkpoint", settings=settings)
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            self.adaptive_runtime = None
            facade = load_spalmer_runtime(
                spec.ref,
                dynamic_residency=bool(settings.get("spalmer_dynamic_residency", True)),
                max_context_tokens=(
                    int(settings["max_context_tokens"])
                    if int(settings.get("max_context_tokens") or 0) > 0
                    else None
                ),
            )
            self.model = facade
            self.tokenizer = facade.tokenizer
            self.spec = spec
            self.notes = list(facade.notes)
            self.cuda = bool(facade.cuda_available)
            self.device = "cuda" if self.cuda else "cpu"
            return list(self.notes)

    def _generate_kwargs(
        self, input_ids, attention_mask, temperature: float, top_p: float, max_new_tokens: int
    ) -> dict:
        settings = self.spec.settings if self.spec else {}
        max_ctx = self.context_limit()
        prompt_len = input_ids.shape[1]
        budget = max_ctx - prompt_len
        if budget < 1:
            raise ValueError(
                f"Prompt ({prompt_len} tokens) leaves no generation room in max context ({max_ctx})."
            )
        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": min(int(max_new_tokens), budget),
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs["temperature"] = max(float(temperature), 1e-5)
            kwargs["top_p"] = float(top_p)
        from ..models.families import SPALMER_MODEL_FAMILY

        spalmer = bool(self.spec and self.spec.model_family == SPALMER_MODEL_FAMILY)
        if spalmer:
            cache_kwargs, cache_notes = {}, [
                "SPALMER generation uses native recurrent KDA/MLA state; Transformers KV cache settings do not apply."
            ]
        else:
            cache_kwargs, cache_notes = _cache_kwargs(settings, self.cuda)
        self.notes.extend(cache_notes)
        kwargs.update(cache_kwargs)
        if (
            not spalmer
            and str(settings.get("cache_implementation") or "dynamic")
            == "quantized_offloaded"
        ):
            import torch

            from .quantized_offload_cache import QuantizedOffloadedCache

            kv_quant = str(settings.get("kv_quant") or "int4")
            kwargs["past_key_values"] = QuantizedOffloadedCache(
                self.model.config,
                nbits=4 if kv_quant == "int4" else 8,
                q_group_size=max(1, int(settings.get("kv_offload_group_size", 64) or 64)),
                residual_length=max(
                    0,
                    int(settings.get("kv_offload_residual_length", 32) or 0),
                ),
                scale_dtype=torch.float16,
                pin_memory=bool(settings.get("kv_offload_pin_memory", True)),
                growth_chunk_size=max(
                    1,
                    int(settings.get("kv_offload_growth_chunk_size", 256) or 256),
                ),
            )

        from transformers import StoppingCriteria, StoppingCriteriaList

        cancel_event = self._cancel

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **_kw) -> bool:
                return cancel_event.is_set()

        kwargs["stopping_criteria"] = StoppingCriteriaList([_Cancel()])
        return kwargs

    def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        telemetry_callback: Callable[[dict[str, object]], None] | None = None,
        prompt_mode: str = "chat",
    ) -> Iterator[str]:
        # Adaptive occupancy is model-attached mutable state. Serializing all
        # generation also prevents fixed-route adapters and the global cancel
        # event from being shared by concurrent GUI clients.
        with self._generation_lock:
            if not self.is_loaded():
                raise RuntimeError("no model loaded")
            self._cancel.clear()

            prompt = format_messages(self.tokenizer, messages, prompt_mode=prompt_mode)
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc["attention_mask"].to(self.model.device)
            runtime = self.adaptive_runtime
            gen_kwargs = self._generate_kwargs(
                input_ids,
                attention_mask,
                temperature,
                top_p,
                max_new_tokens,
            )
            if runtime is not None:
                settings = self.spec.settings if self.spec else {}
                runtime.prepare_request(
                    input_ids,
                    attention_mask,
                    telemetry_callback=telemetry_callback,
                    kv_reserve_bytes=_adaptive_kv_reserve_bytes(
                        self.model.config,
                        settings,
                        prompt_tokens=int(input_ids.shape[1]),
                        answer_tokens=int(gen_kwargs["max_new_tokens"]),
                        cuda=self.cuda,
                    ),
                )
            if runtime is not None:
                from transformers import LogitsProcessorList

                gen_kwargs["logits_processor"] = LogitsProcessorList([runtime.logits_processor()])
                runtime.begin_decode()
            had_cache = "cache_implementation" in gen_kwargs or "past_key_values" in gen_kwargs
            try:
                try:
                    yield from self._stream(gen_kwargs)
                except Exception:
                    if had_cache:
                        for cache_key in (
                            "cache_implementation",
                            "cache_config",
                            "max_cache_len",
                            "prefill_chunk_size",
                            "past_key_values",
                        ):
                            gen_kwargs.pop(cache_key, None)
                        if runtime is not None:
                            runtime.restart_decode_attempt(
                                reason="selected KV cache failed; retrying with dynamic cache"
                            )
                        self.notes.append("Selected KV cache failed; retried with dynamic cache.")
                        yield from self._stream(gen_kwargs)
                    else:
                        raise
            finally:
                if runtime is not None:
                    runtime.end_decode()

    def routing_telemetry(self) -> dict[str, object] | None:
        runtime = self.adaptive_runtime
        if runtime is not None:
            return runtime.telemetry()
        getter = getattr(self.model, "routing_telemetry", None)
        if callable(getter):
            value = getter()
            return value if isinstance(value, dict) else None
        return None

    def adaptive_telemetry(self) -> dict[str, object] | None:
        """Backward-compatible alias for adaptive routing telemetry."""

        return self.routing_telemetry()

    def _stream(self, gen_kwargs: dict) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        from ..mole.inject import begin_prompt_route, end_prompt_route

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        error_box: list[BaseException] = []

        def runner() -> None:
            import torch

            try:
                begin_prompt_route(self.model)
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except BaseException as e:  # noqa: BLE001
                error_box.append(e)
            finally:
                end_prompt_route(self.model)
                streamer.end()

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        t.join(timeout=5)
        if error_box:
            raise error_box[0]


ENGINE = Engine()
