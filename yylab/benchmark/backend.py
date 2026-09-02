"""Direct yyLab model backend used by detached quality benchmark workers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float | None
    finish_reason: str
    routing: dict[str, object] | None
    thinking_tokens: int = 0
    answer_tokens: int | None = None
    thinking_budget_exhausted: bool = False


def _subsequence_index(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    for index in range(len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return None


def _subsequence_last_index(values: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(values):
        return None
    for index in range(len(values) - len(needle), -1, -1):
        if values[index : index + len(needle)] == needle:
            return index
    return None


def _thinking_token_counts(
    values: list[int],
    open_ids: list[int],
    close_ids: list[int],
    *,
    prompt_has_open_thinking: bool,
) -> tuple[int, int, int | None]:
    """Split one generated phase into hidden and visible model tokens."""

    close_offset = _subsequence_index(values, close_ids)
    if close_offset is not None:
        thinking_tokens = close_offset + len(close_ids)
        return thinking_tokens, len(values) - thinking_tokens, close_offset
    if prompt_has_open_thinking or _subsequence_index(values, open_ids) is not None:
        return len(values), 0, None
    return 0, len(values), None


def _format_messages(
    tokenizer,
    messages: list[dict[str, str]],
    enable_thinking: bool,
    prompt_mode: str = "chat",
) -> str:
    """Render a chat-template or literal completion benchmark prompt."""

    normalized_mode = str(prompt_mode or "chat").strip().casefold()
    if normalized_mode == "raw":
        if enable_thinking:
            raise ValueError("thinking requires chat prompt mode")
        parts = [str(message.get("content") or "").strip() for message in messages]
        prompt = "\n\n".join(part for part in parts if part)
        if not prompt:
            raise ValueError("raw benchmark prompt cannot be empty")
        return prompt
    if normalized_mode != "chat":
        raise ValueError(f"unknown benchmark prompt mode: {prompt_mode}")

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        if enable_thinking:
            raise RuntimeError("this tokenizer chat template cannot honor enable_thinking=True") from exc
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            pass
    except (AttributeError, ValueError, NotImplementedError) as exc:
        if enable_thinking:
            raise RuntimeError("this tokenizer chat template cannot honor enable_thinking=True") from exc
        pass
    parts = [f"### {m.get('role', 'user').capitalize()}\n{m.get('content', '')}" for m in messages]
    parts.append("### Assistant\n")
    return "\n\n".join(parts)


class _RouteAccumulator:
    """Accumulate routes across prompt prefill and every decode forward."""

    def __init__(self, model) -> None:
        import torch

        from ..mole.layer import SharedTokenRouter

        self._torch = torch
        self._handles = []
        self._totals = None
        self._selections = None
        self._tokens = 0
        self._calls = 0
        for module in model.modules():
            if isinstance(module, SharedTokenRouter):
                self._handles.append(module.register_forward_hook(self._capture))

    def _capture(self, module, _inputs, output) -> None:
        indices, weights, _logits = output
        count = self._torch.zeros(module.num_experts, dtype=self._torch.float64)
        selections = self._torch.zeros(module.num_experts, dtype=self._torch.float64)
        count.scatter_add_(
            0,
            indices.detach().reshape(-1).cpu(),
            weights.detach().double().reshape(-1).cpu(),
        )
        selections.scatter_add_(
            0,
            indices.detach().reshape(-1).cpu(),
            self._torch.ones(indices.numel(), dtype=self._torch.float64),
        )
        self._totals = count if self._totals is None else self._totals + count
        self._selections = selections if self._selections is None else self._selections + selections
        self._tokens += int(indices.shape[0])
        self._calls += 1

    def reset(self) -> None:
        """Discard phase-one counts before a full-prefix continuation prefill."""

        self._totals = None
        self._selections = None
        self._tokens = 0
        self._calls = 0

    def close(self) -> dict[str, object] | None:
        for handle in self._handles:
            handle.remove()
        if self._totals is None or self._totals.sum().item() <= 0:
            return None
        share = self._totals / self._totals.sum()
        selection_share = self._selections / self._selections.sum()
        if share.numel() > 1:
            entropy = -(share * share.clamp_min(1e-12).log()).sum() / self._torch.log(
                share.new_tensor(float(share.numel()))
            )
        else:
            entropy = share.new_tensor(1.0)
        return {
            "router_entropy": round(float(entropy.item()), 6),
            "dead_experts": int((self._selections == 0).sum().item()),
            "expert_weight_share": [round(float(value), 6) for value in share.tolist()],
            "expert_selection_share": [round(float(value), 6) for value in selection_share.tolist()],
            "routed_token_blocks": self._tokens,
            "router_calls": self._calls,
        }


def _adaptive_generation_phase(
    model,
    generation_kwargs: dict[str, object],
    runtime,
    *,
    settings: dict[str, object] | None = None,
    cuda: bool = False,
) -> tuple[object, dict[str, object] | None]:
    """Run one direct generation phase with adaptive residency held fixed.

    Benchmark generation calls the model directly instead of going through
    :meth:`Engine.generate_stream`, so it must reproduce the adaptive runtime
    request lifecycle here. Keep the caller's kwargs immutable: thinking
    continuations are assembled from them and must not inherit an entropy
    processor from an earlier phase.
    """

    if runtime is None:
        return model.generate(**generation_kwargs), None

    from transformers import LogitsProcessorList

    input_ids = generation_kwargs.get("input_ids")
    if input_ids is None:
        raise ValueError("adaptive benchmark generation requires input_ids")
    from ..infer.engine import _adaptive_kv_reserve_bytes

    runtime.prepare_request(
        input_ids,
        generation_kwargs.get("attention_mask"),
        kv_reserve_bytes=_adaptive_kv_reserve_bytes(
            model.config,
            dict(settings or {}),
            prompt_tokens=int(input_ids.shape[1]),
            answer_tokens=int(generation_kwargs.get("max_new_tokens") or 0),
            cuda=bool(cuda),
        ),
    )

    phase_kwargs = dict(generation_kwargs)
    configured_processors = phase_kwargs.get("logits_processor")
    processors = list(configured_processors) if configured_processors is not None else []
    processors.append(runtime.logits_processor())
    phase_kwargs["logits_processor"] = LogitsProcessorList(processors)

    before = runtime.snapshot()
    before_transfers = (
        before.lane_loads,
        before.h2d_transfers,
        before.h2d_bytes,
    )
    runtime.begin_decode()
    try:
        generated = model.generate(**phase_kwargs)
    finally:
        runtime.end_decode()
        after = runtime.snapshot()
        after_transfers = (
            after.lane_loads,
            after.h2d_transfers,
            after.h2d_bytes,
        )
        if after_transfers != before_transfers:
            raise RuntimeError("adaptive benchmark generation transferred expert weights during decode")
    return generated, dict(runtime.telemetry())


class YyLabBenchmarkBackend:
    """Load one checkpoint and switch efficiently between yyLab benchmark policies."""

    def __init__(
        self,
        *,
        checkpoint_ref: str,
        quant: str,
        adapter_path: str | None,
        adapter_arch: str,
        model_family: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        from ..infer.engine import Engine
        from ..models.families import family_from_reference

        self.checkpoint_ref = checkpoint_ref
        self.quant = quant
        self.adapter_path = adapter_path
        self.adapter_arch = adapter_arch
        self.settings = dict(settings or {})
        declared_family = str(model_family or "").strip().casefold()
        if declared_family and declared_family not in {"hf", "spalmer", "spalmer_v1"}:
            raise ValueError(f"unknown benchmark model family: {model_family}")
        self.model_family = (
            "spalmer"
            if declared_family in {"spalmer", "spalmer_v1"}
            else ("hf" if declared_family else family_from_reference(checkpoint_ref))
        )
        self.engine = Engine()
        self._stack = ""
        self._policy = ""

    def identity(self) -> dict[str, object]:
        value = {
            "backend": "yylab-direct",
            "model_family": self.model_family,
            "checkpoint_ref": self.checkpoint_ref,
            "quant": self.quant,
            "adapter_path": self.adapter_path,
            "adapter_arch": self.adapter_arch,
            "policy": self._policy,
            "generation_controls": [
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "prompt_mode",
                "enable_thinking",
                "max_thinking_tokens",
                "cache_mode",
                "seed",
            ],
        }
        if self.model_family == "spalmer":
            value["expert_placement"] = {
                "requested_physical_offload": bool(
                    self.settings.get("spalmer_expert_offload", True)
                ),
                "requested_cache_size": self.settings.get("spalmer_expert_cache_size"),
                "actual_placement": "reported per generation in routing telemetry",
            }
        return value

    def configure_policy(self, policy: str) -> list[str]:
        from ..infer.engine import LoadSpec
        from ..mole.inject import set_active_expert

        if self.model_family == "spalmer":
            if policy not in {"base", "dynamic"}:
                raise ValueError(
                    "SPALMER benchmarks support its fixed base route or native dynamic "
                    "2–20 expert residency; yyLab adapter/forced-expert policies do not apply"
                )
            stack = f"spalmer:{policy}"
            if self._stack != stack:
                settings = dict(self.settings)
                settings["spalmer_dynamic_residency"] = policy == "dynamic"
                notes = self.engine.load(
                    LoadSpec(
                        ref=self.checkpoint_ref,
                        quant="checkpoint",
                        model_family="spalmer",
                        settings=settings,
                    )
                )
                self._stack = stack
            else:
                notes = []
            self._policy = policy
            return notes

        if policy == "base":
            if self._stack != "base":
                notes = self.engine.load(
                    LoadSpec(
                        ref=self.checkpoint_ref,
                        quant=self.quant,
                        model_family=self.model_family,
                        settings=self.settings,
                    )
                )
                self._stack = "base"
            else:
                notes = []
        else:
            if not self.adapter_path:
                raise ValueError(f"policy {policy!r} requires an adapter")
            if self._stack != "adapter":
                notes = self.engine.load(
                    LoadSpec(
                        ref=self.checkpoint_ref,
                        quant=self.quant,
                        model_family=self.model_family,
                        adapter_path=self.adapter_path,
                        adapter_arch=self.adapter_arch,
                        settings=self.settings,
                    )
                )
                self._stack = "adapter"
            else:
                notes = []
            if policy == "dynamic":
                if self.adapter_arch != "mole":
                    raise ValueError("dynamic routing requires a MoLE adapter")
                set_active_expert(self.engine.model, None)
            elif policy.startswith("expert:"):
                if self.adapter_arch != "mole":
                    raise ValueError("forced experts require a MoLE adapter")
                if getattr(self.engine, "adaptive_runtime", None) is not None:
                    raise ValueError(
                        "forced expert benchmark policies are unavailable for adaptive MoLE v4; "
                        "use 'dynamic' or 'adapter' so expert lanes are prepared before decode"
                    )
                set_active_expert(self.engine.model, int(policy.split(":", 1)[1]))
            elif policy == "adapter" and self.adapter_arch == "mole":
                set_active_expert(self.engine.model, None)
            elif policy != "adapter":
                raise ValueError(f"unknown benchmark policy: {policy}")
        self._policy = policy
        return notes

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
        cache_mode: str,
        seed: int,
        max_thinking_tokens: int = 1024,
        prompt_mode: str = "chat",
    ) -> GenerationResponse:
        import torch

        from ..mole.inject import begin_prompt_route, end_prompt_route

        if not self.engine.is_loaded():
            raise RuntimeError("benchmark backend has no loaded model")
        tokenizer = self.engine.tokenizer
        model = self.engine.model
        prompt = _format_messages(tokenizer, messages, enable_thinking, prompt_mode)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        answer_budget = int(max_new_tokens)
        thinking_budget = int(max_thinking_tokens) if enable_thinking else 0
        if answer_budget < 1:
            raise ValueError("max_new_tokens must be positive")
        if enable_thinking and thinking_budget < 1:
            raise ValueError("max_thinking_tokens must be positive when thinking is enabled")

        open_ids: list[int] = []
        close_ids: list[int] = []
        answer_instruction = ""
        if enable_thinking:
            open_ids = [int(value) for value in tokenizer.encode("<think>", add_special_tokens=False)]
            close_ids = [int(value) for value in tokenizer.encode("</think>", add_special_tokens=False)]
            if not open_ids or not close_ids:
                raise RuntimeError("tokenizer cannot encode Qwen thinking delimiters")
            answer_instruction = (
                "Using the reasoning already completed, answer the original "
                "request directly and concisely. Follow its requested output "
                "format exactly. Do not repeat or summarize the reasoning."
            )

        prompt_values = [int(value) for value in input_ids[0].tolist()]
        prompt_open_offset = _subsequence_last_index(prompt_values, open_ids)
        prompt_close_offset = _subsequence_last_index(prompt_values, close_ids)
        prompt_has_open_thinking = bool(
            enable_thinking
            and prompt_open_offset is not None
            and (prompt_close_offset is None or prompt_open_offset > prompt_close_offset)
        )
        if enable_thinking and not prompt_has_open_thinking:
            # Qwen's thinking template normally leaves generation responsible
            # for the opening marker. Fine-tuned adapters do not always emit
            # it, even though they immediately start reasoning and may later
            # emit </think>. Prefill the marker so budget accounting and answer
            # extraction do not depend on the adapter reproducing the protocol.
            thinking_prefix_ids = [
                int(value) for value in tokenizer.encode("<think>\n", add_special_tokens=False)
            ]
            if _subsequence_index(thinking_prefix_ids, open_ids) is None:
                raise RuntimeError("tokenizer cannot encode a Qwen thinking prefix")
            prefix_tensor = input_ids.new_tensor(thinking_prefix_ids).unsqueeze(0)
            if input_ids.shape[0] > 1:
                prefix_tensor = prefix_tensor.expand(input_ids.shape[0], -1)
            input_ids = torch.cat((input_ids, prefix_tensor), dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], len(thinking_prefix_ids))),
                    ),
                    dim=-1,
                )
            prompt_has_open_thinking = True

        configured_context = self.settings.get("max_context_tokens")
        model_context = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        tokenizer_context = getattr(tokenizer, "model_max_length", None)
        if configured_context:
            max_context_tokens = int(configured_context)
        elif isinstance(model_context, int) and model_context > 0:
            max_context_tokens = model_context
        elif isinstance(tokenizer_context, int) and tokenizer_context > 0 and tokenizer_context < 10_000_000:
            max_context_tokens = tokenizer_context
        else:
            max_context_tokens = 8192
        prompt_tokens = int(input_ids.shape[1])
        reserved_tokens = answer_budget
        continuation_overhead = 0
        if enable_thinking:
            # If reasoning ends without a closing marker, yyLab starts a fresh
            # direct-answer turn containing the completed hidden reasoning.
            # Reserve its chat-control and instruction tokens up front.
            continuation_stub = _format_messages(
                tokenizer,
                messages
                + [
                    {"role": "assistant", "content": "<think>\n</think>"},
                    {"role": "user", "content": answer_instruction},
                ],
                False,
            )
            continuation_stub_ids = tokenizer(continuation_stub, return_tensors="pt")["input_ids"]
            continuation_overhead = max(
                0,
                int(continuation_stub_ids.shape[1]) - prompt_tokens,
            )
            reserved_tokens += thinking_budget + continuation_overhead
        if prompt_tokens + reserved_tokens > max_context_tokens:
            if self.model_family == "spalmer":
                guidance = "increase the SPALMER native context limit or reduce the requested budgets"
            elif enable_thinking:
                guidance = (
                    "select a YaRN context extension or reduce the thinking budget because "
                    "yyLab will not reduce the answer budget"
                )
            else:
                guidance = "select a YaRN context extension or reduce the answer budget"
            raise ValueError(
                "benchmark budgets exceed the configured context: "
                f"prompt {prompt_tokens} + thinking {thinking_budget} + answer {answer_budget}"
                + (f" + continuation {continuation_overhead}" if enable_thinking else "")
                + f" > {max_context_tokens}; {guidance}"
            )

        kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": answer_budget,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": temperature > 0,
            "use_cache": cache_mode != "disabled",
        }
        if temperature > 0:
            kwargs.update(
                temperature=max(float(temperature), 1e-5),
                top_p=float(top_p),
            )
            if top_k > 0:
                kwargs["top_k"] = int(top_k)

        stopper = None
        if enable_thinking:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _ThinkingBudgetStop(StoppingCriteria):
                def __init__(self) -> None:
                    self.saw_close = None

                @staticmethod
                def _ends_with(current_ids, token_ids: list[int]):
                    if current_ids.shape[1] - input_ids.shape[1] < len(token_ids):
                        return current_ids.new_zeros(current_ids.shape[0], dtype=torch.bool)
                    expected = current_ids.new_tensor(token_ids)
                    return (current_ids[:, -len(token_ids) :] == expected).all(dim=1)

                def __call__(self, current_ids, _scores, **_kwargs):
                    batch_size = current_ids.shape[0]
                    if self.saw_close is None:
                        self.saw_close = current_ids.new_zeros(batch_size, dtype=torch.bool)
                    self.saw_close = self.saw_close | self._ends_with(current_ids, close_ids)
                    generated_count = current_ids.shape[1] - input_ids.shape[1]

                    thinking_limit = current_ids.new_zeros(batch_size, dtype=torch.bool)
                    if generated_count >= thinking_budget:
                        thinking_limit = ~self.saw_close

                    answer_limit = current_ids.new_zeros(batch_size, dtype=torch.bool)
                    if generated_count >= answer_budget + len(close_ids):
                        close_start = current_ids.shape[1] - answer_budget - len(close_ids)
                        expected_close = current_ids.new_tensor(close_ids)
                        answer_limit = (
                            current_ids[:, close_start : close_start + len(close_ids)] == expected_close
                        ).all(dim=1)
                    return thinking_limit | answer_limit

            stopper = _ThinkingBudgetStop()
            kwargs["max_new_tokens"] = thinking_budget + answer_budget
            kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])

        adaptive_runtime = getattr(self.engine, "adaptive_runtime", None)
        if adaptive_runtime is not None and self._policy not in {"dynamic", "adapter"}:
            raise RuntimeError(
                "adaptive MoLE v4 benchmark generation requires the 'dynamic' or 'adapter' policy"
            )
        accumulator = (
            _RouteAccumulator(model) if self._policy == "dynamic" and adaptive_runtime is None else None
        )
        adaptive_phases: list[dict[str, object]] = []
        if self.engine.cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        routing = None
        thinking_budget_exhausted = False
        model_generated_tokens = 0
        answer_tokens = 0
        thinking_tokens = 0
        response_text: str | None = None
        finish_completion_ids = None
        begin_prompt_route(model)
        try:
            with torch.no_grad():
                generated, adaptive_phase = _adaptive_generation_phase(
                    model,
                    kwargs,
                    adaptive_runtime,
                    settings=self.settings,
                    cuda=self.engine.cuda,
                )
                if adaptive_phase is not None:
                    adaptive_phases.append(adaptive_phase)
                phase_one_count = int(generated.shape[1] - input_ids.shape[1])
                model_generated_tokens = phase_one_count
                answer_tokens = phase_one_count
                if enable_thinking:
                    phase_one_values = [int(value) for value in generated[0, input_ids.shape[1] :].tolist()]
                    classified_thinking, classified_answer, close_offset = _thinking_token_counts(
                        phase_one_values,
                        open_ids,
                        close_ids,
                        prompt_has_open_thinking=prompt_has_open_thinking,
                    )
                    phase_answer_text = ""
                    if close_offset is not None:
                        answer_start = close_offset + len(close_ids)
                        phase_answer_ids = input_ids.new_tensor(phase_one_values[answer_start:])
                        phase_answer_text = tokenizer.decode(
                            phase_answer_ids,
                            skip_special_tokens=True,
                        ).strip()
                    needs_answer_continuation = bool(close_offset is None or not phase_answer_text)
                    thinking_budget_exhausted = bool(
                        close_offset is None and phase_one_count >= thinking_budget
                    )
                    if needs_answer_continuation:
                        eos_id = tokenizer.eos_token_id
                        reasoning_values = (
                            list(phase_one_values)
                            if close_offset is None
                            else list(phase_one_values[:close_offset])
                        )
                        if reasoning_values and eos_id is not None and reasoning_values[-1] == eos_id:
                            # EOS inside an unclosed hidden block is not a final
                            # answer and must not terminate the follow-up turn.
                            reasoning_values.pop()
                        if reasoning_values[: len(open_ids)] == open_ids:
                            reasoning_values = reasoning_values[len(open_ids) :]
                        reasoning_ids = input_ids.new_tensor(reasoning_values)
                        reasoning_text = tokenizer.decode(
                            reasoning_ids,
                            skip_special_tokens=True,
                        ).strip()
                        continuation_prompt = _format_messages(
                            tokenizer,
                            messages
                            + [
                                {
                                    "role": "assistant",
                                    "content": f"<think>\n{reasoning_text}\n</think>",
                                },
                                {"role": "user", "content": answer_instruction},
                            ],
                            False,
                        )
                        continuation_encoded = tokenizer(continuation_prompt, return_tensors="pt")
                        continuation_ids = continuation_encoded["input_ids"].to(model.device)
                        continuation_mask = continuation_encoded.get("attention_mask")
                        if continuation_mask is None:
                            continuation_mask = torch.ones_like(continuation_ids, dtype=torch.int64)
                        else:
                            continuation_mask = continuation_mask.to(model.device)
                        if int(continuation_ids.shape[1]) + answer_budget > max_context_tokens:
                            guidance = (
                                "increase the SPALMER native context limit or reduce the thinking budget"
                                if self.model_family == "spalmer"
                                else "select a YaRN context extension or reduce the thinking budget"
                            )
                            raise ValueError(
                                "forced benchmark answer continuation exceeds the configured "
                                f"context: prompt {int(continuation_ids.shape[1])} + answer "
                                f"{answer_budget} > {max_context_tokens}; {guidance}"
                            )
                        continuation_kwargs = {
                            key: value
                            for key, value in kwargs.items()
                            if key not in {"input_ids", "attention_mask", "stopping_criteria"}
                        }
                        continuation_kwargs.update(
                            input_ids=continuation_ids,
                            attention_mask=continuation_mask,
                            max_new_tokens=answer_budget,
                        )
                        if accumulator is not None:
                            # Phase two performs a full prefill of the logical
                            # prompt+reasoning prefix, so retaining phase one's
                            # hooks would double-count all of those routes.
                            accumulator.reset()
                        end_prompt_route(model)
                        begin_prompt_route(model)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        generated, adaptive_phase = _adaptive_generation_phase(
                            model,
                            continuation_kwargs,
                            adaptive_runtime,
                            settings=self.settings,
                            cuda=self.engine.cuda,
                        )
                        if adaptive_phase is not None:
                            adaptive_phases.append(adaptive_phase)
                        answer_tokens = int(generated.shape[1] - continuation_ids.shape[1])
                        thinking_tokens = phase_one_count if close_offset is None else classified_thinking
                        model_generated_tokens += answer_tokens
                        finish_completion_ids = generated[0, continuation_ids.shape[1] :]
                        answer_text = tokenizer.decode(
                            finish_completion_ids,
                            skip_special_tokens=True,
                        ).strip()
                        response_text = (f"<think>\n{reasoning_text}\n</think>\n\n{answer_text}").strip()
                    elif close_offset is not None:
                        thinking_tokens = classified_thinking
                        answer_tokens = classified_answer
        finally:
            end_prompt_route(model)
            if accumulator is not None:
                routing = accumulator.close()
            if adaptive_phases:
                adaptive_receipt = dict(adaptive_phases[-1])
                adaptive_receipt["generation_phases"] = adaptive_phases
                routing = dict(routing or {})
                routing["adaptive"] = adaptive_receipt
            telemetry_getter = getattr(self.engine, "routing_telemetry", None)
            native_routing = telemetry_getter() if callable(telemetry_getter) else None
            if native_routing is not None:
                routing = native_routing
        if self.engine.cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        completion_tokens = model_generated_tokens
        if response_text is None:
            finish_completion_ids = generated[0, prompt_tokens:]
            response_text = tokenizer.decode(
                finish_completion_ids,
                skip_special_tokens=True,
            ).strip()
            if prompt_has_open_thinking:
                response_text = f"<think>\n{response_text}"
        eos_id = tokenizer.eos_token_id
        finish_reason = (
            "stop"
            if finish_completion_ids.numel()
            and eos_id is not None
            and int(finish_completion_ids[-1].item()) == eos_id
            else "length"
        )
        return GenerationResponse(
            text=response_text,
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=completion_tokens / elapsed if elapsed > 0 else None,
            finish_reason=finish_reason,
            routing=routing,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            thinking_budget_exhausted=thinking_budget_exhausted,
        )

    def close(self) -> None:
        self.engine.unload()
