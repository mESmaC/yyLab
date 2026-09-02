from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

CHAT_SPECIAL_TOKENS: tuple[str, ...] = (
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
)

CAUSAL_SPECIAL_TOKENS: tuple[str, ...] = ("<|endoftext|>",)
SPECIAL_TOKEN_PROFILES = frozenset({"causal", "chat", "custom"})
NORMALIZATION_MODES = frozenset({"none", "nfkc", "nfkc_lowercase"})
# Backward-compatible public name used by earlier compact-tokenizer callers.
DEFAULT_SPECIAL_TOKENS = CHAT_SPECIAL_TOKENS

GENERIC_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|' + message['role'] + '|>\n' + message['content'] + eos_token + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|assistant|>\n' }}{%- endif %}"""


@dataclass(frozen=True, slots=True)
class CompactTokenizerSpec:
    """Deterministic byte-level BPE tokenizer training contract."""

    vocab_size: int = 2048
    min_frequency: int = 2
    max_documents: int = 100_000
    max_training_bytes: int = 0
    lowercase: bool = False
    normalization: str = "none"
    special_token_profile: str = "chat"
    custom_special_tokens: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, values: dict | None) -> CompactTokenizerSpec:
        raw = dict(values or {})
        allowed = set(cls.__dataclass_fields__)
        selected = {key: raw[key] for key in raw.keys() & allowed}
        if "custom_special_tokens" in selected:
            selected["custom_special_tokens"] = tuple(selected["custom_special_tokens"] or ())
        return cls(**selected)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> CompactTokenizerSpec:
        # ByteLevel's complete alphabet consumes 256 entries before merges.
        minimum = 256 + len(self.resolved_special_tokens())
        if int(self.vocab_size) < minimum:
            raise ValueError(
                f"compact tokenizer vocabulary must be at least {minimum} for byte coverage "
                f"with the {self.special_token_profile!r} special-token profile"
            )
        if int(self.min_frequency) < 1:
            raise ValueError("compact tokenizer minimum frequency must be positive")
        if int(self.max_documents) < 1:
            raise ValueError("compact tokenizer document cap must be positive")
        if int(self.max_training_bytes) < 0:
            raise ValueError("compact tokenizer byte cap cannot be negative")
        self.resolved_normalization()
        return self

    def resolved_normalization(self) -> str:
        mode = str(self.normalization or "none").strip().casefold().replace("+", "_")
        aliases = {"lowercase": "nfkc_lowercase", "nfkc_lower": "nfkc_lowercase"}
        mode = aliases.get(mode, mode)
        # ``lowercase`` predates the explicit normalization selector. Preserve
        # old recipes while allowing new byte-lossless tokenizers by default.
        if self.lowercase and mode == "none":
            mode = "nfkc_lowercase"
        if mode not in NORMALIZATION_MODES:
            choices = ", ".join(sorted(NORMALIZATION_MODES))
            raise ValueError(f"unknown tokenizer normalization {mode!r}; expected one of: {choices}")
        return mode

    def resolved_special_tokens(self) -> tuple[str, ...]:
        profile = str(self.special_token_profile or "chat").strip().casefold()
        if profile not in SPECIAL_TOKEN_PROFILES:
            choices = ", ".join(sorted(SPECIAL_TOKEN_PROFILES))
            raise ValueError(f"unknown special-token profile {profile!r}; expected one of: {choices}")
        if profile == "causal":
            values = CAUSAL_SPECIAL_TOKENS
        elif profile == "chat":
            values = CHAT_SPECIAL_TOKENS
        else:
            values = tuple(str(value).strip() for value in self.custom_special_tokens)
            if not values or any(not value for value in values):
                raise ValueError("custom special-token profile requires at least one non-empty token")
        # Stable de-duplication is important because the count determines the
        # smallest valid byte-complete vocabulary.
        return tuple(dict.fromkeys(values))


def _bounded_texts(
    texts: Iterable[str],
    maximum: int,
    *,
    lowercase: bool,
    max_bytes: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[str]:
    yielded = 0
    consumed_bytes = 0
    for value in texts:
        if yielded >= maximum:
            return
        text = str(value)
        if not text.strip():
            continue
        text = text.lower() if lowercase else text
        raw = text.encode("utf-8")
        if max_bytes:
            remaining = max_bytes - consumed_bytes
            if remaining <= 0:
                return
            if len(raw) > remaining:
                text = raw[:remaining].decode("utf-8", errors="ignore")
                raw = text.encode("utf-8")
                if not text:
                    return
        yielded += 1
        consumed_bytes += len(raw)
        if progress is not None:
            progress(yielded, consumed_bytes)
        yield text
        if max_bytes and consumed_bytes >= max_bytes:
            return


def train_compact_tokenizer(
    texts: Iterable[str],
    output_dir: str | Path,
    spec: CompactTokenizerSpec | dict | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
):
    """Train and save a self-contained fast byte-level BPE tokenizer.

    The iterator is consumed once and bounded by ``max_documents`` so this can
    run directly over yyLab's streaming corpus reader without materializing a
    corpus in RAM. The complete byte alphabet prevents unknown-token holes in
    arbitrary code, math, and multilingual UTF-8 text.
    """

    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    resolved = spec if isinstance(spec, CompactTokenizerSpec) else CompactTokenizerSpec.from_dict(spec)
    resolved.validate()
    special_tokens = resolved.resolved_special_tokens()
    unk_token = "<unk>" if "<unk>" in special_tokens else None
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    normalization = resolved.resolved_normalization()
    if normalization == "nfkc_lowercase":
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    elif normalization == "nfkc":
        tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(resolved.vocab_size),
        min_frequency=int(resolved.min_frequency),
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        _bounded_texts(
            texts,
            int(resolved.max_documents),
            # Normalization belongs to the tokenizer pipeline so the exact same
            # transformation is applied during both training and inference.
            lowercase=False,
            max_bytes=int(resolved.max_training_bytes),
            progress=progress,
        ),
        trainer=trainer,
    )
    if str(resolved.special_token_profile).strip().casefold() == "chat":
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            additional_special_tokens=list(CHAT_SPECIAL_TOKENS[4:]),
            clean_up_tokenization_spaces=False,
        )
        fast.chat_template = GENERIC_CHAT_TEMPLATE
    else:
        # For causal and custom profiles the first configured token is a shared
        # EOS/PAD marker. This keeps a truly tiny byte-complete tokenizer at 257
        # entries while remaining directly usable by Transformers trainers.
        shared_end = special_tokens[0]
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token=shared_end,
            pad_token=shared_end,
            additional_special_tokens=list(special_tokens[1:]),
            clean_up_tokenization_spaces=False,
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(destination)
    return fast
