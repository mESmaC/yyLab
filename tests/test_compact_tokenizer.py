from __future__ import annotations

import pytest

from yylab.train.compact_tokenizer import CompactTokenizerSpec, train_compact_tokenizer


def test_compact_tokenizer_roundtrip_and_chat_template(tmp_path):
    tokenizer = train_compact_tokenizer(
        [
            "The quick brown fox jumps over the lazy dog.",
            "def add(left, right):\n    return left + right",
            "∀ x ∈ ℝ, x² ≥ 0.",
        ],
        tmp_path,
        CompactTokenizerSpec(vocab_size=384, min_frequency=1, max_documents=3),
    )

    text = "code: λ = 7; emoji: 🧪"
    assert tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True) == text
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<|user|>\nhello</s>" in rendered
    assert rendered.endswith("<|assistant|>\n")
    assert (tmp_path / "tokenizer.json").is_file()


def test_compact_tokenizer_spec_rejects_vocabulary_without_byte_coverage():
    with pytest.raises(ValueError, match="byte coverage"):
        CompactTokenizerSpec(vocab_size=128).validate()


def test_minimal_causal_profile_is_257_entries_and_byte_lossless(tmp_path):
    tokenizer = train_compact_tokenizer(
        ["ASCII ＦＵＬＬＷＩＤＴＨ ① λ 数学 🧪"],
        tmp_path,
        CompactTokenizerSpec(
            vocab_size=257,
            min_frequency=1,
            max_documents=1,
            special_token_profile="causal",
            normalization="none",
        ),
    )

    text = "ASCII ＦＵＬＬＷＩＤＴＨ ① λ 数学 🧪"
    assert len(tokenizer) == 257
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.decode(tokenizer.encode(text), skip_special_tokens=False) == text


def test_chat_and_custom_profiles_enforce_exact_byte_alphabet_floor():
    CompactTokenizerSpec(vocab_size=264, special_token_profile="chat").validate()
    CompactTokenizerSpec(
        vocab_size=258,
        special_token_profile="custom",
        custom_special_tokens=("<eos>", "<mask>"),
    ).validate()
    with pytest.raises(ValueError, match="at least 264"):
        CompactTokenizerSpec(vocab_size=263, special_token_profile="chat").validate()


def test_legacy_lowercase_flag_maps_to_explicit_normalization_mode():
    assert CompactTokenizerSpec(lowercase=True).resolved_normalization() == "nfkc_lowercase"


def test_training_stream_honors_byte_cap_and_reports_progress(tmp_path):
    seen: list[tuple[int, int]] = []
    train_compact_tokenizer(
        ["abcdef", "ghijkl"],
        tmp_path,
        CompactTokenizerSpec(
            vocab_size=257,
            min_frequency=1,
            max_documents=10,
            max_training_bytes=8,
            special_token_profile="causal",
        ),
        progress=lambda docs, size: seen.append((docs, size)),
    )
    assert seen[-1] == (2, 8)
