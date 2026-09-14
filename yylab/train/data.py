from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import db
from ..artifacts import corpus_bucket_dir, corpus_dir
from ..ingest.dedup import sha256_hex
from ..ingest.pipeline import _iter_shards
from ..ingest.records import (
    SOURCE_SPLITS,
    canonical_source_split,
    infer_source_split,
    normalize_chat_messages,
)

CHARS_PER_TOKEN = 3.5
DEFAULT_SHUFFLE_BUFFER = 128
DEFAULT_SHUFFLE_BUFFER_BYTES = 64 * 2**20
DEFAULT_TOKENIZE_CHUNK = 8
# Keep each fast-tokenizer input comfortably below the very large raw chunks
# accepted by corpus ingestion.  Packing still joins every returned token into
# fixed model sequences, so this bounds transient CPU/RAM use without dropping
# corpus content.
DEFAULT_TEXT_CHUNK_CHARS = 65_536
CORPUS_MIX_CONCATENATE = "concatenate"
CORPUS_MIX_ROUND_ROBIN = "round_robin"
CORPUS_MIX_TOKEN_BALANCED = "token_balanced"
CORPUS_MIX_MODES = frozenset({CORPUS_MIX_CONCATENATE, CORPUS_MIX_ROUND_ROBIN, CORPUS_MIX_TOKEN_BALANCED})
OBJECTIVE_CAUSAL_LM = "causal_lm"
OBJECTIVE_CHAT_SFT = "chat_sft"
TRAINING_OBJECTIVES = frozenset({OBJECTIVE_CAUSAL_LM, OBJECTIVE_CHAT_SFT})
MAX_CHAT_FAILURE_REASONS = 12
CORPUS_ID_FIELD = "_yylab_corpus_id"
PREPARED_FIELD = "_yylab_prepared"


def _document_chars(record: dict) -> int:
    prepared = record.get(PREPARED_FIELD)
    return int(prepared["chars"]) if prepared else len(str(record.get("text") or ""))


def _shuffle_bytes(record: dict) -> int:
    prepared = record.get(PREPARED_FIELD)
    if prepared:
        return int(prepared["shuffle_bytes"])
    return sys.getsizeof(record) + sys.getsizeof(record.get("text", ""))


_ROLE_HEADER = re.compile(r"(?im)^(system|user|assistant)[ \t]*:[ \t]*")


def normalize_training_objective(value: object) -> str:
    """Resolve the base-training loss objective while keeping old recipes causal."""

    objective = str(value or OBJECTIVE_CAUSAL_LM).strip().casefold().replace("-", "_")
    aliases = {
        "causal": OBJECTIVE_CAUSAL_LM,
        "clm": OBJECTIVE_CAUSAL_LM,
        "pretrain": OBJECTIVE_CAUSAL_LM,
        "continued_pretrain": OBJECTIVE_CAUSAL_LM,
        "sft": OBJECTIVE_CHAT_SFT,
        "chat": OBJECTIVE_CHAT_SFT,
    }
    objective = aliases.get(objective, objective)
    if objective not in TRAINING_OBJECTIVES:
        choices = ", ".join(sorted(TRAINING_OBJECTIVES))
        raise ValueError(f"unknown training objective {objective!r}; expected one of: {choices}")
    return objective


def parse_role_transcript(text: str) -> list[dict[str, str]]:
    """Parse flattened ``role:`` blocks without treating inline colons as roles.

    Corpora produced by yyLab store chat records as ordinary text. A role header
    is recognized only at the start of a line, case-insensitively. Text before
    the first header and empty blocks make the document ineligible rather than
    silently turning unstructured pretraining text into an SFT example.
    """

    if not isinstance(text, str) or not text.strip():
        return []
    matches = list(_ROLE_HEADER.finditer(text))
    if not matches or text[: matches[0].start()].strip(" \t\r\n\ufeff"):
        return []
    messages: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end() : end].strip()
        if not content:
            return []
        messages.append({"role": match.group(1).casefold(), "content": content})
    return messages


def _flat_token_values(value: Any, *, field: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list) or any(isinstance(item, (list, tuple)) for item in value):
        raise ValueError(f"chat template returned an unsupported {field} shape")
    return [int(item) for item in value]


def _template_input_ids(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    return _flat_token_values(rendered, field="input_ids")


def _template_text(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str):
        raise ValueError("chat template did not return text when tokenize=False")
    return rendered


def _offset_assistant_example(
    tokenizer,
    messages: list[dict[str, str]],
    assistant_index: int,
) -> tuple[list[int], list[int]]:
    """Use exact-render offsets when a token prefix ends on a BPE boundary."""

    target_messages = messages[: assistant_index + 1]
    rendered = _template_text(tokenizer, target_messages)
    prompt = _template_text(
        tokenizer,
        messages[:assistant_index],
        add_generation_prompt=True,
    )
    if not rendered.startswith(prompt) or len(rendered) <= len(prompt):
        raise ValueError("chat template changes the context before the assistant response")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer did not return an encoding with offsets")
    input_ids = _flat_token_values(encoded.get("input_ids"), field="input_ids")
    offsets = encoded.get("offset_mapping")
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if isinstance(offsets, list) and len(offsets) == 1 and isinstance(offsets[0], list):
        offsets = offsets[0]
    if not isinstance(offsets, list) or len(offsets) != len(input_ids):
        raise ValueError("tokenizer returned invalid offsets for rendered chat")
    exact_ids = _template_input_ids(tokenizer, target_messages)
    if input_ids != exact_ids:
        raise ValueError("tokenizer offsets do not align with exact chat-template tokens")
    first_content_token = None
    content_start = len(prompt)
    for index, offset in enumerate(offsets):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("tokenizer returned an invalid token offset")
        end = int(offset[1])
        if end > content_start:
            first_content_token = index
            break
    if first_content_token is None:
        raise ValueError("tokenizer produced no assistant content tokens")
    labels = [-100] * first_content_token + input_ids[first_content_token:]
    return input_ids, labels


def _assistant_turn_example(
    tokenizer,
    messages: list[dict[str, str]],
    assistant_index: int,
) -> tuple[list[int], list[int]]:
    """Build one record whose sole target is the selected assistant turn."""

    target_messages = messages[: assistant_index + 1]
    input_ids = _template_input_ids(tokenizer, target_messages)
    try:
        prompt_ids = _template_input_ids(
            tokenizer,
            messages[:assistant_index],
            add_generation_prompt=True,
        )
    except (IndexError, TypeError, ValueError):
        return _offset_assistant_example(tokenizer, messages, assistant_index)
    if len(prompt_ids) >= len(input_ids) or input_ids[: len(prompt_ids)] != prompt_ids:
        return _offset_assistant_example(tokenizer, messages, assistant_index)
    # The generation prompt owns the assistant role/header. Everything the
    # exact completed render adds after it is model output, including thinking
    # scaffolding and the tokenizer's end-of-turn/EOS suffix.
    return input_ids, [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]


def tokenize_chat_examples(
    tokenizer,
    messages: list[dict[str, str]],
) -> list[tuple[list[int], list[int]]]:
    """Render one independent, record-aware training example per assistant."""

    if not messages or not any(message.get("role") == "assistant" for message in messages):
        raise ValueError("chat SFT transcript must contain at least one assistant block")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("chat SFT requires a tokenizer with an apply_chat_template method")
    examples = [
        _assistant_turn_example(tokenizer, messages, index)
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if not examples or any(not any(label != -100 for label in labels) for _, labels in examples):
        raise ValueError("tokenizer chat template produced no supervised assistant tokens")
    return examples


def tokenize_chat_messages(
    tokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    """Render one full chat and label assistant content through its terminator.

    Training uses :func:`tokenize_chat_examples` so turns never share a model
    row. This compatibility helper retains a single-render view for diagnostics.
    """

    if not messages or not any(message.get("role") == "assistant" for message in messages):
        raise ValueError("chat SFT transcript must contain at least one assistant block")
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("chat SFT requires a tokenizer with an apply_chat_template method")

    input_ids = _template_input_ids(tokenizer, messages)
    labels = [-100] * len(input_ids)
    raw_eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = (
        {int(value) for value in raw_eos if value is not None}
        if isinstance(raw_eos, (list, tuple, set, frozenset))
        else ({int(raw_eos)} if raw_eos is not None else set())
    )
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    for ordinal, assistant_index in enumerate(assistant_indices):
        try:
            prompt_ids = _template_input_ids(
                tokenizer,
                messages[:assistant_index],
                add_generation_prompt=True,
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"could not locate assistant turn {ordinal + 1}: {exc}") from exc
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "chat template changes the context before an assistant generation prompt"
            )
        start = len(prompt_ids)
        end = next(
            (index + 1 for index in range(start, len(input_ids)) if input_ids[index] in eos_ids),
            len(input_ids) if ordinal == len(assistant_indices) - 1 else None,
        )
        if end is None or end <= start:
            raise ValueError(f"could not locate the end of assistant turn {ordinal + 1}")
        labels[start:end] = input_ids[start:end]
    if not input_ids or not any(label != -100 for label in labels):
        raise ValueError("tokenizer chat template produced no supervised assistant tokens")
    return input_ids, labels


def normalize_corpus_ids(corpus_ids: object) -> list[str]:
    """Return a deterministic, ordered, duplicate-free corpus selection.

    Older base-run payloads may contain one corpus ID directly instead of a
    list. Treating that string as an iterable would stream one-character IDs,
    so normalize it at the data boundary as well as in the worker. JSON lists
    keep their user-selected order; unordered Python sets are sorted so tests
    and programmatic callers remain reproducible.
    """

    if corpus_ids is None:
        values: Iterable[object] = ()
    elif isinstance(corpus_ids, str):
        values = (corpus_ids,)
    elif isinstance(corpus_ids, (set, frozenset)):
        values = sorted(corpus_ids, key=str)
    elif isinstance(corpus_ids, Iterable):
        values = corpus_ids
    else:
        raise TypeError("corpora must be a corpus ID or an iterable of corpus IDs")

    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("every corpus ID must be a string")
        corpus_id = value.strip()
        if corpus_id and corpus_id not in seen:
            selected.append(corpus_id)
            seen.add(corpus_id)
    return selected


def normalize_corpus_mix(value: object) -> str:
    """Resolve an explicit mixing strategy, defaulting old runs to concatenation."""

    mode = str(value or CORPUS_MIX_CONCATENATE).strip().casefold()
    if mode not in CORPUS_MIX_MODES:
        choices = ", ".join(sorted(CORPUS_MIX_MODES))
        raise ValueError(f"unknown corpus mix mode {mode!r}; expected one of: {choices}")
    return mode


def normalize_source_splits(value: object) -> list[str]:
    """Return a deterministic source-split filter; an empty list means all splits."""

    if value is None:
        values: Iterable[object] = ()
    elif isinstance(value, str):
        values = (value,)
    elif isinstance(value, (set, frozenset)):
        values = sorted(value, key=str)
    elif isinstance(value, Iterable):
        values = value
    else:
        raise TypeError("splits must be a string or an iterable of strings")
    selected: list[str] = []
    seen: set[str] = set()
    for raw in values:
        split = canonical_source_split(raw)
        if split is None:
            choices = ", ".join(sorted(SOURCE_SPLITS))
            raise ValueError(f"unknown source split {raw!r}; expected one of: {choices}")
        if split not in seen:
            selected.append(split)
            seen.add(split)
    return selected


def corpus_data_provenance(
    corpus_ids: object,
    domains: list[str] | None = None,
    *,
    corpus_mix: object = CORPUS_MIX_CONCATENATE,
    splits: object = None,
    cross_corpus_dedup: bool = True,
    corpus_weights: Mapping[str, float] | None = None,
    source_max_ratio: float | None = None,
) -> dict[str, Any]:
    """Snapshot the selected corpus identities and aggregate intake statistics."""

    selected_ids = normalize_corpus_ids(corpus_ids)
    selected_domains = sorted({str(domain) for domain in domains or () if str(domain)})
    domain_set = set(selected_domains)
    selected_splits = normalize_source_splits(splits)
    split_set = set(selected_splits)
    corpora: list[dict[str, Any]] = []
    total_docs = 0
    total_chars = 0
    for corpus_id in selected_ids:
        row = db.q1(
            "SELECT id, name, source_path, status, kind, verdict, stats_json "
            "FROM corpora WHERE id=?",
            (corpus_id,),
        )
        entry: dict[str, Any] = {"id": corpus_id}
        if row:
            import orjson

            stats = orjson.loads(row.get("stats_json") or b"{}")
            domain_stats = stats.get("domains", {})
            split_stats = stats.get("splits", {})

            def domain_totals(values: Mapping[str, Any]) -> tuple[int, int]:
                chosen = [
                    counts
                    for domain, counts in values.items()
                    if not domain_set or domain in domain_set
                ]
                return (
                    sum(int(counts.get("docs", 0)) for counts in chosen),
                    sum(int(counts.get("chars", 0)) for counts in chosen),
                )

            if split_set and split_stats:
                docs = 0
                chars = 0
                for split, values in split_stats.items():
                    if split not in split_set or not isinstance(values, Mapping):
                        continue
                    split_domains = values.get("domains")
                    if domain_set and isinstance(split_domains, Mapping):
                        split_docs, split_chars = domain_totals(split_domains)
                    elif domain_set:
                        split_docs = split_chars = 0
                    else:
                        split_docs = int(values.get("docs", 0))
                        split_chars = int(values.get("chars", 0))
                    docs += split_docs
                    chars += split_chars
            elif split_set and "unspecified" not in split_set:
                # Legacy shard lines have no split key and are therefore read as
                # ``unspecified``. Their catalog statistics follow that contract.
                docs = chars = 0
            else:
                docs, chars = domain_totals(domain_stats)
                if not domain_stats and not domain_set:
                    docs = int(stats.get("docs_written", 0))
                    chars = int(stats.get("chars_written", 0))
            legacy_docs = int(stats.get("docs_written", 0)) or sum(
                int(values.get("docs", 0))
                for values in domain_stats.values()
                if isinstance(values, Mapping)
            )
            available_splits = (
                sorted(str(split) for split in split_stats)
                if split_stats
                else (["unspecified"] if legacy_docs else [])
            )
            cdas_catalog = dict(stats.get("cdas") or {})
            selection_profile = str(
                (stats.get("ingest_config") or {}).get("selection_profile") or ""
            )
            kind = str(row.get("kind") or "ingested")
            verdict = str(row.get("verdict") or "")
            gccf_catalog = dict(stats.get("gccf") or {})
            source_index = dict(stats.get("source_index") or {})
            if gccf_catalog:
                catalog_status = "gccf_catalogued"
            elif source_index:
                catalog_status = "raw_source_index"
            elif cdas_catalog:
                catalog_status = "cdas_catalogued"
            elif kind == "approved":
                catalog_status = "approved_store"
            elif verdict == "approved":
                catalog_status = "legacy_reviewed"
            else:
                catalog_status = "legacy_unreviewed"
            entry.update(
                name=str(row.get("name") or corpus_id),
                source_path=str(row.get("source_path") or ""),
                status=str(row.get("status") or ""),
                kind=kind,
                verdict=verdict,
                estimated_docs=docs,
                estimated_chars=chars,
                available_splits=available_splits,
                selection_profile=selection_profile,
                catalog_status=catalog_status,
                cdas_catalog=cdas_catalog,
                gccf=gccf_catalog,
                source_index=source_index,
            )
            total_docs += docs
            total_chars += chars
        else:
            # Programmatic and legacy runs may point at an artifact whose DB row
            # has since been removed. Keep its ID in the durable provenance; the
            # stream will still fail naturally if no shards exist.
            entry["missing_catalog_entry"] = True
        corpora.append(entry)
    return {
        "schema_version": 3,
        "corpus_ids": selected_ids,
        "corpus_count": len(selected_ids),
        "corpus_mix": normalize_corpus_mix(corpus_mix),
        "cross_corpus_dedup": bool(cross_corpus_dedup),
        "corpus_weights": {
            str(corpus_id): float(weight) for corpus_id, weight in (corpus_weights or {}).items()
        },
        "source_max_ratio": (
            float(source_max_ratio) if source_max_ratio is not None else None
        ),
        "domains": selected_domains,
        "splits": selected_splits,
        "estimated_docs": total_docs,
        "estimated_chars": total_chars,
        "corpora": corpora,
    }


def corpus_training_preflight(
    corpus_ids: object,
    domains: list[str] | None = None,
    *,
    corpus_mix: object = CORPUS_MIX_CONCATENATE,
    splits: object = None,
    cross_corpus_dedup: bool = True,
    corpus_weights: Mapping[str, float] | None = None,
    source_max_ratio: float | None = None,
) -> dict[str, Any]:
    """Return one shared data-lineage preflight for the UI and worker."""

    provenance = corpus_data_provenance(
        corpus_ids,
        domains,
        corpus_mix=corpus_mix,
        splits=splits,
        cross_corpus_dedup=cross_corpus_dedup,
        corpus_weights=corpus_weights,
        source_max_ratio=source_max_ratio,
    )
    errors: list[str] = []
    warnings: list[str] = []
    for corpus in provenance["corpora"]:
        name = str(corpus.get("name") or corpus.get("id") or "unknown corpus")
        if corpus.get("missing_catalog_entry"):
            errors.append(f"Missing corpus: {name}")
            continue
        if str(corpus.get("status") or "") != "ready":
            errors.append(f"Corpus {name!r} is not ready")
        if str(corpus.get("kind") or "").casefold() == "rejected" or str(
            corpus.get("verdict") or ""
        ).casefold() == "rejected":
            errors.append(f"Corpus {name!r} is rejected")
        catalog_status = str(corpus.get("catalog_status") or "")
        if catalog_status in {"legacy_reviewed", "legacy_unreviewed", "approved_store"}:
            warnings.append(
                f"Corpus {name!r} has no CDAS intake catalog ({catalog_status.replace('_', ' ')})"
            )
        if corpus.get("selection_profile") == "preserve" and not corpus.get("gccf"):
            warnings.append(f"Corpus {name!r} used the preserve profile; no artifact policy was applied")
        issues = int((corpus.get("cdas_catalog") or {}).get("issues") or 0)
        if issues:
            warnings.append(f"Corpus {name!r} has {issues:,} catalog issue(s)")
    if (
        provenance["corpora"]
        and not errors
        and int(provenance.get("estimated_docs") or 0) < 1
    ):
        errors.append("Selected corpus/domain/split filters contain no estimated training documents")
    return {"provenance": provenance, "errors": errors, "warnings": warnings}


def tokenizer_stats_key(model_type: str | None) -> str | None:
    """Return the closest intake-tokenizer counter for a model family."""

    normalized = (model_type or "").casefold().replace("_", "").replace("-", "")
    if normalized.startswith("qwen"):
        return "qwen25"
    if normalized.startswith("llama"):
        return "llama3"
    if normalized.startswith("mistral"):
        return "mistral"
    if normalized.startswith("phi3"):
        return "phi3"
    if normalized.startswith("gemma"):
        return "gemma3"
    if normalized.startswith("gpt2"):
        return "gpt2"
    return None


@dataclass(frozen=True)
class ExpertDataPlan:
    """One isolated expert's deterministic slice of the selected corpora."""

    name: str
    domains: tuple[str, ...]
    estimated_chars: int
    shard_index: int | None = None
    num_shards: int | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domains": list(self.domains),
            "estimated_chars": self.estimated_chars,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ExpertDataPlan:
        return cls(
            name=str(value["name"]),
            domains=tuple(value.get("domains") or ()),
            estimated_chars=max(0, int(value.get("estimated_chars") or 0)),
            shard_index=(None if value.get("shard_index") is None else int(value["shard_index"])),
            num_shards=(None if value.get("num_shards") is None else int(value["num_shards"])),
        )


def corpus_data_dir(corpus_id: str) -> Path:
    """Live ``data`` directory of a corpus, resolved through its generation pointer."""

    return corpus_bucket_dir(corpus_id, "data", root=corpus_dir(corpus_id))


def corpus_shards(corpus_id: str, data_dir: Path | None = None) -> list[str]:
    cdir = data_dir if data_dir is not None else corpus_data_dir(corpus_id)
    return sorted(p.name for p in cdir.glob("shard-*.jsonl.zst"))


def iter_corpus_docs(
    corpus_ids: object,
    domains: list[str] | None = None,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
    seed: int = 0,
    shard_index: int | None = None,
    num_shards: int | None = None,
    shuffle_buffer_bytes: int = DEFAULT_SHUFFLE_BUFFER_BYTES,
    corpus_mix: str = CORPUS_MIX_CONCATENATE,
    splits: object = None,
    cross_corpus_dedup: bool = True,
    corpus_weights: Mapping[str, float] | None = None,
    source_max_ratio: float | None = None,
    annotate_source: bool = False,
    corpus_reader: Callable[[str], Iterator[dict]] | None = None,
) -> Iterator[dict]:
    if (shard_index is None) != (num_shards is None):
        raise ValueError("shard_index and num_shards must be supplied together")
    if num_shards is not None and (num_shards < 1 or not 0 <= shard_index < num_shards):
        raise ValueError("document shard must be within [0, num_shards)")
    selected_corpora = normalize_corpus_ids(corpus_ids)
    mix_mode = normalize_corpus_mix(corpus_mix)
    domain_set = set(domains) if domains else None
    selected_splits = normalize_source_splits(splits)
    split_set = set(selected_splits) if selected_splits else None
    weights = dict(corpus_weights or {})
    for corpus_id, value in weights.items():
        if corpus_id not in selected_corpora:
            continue
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("every selected corpus weight must be finite and positive")
    if source_max_ratio is not None:
        source_max_ratio = float(source_max_ratio)
        if not math.isfinite(source_max_ratio) or not 0 < source_max_ratio <= 1:
            raise ValueError("source_max_ratio must be in (0, 1]")

    def corpus_stream(corpus_id: str) -> Iterator[dict]:
        # Resolve the generation once per stream so a re-import published
        # mid-run cannot mix old and new shards into the same epoch.
        data_dir = corpus_data_dir(corpus_id)
        if corpus_reader is not None:
            records = corpus_reader(corpus_id)
        elif (corpus_dir(corpus_id) / "text-source.json").exists():
            from ..ingest.text_source import iter_text_source_corpus

            records = iter_text_source_corpus(corpus_id, splits=selected_splits or None)
        elif (corpus_dir(corpus_id) / "gccf-source.json").exists():
            from ..ingest.gccf import iter_source_corpus

            records = iter_source_corpus(corpus_id, splits=selected_splits or None)
        else:
            shards = corpus_shards(corpus_id, data_dir=data_dir)
            if not shards:
                return
            records = _iter_shards(data_dir, shards)
        try:
            for line in records:
                split = infer_source_split(line.get("source", ""), line.get("split"))
                if (domain_set is None or line.get("domain") in domain_set) and (
                    split_set is None or split in split_set
                ):
                    if line.get("split") != split or annotate_source:
                        line = dict(line)
                    if line.get("split") != split:
                        line["split"] = split
                    if annotate_source:
                        line[CORPUS_ID_FIELD] = corpus_id
                    yield line
        finally:
            close = getattr(records, "close", None)
            if close is not None:
                close()

    def shuffled_stream(
        items: Iterator[dict],
        *,
        rng: random.Random,
        max_items: int,
        max_bytes: int,
    ) -> Iterator[dict]:
        if max_items <= 1:
            yield from items
            return
        buf: list[dict] = []
        buffered_bytes = 0
        for item in items:
            buf.append(item)
            buffered_bytes += _shuffle_bytes(item)
            if len(buf) >= max_items or buffered_bytes >= max_bytes:
                index = rng.randrange(len(buf))
                selected = buf[index]
                buffered_bytes -= _shuffle_bytes(selected)
                yield selected
                buf[index] = buf[-1]
                buf.pop()
        rng.shuffle(buf)
        yield from buf

    def raw_stream() -> Iterator[dict]:
        if mix_mode == CORPUS_MIX_CONCATENATE:
            for corpus_id in selected_corpora:
                yield from corpus_stream(corpus_id)
            return

        # Each source is independently shuffled before either document- or
        # character-balanced scheduling.  This retains bounded memory and a
        # deterministic order for every seed.
        source_count = max(1, len(selected_corpora))
        per_source_items = max(1, shuffle_buffer // source_count)
        per_source_bytes = max(1, shuffle_buffer_bytes // source_count)
        active = [
            iter(
                shuffled_stream(
                    corpus_stream(corpus_id),
                    rng=random.Random(seed if source_count == 1 else f"{seed}:{index}:{corpus_id}"),
                    max_items=per_source_items,
                    max_bytes=per_source_bytes,
                )
            )
            for index, corpus_id in enumerate(selected_corpora)
        ]
        if mix_mode == CORPUS_MIX_ROUND_ROBIN:
            while active:
                remaining: list[Iterator[dict]] = []
                for iterator in active:
                    try:
                        yield next(iterator)
                    except StopIteration:
                        continue
                    remaining.append(iterator)
                active = remaining
            return

        # Weighted fair queuing by model-visible characters is a cheap online
        # proxy for tokens.  A giant SVG or generated source file therefore uses
        # up that corpus's credit instead of receiving the same scheduling weight
        # as one short prose record.  This directly prevents document-balanced
        # round robin from becoming token-dominated by one source.
        pending_items: dict[int, dict] = {}
        emitted = [0] * len(active)
        total_emitted = 0
        exhausted: set[int] = set()
        effective_weights = [float(weights.get(corpus_id, 1.0)) for corpus_id in selected_corpora]
        while len(exhausted) < len(active):
            for index, iterator in enumerate(active):
                if index in exhausted or index in pending_items:
                    continue
                try:
                    pending_items[index] = next(iterator)
                except StopIteration:
                    exhausted.add(index)
            if not pending_items:
                break
            ranked = sorted(
                pending_items,
                key=lambda index: (
                    emitted[index] / effective_weights[index],
                    index,
                ),
            )
            chosen: int | None = None
            if source_max_ratio is not None and len(active) > 1 and total_emitted:
                for index in ranked:
                    cost = max(1, _document_chars(pending_items[index]))
                    projected_share = (emitted[index] + cost) / (total_emitted + cost)
                    if projected_share <= source_max_ratio:
                        chosen = index
                        break
                if chosen is None:
                    if len(pending_items) < 2:
                        # Other sources are exhausted and this source has reached
                        # its explicit cap. Stopping is preferable to flooding
                        # the remaining token budget from one source.
                        break
                    # No whole pending document fits yet. Accumulate the smallest
                    # available counterweight instead of admitting one enormous
                    # document that would permanently dominate a capped run.
                    chosen = min(
                        pending_items,
                        key=lambda index: (
                            _document_chars(pending_items[index]),
                            emitted[index] / effective_weights[index],
                            index,
                        ),
                    )
            elif source_max_ratio is not None and len(active) > 1:
                chosen = min(
                    pending_items,
                    key=lambda index: (
                        _document_chars(pending_items[index]),
                        index,
                    ),
                )
            if chosen is None:
                chosen = ranked[0]
            item = pending_items.pop(chosen)
            cost = max(1, _document_chars(item))
            emitted[chosen] += cost
            total_emitted += cost
            yield item

    def stream() -> Iterator[dict]:
        eligible_index = 0
        seen_hashes: set[bytes] = set()
        for line in raw_stream():
            if cross_corpus_dedup:
                text = line.get("text")
                prepared = line.get(PREPARED_FIELD)
                if (prepared and not prepared["chars"]) or (
                    not prepared and (not isinstance(text, str) or not text)
                ):
                    continue
                # Hard suppression uses exact model-visible bytes. Normalized,
                # structural, and LSH identities remain evidence because case
                # folding can change code, symbols, and natural-language meaning.
                fingerprint = bytes.fromhex(prepared["text_sha256"] if prepared else sha256_hex(text))[:16]
                if fingerprint in seen_hashes:
                    continue
                seen_hashes.add(fingerprint)
            selected = num_shards is None or eligible_index % num_shards == shard_index
            eligible_index += 1
            if selected:
                yield line

    if mix_mode in {CORPUS_MIX_ROUND_ROBIN, CORPUS_MIX_TOKEN_BALANCED}:
        # Each source was shuffled independently above; preserving the strict
        # interleave here prevents a later global buffer from reclustering it.
        yield from stream()
        return
    if shuffle_buffer <= 1:
        yield from stream()
        return
    rng = random.Random(seed)
    buf: list[dict] = []
    buffered_bytes = 0
    for item in stream():
        buf.append(item)
        buffered_bytes += _shuffle_bytes(item)
        if len(buf) >= shuffle_buffer or buffered_bytes >= shuffle_buffer_bytes:
            i = rng.randrange(len(buf))
            selected = buf[i]
            buffered_bytes -= _shuffle_bytes(selected)
            yield selected
            buf[i] = buf[-1]
            buf.pop()
    rng.shuffle(buf)
    yield from buf


class PackedBatches:
    def __init__(
        self,
        docs: Iterable[dict],
        tokenize: Callable[[list[str]], list[list[int]]],
        seq_len: int,
        eos_token_id: int,
        micro_batch: int,
        max_batches: int | None = None,
        tokenize_chunk: int = DEFAULT_TOKENIZE_CHUNK,
        max_text_chars: int = DEFAULT_TEXT_CHUNK_CHARS,
    ) -> None:
        if seq_len < 1 or micro_batch < 1 or tokenize_chunk < 1 or max_text_chars < 1:
            raise ValueError("sequence, batch, tokenization chunk, and text chunk sizes must be positive")
        self.docs = docs
        self.tokenize = tokenize
        self.seq_len = seq_len
        self.eos = [eos_token_id] if eos_token_id is not None else []
        self.micro_batch = micro_batch
        self.max_batches = max_batches
        self.tokenize_chunk = tokenize_chunk
        self.max_text_chars = max_text_chars
        self.sequences_made = 0
        self.sequences_forwarded = 0
        self.tokens_seen = 0
        self.boundary_tokens = 0
        self.forwarded_tokens = 0
        self.docs_seen = 0
        self.source_docs_seen: Counter[str] = Counter()
        self.source_content_tokens: Counter[str] = Counter()
        self.source_boundary_tokens: Counter[str] = Counter()
        self.source_forwarded_tokens: Counter[str] = Counter()

    def data_stats(self) -> dict[str, Any]:
        rendered = self.tokens_seen + self.boundary_tokens
        target_slots = self.sequences_forwarded * max(self.seq_len - 1, 0)
        sources = sorted(
            set(self.source_docs_seen) | set(self.source_content_tokens) | set(self.source_forwarded_tokens)
        )
        return {
            "docs_seen": self.docs_seen,
            "rendered_tokens": self.tokens_seen,
            "sequences": self.sequences_forwarded,
            "tokens": {
                "rendered_content_tokens": self.tokens_seen,
                "inserted_document_boundary_tokens": self.boundary_tokens,
                "rendered_token_positions": rendered,
                "covered_rendered_positions": self.forwarded_tokens,
                "unforwarded_rendered_positions": max(
                    0,
                    rendered - self.forwarded_tokens,
                ),
                "forwarded_real_inputs": self.forwarded_tokens,
                "forwarded_model_slots": self.forwarded_tokens,
                "padding_inputs": 0,
                "packing_replay_inputs": 0,
                "model_target_slots": target_slots,
                "optimizer_supervised_targets": target_slots,
                "ignored_target_slots": 0,
            },
            "per_corpus": {
                source: {
                    "documents_seen": self.source_docs_seen[source],
                    "rendered_content_tokens": self.source_content_tokens[source],
                    "inserted_document_boundary_tokens": self.source_boundary_tokens[source],
                    "forwarded_real_inputs": self.source_forwarded_tokens[source],
                }
                for source in sources
            },
        }

    def __iter__(self) -> Iterator[dict]:
        carry: list[int] = []
        carry_start = 0
        batch: list[list[int]] = []
        batch_exposures: list[Counter[str]] = []
        # Keep the document boundary alongside each bounded raw-text chunk.
        # Tokenization may happen in several calls for one long document, but
        # packing must still add exactly one EOS for that original document.
        pending_texts: list[tuple[Any, bool, str]] = []
        source_runs: deque[list[Any]] = deque()
        batches_made = 0

        def append_source(source: str, count: int) -> None:
            if count < 1:
                return
            if source_runs and source_runs[-1][0] == source:
                source_runs[-1][1] += count
            else:
                source_runs.append([source, count])

        def consume_sources(count: int) -> Counter[str]:
            exposure: Counter[str] = Counter()
            remaining = count
            while remaining:
                if not source_runs:
                    raise RuntimeError("packing source attribution fell behind token carry")
                source, available = source_runs[0]
                take = min(remaining, int(available))
                exposure[str(source)] += take
                remaining -= take
                available -= take
                if available:
                    source_runs[0][1] = available
                else:
                    source_runs.popleft()
            return exposure

        def mark_forwarded(exposures: list[Counter[str]]) -> None:
            self.sequences_forwarded += len(exposures)
            tokens = len(exposures) * self.seq_len
            self.forwarded_tokens += tokens
            for exposure in exposures:
                self.source_forwarded_tokens.update(exposure)

        def flush_texts() -> Iterator[dict]:
            nonlocal carry, carry_start, batches_made
            if not pending_texts:
                return
            texts = [text for text, _is_doc_end, _source in pending_texts]
            boundaries = [is_doc_end for _text, is_doc_end, _source in pending_texts]
            sources = [source for _text, _is_doc_end, source in pending_texts]
            prepared = not isinstance(texts[0], str)
            encodings = [value.read_ids() for value in texts] if prepared else self.tokenize(texts)
            for ids, is_doc_end, source in zip(
                encodings,
                boundaries,
                sources,
                strict=True,
            ):
                self.tokens_seen += len(ids)
                self.source_content_tokens[source] += len(ids)
                carry.extend(ids)
                append_source(source, len(ids))
                if is_doc_end:
                    carry.extend(self.eos)
                    self.boundary_tokens += len(self.eos)
                    self.source_boundary_tokens[source] += len(self.eos)
                    append_source(source, len(self.eos))
                while len(carry) - carry_start >= self.seq_len:
                    end = carry_start + self.seq_len
                    batch.append(carry[carry_start:end])
                    batch_exposures.append(consume_sources(self.seq_len))
                    carry_start = end
                    self.sequences_made += 1
                    if len(batch) >= self.micro_batch:
                        result = {"input_ids": batch[: self.micro_batch]}
                        del batch[: self.micro_batch]
                        exposures = batch_exposures[: self.micro_batch]
                        del batch_exposures[: self.micro_batch]
                        batches_made += 1
                        mark_forwarded(exposures)
                        yield result
                        if self.max_batches is not None and batches_made >= self.max_batches:
                            pending_texts.clear()
                            return
                # Avoid repeatedly copying the whole remainder for every sequence,
                # while still releasing consumed token references on long documents.
                if carry_start and (carry_start >= 65_536 or carry_start * 2 >= len(carry)):
                    carry = carry[carry_start:]
                    carry_start = 0
            pending_texts.clear()

        for doc in self.docs:
            self.docs_seen += 1
            source = str(doc.get(CORPUS_ID_FIELD) or "unknown")
            self.source_docs_seen[source] += 1
            prepared = doc.get(PREPARED_FIELD)
            if prepared:
                chunks = prepared["units"]
                parts = ((chunk, index == len(chunks) - 1) for index, chunk in enumerate(chunks))
            else:
                text = doc["text"]
                parts = (
                    (text[start : start + self.max_text_chars], start + self.max_text_chars >= len(text))
                    for start in range(0, max(len(text), 1), self.max_text_chars)
                )
            for part, is_end in parts:
                pending_texts.append((part, is_end, source))
                if len(pending_texts) >= self.tokenize_chunk:
                    yield from flush_texts()
                    if self.max_batches is not None and batches_made >= self.max_batches:
                        return

        yield from flush_texts()
        if batch and (self.max_batches is None or batches_made < self.max_batches):
            mark_forwarded(batch_exposures)
            yield {"input_ids": list(batch)}


def render_chat_record(tokenizer, doc: dict) -> tuple[dict, list]:
    """Render one record identically for live packing and offline preparation."""
    raw_messages = doc.get("messages")
    status = {"structured": isinstance(raw_messages, list), "skipped": False, "error": None}
    try:
        if status["structured"]:
            messages = normalize_chat_messages(raw_messages)
            if not messages:
                raise ValueError("structured messages normalized to an empty chat")
        else:
            messages = parse_role_transcript(doc.get("text", ""))
        if not messages or not any(message["role"] == "assistant" for message in messages):
            if status["structured"]:
                raise ValueError("structured chat has no assistant turn")
            status["skipped"] = True
            return status, []
        return status, tokenize_chat_examples(tokenizer, messages)
    except Exception as exc:  # same per-record rejection contract as the live packer
        message = (str(exc).splitlines() or [""])[0].strip()[:180]
        status.update(
            skipped=True,
            error=f"{type(exc).__name__}: {message or 'unrenderable chat record'}",
        )
        return status, []


class ChatSFTBatches:
    """Stream fixed-length chat batches with loss restricted to assistant turns."""

    def __init__(
        self,
        docs: Iterable[dict],
        tokenizer,
        seq_len: int,
        micro_batch: int,
        max_batches: int | None = None,
    ) -> None:
        if seq_len < 2 or micro_batch < 1:
            raise ValueError("chat SFT sequences need at least two tokens and a positive batch size")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("chat SFT fixed-length batching requires a tokenizer pad or EOS token")
        self.docs = docs
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.micro_batch = int(micro_batch)
        self.max_batches = max_batches
        self.pad_token_id = int(pad_token_id)
        self.docs_seen = 0
        self.parseable_docs = 0
        self.skipped_docs = 0
        self.structured_docs = 0
        self.legacy_transcript_docs = 0
        self.unrenderable_docs = 0
        self.failure_reasons: dict[str, int] = {}
        self.failure_reason_overflow = 0
        self.examples_rendered = 0
        self.tokens_seen = 0
        self.supervised_tokens_seen = 0
        self.tokens_batched = 0
        self.supervised_tokens_batched = 0
        self.packing_replay_tokens = 0
        self.covered_rendered_tokens = 0
        self.padding_tokens = 0
        self.sequences_made = 0
        self.unsupervised_sequences_skipped = 0
        self.source_docs_seen: Counter[str] = Counter()
        self.source_examples_rendered: Counter[str] = Counter()
        self.source_rendered_tokens: Counter[str] = Counter()
        self.source_forwarded_tokens: Counter[str] = Counter()
        self.source_replay_tokens: Counter[str] = Counter()
        self.source_padding_tokens: Counter[str] = Counter()
        self.source_supervised_tokens: Counter[str] = Counter()

    def data_stats(self) -> dict[str, Any]:
        model_slots = self.sequences_made * self.seq_len
        target_slots = self.sequences_made * (self.seq_len - 1)
        sources = sorted(
            set(self.source_docs_seen) | set(self.source_rendered_tokens) | set(self.source_forwarded_tokens)
        )
        return {
            "docs_seen": self.docs_seen,
            "parseable_docs": self.parseable_docs,
            "skipped_docs": self.skipped_docs,
            "structured_docs": self.structured_docs,
            "legacy_transcript_docs": self.legacy_transcript_docs,
            "unrenderable_docs": self.unrenderable_docs,
            "failure_reasons": dict(self.failure_reasons),
            "failure_reason_overflow": self.failure_reason_overflow,
            "rendered_tokens": self.tokens_seen,
            "rendered_supervised_tokens": self.supervised_tokens_seen,
            "batched_tokens": self.tokens_batched,
            "batched_supervised_tokens": self.supervised_tokens_batched,
            "examples_rendered": self.examples_rendered,
            "sequences": self.sequences_made,
            "unsupervised_sequences_skipped": self.unsupervised_sequences_skipped,
            "tokens": {
                "rendered_content_tokens": self.tokens_seen,
                "inserted_document_boundary_tokens": 0,
                "rendered_token_positions": self.tokens_seen,
                "covered_rendered_positions": self.covered_rendered_tokens,
                "unforwarded_rendered_positions": max(
                    0,
                    self.tokens_seen - self.covered_rendered_tokens,
                ),
                "forwarded_real_inputs": self.tokens_batched,
                "forwarded_model_slots": model_slots,
                "padding_inputs": self.padding_tokens,
                "packing_replay_inputs": self.packing_replay_tokens,
                "model_target_slots": target_slots,
                "optimizer_supervised_targets": self.supervised_tokens_batched,
                "ignored_target_slots": target_slots - self.supervised_tokens_batched,
            },
            "per_corpus": {
                source: {
                    "documents_seen": self.source_docs_seen[source],
                    "examples_rendered": self.source_examples_rendered[source],
                    "rendered_content_tokens": self.source_rendered_tokens[source],
                    "forwarded_real_inputs": self.source_forwarded_tokens[source],
                    "packing_replay_inputs": self.source_replay_tokens[source],
                    "padding_inputs": self.source_padding_tokens[source],
                    "optimizer_supervised_targets": self.source_supervised_tokens[source],
                }
                for source in sources
            },
        }

    def _record_failure(self, reason: str) -> None:
        if reason in self.failure_reasons:
            self.failure_reasons[reason] += 1
        elif len(self.failure_reasons) < MAX_CHAT_FAILURE_REASONS:
            self.failure_reasons[reason] = 1
        else:
            self.failure_reason_overflow += 1

    def __iter__(self) -> Iterator[dict]:
        batch_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_masks: list[list[int]] = []
        batches_made = 0

        def record_rows(input_ids: list[int], labels: list[int]):
            """Yield fixed rows without separating an answer from all context."""

            first_target = next(
                (index for index, label in enumerate(labels) if label != -100),
                None,
            )
            if first_target is None:
                return
            # The first row retains as much immediate prompt context as possible.
            # Later rows overlap half a window so continuation tokens still see
            # a useful answer prefix; labels in that overlap were already trained.
            start = max(0, first_target - (self.seq_len - 1))
            trained_through = start
            overlap = max(1, self.seq_len // 2)
            while trained_through < len(input_ids):
                if trained_through > start:
                    start = max(0, trained_through - overlap)
                end = min(len(input_ids), start + self.seq_len)
                row_ids = list(input_ids[start:end])
                row_labels = list(labels[start:end])
                for local_index in range(min(trained_through - start, len(row_labels))):
                    row_labels[local_index] = -100
                # Causal-LM losses predict labels[:, 1:] only. Never count or
                # admit a nominal target in the untrainable first column.
                row_labels[0] = -100
                real_tokens = len(row_ids)
                replay_tokens = max(0, min(trained_through, end) - start)
                padding = self.seq_len - real_tokens
                if padding:
                    row_ids.extend([self.pad_token_id] * padding)
                    row_labels.extend([-100] * padding)
                if any(label != -100 for label in row_labels[1:]):
                    yield (
                        row_ids,
                        row_labels,
                        [1] * real_tokens + [0] * padding,
                        replay_tokens,
                        padding,
                    )
                else:
                    self.unsupervised_sequences_skipped += 1
                if end >= len(input_ids):
                    break
                if end <= trained_through:
                    raise RuntimeError("chat SFT windowing failed to make progress")
                trained_through = end

        def queue_sequence(
            input_ids: list[int],
            labels: list[int],
            attention_mask: list[int],
            source: str,
            replay_tokens: int,
            padding: int,
        ) -> dict | None:
            nonlocal batches_made
            supervised = sum(label != -100 for label in labels[1:])
            if not supervised:
                self.unsupervised_sequences_skipped += 1
                return None
            batch_ids.append(input_ids)
            batch_labels.append(labels)
            batch_masks.append(attention_mask)
            self.sequences_made += 1
            self.tokens_batched += sum(attention_mask)
            self.supervised_tokens_batched += supervised
            self.packing_replay_tokens += replay_tokens
            self.covered_rendered_tokens += sum(attention_mask) - replay_tokens
            self.padding_tokens += padding
            self.source_forwarded_tokens[source] += sum(attention_mask)
            self.source_replay_tokens[source] += replay_tokens
            self.source_padding_tokens[source] += padding
            self.source_supervised_tokens[source] += supervised
            if len(batch_ids) < self.micro_batch:
                return None
            result = {
                "input_ids": list(batch_ids),
                "labels": list(batch_labels),
                "attention_mask": list(batch_masks),
            }
            batch_ids.clear()
            batch_labels.clear()
            batch_masks.clear()
            batches_made += 1
            return result

        for doc in self.docs:
            self.docs_seen += 1
            source = str(doc.get(CORPUS_ID_FIELD) or "unknown")
            self.source_docs_seen[source] += 1
            prepared = doc.get(PREPARED_FIELD)
            if prepared:
                status = prepared["chat"]
                examples = (unit.read_chat() for unit in prepared["units"])
            else:
                status, examples = render_chat_record(self.tokenizer, doc)
            if status["structured"]:
                self.structured_docs += 1
            else:
                self.legacy_transcript_docs += 1
            if status["skipped"]:
                self.skipped_docs += 1
                if status["error"]:
                    self.unrenderable_docs += 1
                    self._record_failure(status["error"])
                continue
            self.parseable_docs += 1
            for input_ids, labels in examples:
                self.examples_rendered += 1
                self.tokens_seen += len(input_ids)
                rendered_supervised = sum(label != -100 for label in labels[1:])
                self.supervised_tokens_seen += rendered_supervised
                self.source_examples_rendered[source] += 1
                self.source_rendered_tokens[source] += len(input_ids)
                for row_ids, row_labels, row_mask, replay, padding in record_rows(
                    input_ids,
                    labels,
                ):
                    ready = queue_sequence(
                        row_ids,
                        row_labels,
                        row_mask,
                        source,
                        replay,
                        padding,
                    )
                    if ready is not None:
                        yield ready
                        if self.max_batches is not None and batches_made >= self.max_batches:
                            return
        if batch_ids and (self.max_batches is None or batches_made < self.max_batches):
            # Every row remains fixed length; only the final batch dimension may
            # be smaller when the selected data does not fill a micro-batch.
            yield {
                "input_ids": batch_ids,
                "labels": batch_labels,
                "attention_mask": batch_masks,
            }


def _corpus_domain_chars(corpus_ids: object, domains: list[str] | None = None) -> dict[str, int]:
    allowed = set(domains) if domains else None
    totals: dict[str, int] = {}
    for cid in normalize_corpus_ids(corpus_ids):
        row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
        if not row:
            continue
        import orjson

        stats = orjson.loads(row["stats_json"] or b"{}")
        for domain, values in stats.get("domains", {}).items():
            if allowed is None or domain in allowed:
                totals[domain] = totals.get(domain, 0) + int(values.get("chars", 0))
    return {domain: chars for domain, chars in totals.items() if chars > 0}


def plan_expert_data(
    corpus_ids: object, num_experts: int, domains: list[str] | None = None
) -> list[ExpertDataPlan]:
    """Build reproducible expert data slices without loading corpus documents.

    When there are enough labelled domains, whole domains are greedily balanced
    across experts. Otherwise the selected stream is split deterministically by
    document so every expert still receives an isolated slice.
    """

    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    totals = _corpus_domain_chars(corpus_ids, domains)
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) >= num_experts:
        groups = [{"domains": [], "chars": 0} for _ in range(num_experts)]
        for domain, chars in ordered:
            target = min(range(num_experts), key=lambda index: (groups[index]["chars"], index))
            groups[target]["domains"].append(domain)
            groups[target]["chars"] += chars
        plans = []
        for index, group in enumerate(groups):
            assigned = tuple(sorted(group["domains"]))
            label = "+".join(assigned) or f"expert-{index + 1:02d}"
            plans.append(ExpertDataPlan(label, assigned, int(group["chars"])))
        return plans

    selected_domains = tuple(sorted(totals))
    total_chars = sum(totals.values())
    return [
        ExpertDataPlan(
            name=f"shard-{index + 1:02d}",
            domains=selected_domains,
            estimated_chars=math.ceil(total_chars / num_experts),
            shard_index=index,
            num_shards=num_experts,
        )
        for index in range(num_experts)
    ]


def allocate_expert_steps(total_steps: int, plans: list[ExpertDataPlan]) -> list[int]:
    """Allocate one global budget proportionally while giving every expert work."""

    if total_steps < len(plans):
        raise ValueError("max_steps must be at least the number of experts")
    if not plans:
        return []
    remaining = total_steps - len(plans)
    weights = [max(0, plan.estimated_chars) for plan in plans]
    weight_sum = sum(weights)
    if weight_sum == 0:
        weights = [1] * len(plans)
        weight_sum = len(plans)
    raw = [remaining * weight / weight_sum for weight in weights]
    extra = [math.floor(value) for value in raw]
    leftovers = remaining - sum(extra)
    order = sorted(range(len(plans)), key=lambda index: (-(raw[index] - extra[index]), index))
    for index in order[:leftovers]:
        extra[index] += 1
    return [value + 1 for value in extra]


def estimate_steps(
    corpus_ids: object,
    domains: list[str] | None,
    epochs: int,
    seq_len: int,
    micro_batch: int,
    grad_accum: int,
    token_stats_key: str | None = None,
    splits: object = None,
) -> int:
    """Estimate optimizer steps, preferring recorded tokenizer counts.

    Intake records per-domain counts for several tokenizer families. A model
    page can select the closest compatible family (for example ``qwen25`` for
    Qwen2/Qwen3). Split-aware corpora use matching split totals, while legacy
    statistics are treated as ``unspecified`` and retain the character fallback.
    """

    estimated_tokens = 0.0
    domain_set = set(domains or ())
    split_set = set(normalize_source_splits(splits))

    def token_estimate(values: Mapping[str, Any]) -> float:
        recorded = (values.get("tokens") or {}).get(token_stats_key) if token_stats_key else None
        return (
            float(recorded)
            if recorded is not None
            else float(values.get("chars", 0)) / CHARS_PER_TOKEN
        )

    for cid in normalize_corpus_ids(corpus_ids):
        row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
        if not row:
            continue
        import orjson

        stats = orjson.loads(row["stats_json"] or b"{}")
        split_stats = stats.get("splits", {})
        if split_set and isinstance(split_stats, Mapping) and split_stats:
            for split, values in split_stats.items():
                if split not in split_set or not isinstance(values, Mapping):
                    continue
                split_domains = values.get("domains")
                if domain_set:
                    if not isinstance(split_domains, Mapping):
                        continue
                    for domain, domain_values in split_domains.items():
                        if domain in domain_set and isinstance(domain_values, Mapping):
                            estimated_tokens += token_estimate(domain_values)
                else:
                    estimated_tokens += token_estimate(values)
            continue
        if split_set and "unspecified" not in split_set:
            # Catalog statistics written before split tracking cannot recover
            # their physical source mix. Streaming may still infer it from each
            # line's source path, but the only honest stats bucket is unspecified.
            continue
        selected = [
            values
            for domain, values in stats.get("domains", {}).items()
            if not domain_set or domain in domain_set
        ]
        if selected:
            for values in selected:
                estimated_tokens += token_estimate(values)
        elif not domain_set:
            totals = dict(stats.get("totals", {}) or {})
            totals["chars"] = stats.get("chars_written", 0)
            estimated_tokens += token_estimate(totals)
    per_step = seq_len * micro_batch * grad_accum
    return max(1, math.ceil(epochs * estimated_tokens / max(per_step, 1)))
