from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TERM_RE = re.compile(r"[\w./:\\-]+", re.UNICODE)
_STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
}


def _normal(value: object) -> str:
    return " ".join(str("" if value is None else value).strip().lower().split())


@dataclass(frozen=True, slots=True)
class FactorizedAddress:
    """yyLab's payload-blind address A=(T, X, Y).

    ``T`` is the table/type, ``X`` is the entity/key, and ``Y`` is the
    condition/qualifier.  ``A`` names the complete address; it is not a fourth
    coordinate in the existing ATXY experiment.
    """

    t: str = ""
    x: str = ""
    y: str = ""

    @property
    def present(self) -> bool:
        return bool(self.t or self.x or self.y)

    @classmethod
    def from_value(cls, value: object) -> FactorizedAddress:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                t=_normal(value.get("t", value.get("type", value.get("table", "")))),
                x=_normal(value.get("x", value.get("entity", value.get("key", "")))),
                y=_normal(
                    value.get(
                        "y",
                        value.get("condition", value.get("qualifier", value.get("time", ""))),
                    )
                ),
            )
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return cls(*(_normal(part) for part in value))
        return cls()


def message_address(message: dict) -> FactorizedAddress:
    """Read an optional address without making metadata mandatory for chat."""

    for key in ("atxy", "address"):
        address = FactorizedAddress.from_value(message.get(key))
        if address.present:
            return address
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        for key in ("atxy", "address"):
            address = FactorizedAddress.from_value(metadata.get(key))
            if address.present:
                return address
    return FactorizedAddress()


def group_address(messages: list[dict]) -> FactorizedAddress:
    """Return the most specific explicit address carried by a turn."""

    addresses = [message_address(message) for message in messages]
    addresses = [address for address in addresses if address.present]
    if not addresses:
        return FactorizedAddress()
    return max(addresses, key=lambda address: sum(bool(v) for v in (address.t, address.x, address.y)))


def address_similarity(query: FactorizedAddress, candidate: FactorizedAddress) -> float:
    """Exact factor agreement with conflicts penalized rather than ignored."""

    if not query.present or not candidate.present:
        return 0.0
    weights = (0.30, 0.45, 0.25)
    shared_weight = 0.0
    signed_match = 0.0
    for weight, left, right in zip(weights, (query.t, query.x, query.y), (candidate.t, candidate.x, candidate.y)):
        if not left or not right:
            continue
        shared_weight += weight
        signed_match += weight if left == right else -weight
    if shared_weight <= 0.0:
        return 0.0
    return max(0.0, signed_match / shared_weight)


def content_terms(messages: list[dict]) -> set[str]:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    return {
        term
        for raw in _TERM_RE.findall(text.lower())
        if len(term := raw.strip("._-/\\:")) > 1 and term not in _STOP_TERMS
    }


def lexical_similarity(query_terms: set[str], candidate_terms: set[str]) -> float:
    if not query_terms or not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / math.sqrt(len(query_terms) * len(candidate_terms))


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    index: int
    messages: tuple[dict, ...]
    atxy_score: float
    lexical_score: float
    recency: float
    query_echo: bool = False

    @property
    def proposal_score(self) -> float:
        # ATXY is a candidate generator, not an acceptance objective.  A
        # matching exact address therefore leads the shortlist while online
        # marginal NLL still decides whether the payload enters the prompt.
        return 4.0 * self.atxy_score + self.lexical_score + 0.05 * self.recency


def rank_context_candidates(groups: list[list[dict]], query_group: list[dict]) -> list[ContextCandidate]:
    """Rank old turns for bounded marginal-NLL evaluation.

    No payload score here is treated as final relevance.  The cheap lexical and
    address factors only decide which candidates receive the expensive model
    probe first.
    """

    query_terms = content_terms(query_group)
    query_address = group_address(query_group)
    query_text = _normal("\n".join(str(message.get("content") or "") for message in query_group))
    denominator = max(1, len(groups))
    candidates = []
    for index, group in enumerate(groups):
        candidate_terms = content_terms(group)
        candidate_user_text = _normal(
            "\n".join(
                str(message.get("content") or "")
                for message in group
                if str(message.get("role") or "") == "user"
            )
        )
        payload_terms = content_terms(
            [message for message in group if str(message.get("role") or "") != "user"]
        )
        near_query_copy = bool(
            query_text
            and candidate_user_text
            and (
                query_text == candidate_user_text
                or query_text in candidate_user_text
                or (
                    query_terms
                    and len(query_terms & candidate_terms) / len(query_terms) >= 0.95
                    and lexical_similarity(query_terms, candidate_terms) >= 0.9
                )
            )
        )
        candidates.append(
            ContextCandidate(
                index=index,
                messages=tuple(dict(message) for message in group),
                atxy_score=address_similarity(query_address, group_address(group)),
                lexical_score=lexical_similarity(query_terms, candidate_terms),
                recency=(index + 1) / denominator,
                query_echo=near_query_copy and not bool(payload_terms - query_terms),
            )
        )
    return sorted(candidates, key=lambda candidate: (-candidate.proposal_score, -candidate.index))
