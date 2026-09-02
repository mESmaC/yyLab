# Ranked-Precedence Distributional Tokenizer

Component: C01 — Tokenization and input representation
Design state: Draft / captured candidate
Last updated: 2026-09-01

This document preserves the current tokenizer proposal as a component of the
custom Transformer/LLM architecture. It is a design specification, not a claim
that yyLab's present tokenizer implementation already follows it.

Implementation posture: keep the draft compact and use the easiest faithful
implementation. Unspecified details are pivot room, not missing requirements.
Production planning, exhaustive conformance work, and formal verification are
outside the intended scope unless later requested.

## 1. Purpose and scope

A from-scratch, static-vocabulary tokenizer for English prose and programming
languages.

Principles:

- Determinism where grammar is known, statistics where it is not.
- Model signal decides splits; frequency is fallback only.
- Minimize singleton tokens, but never force their count to zero. Rare
  single-token representations are required for coverage.
- Guarantee total coverage: every UTF-8 input tokenizes through the fallback
  tiers.
- Deliberately target English only. At small corpus scale, multilingual data is
  treated as noise in the intended distribution.

## 2. Runtime tier stack

Input is matched left-to-right. The highest eligible tier claims first, and the
longest match wins within a tier. This generalizes maximal munch with ranked
precedence.

1. **Lexer tier** — grammar-known tokens, including multi-character operators
   such as `==`, `=>`, and `//`.
2. **Phrase tier** — distributionally justified multi-word tokens.
3. **Word tier** — distributional word splits.
4. **Salvage tier** — frequency-merged residual material.
5. **Language fallback tier** — English alphabet plus English bigrams.
6. **Atom tier** — whitespace, punctuation, digits, and letters; atomic and
   never merged at this tier.
7. **Byte backstop** — all 256 bytes for universal coverage.

Whitespace, punctuation, and special characters are each single tokens. Spaces
inside a phrase token are consumed by that phrase. Only unclaimed spaces reach
the atom tier.

## 3. Vocabulary construction

Inputs are English prose and source code, weighted by their intended deployment
ratio.

### Phase 0 — Corpus preparation

- Weight and mix corpora by deployment target.
- Fit Heaps' law to confirm the corpus is large enough for vocabulary
  stabilization.

### Phase 1 — Lexer tier

- Enumerate keywords, operators, and delimiters per programming language.
- Reserve these deterministically rather than inferring them statistically.
- Reserve multi-character operators explicitly so the splitter cannot shred
  them.

### Phase 2 — Code-region routing

- Route structured regions to the lexer tier.
- Route identifiers through the identifier path in Phase 3.
- Route comments and docstrings through the English distributional path.
- Keep string-literal delimiters atomic and route their contents through the
  out-of-distribution fallback path.

### Phase 3 — Identifier preprocessing

Split `camelCase` and `snake_case` boundaries before distributional analysis.

### Phase 4 — Recursive distributional splitting

- Recursively split spans top-down.
- Score sub-spans using conditional surprisal under a proxy model.
- Continue until segments are predictable enough to stop splitting.
- Preserve multi-character, highly predictable sequences as single tokens,
  including justified two- or three-word phrases.
- Assign phrase tokens the higher precedence rank so their internal whitespace
  is consumed before whitespace reaches the atom tier.

### Phase 5 — Fallthrough logging

Perform one left-to-right pass. Record every character that no earlier tier
claims in the residual stream. The pass produces tokenization output and
fallthrough evidence without rescanning the corpus.

### Phase 6 — Frequency salvage

- Work bottom-up through the residual material.
- Pair residual characters into bigrams and continue toward quadrigrams.
- Score merges with PMI or proxy-model surprisal rather than raw counts, which
  can over-merge boundary artifacts.
- Extend to variable length while surprisal continues to fall.
- Minimize singleton residuals while retaining atoms that never clear the merge
  threshold as singletons by construction.

### Phase 7 — Reserved tiers

Append:

- the English alphabet and English bigrams for language fallback;
- whitespace, punctuation, and digit atoms;
- the 256-byte backstop.

### Phase 8 — Utilization audit

Count token firings on a held-out split and flag dead tokens.

## 4. Runtime behavior

### Encode

Encoding is a single left-to-right pass through the ranked tier stack, with
fallthrough evidence logged.

Precedence matching matters only during encoding, where the complete input is
available. There is no decode-time lookahead issue: the language model emits a
phrase token directly as one vocabulary entry, without performing
segmentation.

Known greedy-precedence behavior: an early long match can occasionally block a
globally better later split. If compression measurements show that behavior,
Viterbi/dynamic-programming search over the same tiered candidates is the
designated alternative. It composes with the tier stack.

### Decode

Decoding is ordinary token-to-byte/text reconstruction. Autoregressive token
emission does not invoke segmentation.

## 5. Fallthrough stream and predictive tokenization

The residual stream feeds two consumers:

1. **Salvage vocabulary statistics** — residual letters are merged during
   Phase 6.
2. **Predictive statistics** — atom adjacency, such as a comma predicting a
   following space, informs boundary prediction and out-of-distribution hints
   without creating merged atom vocabulary entries.

Corpus-bigram statistics gathered during construction guide out-of-distribution
segmentation. Unseen patterns decompose through the bigram table, with the
language fallback and byte tiers as the coverage floor. Every character becomes
either token material or predictive training evidence.

## 6. Statistical toolbox

| Method | Role |
|---|---|
| Raw and relative frequency | Salvage pass and utilization audit |
| Zipf's law, `f(r) proportional to r^-alpha` | Vocabulary-size planning and sanity checks |
| TF-IDF | Surface domain-distinctive phrase candidates |
| PMI | Score Phase-6 salvage merges |
| Co-occurrence and embeddings | Score distributional fit in Phase 4 |
| N-gram models | Proxy surprisal and OOD bigram fallback |
| Entropy and perplexity | Phase-4 split objective and evaluation |
| Heaps' law, `V(n) = K * n^beta` | Phase-0 corpus-sufficiency check |
| Burstiness | Identify domain-concentrated tokens |

## 7. Evaluation metrics

- Compression rate in bits per character as a language-independent baseline.
- Token-stream entropy as a cheap proxy for downstream model quality.
- Vocabulary utilization on a held-out split.
- Singleton count, minimized but never forced to zero.
- Downstream language-model loss as the final arbiter.

## 8. Open knobs

- Case policy; default is case-preserving.
- Salvage merge threshold.
- Variable-length extension versus fixed doubling.
- Prose-to-code corpus weighting.
- Optional leading-space merges when prose sequence length becomes important.
- Greedy ranked matching versus the Viterbi/DP alternative when measured
  compression justifies it.

## 9. Lifecycle

Corpus changes alter the tokenizer and invalidate models trained with it.
Tokenizers must therefore be versioned. Freeze version 1, and make later
vocabulary merges append-only.
