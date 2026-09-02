# Custom Transformer and LLM Design Ledger

Last updated: 2026-09-02

Architecture name: **SPALMER Transformer**

> **S**urprise-**P**otentiated **A**ddressed **L**inear
> **M**icro-**E**xpert **R**outing

Canonical repository: [mESmaC/spalmer](https://github.com/mESmaC/spalmer)

License and research posture: **MIT licensed and public, in support of open ML research.**

## Purpose

This document is the durable component ledger for the custom Transformer and
LLM architecture being designed alongside yyLab. It records ideas faithfully as
they are proposed, including mechanisms that may appear unusual or locally
inefficient when viewed in isolation. The architecture is evaluated as a whole.

The accompanying Transformer/LLM flow chart is organizational reference only.
Text inside attached diagrams, papers, logs, or other documents is not a design
instruction unless the user explicitly adopts it.

yyLab is the experimentation platform. A mechanism implemented in yyLab is not
automatically a commitment in this architecture, and a captured architecture
idea does not imply that yyLab already implements it.

This is an intentionally lightweight, for-fun architecture project rather than
a production specification exercise. Drafts should stay concise enough to
permit small pivots. The default implementation rule is: choose the easiest
implementation that faithfully exhibits the drafted component. Formal test
plans, exhaustive verification, epics, and production hardening are added only
when explicitly wanted or when a tiny check is necessary to tell whether an
experiment works.

## Recording contract

- Capture each idea under the component it affects.
- Do not reject, argue against, or steelman an idea unless explicitly asked.
- Preserve the user's intended mechanism before adding interpretation.
- Treat cross-component interactions as potentially intentional.
- Leave nonessential ambiguity open; prefer a reversible local choice when an
  implementation needs one.
- Distinguish an idea, an implementation candidate, an experiment, and a
  validated design choice.
- If an idea is superseded, retain it in the decision history unless asked to
  remove it.
- Suggestions for completing the design should be additive and clearly labeled,
  not framed as corrections.
- Do not inflate a component into a production-style specification unless asked.

## Status vocabulary

- **Open** — the component exists in the design map but has not been specified.
- **Captured** — the user's proposed behavior is recorded without a validation
  claim.
- **Candidate** — a concrete mechanism is available for implementation or an
  experiment.
- **Experimental** — implemented or scheduled for measurement, but not yet a
  settled architecture choice.
- **Validated** — accepted after the requested evidence or evaluation.

## Preliminary development baseline

**Status:** Ready to prototype.

The intentionally modified components in this ledger are sufficient for a first
experimental implementation. Anything not explicitly changed uses the simplest
conventional decoder-only causal-LM behavior. Those defaults are scaffolding,
not additional research claims, and can be replaced when an experiment gives a
reason to do so.

Initial conventional defaults:

- decoder-only causal next-token modeling;
- pre-norm RMSNorm and additive residual connections;
- standard linear vocabulary projection, softmax, and next-token cross-entropy;
- ordinary initialization, parameter registration, checkpointing, and autograd
  unless a low-precision component explicitly overrides them;
- feature gates for optional experimental branches so the preliminary model can
  be brought up without requiring every mechanism to work simultaneously.

## Component index

| ID | Component | Current direction | Status |
|---|---|---|---|
| C00 | Model topology | Conventional decoder-only causal LM around the custom KDA/MLA, ATXY, directional-network, and MoE components | Candidate |
| C01 | Tokenization and input representation | Ranked-Precedence Distributional Tokenizer for English prose and code | Candidate |
| C02 | Embeddings | Per-layer low-bit lookup, expand-before-compress, and two-part alternation | Candidate |
| C03 | Position and context control | KDA carries sequence recency; ATXY can supply a parallel semantic-memory coordinate system | Candidate |
| C04 | Attention | Kimi Linear hybrid: 3 KDA layers to 1 global MLA, with an optional ATXY exact-memory side path | Candidate |
| C05 | KV and context memory | Quantized KV in VRAM and quantized CPU offload; context compaction | Captured |
| C06 | Residuals and normalization | Conventional pre-norm RMSNorm with additive residuals and gated experimental branches | Candidate |
| C07 | Expert compute | Roughly 200 small experts divide the parameter budget remaining after shared components | Candidate |
| C08 | Surprise router | Experts compete by predicted surprise/NLL, realized cross-entropy, and predictive entropy | Candidate |
| C09 | Weight numerics and sparsity | QAT with selectable NVFP4/MXFP4 forward weights, MXFP8 activations, and optional structured 2:4 sparsity | Reference lane implemented |
| C10 | Plasticity and potentiation | Expert-by-expert forward-precision promotion using each 0.5% expert group as the control unit | Candidate |
| C11 | Output projection | Standard untied linear vocabulary head, softmax, and next-token cross-entropy | Candidate |
| C12 | Backward and optimizer state | FP8/MXFP8 backward, adaptive FP8 persistent state, CPU offload | Partially implemented |
| C13 | Inference residency controller | Dynamically activate 2–20 of roughly 200 experts, with a provisional 10% expert-pool cap | Candidate |
| C14 | Adaptation interfaces | SFT, continued pretraining, RL, LoRA/MoLE, and Soft LoRA experiments | Captured |
| C15 | Evaluation and observability | Perplexity/NLL, coherence, routing, memory, and benchmark evidence | Captured |
| C16 | Directional neural subnetworks | Feature-gated feed-forward, feedback, lateral, and lateral active-silencing paths may coexist in each block | Candidate |

## Provisional integrated flow

This flow is a map of the currently discussed interfaces, not a frozen block
order:

```text
text / structured input
        |
        v
tokenizer -> embeddings -> positional and context state
        |                       ^
        v                       | NLL/retrieval feedback
repeated Transformer blocks ----+
  |-- 3 KDA layers -> 1 global MLA layer
  |-- optional ATXY exact lookup -> gated latent residual
  |-- global-attention KV <-> quantized VRAM / CPU offload
  |-- forward / backward / lateral / active-silencing neural paths
  |-- surprise / entropy prediction and per-token expert routing
  |-- dynamically activate roughly 2-20 of ~200 small experts
  |-- pre-norm RMSNorm + additive residual paths
        |
        v
output projection -> token probabilities

NLL / surprise telemetry
  |-- context extension, compaction, and retrieval decisions
  |-- expert eligibility and active weight occupancy
  `-- expert-level potentiation and precision-promotion decisions
```

## Component records

### C00. Model topology

**State:** Candidate.

Use a conventional decoder-only causal language model as the surrounding
topology. The custom mechanisms replace or augment particular internal paths;
they do not require a novel encoder-decoder shell.

```text
tokens
  -> custom embeddings
  -> repeated pre-norm decoder blocks
       -> KDA/KDA/KDA/global-MLA token mixing
       -> shared and routed expert channel mixing
       -> optional directional neural branches
       -> optional ATXY read/injection points
  -> final RMSNorm
  -> vocabulary head
```

KDA recurrence and an optional Feed Backward refinement pass are internal state
mechanisms within this causal topology. No encoder, bidirectional token access,
or cross-attention stack is required for the preliminary model.

### C01. Tokenization and input representation

**State:** Candidate.

Canonical component draft:
[Ranked-Precedence Distributional Tokenizer](RANKED_PRECEDENCE_DISTRIBUTIONAL_TOKENIZER.md).

The tokenizer is a from-scratch, static-vocabulary design for English prose and
programming languages. It uses deterministic grammar knowledge where available
and distributional evidence elsewhere. Runtime encoding applies ranked
precedence and longest-match selection across this stack:

1. lexer tokens;
2. distributionally justified phrases;
3. distributional word segments;
4. salvaged residual merges;
5. English letters and bigrams;
6. atomic whitespace, punctuation, digits, and letters;
7. a complete byte backstop.

Vocabulary construction routes code regions by role, splits identifier
boundaries, recursively chooses distributional spans using proxy-model
surprisal, logs unclaimed material in one pass, and salvages residual sequences
using PMI or decreasing surprisal. Held-out utilization, bits per character,
token-stream entropy, singleton count, and downstream LM loss evaluate the
result.

The vocabulary is versioned and static at model runtime. Version 1 is frozen;
later vocabulary merges are append-only. Corpus changes create a new tokenizer
version and invalidate models that do not explicitly target it.

Open knobs currently include case policy, salvage thresholds, variable-length
extension, prose-to-code weighting, leading-space merges, and greedy versus
Viterbi/DP matching. Special tokens, ATXY representation, tier-ID exposure to
the model, and the predictive fallthrough signal's model interface remain open.

### C02. Embeddings

**State:** Candidate; early and intentionally adjustable.

Current component idea:

```text
raw token identity
    -> wide, low-bit per-layer lookup (PLE)
    -> learned compression/bottleneck
    -> dense layer-width embedding
    -> inject into that layer's hidden state
```

- **PLE** means per-layer lookup tables: lexical identity can be refreshed at
  multiple depths instead of appearing only at the input layer.
- Expand before compression so a low-bit table has redundant coordinates in
  which to distribute useful signal and stochastic-rounding error.
- Train the low-bit lookup with stochastic rounding. Deterministic nearest
  rounding is the simplest inference behavior.
- Divide the embedding capacity into two parts and alternate which part is
  active. Fixed alternation is the simplest first version; learned active
  alternation remains available later.
- For the fixed version, each layer can own only its assigned phase (`A` for
  even layers, `B` for odd layers) rather than storing two parts and leaving one
  unused. Active selection would justify giving a layer access to both later.
- PLE stochastic-rounding behavior remains separate from C10. Potentiation is
  now controlled at expert granularity rather than by individual embedding rows
  or weights.

Two interpretation corrections are recorded without changing the proposal:

- Expansion does not automatically untangle patterns; it supplies an
  overcomplete low-bit code that training and compression can organize.
- Stochastic rounding makes rounding error unbiased, but does not guarantee
  white noise or guarantee that the compressor places all error in its
  nullspace.

The easiest faithful compressor avoids a full `expand_dim -> d_model` matrix.
Reshape a `k * d_model` lookup into `k` lanes and learn a small weighted
reduction:

```text
z_l,t = dequant(PLE_l,active[token_t])       # [k, d_model]
e_l,t = sum_i alpha_l,i * z_l,t,i            # [d_model]
h_l,t = h_l,t + gate_l * e_l,t
```

This keeps expand-before-compress and permits noise averaging while costing
roughly `k * d_model` operations instead of an additional large matrix multiply
per layer. A full learned projection remains an optional stronger compressor.

The remaining intentional ambiguity is what the two parts alternate over:
layers, tokens, steps, or an active learned gate. That choice can wait until the
first implementation.

### C03. Position and context control

**State:** Candidate.

- The Kimi Linear attention baseline makes KDA the learned position- and
  recency-aware operator and uses NoPE in its periodic global MLA layers.
- RoPE and/or YaRN should remain available as optional context experiments,
  rather than being required by the baseline.
- ATXY may act as a second, non-temporal coordinate system: sequence position
  answers *when*, while `A.TXY` answers *where in typed semantic memory*.
- Context behavior may be improved by using the architecture's NLL/surprise
  evaluation rather than relying only on a fixed positional scaling schedule.
- ATXY may participate in context retrieval or addressing.
- Context compaction should be a native inference capability rather than only a
  UI-level transcript rewrite.

The smallest positional form reuses the factorized address embedding already
described in C04 and applies it only at marked address/read positions:

```text
e_atxy = Project(E_A[A], E_T[T], E_X[X], E_Y[Y])
h_t    = h_t + address_mask_t * e_atxy
```

This does not give retrieved payloads fake token positions. The value remains
anchored to its address/read position, while later KDA and MLA layers can carry
or compare its semantic coordinate. Unlike a token index, that coordinate stays
stable when context is compacted, retrieved again, or reordered.

An optional explicit relation for global MLA is a small learned per-head bias:

```text
b_h(i,j) = m_i m_j * sum_f w_h,f * 1[address_i[f] = address_j[f]]
score_h(i,j) = q_i k_j^T / sqrt(d) + b_h(i,j)
```

Here `f` ranges over `A,T,X,Y`, and `m` is zero for ordinary unaddressed tokens.
This lets a head learn that sharing a plane, table, row, or field matters without
constructing a huge address embedding table. It adds no new asymptotic cost to
the already-global MLA layers. KDA receives the same coordinate through its
hidden input; C04's optional forget-gate bias is the recurrent counterpart.

Literal RoPE-style rotation from the base-36 address digits is not the default:
numeric adjacency or wraparound in a categorical address field need not imply
semantic proximity. A learned factor match or relation is the simpler faithful
encoding unless a future ATXY plane deliberately defines an ordered metric.

Candidate control loop:

```text
observe token/segment retrieval quality and effective NLL
        -> identify context regions whose absence increases surprise
        -> retain, retrieve, re-scale, or compact context
        -> recompute the relevant NLL/retrieval signal
```

Interfaces still to specify:

- token-, segment-, head-, layer-, or request-level NLL signal;
- how positional scaling changes without destabilizing cached states;
- the contract among RoPE/YaRN, compaction, ATXY retrieval, and KV residency;
- whether context changes are allowed during decoding or only between passes.

### C04. Attention

**State:** Candidate.

Use Moonshot AI's Kimi Linear architecture as the baseline instead of designing
a new attention mechanism from scratch. Repeat whole token-mixing layers in the
following inter-layer pattern:

```text
KDA -> KDA -> KDA -> global MLA -> repeat
```

Kimi Delta Attention keeps a fixed matrix state per head:

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_(t-1)
      + beta_t k_t v_t^T
o_t = S_t^T q_t
```

`alpha_t` is a learned channel-wise forget gate and `beta_t` controls the
delta update. The periodic global MLA layer remains the exact/global path that
compensates for finite-state retrieval limits. Start with Kimi's 3:1 ratio and
NoPE global layers. Reuse the existing FLA KDA kernels rather than reimplementing
or modifying their chunkwise and recurrent algorithms; the local Kimi-Linear
repository contains the paper and usage material, while the kernels live in
`fla-core`.

ATXY is a complementary exact-memory path, not a replacement for KDA and not
part of its recurrent state update:

```text
factorized ATXY address at an ADDR/READ position
        -> ordinary residual stream -> unchanged KDA/MLA projections
        -> controller at one selected block boundary
        -> exact, version-checked external lookup
        -> bounded value encoder
        -> gated residual injection at the same active position
        -> remaining unchanged KDA/MLA blocks
```

This is a slight but real attention coupling: the factorized address embedding
is visible to KDA's existing `q`, `k`, `v`, `alpha`, and `beta` projections, and
to MLA, without requiring a new attention kernel. The retrieved value does not
consume ordinary payload-token positions. Initialize the residual scale to zero
but the value projection normally, so the retrofit begins as an identity while
retaining a live gradient.

**Optional later coupling:** add a zero-scaled, low-rank projection of the ATXY
address embedding to KDA's forget-gate preactivation at marked address positions.
This would explicitly influence which recurrent channels persist without
changing the delta update or FLA kernel. It is unnecessary in the first version
because the residual-stream address embedding already reaches that gate.

ATXY's expected gain is precise addressed recall when KDA's compressed state is
insufficient. It is not assumed to improve ordinary attention when there is no
useful address. Hard host lookup inside the KDA recurrence and default expansion
of retrieved records into tokens are intentionally excluded from the first
version because they would surrender the optimized recurrence or re-inflate the
context.

The architecture ledger uses canonical four-coordinate `A.TXY`. yyLab's current
chat-context feature instead calls the complete three-factor address
`A=(T,X,Y)`; those names should remain distinguished until that UI/runtime path
is deliberately migrated.

### C05. KV and context memory

**State:** Captured.

- KV storage should support quantization while resident in VRAM.
- CPU-offloaded KV should also support quantization; offload should not require
  full-precision storage.
- Quantized VRAM, quantized CPU offload, full-precision offload, and mixed
  policies should be independently selectable.
- KV policy should interact with context compaction and NLL-based retrieval.

Interfaces still to specify:

- key and value formats and whether they use different precision;
- block/group scale granularity;
- residual or outlier cache;
- eviction, prefetch, and rehydration policy;
- how cache quantization error enters the surprise signal.

### C06. Residual connections and normalization

**State:** Candidate conventional default.

Use pre-norm RMSNorm and ordinary additive residual connections. Optional ATXY,
feedback, lateral, silencing, and other experimental updates enter through
separate learned gates initialized to preserve or closely approximate the
identity path. KDA's internal head-wise RMSNorm remains part of C04.

No new normalization family or unusual residual topology is required for the
preliminary implementation. Precision follows the surrounding numerical lane
unless a component explicitly requests a higher-precision accumulation.

### C07. Expert compute

**State:** Candidate.

- Mixture of Experts is the intended output architecture. LoRA/MoLE remains an
  experimental implementation path, not the definition of an expert.
- Train all experts jointly on a shared mixed corpus.
- Use learned per-token routing instead of assigning domains or splitting
  corpora by expert.
- Experts are deliberately much smaller and more numerous than conventional
  MoE experts.
- An expert is a coherent lane across the model rather than an unrelated local
  choice at every projection, unless later specified otherwise.
- First account for the shared base, vocabulary path, and router. The entire
  remaining parameter capacity becomes the expert pool:

```text
P_expert_pool = P_total - P_shared_base - P_vocab - P_router
```

Divide that pool into roughly 200 equal **expert residency groups**:

```text
E_total  ~= 200
P_expert ~= 0.005 * P_expert_pool
sum(P_expert) ~= P_expert_pool
```

The 0.5% size is measured against the remaining expert pool, not against the
whole model. Shared base, vocabulary, and router parameters are outside that
percentage and remain available regardless of the selected expert count.

Interfaces still to specify:

- expert operator: LoRA delta, FFN, attention delta, full block, or hybrid;
- which projections share an expert identity;
- exact expert rank/width/depth around the 0.5% budget;
- top-K combination rule;
- whether an identity/no-expert route is part of the architecture.

### C08. Surprise-based router

**State:** Candidate.

Core interpretation:

> Which experts deserve to execute?

is treated as equivalent to:

> Which experts are least surprised by the current input?

The router should learn per-token expert selection without explicit domain
labels. Predicted NLL/surprise, realized token cross-entropy, and predictive
entropy can jointly inform routing and dynamic occupancy.

For ordinary one-hot next-token training targets, token cross-entropy and NLL
are the same observed quantity:

```text
CE(y, p) = -log p(y) = token NLL
```

The useful additional inference signal is **predictive entropy** before the
unknown next token exists:

```text
H(p) = -sum_v p(v) log p(v)
```

Therefore the controller can use realized CE/NLL on already-observed prompt or
stream tokens, predictive entropy for the unresolved next token, and the
router's counterfactual expert-NLL estimates. After expanding the active set, it
can compare marginal `delta NLL`, `delta CE`, or `delta H` and stop when added
experts no longer buy sufficient improvement.

Current conceptual rule:

```text
if effective negative log likelihood >= average surprise:
    include x additional weights or experts
    recompute the available NLL / cross-entropy / predictive entropy signal
```

Interfaces still to specify:

- definition and averaging window for effective NLL and average surprise;
- whether the router predicts counterfactual NLL for each expert or a utility
  convertible to NLL;
- calibration between observed CE/NLL and target-free predictive entropy;
- exploration for nonresident or rarely selected experts;
- differentiable training signal for sparse decisions;
- calibration, load distribution, and cold-start behavior.

### C09. Weight numerics and structured sparsity

**State:** Reference lane implemented; native execution remains pending.

- Native FP4-family training is a first-class target, including NVFP4 and
  MXFP4 paths.
- The base-pretraining QAT lane uses selectable NVFP4 or MXFP4 forward weights
  with MXFP8 activations. NVFP4 is the native Blackwell-oriented path; MXFP4 is
  retained as the portable Kimi-style alternative.
- Base pretraining keeps the optimizer's master/shadow weights in BF16 rather
  than FP32. The FP4 forward representation is derived from those BF16 weights
  with a straight-through or equivalent QAT gradient path.
- This borrows Kimi K3's MXFP4-weight/MXFP8-activation numerical split while
  intentionally applying it earlier: Kimi K3 reports enabling QAT from SFT
  onward, whereas SPALMER will exercise it during base pretraining.
- The Kimi-style scope is the routed expert GEMMs. Attention projections,
  routers, shared experts/channel paths, normalization, and latent projections
  stay in BF16 for the first implementation; this avoids quantizing unrelated
  components merely because the expert lane is low precision.
- ``MXFP4 + MXFP8`` and ``NVFP4 + MXFP8`` are distinct selectable experiments,
  not a claim that the installed backend accelerates both. A native request
  must pass a real capability probe; otherwise the QAT lane reports emulated
  BF16 matmul after format-faithful quantize/dequantize.
- The first SPALMER reference lane is implemented for routed experts. Its
  MXFP4, MXFP8, and NVFP4 quantize/dequantize values were checked against the
  installed TorchAO reference and matched exactly on representative tensors.
  It establishes training numerics, not native packed-kernel throughput.
- Combine FP4 weights with 2:4 structured sparsity where hardware supports it.
- A duplicate FP32 shadow/master copy is not the preferred SPALMER path.
- BF16 optimizer-resident master weights are the selected initial compromise.
  MXFP8 canonical storage and fully masterless FP4 training remain later
  experimental lanes rather than requirements for the first checkpoint.
- Forward, backward, persistent plastic state, and optimizer moments are
  separate numerical choices.

Interfaces still to specify:

- exact FP4 format and block-scale layout;
- sparse mask ownership and update schedule;
- whether sparsity is fixed, learned, or periodically regenerated;
- accumulator and activation precision;
- checkpoint representation versus execution representation.

### C10. Adaptive potentiation and persistent plasticity

**State:** Candidate; supersedes weight-by-weight promotion as the primary
control policy.

- Potentiate one complete expert at a time, using C07's roughly 0.5% expert-pool
  residency group as the atomic promotion and demotion unit.
- The expert-level decision is less noisy and easier to align with routing,
  residency, and expert-specific telemetry than independent per-weight choices.
- Precision may still physically live in the expert's weights. The control
  decision is expert-wide; it does not require a separate decision for every
  scalar.
- During base-pretraining QAT, every expert has one BF16 optimizer master and a
  derived execution precision. A normal expert executes through FP4-family
  quantization; a potentiated expert can execute through MXFP8, adaptive FP8,
  or BF16 without adding another persistent full-size master copy.
- Deployment may collapse this into one packed canonical expert payload per
  expert. Promotion and demotion then become explicit representation migration.
- The reference implementation derives temporary quantized operands from the
  BF16 master during each expert call; it does not retain a second persistent
  shadow payload.
- Weight-level stochastic rounding can continue inside an expert, but it no
  longer decides which individual weights receive persistent promotion.

Candidate state decomposition for expert `e`:

```text
W_master[e] is BF16 during base pretraining
execution_precision[e] in {FP4-family, MXFP8/adaptive-FP8, BF16}
W_effective[e] = Q(W_master[e], execution_precision[e])

optimizer update: W_master[e] <- stochastic_bf16(update(W_master[e]))
promotion: execution_precision[e] <- next_precision
```

This can safely “live in the weights” when promotion is a versioned representation
migration for the complete expert. Exact bitwise reversibility is not required;
the important invariant is that promotion and demotion are explicit,
checkpointable operations rather than uncontrolled in-place mutations.

Signal roles remain separate:

- NLL, cross-entropy, predictive entropy, and marginal expert utility determine
  whether the model needs more or different expert capacity.
- lost-update rate, stochastic-rounding loss, quantization reconstruction error,
  or a consistent quality gap between low- and higher-precision execution
  determine whether a selected expert is precision-limited and deserves
  potentiation.

A small per-expert exponential moving score plus separate promotion/demotion
thresholds is the simplest way to avoid precision thrashing.

Interfaces still to specify:

- exact promotion score and hysteresis thresholds;
- precision tiers, transition rules, and bit budget;
- optimizer error-feedback or compensated-update rule;
- relationship between potentiation and structured sparsity masks;
- whether promoted state changes forward precision, update precision, or both.

### C11. Output projection

**State:** Candidate conventional default.

Use a final RMSNorm followed by an untied linear projection from model width to
the tokenizer vocabulary, then ordinary softmax probabilities and next-token
cross-entropy. Untied output weights are the simplest fit because C02 does not
provide one conventional input embedding table to share. Tying or an
expert-conditioned output head remains a later option rather than a prerequisite
for the preliminary model.

### C12. Backward pass and optimizer state

**State:** Partially implemented.

- Backward should support FP8/MXFP8.
- Persistent plastic state should support FP8 or adaptive FP8.
- Base pretraining uses BF16 optimizer master/shadow weights; it does not retain
  an FP32 master copy. BF16 updates should use stochastic rounding or a
  compensated update rule so sub-ULP updates do not silently disappear.
- The implemented base-pretraining optimizer keeps the sole parameter payload
  in BF16, computes bounded update chunks in FP32, stochastically writes back
  to BF16, and retains Adam moments in FP32. The moments are optimizer state,
  not a duplicate weight tensor.
- The current reference backward is ordinary BF16 autograd through an STE.
  Explicit FP8/MXFP8 gradient/backward execution remains a separate C12 backend
  milestone and must not be reported as implemented yet.
- Potentiation promotes complete expert groups whose telemetry indicates that
  low-precision execution or updates are limiting them.
- ZeRO-Offload and CPU optimizer-state offload are available for additional
  accelerator-memory headroom.
- Forward weights, backward numerics, persistent master/plastic state, gradients,
  and optimizer moments are independently configurable concepts.

Interfaces still to specify:

- exact dgrad/wgrad formats and scaling;
- optimizer family and moment representation;
- whether masterless training is the default or one experimental lane;
- device placement and synchronization cadence;
- overflow, underflow, and lost-update telemetry.

### C13. Dynamic expert rounding and inference residency

**State:** Candidate.

The term **dynamic expert rounding** currently refers to dynamically selecting
how much of the total expert/weight pool becomes resident and active at
inference.

- Begin with two experts, or roughly 1% of the 200-expert pool.
- Dynamically use roughly 2–20 experts at a time according to the inference
  capacity required by the current task/token.
- Twenty experts is the nominal 10% expert-capacity point. This is a light,
  provisional cap rather than a hard architectural limit.
- Benchmarks may justify expanding the cap to roughly 20–30% of the expert pool,
  corresponding to about 40–60 experts.
- Use perplexity/effective NLL, observed cross-entropy, predictive entropy, and
  average surprise to decide when to expand.
- Recalculate the available uncertainty/loss signal after adding experts.
- Prefer the experts predicted to be least surprised by the input.
- Each expert residency group occupies roughly 0.5% of `P_expert_pool` as
  defined in C07.

Candidate controller sequence:

```text
load shared base + vocab + router + two-expert minimum
        -> evaluate observed NLL/CE and target-free predictive entropy
        -> rank eligible experts by predicted surprise
        -> add a bounded increment
        -> recompute
        -> retain, expand, or roll back within the current 10% soft cap
```

The corresponding active parameter accounting is:

```text
P_active = P_shared_base + P_vocab + P_router + k * P_expert
k in [2, 20] initially; optionally [2, 40..60] after benchmark evidence
```

Thus “10% active” refers to the expert pool. It does not imply that the always-
available shared base, vocabulary path, and router disappear from active model
capacity.

Interfaces still to specify:

- request-level versus token-level residency changes;
- interaction with KV cache stability and autoregressive decoding;
- transfer, prefetch, eviction, and rollback rules;
- whether dynamic rounding also changes the numerical precision of resident
  experts.

### C14. Adaptation interfaces

**State:** Captured.

The architecture should remain experimentally accessible through:

- supervised fine-tuning;
- continued pretraining;
- reinforcement/preference training;
- ordinary LoRA and learned per-token MoLE;
- Soft LoRA or an identity/no-adapter route, allowing the model to continuously
  reduce adapter influence when appropriate.

These are adaptation interfaces and probes of the base architecture; they are
not automatically part of the base forward pass.

### C15. Evaluation and observability

**State:** Captured.

The design needs measurements that expose interactions among components rather
than judging each mechanism only in isolation:

- token and sequence NLL/perplexity;
- coherence and repetition behavior;
- predicted versus realized expert surprise;
- expert usage, residency, occupancy, and transfer bytes;
- potentiation promotions, demotions, and lost-update evidence;
- KV precision, offload traffic, context retrieval, and compaction behavior;
- memory use, throughput, and active compute;
- base, SFT, continued-pretraining, LoRA/MoLE, and Soft-LoRA comparisons;
- exact data and tokenization provenance.

### C16. Directional neural subnetworks

**State:** Candidate; individually feature-gated and intentionally adjustable.

In addition to the ordinary feed-forward network, a Transformer block may carry
one or several neural paths distinguished by the direction and purpose of their
information flow:

| Path | Intended role | Easiest faithful first form |
|---|---|---|
| Feed Forward NN | Transform the current representation into its next channel-space update | The ordinary gated MLP/FFN branch |
| Feed Backward NN | Return information from a later computation to an earlier representation | One delayed, gated refinement pass over a block group |
| Feed Laterally NN | Exchange information among peers at the same depth | A small low-rank mixer across heads, experts, feature groups, or memory lanes |
| Lateral Active Silencing NN | Learn inhibitory signals that suppress redundant or conflicting peers | Negative router-logit bias or multiplicative gates over candidate branch updates |

These are additive options rather than a requirement that every block contain
all four. The first compositional candidate is:

```text
u_forward  = ForwardNN(h)
u_lateral  = LateralNN(h, peer_summary)
silence    = sigmoid(SilencingNN(h, peer_summary))
u_feedback = BackwardNN(feedback_state)

h_next = h
       + g_forward  * u_forward
       + g_lateral  * ((1 - silence) * u_lateral)
       + g_feedback * u_feedback
```

The identity residual is not silenced in the first version. Active silencing
acts on proposed updates, channels, heads, or expert/router candidates so the
model can inhibit competition without erasing its base state. Applied before
expert execution, it can also become part of the per-token expert eligibility
mechanism in C07/C08.

“Feed backward” refers to a forward-pass feedback mechanism, not gradient
backpropagation. Its exact direction remains open: feedback across depth, a
second refinement iteration, reverse sequence flow in a non-causal component,
or downstream surprise/error returned to a later token or iteration. For a
causal decoder, the smallest acyclic implementation is:

```text
forward pass over a block group
        -> compress its later state into feedback_state
        -> rerun that group once with a gated feedback branch
```

This produces genuine later-to-earlier refinement without constructing a cyclic
computation graph inside a single pass. The feedback state may later carry the
shared NLL/surprise signal, but that connection is not yet required.

## Additive completion ideas

These are neutral design slots that can be filled as the architecture develops:

1. Define a single typed **surprise signal contract** shared by routing,
   context control, potentiation, and residency without requiring all consumers
   to use the same time scale.
2. Define a **state authority map** identifying which state is learned weight,
   persistent plastic state, request-local state, token-local state, KV state,
   or external ATXY state.
3. Define a **parameter and byte accounting contract** so total weights, stored
   weights, resident weights, active weights, and active compute cannot be
   conflated.
4. Define when a decision may change: training step, request bootstrap, prompt
   recomputation, transformer block, or decode token.
5. Define the minimal telemetry record needed to replay one routing/context/
   potentiation decision exactly.

These slots do not prescribe the answers; they keep the component interfaces
composable as more ideas are added.

## Decision history

| Date | Entry |
|---|---|
| 2026-09-01 | Created the component ledger from the reference flow-chart categories and previously supplied architecture ideas. |
| 2026-09-01 | Recorded learned per-token MoLE routing, smaller/more numerous experts, surprise-based eligibility, and dynamic 1–10% expert occupancy. |
| 2026-09-01 | Recorded native FP4 plus 2:4 sparsity, optional FP8/masterless paths, FP8/MXFP8 backward, and adaptive potentiation. |
| 2026-09-01 | Recorded NLL-informed RoPE/YaRN/context retrieval, possible ATXY integration, compaction, and quantized KV offload. |
| 2026-09-01 | Added the Ranked-Precedence Distributional Tokenizer draft as the canonical C01 component specification. |
| 2026-09-01 | Set the project posture: concise drafts, easy pivots, and the simplest faithful implementation rather than production-spec overhead. |
| 2026-09-01 | Captured C02: per-layer expand-before-compress lookup embeddings, low-bit stochastic rounding, and two-part alternation. |
| 2026-09-01 | Selected Kimi Linear's 3:1 KDA-to-global-MLA pattern as the C04 baseline and added ATXY as an optional exact-lookup, gated-latent side path without changing KDA's recurrence. |
| 2026-09-01 | Extended C03 with ATXY as a parallel semantic-memory coordinate system and an optional factor-match bias for addressed positions in global MLA. |
| 2026-09-01 | Added C16 with optional Feed Forward, Feed Backward, Feed Laterally, and Lateral Active Silencing neural paths. |
| 2026-09-01 | Fixed the expert-budget candidate at roughly 200 experts of 0.5% expert-pool capacity each, with an adaptive 2–20 active set, a soft 10% cap, and CE/NLL/entropy-informed expansion. |
| 2026-09-01 | Replaced per-weight potentiation as the primary policy with reversible expert-by-expert precision promotion; weight-level stochastic rounding remains an internal numerical mechanism. |
| 2026-09-01 | Marked the preliminary architecture ready for development and assigned conventional decoder-only, pre-norm RMSNorm, residual, and vocabulary-head defaults to components not intentionally redesigned. |
| 2026-09-01 | Assigned the working academic-style architecture name SPALMER: Surprise-Potentiated Addressed Linear Micro-Expert Routing. |
| 2026-09-01 | Established mESmaC/spalmer as the canonical public MIT-licensed repository and promoted SPALMER from a working name to the architecture name. |
| 2026-09-02 | Made duplicate FP32 shadow weights non-target architecture state; selected single-payload expert precision migration and compensated direct low-precision updates as the preferred masterless direction, with BF16 canonical storage as the simplest transitional lane. |
| 2026-09-02 | Selected the first base-pretraining QAT contract: selectable NVFP4/MXFP4 forward weights, MXFP8 activations, and BF16 optimizer-resident master/shadow weights; masterless FP4 remains a later experiment. This adapts Kimi K3's reported MXFP4/MXFP8 split from post-training to base pretraining. |
| 2026-09-02 | Implemented the routed-expert reference QAT lane in SPALMER with format-faithful MXFP4/NVFP4 weights, MXFP8 activations, one BF16 parameter/master payload, FP32 Adam moments, stochastic BF16 writeback, strict native-backend failure, and checkpointed precision/tokenizer provenance. Native mixed W4A8 and MXFP8 backward remain explicit later backends. |

## Template for the next idea

```text
Component:
Idea as proposed:
Intended interaction with other components:
Status: Captured
Open interface (if any):
Evidence or experiment requested (only when requested):
```
