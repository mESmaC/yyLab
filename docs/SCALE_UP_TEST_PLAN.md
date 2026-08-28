# Pre-Scale Test Plan

What must be validated before moving the memory system to a scaled-up model and
real use. Organized by priority. Metric used throughout: **oracle-capture**
`= (R_selected − R_empty) / (R_oracle − R_empty)` under answer-side reward
`R_y(c) = NLL(y|x) − NLL(y|x,c)`, scored on `y` tokens only.

## Current validated state (0.5B, synthetic)

- Predictive gain exists; relevance > recency; dilution real (4 regimes).
- Answer-side `R_y` rejects the input-echo failure (0/20 vs 20/20).
- Listwise `R_y` content-aware controller beats cosine ~7.6x.
- Blind factorized-address routing matches content-aware (~0.986), payload-free.

Everything below is what is **not** yet validated.

---

## Tier 1 — Blocking (must pass before scale-up)

### T1. Multi-seed stability
- **What:** re-run blind + listwise controllers over 3–5 seeds; report mean ± std
  of oracle-capture and fact-hit rate.
- **Why:** current results are single-seed; confirm 0.986 / 0.84 aren't lucky draws.
- **Pass:** capture stays within a tight band across seeds; no seed collapses.

### T2. Statistical significance
- **What:** raise eval to n ≥ 300 samples; report confidence intervals on capture.
- **Why:** n=20–24 is too small to claim stable differences between controllers.
- **Pass:** CIs for blind vs cosine vs random do not overlap where we claim a win.

### T3. Address-space scaling
- **What:** routing accuracy vs number of candidate addresses: 5 → 50 → 500 → 5000.
- **Why:** argmax over 5 is easy; over 5000 is the real regime. Blind capture may
  degrade sharply with fan-out. This is the highest-risk unknown.
- **Pass:** blind capture stays acceptably high (target ≥ 0.7) at ≥ 500 addresses,
  or we identify the break-point and fix routing (hierarchical factor heads / beam).

### T4. Automated address assignment
- **What:** assign `(A,T,X,Y)` from memory content via an ingestion step (not
  hand-structured), then re-run blind routing.
- **Why:** every result so far used a hand-laid coordinate space. Real use requires
  the system to build addresses. This is the biggest integration unknown.
- **Pass:** blind capture on auto-assigned addresses stays close to hand-assigned.

### T5. Real memories + real QA
- **What:** replace planted facts with a real corpus; build held-out questions with
  known answers; measure end-to-end answer NLL/accuracy with blind-routed injection.
- **Why:** all evidence so far is synthetic templates. Real data is messier.
- **Pass:** retrieval measurably improves real-answer quality vs no-retrieval.

---

## Tier 2 — Important (needed to claim "improvement over long context")

### T6. Realistic baselines
- **What:** compare against a well-tuned RAG pipeline (strong embedding retriever +
  reranker), full-context stuffing at several lengths, and BM25.
- **Why:** beating cosine/empty isn't enough to claim an improvement. The claim is
  "better than long context," so we must beat a competent long-context/RAG baseline.
- **Pass:** blind-routed memory ≥ strong-RAG at lower retrieved-token cost, and
  ≥ full-context on needle-at-depth tasks.

### T7. Multi-address retrieval
- **What:** retrieve 2+ addresses when the answer needs multiple memories.
- **Why:** current system retrieves exactly one. Real questions often need several.
- **Pass:** multi-address selection beats single-address on multi-fact questions.

### T8. Sequential retrieval + stop action
- **What:** controller retrieves addresses one at a time, accumulating reward
  `r_t = NLL(y|x,C_{<t}) − NLL(y|x,C_{≤t}) − λ·cost(c_t)`, and learns to stop.
- **Why:** this is the bandit/RL regime; teaches economical retrieval. First place
  cost-awareness and "when to stop" get tested.
- **Pass:** controller stops near the marginal-value knee; token cost drops vs
  fixed-k without losing capture.

### T9. Multi-hop reasoning
- **What:** questions requiring retrieve-fact-1 → use-it → retrieve-fact-2.
- **Why:** the hardest realistic use; stresses retrieval + composition together.
- **Pass:** above a meaningful floor vs single-shot retrieval.

### T10. Value-binding / exact recall
- **What:** distinguish `channel 7 = 4.871` from `channel 7 = 9.2` (same address
  structure, different value) — the residual error seen in the listwise run.
- **Why:** exact-value recall is where content-aware controllers still err and where
  ATXY's exact-coordinate design should help.
- **Pass:** fact-vs-wrong-value discrimination improves over current (5/16 misses).

---

## Tier 3 — Scale-up-model specifics

### T11. Base-model size transfer
- **What:** re-run the core reward + routing experiments on a larger base
  (1.5B / 3B, VRAM permitting on the 2050 with NF4).
- **Why:** `R_y` is computed by the base model; a stronger model changes the reward
  landscape. Confirm the mechanism doesn't depend on the 0.5B's quirks.
- **Pass:** qualitative results (echo rejection, routing capture) hold on the larger
  base.

### T12. Budget + latency at scale
- **What:** measure VRAM (model + injected context + KV) and end-to-end latency of
  the retrieve → inject → generate loop at realistic memory sizes.
- **Why:** the efficiency claim needs real cost numbers, not just accuracy.
- **Pass:** fits target VRAM; latency acceptable; cost advantage over full-context
  quantified.

### T13. Silent-miss characterization
- **What:** measure how often retrieval misses needed info entirely (the price of
  selection vs full-presence), and whether a fallback (broaden radius / escalate)
  recovers it.
- **Why:** the main downside of selection over full context is silent failure.
- **Pass:** miss rate quantified; a fallback reduces it at bounded extra cost.

---

## Adapter-with-memory (the "train an adapter that uses the memory system" idea)

**Yes, this is possible — and yyLab already has the machinery.** Three levels:

### A1. Memory-reading adapter (immediately buildable)
- Freeze the base; train a LoRA/MoLE adapter on samples `(x=query, c=retrieved
  memory, y=answer)` with input `[c][x]`, loss on `y` tokens.
- Teaches the adapter to *extract the answer from injected memory* (the base wasn't
  trained to read this memory format). This is RA-DIT / RePlug-style LM-side training.
- Retriever = our blind controller (frozen) during this stage.
- **Uses existing infra directly:** MoLE training loop, corpus intake, `R_y` reward,
  `engine.nll_ids`.
- **Required ablation:** with-memory vs without-memory accuracy gap, to prove the
  adapter actually *uses* the memory rather than ignoring it.

### A2. Co-adaptation loop (the frontier)
- Alternate: (a) train the controller to fetch memory that reduces the adapter's
  loss; (b) train the adapter to better use fetched memory. `R_y` couples them.
- This is the full vision: a memory system and a model that learns to use it,
  improving together.
- **Risk:** non-stationarity (the reward target moves as the adapter trains). Stage it
  carefully; keep A1 working as the stable baseline.

### A3. Memory as latent / soft tokens
- Inject memory as learned latent vectors (soft prompts) instead of text, which the
  adapter learns to read. Aligns with the latent-injection idea and reduces injected
  token cost.
- **Harder;** treat as later research, after A1/A2 show signal.

**Ordering:** A1 first (cheap, uses existing infra), validate the with/without-memory
gap, then A2. A3 is optional research.

---

## Suggested sequencing

1. **T1 + T2** (cheap, close the statistical gap).
2. **T3 + T4** (the two highest-risk unknowns: routing at scale, auto addresses).
3. **A1** (memory-reading adapter) — validates the system as something a model can
   actually *use*, not just retrieve into.
4. **T5 + T6** (real data + real baselines) — the "is it an improvement" gate.
5. **T7–T10** (multi-address, sequential, multi-hop, value-binding).
6. **T11–T13** (scale-up-model transfer, budget/latency, silent-miss).
7. **A2** co-adaptation once A1 + routing are stable.

The single most informative next tests are **T3 (address-space scaling)** and
**T4 (automated address assignment)** — everything else either confirms statistics
we mostly trust or builds on these two.
