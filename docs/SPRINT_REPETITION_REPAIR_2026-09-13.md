# Sprint repetition repair: evidence and current status

**Status on 2026-09-13: the training/data/formatting safeguards are implemented,
but repetition is not established as fixed. No usable replacement checkpoint is
claimed.** A short data-only intervention failed to remove greedy loops. A
separate, targeted unlikelihood experiment reduced exact repetition on the
original diagnostic prompts and an independent fresh prompt set. Fragmented
words, repeated sentence structures and poor answers remain, so this candidate
is not accepted as a usable repair.

This follows the [repetition audit](SPRINT_REPETITION_AUDIT_2026-09-13.md).
That audit found untrained chat markers, misleading legacy mixture behavior,
templated training/validation content and no generated-quality checks. It also
reproduced repetition without PLE/KDA and with independent FP32 computation.
Those findings do not identify a single universal numerical failure. The work
below distinguishes safeguards against known pipeline failures from experiments
that directly test whether model behavior changes.

## Production safeguards

### A source cannot silently take over the token budget

New causal recipes in Train · Sprint default to **Fixed token shares**.
`corpus_mix="fixed_token_quota"` allocates complete rows from explicit relative
`corpus_weights`; missing source weights are 1. Largest-remainder allocation
preserves the total and rounds each source within one sequence. Rows are sampled
across complete source pools and globally shuffled before the first optimizer
step. A row contains one source, so input and shifted target shares are exact.

`quota_max_source_epochs` defaults to 1. Insufficient data raises an error with
required/available counts; the packer does not silently fill the remainder from
another corpus. A finite larger allowance explicitly permits replay, optionally
per source. Source pools survive a failed quota-admission check and are reused
when changing weights, budgets, seeds or replay allowances. Changed corpus
selection/contents, sequence length or tokenizer needs different pools.

The versioned allocation and replay contract belongs to cache identity. Each
materialized row records its corpus and source-pool row index. Saved
`consumed_data_stats` counts source inputs, supervised targets, covered positions
and replay for the actual consumed prefix, including a stopped run. These are
pool-position counts, not semantic novelty. Existing exact-text deduplication is
preserved; held-out/semantic decontamination is not added by this scheduler.

Legacy recipes remain legacy recipes. `token_balanced` is now labeled
**Balance while sources last (characters)** in the page. Its previous cache keys
and exact-resume behavior are not reinterpreted. The prior 150M run's separately
materialized `fixed_token_mix` contract also remains supported unchanged.

Implementation: [quota packer](../yylab/train/sprint_mix.py),
[training integration](../yylab/train/sprint.py),
[focused quota tests](../tests/test_sprint_token_quota.py).

### Inference follows recorded training formats

New bundles include `prompt_format.json`. Plain causal training defaults to
literal text continuation, with no invented role separators or extra BOS.
Each text-continuation prompt is independent. A generic tokenizer chat template
does not justify claiming that a pretrained checkpoint learned those markers.
Older Sprint causal metadata is interpreted read-only; its weights and bundle
are not rewritten to change the default.

Chat SFT records its exact template and checksum. Causal continuation inherits
an established parent chat contract; new chat SFT upgrades a raw contract to the
actual training template. Conflicting inherited templates fail, and exact resume
cannot replace the recorded contract. Unknown external/adapter formats remain
unverified. Generated chat diagnostics require a recorded chat contract and use
its template. None of this teaches existing weights a new role format.

Implementation: [format contract](../yylab/infer/prompt_format.py),
[inference](../yylab/infer/engine.py), [format tests](../tests/test_prompt_format.py).

### Generated behavior and numerical state are visible during training

`generation_eval` defaults to enabled with its own interval of **500 optimizer
updates**, plus a final check. It is independent of loss evaluation and saves.
The default four prompts produce greedy and sampled outputs of up to 64 tokens.
They cover prose, instruction following, copying and a small word problem.
There are no repetition penalties or ngram blockers. The receipt includes the
rendered prompts, token IDs, output text, duplicate ngrams, exact tail loops,
length/EOS outcomes, raw probabilities/entropy and simple answer checks.
The default copy/math checks require the complete generated answer to match;
a correct first line followed by garbage fails. Explicit legacy first-line
checks remain available with their limited scope recorded in the receipt.

Receipts live at `<run>/generation-eval/step-XXXXXXXX.json`; progress/metrics link
the latest one and warn on detected repetition. Runs displays the saved prompts,
generated text and gradient telemetry, retaining warnings after completion or
stopping. The probes preserve RNG and
model train/eval states. Short or skipped outputs are explicit; passing these
small probes does not establish usable prose, reasoning or assistant quality.
Repetition warns rather than automatically ending a run. Nonfinite generated
logits fail the run. To disable diagnostics, uncheck **Check generated text for
repetition** or set `"generation_eval": {"enabled": false}`. Cadence and bounded
prompt/decoding options are configurable.

Pre-clipping gradient norms and the fraction of clipped updates are logged when
clipping is enabled, using the norm already computed by clipping. Weight/master
and optimizer-moment finiteness is checked before checkpoint publication.
These checks expose failures; they do not alter the numerical training recipe
or penalize repetition. Exact quota-prefix accounting is performed for saves
and final reporting, outside the per-update hot path.

Implementation: [generation diagnostics](../yylab/train/sprint_quality.py),
[trainer health integration tests](../tests/test_sprint_health_integration.py),
[diagnostic tests](../tests/test_sprint_quality.py).

## Source allocation is not topic allocation

The 150M's 20% allocation to the corpus named elementary math does **not** mean
only 20% of its training was mathematical. Reasoning, Qwen and FABLE also contain
math. Their ingest manifests name them as reasoning distillation, Qwen
distillation and FABLE/Claude traces, but apply the broad domain override
`conversation`; those labels cannot measure topic shares. Their ingest recipes
used `selection_profile="preserve"` and disabled the ingest quality filter.

A separate [manual snippet review](../artifacts/repetition-repair-20260913/source-topic-spotcheck.json)
sampled **128 actual training row exposures**: 64 reasoning, 32 FABLE and 32 Qwen,
uniformly without replacement within each source. The first 256 tokens of each
2048-token row were decoded, for 32,768 inspected tokens. The assistant read and
categorized each snippet; this was not an automated topic classifier or an
independent human audit. The [sampling procedure](../artifacts/repetition-repair-20260913/source_topic_spotcheck.py)
records row IDs, pool IDs, hashes and the seed.

| Source | Math-dominant snippets | Substantial math mixed with another topic | Code/agent traces | Other snippets | Total |
|---|---:|---:|---:|---:|---:|
| Reasoning | 23 | 11 | 0 | 30 | 64 |
| FABLE | 3 | 0 | 23 | 6 | 32 |
| Qwen | 6 | 0 | 17 | 9 | 32 |

Reasoning includes many elementary word problems, plus algebra, trigonometry and
other mathematics. It also includes entailment, classification, reading
comprehension and general knowledge. FABLE/Qwen snippets were predominantly
coding, agent traces and general/scientific responses in this sample; four of
the Qwen coding snippets implement numerical algorithms. The other category
also includes qualitative science/finance and one symbolic-counting snippet.

This confirms substantial math **outside the named math corpus**. It neither
establishes nor rules out a majority-math overall token budget: the snippets
are short, can cross document boundaries, are stratified rather than weighted
by the mixture, and have subjective category boundaries. No full-mixture math
percentage is inferred from them. The earlier 34.6% reasoning-document heuristic
also cannot be used as an exact token share.

Another unambiguous bad training answer was found in reasoning sample 27
(training row 230055, pool row 194973): it states `27 × (5/18) = 15 m/s`, then
answers that a 330-meter train takes 22 seconds. The conversion is **7.5 m/s**,
which gives **44 seconds** under its stated relative-speed calculation. This
is a concrete source error, not an estimated corpus-wide error rate.

## Experiment 1: can Sprint learn a clean bounded task?

[Receipt](../artifacts/repetition-repair-20260913/controlled-learning.json) ·
[procedure](../artifacts/repetition-repair-20260913/controlled_learning.py) ·
[data and schedule](../artifacts/repetition-repair-20260913/controlled-data.json)

Three fresh models learned procedurally generated context-retrieval examples:
answer a named person's object location or coat color from the preceding text.
There were 512 distinct training examples and 128 held-out examples; evaluation
used the first 16 training and first 32 held-out examples. All arms used the same
8K tokenizer and seeded batches, FP32 weights/GPU optimizer, BF16 CUDA compute,
400 updates, batch 16 and sequence length 128. Each consumed **819,200 input
slots and 339,155 supervised targets**, with padding ignored in the loss.

| Model | Parameters | Final logged loss | Train answers | Held-out answers | Held-out EOS endings | Mean generated duplicate 4-grams |
|---|---:|---:|---:|---:|---:|---:|
| HF GPT-2 reference | 1,858,304 | 0.174710 | 8/16 | 11/32 | 32/32 | 0 |
| Sprint SDPA | 1,836,160 | 0.165402 | 10/16 | 20/32 | 32/32 | 0 |
| Sprint KDA + regular PLE | 5,086,537 | 0.163191 | 10/16 | 20/32 | 32/32 | 0 |

This establishes that the tested Sprint paths can learn useful conditional
behavior and stop with EOS on a small task. It does not establish general prose
quality, a successful 150M training recipe or architecture superiority. The
models are not matched in parameter count; GPT-2 also differs in normalization
and positions. Held-out examples share the procedural templates with training.

## Experiment 2: a data-only continuation did not remove the loops

[Receipt](../artifacts/repetition-repair-20260913/data-intervention.json) ·
[procedure](../artifacts/repetition-repair-20260913/data_intervention.py)

Both arms independently started from the original 150M PLE + KDA weights,
`ckpt_20260913_014748_0f6e04`, with fresh identical optimizers/schedules. One used
random rows from its original mixed dataset; the other used the existing
English-prose-only pool. The name `clean_prose` identifies that pool, not a claim
of comprehensive editorial or semantic cleaning. Sampling was without
replacement from the full respective pools.

Each arm ran **512 updates, 8,388,608 input slots and 8,372,224 supervised
targets**, microbatch 16 × accumulation 2, sequence length 512, peak LR 0.00003,
20 warmup updates and minimum LR ratio 0.5. The original run used sequence 2048;
both diagnostic arms changed to 512 equally. The only difference between the
two adaptation arms was the selected training data.

Evaluation used 12 predeclared prose prompts, greedy and temperature-0.8/top-p-0.95
sampling, up to 128 new tokens, without repetition penalties. Here `rep4` is the
fraction of repeated generated 4-gram occurrences; “high repetition” means
`rep4 >= 0.30`. These experimental counts are separate from the production
health evaluator's additional tail-loop test.

| Checkpoint | Mean greedy rep4 | High-repetition greedy outputs | Mean sampled rep4 | Prose validation loss | Math validation loss | Qwen validation loss |
|---|---:|---:|---:|---:|---:|---:|
| Original 150M | 0.598180 | 9/12 | 0.094420 | 3.087764 | 0.096249 | 1.121055 |
| Original-mixture continuation | 0.689347 | 10/12 | 0.091784 | 3.089979 | 0.096641 | 1.122755 |
| Prose-only continuation | 0.640000 | 12/12 | 0.052667 | 3.065594 | 0.108481 | 1.146548 |

**The data-only intervention failed its repetition objective.** Prose validation
loss improved slightly in the prose arm while every greedy probe exceeded the
repetition threshold. A falling held-out loss is therefore insufficient as the
repair acceptance criterion. This short continuation does not prove that corpus
quality is irrelevant or that a fresh better-data training would fail.

## Experiment 3: targeted unlikelihood reduced repetition

[Receipt](../artifacts/repetition-repair-20260913/unlikelihood-intervention.json) ·
[procedure](../artifacts/repetition-repair-20260913/unlikelihood_intervention.py) ·
[rollout source rows](../artifacts/repetition-repair-20260913/ul-rollout-source.json)

[Welleck et al., *Neural Text Generation with Unlikelihood Training*](https://arxiv.org/abs/1908.04319)
propose lowering the probabilities of undesirable outputs during training and
study token- and sequence-level objectives for repetitive generation. This
experiment independently implements a conservative variant; it is not a faithful
replication of that paper's full training recipe or reported quality results.

The variant penalizes only the final token of an exact generated 4-gram from its
third occurrence onward. Counting excludes prompt tokens and positions at/after
the first generated EOS. The auxiliary loss is the mean `-log(1-p)` across marked
positions, added as **CE + 0.1 × UL**. Positive cross entropy remains on real
prose rows. Every eight updates, eight fresh 64-token prose prefixes produce
96-token greedy rollouts; their negative masks are reused for those eight
updates. Evaluation prompts are not rollout training prompts.

Both arms again started from the original 150M weights, with matched fresh
optimizer/schedule and prose CE rows. They ran **128 updates and 2,097,152 CE input
slots / 2,093,056 CE supervised targets per arm**. Microbatch, accumulation,
sequence, LR, warmup and final LR ratio matched the preceding study. The UL arm
additionally forwarded **163,840 auxiliary training slots** and accumulated
**14,296 negative-position exposures**. Its 16 rollout refreshes generated 12,288
suffix slots; autoregressive rollout recomputation adds work beyond these CE
and auxiliary budgets. The arms are update/CE-data matched, not compute matched.

| Arm | Mean greedy rep4 | High-repetition greedy outputs | Mean sampled rep4 | Prose validation loss | Math validation loss | Qwen validation loss |
|---|---:|---:|---:|---:|---:|---:|
| CE control | 0.572000 | 10/12 | 0.092667 | 3.077130 | 0.098607 | 1.128371 |
| CE + targeted UL | 0.183333 | 1/12 | 0.014667 | 3.093289 | 0.098690 | 1.141474 |

Both arms averaged 128 generated tokens in each decoding mode, so this measured
reduction did not come from simply terminating the probes early. The UL arm
reduced greedy rep4 by approximately **68% versus its matched control**, but
prose and Qwen loss regressed slightly. Generated text still contains nonsensical
or weakly connected content, including broken-word substitutions and repeated
sentence structures (for example, the storm continuation degrades into
"pudd danc epigram"). Avoiding exact repeated 4-grams can improve this metric
while leaving the underlying prose failure visible. Less repetition does not establish coherence,
instruction following, reasoning accuracy or a usable checkpoint.

This is one short paired adaptation seed on the same 12 prompts used in the
preceding diagnostic. The following section adds independent fresh-prompt
validation; broader seed/quality-regression checks remain incomplete. The helpers in
[sprint_unlikelihood.py](../yylab/train/sprint_unlikelihood.py) and
[their numerical/masking tests](../tests/test_sprint_unlikelihood.py) support
experiments; ordinary Sprint training has no enabled UL objective or UI knob.

## Independent validation: the metric improvement is real but insufficient

[Independent receipt](../artifacts/repetition-repair-20260913/independent-ul-validation.json)
contains a predeclared fresh set of 24 prose prompts and 8 instruction prompts,
scored independently for the original model, CE control and targeted UL copy.
Each used greedy and temperature-0.8/top-p-0.95 sampling, up to **192 new tokens**,
without repetition penalties, ngram blocking or generation caching. This is
192 outputs total, with no training. Checkpoint integrity checks before/after
confirmed all files unchanged; all output logits were finite.

| Checkpoint | Fresh prose greedy mean rep4 | Severe prose loops | All 32 greedy mean rep4 | Severe loops across all 32 | Exact first-line checks passed, greedy |
|---|---:|---:|---:|---:|---:|
| Original 150M | 0.502446 | 17/24 | 0.427822 | 20/32 | 1/6 |
| CE control | 0.663245 | 23/24 | 0.573965 | 28/32 | 1/6 |
| CE + targeted UL | 0.296296 | 10/24 | 0.313252 | 15/32 | 1/6 |

The independent test confirms reduced exact repetition. It also still finds
severe repetition in **10/24 fresh prose outputs and 15/32 outputs overall**.
The UL prose outputs reached the full 192-token limit on average rather than
escaping the metric by terminating early. All three models passed only one of
the six first-line checks in each decoding mode. The independent
[text review](../artifacts/repetition-repair-20260913/independent-ul-review.md)
found **0/6 exact complete greedy answers in every arm**: the nominal passing
first line was followed by incorrect or unwanted text. Raw samples retain broken
word fragments and repeated sentence structures; lower exact 4-gram repetition
does not make those passages coherent. **The targeted candidate fails the
usable-checkpoint goal.** The metric reduction demonstrates that this objective
can change exact repetition; it does not establish a successful quality repair.

## Experiment 4: a whole-span variant also failed the greedy criterion

[Receipt](../artifacts/repetition-repair-20260913/unlikelihood-span-intervention.json) ·
[procedure](../artifacts/repetition-repair-20260913/unlikelihood_span_intervention.py)

Another isolated copy started from the original weights and used the same
128-update CE rows, schedule and rollout prefixes: **2,097,152 CE input slots
and 2,093,056 CE targets**. This variant marks all four tokens of a repeated
4-BPE-token span from its second occurrence, requiring a prior nonoverlapping
occurrence wholly inside the generated suffix. Prompt/EOS positions remain
excluded. Its mean negative-position loss used the lower coefficient **0.02**.
There were **53,704 negative-position exposures**.

The result still had **11/12 severe greedy loops**, with mean rep4 **0.516000**,
versus 10/12 and 0.572000 for the matched CE control. Sampling had 0/12 severe
loops and mean rep4 0.011333; both decoding modes averaged 128 new tokens.
Validation losses were prose **3.085349**, math **0.098659** and Qwen **1.134952**.
It failed the initial greedy acceptance criterion and was rejected before
running the independent fresh-prompt suite. Neither UL variant is promoted into
ordinary training or exposed as a production UI feature.

## Preservation and next evidence

The final scoped CPU/regression run passed **535 tests** with 16 existing PEFT
warnings; the [test log](../artifacts/repetition-repair-20260913/final-regression-tests.log)
records the run. Ruff passed for the touched production and new test modules.
These checks cover implementation contracts, including format inheritance and
diagnostic-template binding. They do not establish learned model quality.

The [CUDA health integration log](../artifacts/repetition-repair-20260913/cuda-health-tests.log)
records one further passing test (40.65 seconds, 14 existing deprecation
warnings): an actual eager generation check between compiled FLA KDA + PLE
optimizer updates preserves CUDA RNG, weights and gradients exactly, and the
next update agrees with a restored control within the test's tolerances.

The original registered 150M checkpoint remains the source artifact. Adaptation
arms are separate, unregistered diagnostic copies under
`artifacts/repetition-repair-20260913/`; they are not published replacements or
approved models for Antidoom. No new billion-token training run is claimed here.

Further interventions must improve actual prose and preserve capability on
independent prompts, with results reported alongside repetition metrics. The
present independent evaluation rules out accepting this targeted candidate as
the missing usable-checkpoint fix.

The subsequent [data duplication and diversity audit](SPRINT_DATA_DUPLICATION_2026-09-13.md)
measures the complete saved payload, distinguishes explicit replay from repeated
content and templates, and documents a conservative preparation filter plus the
remaining source-coverage gaps. It does not change this study's quality outcome.
