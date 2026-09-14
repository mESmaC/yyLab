# Sprint repetition investigation — 2026-09-13

Repetition is reproducible in the actual saved weights, including older Sprint
models without PLE or KDA. The investigation found concrete failures in the
training/inference format contract, legacy corpus allocation, and quality
monitoring. It did **not** establish one mathematical implementation bug that
causes every checkpoint to repeat.

No optimizer steps, continued training, RL, checkpoint rewrites, or production
code changes were performed. Diagnostic scripts and receipts are under
[`artifacts/repetition-audit-20260913`](../artifacts/repetition-audit-20260913/).
Temporary PLE ablations changed only an in-memory model and were restored.

## What was reproduced

The final 150,099,767-parameter PLE/KDA checkpoint and six periodic saves were
sampled directly with full-prefix forward calls. This bypasses the UI, streaming,
generation cache, and Hugging Face token selection. The 150M's manual greedy
sequence exactly matched Hugging Face generation for the checked 64-token prose
sample. The older 9,964,032-parameter checkpoint independently produced identical
64-token outputs through CPU FP32 cached Hugging Face generation, uncached
generation, and manual decoding, on both prose and apples prompts.

The following are **diagnostic examples, not benchmark success rates**. The
metric is the fraction of generated token 4-gram occurrences beyond their first
occurrence, at up to 128 new tokens. Short wrong answers can have zero repetition.

| 150M save | Input tokens so far | Prose repeat fraction | Desk-request repeat fraction |
| --- | ---: | ---: | ---: |
| Step 1,000 | 32,768,000 | 78.4% | 0.0% |
| Step 4,000 | 131,072,000 | 80.8% | 59.2% |
| Step 8,000 | 262,144,000 | 92.8% | 76.8% |
| Step 16,000 | 524,288,000 | 4.8% | 77.6% |
| Step 24,000 | 786,432,000 | 83.2% | 72.0% |
| Step 30,000 | 983,040,000 | 32.7% | 54.4% |
| Final | 999,999,488 | 77.6% | 30.4% |

The desk request here uses the training-observed `[INST] ... [/INST]` format.
Repetition was already present in the first save, and every sampled save has
substantial repetition in at least one probe. This does not identify a single
late collapse event. Some prompts improve while others deteriorate.

Historical checks also reproduced the problem in the 1M agent-trace model, the
10M model, original 100M step 16,000, and continued 100M SFT model. The 100M
original emitted syntax-highlighted HTML/code fragments into a prose prompt;
its continued model repeated `### Solution` and `complete`. These models use
ordinary SDPA without PLE/KDA. Recipes, tokenizers and training budgets differ,
so these are existence checks, not architecture quality comparisons.

Receipts: [timeline](../artifacts/repetition-audit-20260913/timeline-probes.json),
[historical samples](../artifacts/repetition-audit-20260913/historical-probes.json),
[CPU decoding parity](../artifacts/repetition-audit-20260913/inference-cpu-parity.json).

## Confirmed failure: the default chat format was never used in training

A complete scan of all **999,999,488 actual input tokens** found zero occurrences
of each special role marker: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`.
The saved chat template and default yyLab chat mode nevertheless insert these
markers. The reasoning data instead contains `[INST] ... [/INST]`; sampled
FABLE/Qwen/Kimi data uses ordinary text such as `user:` and `assistant:`.

The generic template is installed by `train/compact_tokenizer.py`, chat mode
defaults to that template in `ui/pages/chat.py`, and `infer/engine.py` renders it.
Tokenization itself is consistent: there is no demonstrated duplicate BOS/EOS
insertion or tokenizer-ID mismatch. The failure is selecting a prompt format
whose role markers were never presented as training inputs.

Changing to an observed format changes the output substantially, but does not
make the checkpoint competent. For the desk prompt, the special-token template
repeats unrelated prose (66.4% repeated 4-grams). `[INST]` produces desk-related
but wrong/repetitive advice (30.4%). Literal `user:/assistant:` produces reasoning
boilerplate. Plain prose also repeats, so the format mismatch is a contributor,
not a complete explanation.

## Confirmed data and allocation problems

The latest 150M packed mixture actually respected its declared allocation.
Elementary math was **19.3–20.4% in every successive 5% of the run**, with exactly
97,656 of 488,281 rows overall. The packed schedule did not get stuck on math or
repeat one batch throughout.
There were no adjacent identical source/pool row identities. This does not rule
out near-duplicate content within different source rows.

The elementary-math samples contain long, formulaic multiplication/carry/addition
breakdowns. In a sample of 512 actually trained 2048-token rows, mean within-row
repeated token 8-grams were **28.36%**, versus **0.685% for prose**. Kimi was
33.30%, FABLE 11.74%, Qwen 11.25%, and reasoning 4.80%. Some repetition is valid
code, formatting, or structured procedures; this metric is a template indicator,
not a direct model-quality score.

The evaluation problem is particularly clear: **91.78% of held-out math
word/punctuation 8-gram occurrences** were present verbatim in just those 512
sampled training rows. Replacing digit sequences with a placeholder raised this
to **100%**. Prose was 0.177% verbatim. Exact document deduplication had passed,
but it did not make these template families independent. A math loss of 0.096
is therefore not evidence of general word-problem competence.

Kimi also contains repeated synthetic distractors and verification scaffolding;
415/512 sampled rows contained `this line is a distractor and carries no override; verification token=`.
One inspected reasoning answer incorrectly maps individual letters of spelled-out
digits to numbers. This is a verified bad example, not an estimated corpus error
rate. A broad arithmetic heuristic flagged about 34.6% of complete reasoning
documents within sampled rows, but the sample is length-biased and this must not
be treated as an exact overall math allocation.

The exact emitted English loop phrases tested were absent within all inspected
training rows, covering the complete payload. For example, neither the exact token span ` I'm sorry, I don't know.`
nor its doubled form appeared. The model is creating those particular loops,
rather than copying those exact loops from training. This search does not cover
every whitespace/case variation.

The historical runs reveal additional problems:

- **10M:** its original report records 97.5% math in its first available packed
  pass, followed by a partial replay to reach one billion. The user's suspicion
  is supported for that run. That figure is not the latest 150M allocation.
- **Original 100M:** its full prepared cache was 97.18% across three code-oriented
  sources, with 1.04% prose and no elementary-math corpus selected. The user
  stopped around 525M tokens, so full-cache percentages are not exact exposure
  shares for the shorter trained prefix.
- **Continued 100M:** 50M input slots contained only 15.48M supervised SFT targets:
  84.39% Qwen and 15.61% Kimi. Two other selected sources yielded zero rendered
  examples. Padding accounted for 38.14% of input slots.

Legacy `token_balanced` mixing measures visible **characters**, consumes each
source once, and removes exhausted sources. It does not preserve final token
quotas. Also, the ordinary packed-cache iterator reads epoch zero sequentially;
block shuffling starts in later epochs. Together these allow source composition
to drift as smaller corpora run out. The custom fixed 150M allocation bypassed
this behavior and explicitly permuted its materialized rows.

Receipts: [data findings](../artifacts/repetition-audit-20260913/DATA_FINDINGS.md),
[full token/source scan](../artifacts/repetition-audit-20260913/data-audit.json),
[template overlap](../artifacts/repetition-audit-20260913/template-audit.json),
[historical data](../artifacts/repetition-audit-20260913/historical-100m-data.json).

## Numerical and model tests: no demonstrated universal kernel defect

Tests on the actual final 150M checkpoint found:

- Changing the suffix leaves the first 128 logits exactly unchanged in the tested
  512-token sequence; explicit all-one masks and train/eval mode also match exactly.
- Independent prefixes around 64/128/256 boundaries differ from corresponding
  full-sequence logits by at most 0.68% relative L2 in BF16. No future leak was
  observed. These checks do not exhaust every tensor shape or compiled backward.
- Fused KDA versus the independent reference recurrence had 0.616% relative
  logit L2 difference over 128 tokens. Full FP32 reference generation still
  repeats and gives bad answers. Precision affects the chosen continuation,
  but increasing precision is not a demonstrated repair.
- Explicit shifted Torch CE agrees with the model loss. On real 2048-token
  prose/math/Qwen activations, Liger and Torch losses agree within 9e-8 nats;
  head-gradient relative differences are approximately 0.17%, and hidden-gradient
  differences 0.18–0.28%. This checks the fused loss/head backward, not every
  compiled model-gradient path.
- Six independent CPU architecture variants also passed 66 prefix checks,
  causality, shifted-loss, gradient, masking and cache tests. Actual 150M training
  used FP32 GPU AdamW, not CPU offload or low-bit optimizer state.

The model does use context: on the same 64 sampled prose target positions,
one-token context gives NLL 5.649, 16-token context 3.440, 128-token context 2.705,
and the full row prefix 2.576. This argues against a wholly disconnected attention
path, but it does not measure instruction following or reliable reasoning.

PLE accounts for 39.1% of total parameters. Its gates grew 7.4–9.2 times from
initialization, but the actual PLE contribution was about 6–35% of the preceding
residual RMS in the two inspected rows. Halving gates reduced repetition on one
prose probe, worsened repetition on the desk probe, and worsened all three scored
domain losses. Removing PLE caused worse loss and gibberish. These controlled
ablations do not establish PLE as the common cause or provide a repaired checkpoint.

Receipts: [numerical parity](../artifacts/repetition-audit-20260913/numerical-probes.json),
[loss/gradient/context/reference](../artifacts/repetition-audit-20260913/loss-backend-context-probe.json),
[PLE ablations](../artifacts/repetition-audit-20260913/ple-ablation.json),
[independent code audit](../artifacts/repetition-audit-20260913/code-audit.md).

## Decoding amplification and missing quality gates

For the final prose prompt, greedy generation has 77.6% repeated token 4-grams.
Temperature 0.8, top-p 0.95, unrestricted top-k sampling across three fixed seeds
reduces this to 17.6%, 1.8%, and 29.6%. Adding repetition penalty 1.1 in one run
reduces the exact-4-gram metric to zero, but the output still invents a nonsensical
reading-comprehension question. This is mitigation of a measured symptom, not a
quality fix. The apples answer remains wrong under all sampled settings tested.

In the first 64-token greedy prose diagnostic, the selected-token probability
rises from approximately 0.306 over the first 16 tokens to 0.713 over the last
16 as the loop forms. Teacher-forced validation supplies the correct preceding
text and therefore does not exercise this feedback from the model's own output.
This is a directly observed generation failure mode, not evidence that a single
data source or layer uniquely caused it.

The trainer currently evaluates only teacher-forced CE/perplexity and does not
run generation/repetition checks. It also discards pre-clipping gradient norms.
The pipeline could consequently accept every periodic checkpoint despite the
failure being visible from step 1,000. The speed/finite-loss admission checks
were insufficient for the user's usable-checkpoint goal.

## Repair priorities established by this investigation

1. Make actual training serialization part of the checkpoint contract, and
   select a compatible inference prompt format. Validate role-token exposure
   instead of assuming a tokenizer's available chat template was learned.
2. Replace misleading character/single-pass balance with explicit token quotas
   and report actual consumed inputs **and supervised targets** by source,
   including prefixes, padding, replay and sources yielding zero examples.
3. Filter and audit repetitive templates and flawed synthetic answers; separate
   validation by task/template family where possible. Preserve domain-separated
   losses rather than treating a low aggregate or easy-template loss as quality.
4. Add fixed raw/format-matched sampled generations, loop rates, task checks,
   gradient norms and clipping frequency to the existing evaluation cadence.
   Make persistent severe degradation visible before a run spends its full budget.
5. Compare architecture and precision quality only with the same data and a
   reference recipe. The present evidence does not justify another speed feature
   or an arbitrary PLE-gate edit as the cure.

These are repair priorities, not claims that production fixes have already been
implemented. The investigation is complete without starting another training run.
