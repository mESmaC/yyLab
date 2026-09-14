# Sprint 150M PLE + KDA: one-billion-token base training

Status: **completed and independently verified**. Training launched at 01:47:49
and finished at 06:19:09 local time on 2026-09-13, as
`run_20260913_014749_780f46`. The final checkpoint is
`artifacts/checkpoints/ckpt_20260913_014748_0f6e04/`, containing the native model,
tokenizer, chat template and resumable optimizer state. All 488,281 scheduled
rows and 999,999,488 input slots were consumed. The
[training launch receipt](../artifacts/sprint-150m-ple-kda-1b-20260913/training-launch.json),
[exact recipe](../artifacts/sprint-150m-ple-kda-1b-20260913/training-recipe.json) and
[live progress](../artifacts/runs/run_20260913_014749_780f46/progress.json) identify
the real run. A
[frozen source manifest](../artifacts/sprint-150m-ple-kda-1b-20260913/source-training/manifest.json)
accompanies the launch.

The full run saved its first periodic checkpoint at update 1,000, after
32,768,000 input slots. Prose validation loss improved from 5.390714 at update
500 to 4.284482 at update 1,000. The
[saved-bundle readback](../artifacts/sprint-150m-ple-kda-1b-20260913/step1000-bundle-validation.json)
loaded all 150,099,767 parameters exactly on CPU, verified finite weights and
tied embeddings, and generated four tokens through the uncached fallback. This
is an early checkpoint compatibility check, not final model evaluation.

The first save is under
`artifacts/runs/run_20260913_014749_780f46/checkpoints/step-00001000/`.
The completion follow-up, `finish-sprint-150m-ple-kda-training`, monitored
the worker and ran the prepared final evaluator after full-budget completion.
It pauses after reporting the results. Antidoom and release publication are not
part of this run.

The completed 120-step real-corpus pilot is `run_20260913_014414_3cc318`; its
[launch receipt](../artifacts/sprint-150m-ple-kda-1b-20260913/pilot-launch.json),
[recipe](../artifacts/sprint-150m-ple-kda-1b-20260913/pilot-recipe.json) and
[summary](../artifacts/runs/run_20260913_014414_3cc318/summary.json) are separate
from the full run. Pilot results below do not establish final trained-model quality.

## Selected model and training recipe

The production estimator reports **150,099,767 unique parameters**. The model
has width 1,024, eight physical layers, eight heads of dimension 128, a GELU
feed-forward width of 2,816 and a vocabulary of 8,192 tokens. Its 3:1 hybrid has
six KDA layers and two full causal SDPA layers. Regular PLE adds a separate token
table before each of the seven later layers; the ordinary input embedding stays
tied to the output head. Maximum context and training sequence length are both
2,048. The complete model configuration and allocation are recorded in the
[selected microbatch probe](../artifacts/sprint-150m-ple-kda-1b-20260913/gpu-probe-micro8.json).

| Parameter component | Unique parameters |
| --- | ---: |
| Tied input/output embedding | 8,388,608 |
| Attention, including KDA controls and convolution | 36,836,144 |
| Feed-forward layers | 46,137,344 |
| Block/final normalization | 17,408 |
| Seven regular PLE tables and gates | 58,720,263 |
| **Total** | **150,099,767** |

This is a manual shape. Its parameter count is slightly above the nominal 150M
label; it preserves a 2.75-times-width feed-forward layer rather than shrinking
it to the very narrow FFNs in some earlier equal-budget speed comparisons.
This allocation has not yet been shown to match a standard Transformer's quality.

The selected recipe uses FP32 stored parameters, BF16 CUDA compute and fused GPU
AdamW with FP32 moments. There is no FP4/FP8 weight recipe, quantized optimizer
state or CPU optimizer offload in this run. It uses `torch.compile` in default
mode, automatic FLA dispatch, `kda_disable_recompute=True`, no block activation
checkpointing and four CPU threads. The
[saved training recipe](../artifacts/sprint-150m-ple-kda-1b-20260913/training-recipe.json)
records the full-run settings:

| Setting | Selected value |
| --- | --- |
| Microbatch × accumulation | 8 × 2 sequences |
| Input slots per full optimizer update | 32,768 |
| Requested token budget | 1,000,000,000 input slots |
| Scheduled rows / input slots | 488,281 / 999,999,488 |
| Supervised next-token targets | 999,511,207 |
| Optimizer updates | 30,518; final update has nine sequences |
| Peak learning rate / minimum ratio | 0.0003 / 0.1 |
| Warmup | 500 updates |
| Adam betas / epsilon | (0.9, 0.95) / 1e-8 |
| Weight decay / gradient clipping | 0.1 / 1.0 |
| Logging / evaluation / saving | Every 20 / 500 / 1,000 updates |
| Validation | English-prose validation split, up to 16 microbatches |
| Tokenizer | `tok_20260911_121147_1f5f01` |

Input slots include inserted document EOS tokens. Each independent 2,048-token
row supplies 2,047 next-token targets, so the target count is lower than the
input budget. Rounding drops 512 requested input slots. The pilot uses 120
updates and a 20-update warmup, and is not part of the final run's token budget.

## Fixed data allocation

The [fixed data config](../artifacts/sprint-150m-ple-kda-1b-20260913/fixed-data-config.json)
and [preparation receipt](../artifacts/sprint-150m-ple-kda-1b-20260913/preparation-receipt.json)
identify the selected registered corpora, source snapshots, tokenizer, full-pass
pool checksums, deterministic sampler and final cache. Nominal percentages are
rounded to complete rows using largest remainders.

| Selected source | Nominal mix | Rows | Input slots | Repeated row exposures |
| --- | ---: | ---: | ---: | ---: |
| Reasoning traces/distillation | 45% | 219,727 | 450,000,896 | 0 |
| Elementary math explanations | 20% | 97,656 | 199,999,488 | 0 |
| Fable | 19% | 92,773 | 189,999,104 | 0 |
| Qwen distillation | 10% | 48,828 | 99,999,744 | 27,586 |
| English prose | 5% | 24,414 | 49,999,872 | 19,508 |
| Kimi distillation | 0.98% | 4,785 | 9,799,680 | 1,925 |
| English quotes | 0.02% | 98 | 200,704 | 29 |
| **Total** | **100%** | **488,281** | **999,999,488** | **49,048** |

All selected validation/test document hashes are collected before training
sources. Training then excludes matching exact rendered text and resolves exact
training duplicates in declared corpus order. Each retained corpus is packed
through a full pass before sampling: no first-prefix shortcut is used. A seeded
permutation samples pool rows without replacement within each cycle; small
pools deliberately repeat complete cycles plus a sampled remainder. A final
global permutation mixes the selected rows. This gives 439,233 distinct selected
pool rows, or 899,549,184 input positions, plus 100,450,304 repeated input
positions. These are row-exposure counts, not claims of semantic uniqueness.

The [independent data validation](../artifacts/sprint-150m-ple-kda-1b-20260913/data-validation.json)
reproduced the sampler, compared **all 488,281 payload rows** with their selected
source-pool rows, verified disjoint retained document hash inventories against
223,639 unique held-out hashes, and reconciled token/EOS/tail counters. Exact-text
deduplication does not establish prompt-family or semantic decontamination.
The validation pool has 120 complete rows (245,760 input slots and 245,640 targets).
That held-out prose loss is a limited monitoring signal rather than an evaluation
of every training domain.

The materialized training cache key is
`8f3dd351394725bec979955c18e625d5c583a10432c6e241a566be211b704064`.
The production `fixed_token_mix` contract prevents the worker from replacing this
allocation with ordinary concatenation if the cache is missing. Normalization
receipts record canonicalization of `domains=[]` to `None` without changing token
payloads, and verify reuse by the ordinary training worker. Exact resume retains
the descriptor, cache identity and consumed row cursor.

## Throughput and correctness evidence

These are complete-step **synthetic admission probes** on the Windows RTX 5070
with PyTorch 2.11.0+cu128. Each excludes three warmup updates and measures twelve
updates at the same 32,768 input slots/update. They include training forward,
backward and optimizer work; preparation, compilation, evaluation and saves are
outside the reported steady-state rate.

| Candidate | Input slots/second | Peak PyTorch allocated memory | Receipt |
| --- | ---: | ---: | --- |
| Microbatch 4 × accumulation 4 | 56,383 | 5.99 GiB | [micro4](../artifacts/sprint-150m-ple-kda-1b-20260913/gpu-probe-micro4.json) |
| **Microbatch 8 × accumulation 2** | **58,094** | **9.48 GiB** | [micro8](../artifacts/sprint-150m-ple-kda-1b-20260913/gpu-probe-micro8.json) |

Both probes completed with finite parameters. Allocator peaks are not physical
GPU residency measurements. The receipts retain source hashes and have different
KDA source revisions; this table records observed candidates, not an isolated
causal microbatch speedup. Repeated synthetic random batches also do not measure
real-corpus PLE lookup behavior, data transfer stalls or long-run learning.

The pilot completed all 120 updates and **3,932,160 input slots**, with finite
weights/state and both a periodic and final checkpoint. Its last reported
training-loss window was **4.096497**. Full prose-validation loss fell from
**7.111844** at update 40 to **6.650670** at update 80 and **6.510159** at update
120. The
[pilot metrics](../artifacts/runs/run_20260913_014414_3cc318/metrics.jsonl)
give **57,913 input slots/second** over updates 21–120 using the difference in
cumulative training time, which excludes evaluation/save overhead. The full
pilot wall time was **152.48 seconds**, or **25,787 input slots/second** including
startup and other overhead. This short run validates admission and an initial
learning signal; it does not establish final quality. The full run starts from
scratch with its own 500-update warmup.

Validation completed before the full run:

- **123 CPU tests passed** across the Sprint training/UI, PLE training and new
  [KDA training tests](../tests/test_sprint_kda_training.py). The new tests include
  actual tiny full-PLE KDA training, exact checkpoint resume and continuation
  with and without activation checkpointing, legacy SDPA inheritance, planner
  counts, precision guards and UI architecture propagation.
- **36 CPU posttraining/runtime tests passed**, including the
  [checkpoint/SFT/FTPO/chat integration tests](../tests/test_sprint_posttrain.py).
  Saved full/QR PLE + KDA models support masked SFT updates and resaving, PEFT
  policy/reference objectives, adapter roundtrips, mining scores and NLL.
- **Six CUDA tests passed**, recorded in the
  [validation log](../artifacts/sprint-150m-ple-kda-1b-20260913/cuda-validation.log).
  The [numerical diagnostic](../artifacts/sprint-150m-ple-kda-1b-20260913/gpu-numerics.json)
  separately compares fused BF16 recurrence with the reference, including plain,
  padded and segmented inputs. Its measured full-model gradient relative norms
  are about 1.93–2.23%; these bounded numerical observations do not establish
  long-run convergence equivalence.

## Checkpoint use and evaluation

The saved output is an initial base checkpoint for continued pretraining and
later SFT/Antidoom work. Periodic and final saves retain tokenizer, architecture,
PLE/KDA weights and exact-resume state. **Continue checkpoint** starts a new
optimizer and schedule; **Resume optimizer state** preserves the existing one.
The prepared Antidoom recipe uses up to 512 prompt tokens plus 128 generated
tokens, which fits this model's 2,048-position capacity. No Antidoom/RL training
is launched by this base-training recipe.

KDA sampling currently recomputes the full prefix per generated token. Chat and
Antidoom select the uncached path; external recurrent/KV caches are unsupported.
Antidoom's tied-head LoRA remains adapter-only. A successful compatibility test
does not establish chat ability or resistance to repetition. One billion input
tokens is an initial base-training budget; completed loss, held-out metrics and
samples must be assessed before claiming model or chat quality. The final
diagnostics below do not establish an assistant-ready model.

The subsequent [repetition investigation](SPRINT_REPETITION_AUDIT_2026-09-13.md)
reproduced the problem in early/intermediate saves and older non-PLE/SDPA models.
It found that none of the saved chat template's special role tokens occurred in
this run's actual training inputs. Plain-text repetition also persists through
independent FP32 reference generation, so the format mismatch is not the sole
cause. The investigation documents template-heavy data, misleading historical
corpus balancing, and missing generation quality checks, with read-only receipts.

## Verified completion results

The [completion audit](../artifacts/sprint-150m-ple-kda-1b-20260913/completion-verification.json)
reconciles the [worker summary](../artifacts/runs/run_20260913_014749_780f46/summary.json),
saved training state and [final evaluation](../artifacts/sprint-150m-ple-kda-1b-20260913/final-evaluation.json).
An independent CPU audit found all 124 parameter tensors and all 248 Adam
moment tensors finite, with every optimizer step counter at 30,518. The final
evaluator loaded the native checkpoint with no missing, unexpected or mismatched
weights, verified all frozen source hashes, and confirmed that model/tokenizer/
trainer files did not change during scoring. No further training or RL ran.

| Completion metric | Result |
| --- | ---: |
| Unique trained parameters | 150,099,767 |
| Completed rows / updates | 488,281 / 30,518 |
| Input slots / supervised targets | 999,999,488 / 999,511,207 |
| Training-loop time | 16,159.35 s — 4 h 29 m 19 s |
| Complete worker wall time | 16,274.42 s — 4 h 31 m 14 s |
| Training-loop input throughput | 61,883.64 slots/s |
| Complete-run input throughput | 61,446.08 slots/s |
| Last reported training loss, final 18 updates | 1.104806 |
| Target-weighted loss, final 518 updates | 1.062331 |

The final 518-update average covers 16,951,207 supervised targets. These losses
describe the mixed training distribution, not general reasoning accuracy. Worker
wall time includes cache admission, compilation, evaluation and saves; the
separate one-time corpus materialization took 387.51 s and is excluded from both
throughput figures. The earlier synthetic probes and pilot are also excluded.
The log contains a late compilation-cache limit warning for the final partial
batch; training still completed its exact budget and saved successfully.

The final evaluator used FP32 stored weights, BF16 CUDA compute and ordinary
Torch cross-entropy. Its six domain scores and six fixed greedy generations
completed with finite outputs in 31.97 s of scoring/check time. Domain loss is
mean next-token negative log likelihood in nats, weighted by actual targets.

| Held-out domain | Targets | Loss | Perplexity |
| --- | ---: | ---: | ---: |
| English prose validation | 245,640 | 3.087764 | 21.927991 |
| English prose test | 325,473 | 3.365875 | 28.958837 |
| Elementary math validation | 131,008 | 0.096249 | 1.101033 |
| Qwen validation | 131,008 | 1.121055 | 3.068089 |
| Kimi validation | 131,008 | 0.867986 | 2.382108 |
| FABLE validation | 131,008 | 1.177853 | 3.247394 |

The samples are a substantial quality limitation. Plain English starts with
recognizable prose but loops on “I'm sorry, I don't know.” The chat version
ignores the requested three sentences about a rainy morning and repeats unrelated
prose. Neither version of the apples word problem produces the correct answer
14: the plain version initially answers 3, then writes an invalid symbolic
breakdown; the chat version generates unrelated repetitive text. The desk-tidying
request produces reasoning boilerplate in plain format and a repeated sentence
fragment in chat format, without giving the two requested suggestions.

These six greedy samples demonstrate weak instruction following and substantial
repetition. The low math validation loss does not establish word-problem-solving
ability. This is a complete pretrained checkpoint available for continued
pretraining and later posttraining; it should not be described as a usable
assistant yet. Full prompts, generated token IDs, decoded outputs, exact settings
and dataset/source hashes remain in the final evaluation receipt.
