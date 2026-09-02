# SPALMER 10M Training Validation — 2026-09-02

## Outcome

The first yyLab-driven SPALMER checkpoint completed successfully and validates
the base-training pipeline. The checkpoint is structurally sound, loads
strictly, has finite held-out metrics, produces coherent corpus-style text with
unrestricted top-2 routing, used the GPU, and demonstrably kept its complete
FP32 Adam moments in pinned CPU memory.

Two inference defects were discovered after training. Both were isolated
without retraining:

1. fla-core 0.5.2's fused recurrent KDA kernel reads unmasked padded gate lanes
   when the head width is not a power of two. The model's width is 12, which
   caused intermittent NaN states. SPALMER now uses stable `chunk_kda(T=1)` for
   non-power-of-two widths and retains the faster fused kernel for safe widths.
2. A later residency-controller change incorrectly made the number of resident
   candidates equal the per-token top-k. SPALMER now keeps resident candidates
   `m` and active experts `k` independent and preserves this checkpoint's
   trained `k=2` during residency probes.

The remaining limitation is architectural rather than a transport failure:
model 1 learned diffuse top-2 routing across the full 200-expert pool. A single
request-global subset of 20, 40, or 60 experts does not preserve its output
quality. Physical CPU expert offload is exact under the same logical subset,
but the current request-global selection policy is not yet a quality-preserving
replacement for unrestricted routing.

Model 2 was not started. A second identical 100M-token run would not isolate or
repair this inference-policy mismatch.

## Successful run

| Field | Result |
|---|---:|
| Run | `run_20260902_153700_47d53b` |
| Checkpoint | `ckpt_20260902_153700_7e1e9d` |
| Requested parameters | 10,000,000 |
| Actual parameters | 10,006,088 (+0.06088%) |
| Vocabulary | 4,096 |
| Shape | 8 layers, width 96, 8 heads |
| Token mixers | 6 KDA + 2 MLA |
| KDA head width | 12 |
| Experts | 200 identities, width 6, trained top-2 |
| Context | 512 tokens |
| Microbatch / accumulation | 32 / 2 |
| Tokens per optimizer step | 32,768 |
| Steps | 3,052 / 3,052 |
| Requested training tokens | 100,000,000 |
| Actual supervised tokens | 100,007,936 |
| Wall time | 3,028.55 seconds (50m 29s) |
| Final throughput | 33,021.8 tokens/s |
| Device | CUDA, BF16 parameters and gradients |
| Expert numerics | MXFP4 weight + MXFP8 activation reference QAT |
| Optimizer | AdamW; FP32 moments in pinned CPU memory |
| Seed | 42 |

The completed checkpoint is
`artifacts/checkpoints/ckpt_20260902_153700_7e1e9d/spalmer.pt`.

## Training health

- Model loss decreased from 8.33869 to 0.16857; the final complete objective
  was 0.17916.
- The early-50-step model-loss median was 4.99904 and the late-50-step median
  was 0.17245.
- Final gradient norm was 0.12506 and all 612 telemetry rows were finite and
  strictly monotonic in step, token count, and elapsed time.
- Objective decomposition matched its recorded components within `2.63e-7`.
- Actual grouped-expert padding stayed between 1.432x and 1.578x. The removed
  global-maximum padding strategy would have reached 77.631x on the observed
  skewed route.
- All 200 expert utilization, predicted-NLL, and quantization-error EMAs were
  finite and nonzero. The final promoted identities were 62 and 107.
- Observations reconciled exactly: 3,052 potentiation updates and 6,104
  surprise observations for two microbatches per optimizer step.
- Live training VRAM remained approximately 5.19 GiB. A 10-second hardware
  sample observed 27–64% SM utilization (about 39% average), reasonable for a
  10M model using reference QAT and a CPU-offloaded optimizer.

## Optimizer offload proof

The placement receipt was captured after optimizer step 1:

- 486 moment tensors, all FP32 on CPU;
- 80,048,704 moment bytes, exactly `10,006,088 × 2 × 4`;
- pinned host memory enabled;
- model parameters remained BF16 on `cuda:0`;
- bounded update chunks of 1,048,576 elements.

This validates optimizer-state offload during training. Expert parameter
masters were accelerator-resident during training; physical expert-weight
offload is currently an inference-only backend.

## Data and tokenization reconciliation

The run consumed only the `elementary_math-v1` training split:

- 68,176 documents;
- 100,135,262 rendered content tokens;
- 68,171 inserted document-boundary tokens;
- 195,328 forwarded sequences;
- 100,007,936 supervised targets;
- zero padding inputs, packing replays, ignored targets, or discarded
  incomplete accumulation batches;
- 169 rendered positions remained after the final complete batch and were not
  forwarded;
- every persisted reconciliation invariant is true;
- the run and checkpoint tokenized-view manifests are byte-identical.

There is no evidence here of malformed ingestion or token accounting. Exact
byte accounting also succeeded on the sampled validation and test material.
The HF tokenizer's overlapping offsets for some synthetic multi-byte Unicode
remain a general BPB-metadata limitation, not a path used by this training run.

## Checkpoint and held-out evaluation

Static validation passed 80 checks with no artifact failure:

- format `spalmer.prototype.checkpoint.v5`;
- file size 20,122,953 bytes;
- SHA-256
  `a457c7d7164d90254e35c32f492d6bb8083fbae9b63d90da781c007e84599083`;
- strict CPU load succeeded;
- all 243 parameter tensors were BF16, finite, and accounted for;
- parameter-plan fingerprint
  `99daebe421fd431a853c47979484f83e2d311d86e3a596d07a4005df56f67a1f`.

Held-out GPU results used 16,384 targets per split:

| Split | NLL | Perplexity | Bits/byte |
|---|---:|---:|---:|
| Validation | 0.170782 | 1.18623 | 0.11716 |
| Test | 0.173731 | 1.18974 | 0.11852 |

All values, recurrent states, and logits were finite after the KDA correction.
Every expert was exercised across the held-out batches, with mean active top-k
exactly 2. Peak PyTorch CUDA allocation for this evaluation was approximately
343.8 MB.

## Output inspection

With unrestricted 200-candidate/top-2 routing, three held-out-style prompts
each generated 96 finite tokens with no replacement characters or repeated
8-grams. The model learned the expected procedural register:

- addition continued with `Step 1:` and a column-addition explanation;
- subtraction continued with a column-subtraction explanation;
- multiplication continued with a long-multiplication explanation.

The arithmetic and operand grounding were unreliable. Examples included
`7 + 2 = 9` for the wrong source digits and incorrect borrow/multiplier choices.
For a 10M checkpoint, this is sufficient evidence of coherent learned form but
not mathematical capability.

## Recurrent KDA defect and correction

One 66-token prompt reproduced the original failure in 28 of 30 clean trials.
Prefill was finite, then fla-core's `fused_recurrent_kda` first produced 144
NaNs in layer 1's FP32 `[1, 8, 12, 12]` recurrent state. Full-prefix FLA and the
reference recurrence remained finite.

The installed kernel rounds key width 12 to 16 but loads the four padded gate
lanes without the key mask. The corrected dispatch is:

- non-power-of-two head width: optimized `chunk_kda(T=1)`;
- power-of-two head width: faster `fused_recurrent_kda`.

The formerly failing prompt then passed 30/30 source-runtime trials and 10/10
fresh installed-package/offload trials. A separate width-16 control completed
2,048 chained fused steps with finite output and close chunk-kernel agreement.

## Physical expert offload

The inference offload backend passed placement, transfer, restoration, and
numeric-parity checks:

- eight complete expert master banks stayed in pinned CPU memory;
- master bytes: 5,529,600;
- CUDA cache capacity: 20 identities;
- live two-identity cache bytes: 55,296;
- every layer held the same resident identities;
- 576 transferred rows and 560 evictions left exactly 16 live rows
  (`8 layers × 2 identities`);
- cumulative transfer bytes: 1,990,656;
- peak PyTorch CUDA allocation: 30,606,848 bytes;
- fixed-seed guarded sampling reproduced exactly.

Under an identical dynamic residency decision, the all-GPU and CPU-offloaded
runtimes selected the same identities, produced bit-identical prefill logits
(maximum difference 0), and emitted identical 96-token greedy continuations.
The offload transport is therefore not responsible for quality changes.

## Residency-policy finding

Training allowed each token to route top-2 over all 200 identities. The former
inference controller instead made resident count equal active top-k, so a
two-resident request executed both experts for every token and a four-resident
request executed an untrained top-4. That regression is now corrected: resident
candidate count `m` and active top-k `k` obey `2 <= k <= m`, and model 1 remains
at its trained `k=2`.

Decoupling is necessary but does not make a request-global cap sufficient for
this checkpoint. Read-only cap sweeps at fixed top-2 found:

| Resident candidates | Result versus unrestricted routing |
|---:|---|
| 20 (10%) | All tested selection policies degraded semantics; an oracle set covered only 25.2% of route events |
| 40 (20%) | All three prompts degraded; subtraction repeated 16 eight-grams |
| 60 (30%) | All three prompts degraded; subtraction repeated 10 eight-grams |
| 100 (50%) | Subtraction and multiplication partly recovered; addition still diverged immediately |
| 200 (100%) | Coherent procedural form, no repeated eight-grams in the three primary probes |

Teacher-forced prompt NLL sometimes preferred the more repetitive constrained
continuation, so prompt NLL alone is not an adequate free-generation quality
criterion for this policy.

The two viable follow-ups are:

1. expose the same request-global candidate constraint during training so the
   router and experts learn to work inside it; or
2. keep full logical top-2 routing and add bounded token/layer-aware expert
   paging or predictive prefetch so the physical cache can remain small.

Until one is implemented, unrestricted top-2 should be used when evaluating
checkpoint quality. Bounded request-global residency remains useful for
placement/mechanics experiments but should not be treated as quality-equivalent.

## Defects found during the smoke cycle

Preflight runs exposed and resolved the following before or while producing
model 1:

- an oversized microbatch that caused WDDM/shared-memory spill;
- an FLA `dt_bias` gradient-shape mismatch;
- retained accumulation output graphs;
- global-max grouped-expert padding that amplified skewed routes;
- generation options that were accepted but silently ignored;
- non-power-of-two fused recurrent KDA NaNs;
- coupled resident-count/per-token-top-k semantics.

All aborted/preflight workers were stopped. Only the successful model-1 run was
allowed to complete, and no second training was launched.

## Current verification and revisions

SPALMER:

- `d826b80` — physical expert and optimizer offload;
- `6c8bfd2` — FLA backward and bounded accumulation;
- `5650ad3` — bounded grouped expert routing;
- `87efaf8` — safe recurrent KDA dispatch;
- `a9cacd8` — independent resident candidates and token top-k.

yyLab:

- `7b9ba26` — physical SPALMER offload integration;
- `b49d221` — first-step optimizer-offload validation;
- `547d19f` — bounded-execution telemetry;
- `0d8fc4a` — generation guard correctness;
- this validation revision pins SPALMER `a9cacd8`, separates active/resident
  telemetry, and records this report.

Final local verification before commit:

- SPALMER: 336 passed, 1 skipped; Ruff passed.
- yyLab: 547 passed; Ruff passed.

## Remaining boundaries

- MXFP4/MXFP8 execution is the format-faithful reference QAT lane, not a native
  packed W4A8 GEMM.
- Backward and accelerator parameters remain BF16; Adam moments remain FP32.
- Exact optimizer resume is false because optimizer moments are not yet stored
  in the checkpoint.
- Physical expert-weight offload is inference-only; training offloaded the
  optimizer state, not expert parameters.
- No periodic held-out pass ran inside this smoke run; held-out validation was
  performed immediately after checkpoint completion.
- Width-12 KDA decode uses the stable chunk kernel. Future experiment shapes
  with power-of-two KDA head widths retain the faster fused recurrent path.
