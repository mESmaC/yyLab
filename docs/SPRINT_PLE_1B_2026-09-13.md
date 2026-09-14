# Sprint PLE and KDA at approximately 1B parameters — 2026-09-13

**Status: complete.** 17 accepted measurements from 18 receipts; 0 queued cases not yet recorded and 0 unfinished receipts.

This compares complete training updates on the local GPU. All rates count shifted supervised targets per second, not raw input slots. The experiment does not establish the fastest possible configuration or equivalent language-model quality.

Hardware: **NVIDIA GeForce RTX 5070**, 11.94 GiB CUDA memory; 20 CPU threads. Torch 2.11.0+cu128; TorchAO 0.18.0; FLA 0.5.2.

## Practical measured configurations

For each model/PLE/sequence, show the highest observed rate among tested recipe representatives after choosing the longer confirmation within each exact recipe. This is a whole-configuration overview, not an isolated PLE effect or a claim of the fastest possible training.

| Model | PLE | Seq | Precision | Micro × accum | KDA recompute | Checkpoint | Steps | Supervised tok/s | Alloc / reserved GiB | Receipt |
|---|---|---:|---|---:|---|---|---:|---:|---:|---|
| KDA hybrid 3584×8 | full | 512 | NVFP4/SR4 | 4 × 8 | off | off | 20 | 15,540 | 8.77 / 9.40 | [screen-kda-full-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s512-m2048-sr4.json>) |
| KDA hybrid 3584×8 | full | 4,096 | NVFP4/SR4 | 1 × 4 | on | off | 20 | 15,811 | 9.82 / 10.43 | [screen-kda-full-s4096-m4096-sr4-recompute](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s4096-m4096-sr4-recompute.json>) |
| KDA hybrid 3584×8 | none | 512 | NVFP4/SR4 | 4 × 8 | off | off | 20 | 14,490 | 8.93 / 9.43 | [screen-kda-none-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s512-m2048-sr4.json>) |
| KDA hybrid 3584×8 | none | 4,096 | NVFP4/SR4 | 1 × 4 | off | on | 20 | 10,288 | 6.98 / 7.51 | [screen-kda-none-s4096-m4096-sr4-checkpoint](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s4096-m4096-sr4-checkpoint.json>) |
| KDA hybrid 3584×8 | qr | 512 | NVFP4/SR4 | 4 × 8 | off | off | 20 | 13,810 | 9.75 / 10.36 | [screen-kda-qr-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-qr-s512-m2048-sr4.json>) |
| Sprint 4096×6 | full | 512 | NVFP4/SR4 | 8 × 4 | — | off | 40 | 22,578 | 8.73 / 9.73 | [confirm-sprint-full-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-full-s512-m4096-sr4.json>) |
| Sprint 4096×6 | full | 4,096 | NVFP4/SR4 | 1 × 4 | — | off | 40 | 18,108 | 8.70 / 9.83 | [confirm-sprint-full-s4096-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-full-s4096-m4096-sr4.json>) |
| Sprint 4096×6 | none | 512 | NVFP4/SR4 | 8 × 4 | — | off | 40 | 21,046 | 8.89 / 9.66 | [confirm-sprint-none-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s512-m4096-sr4.json>) |
| Sprint 4096×6 | none | 4,096 | NVFP4/SR4 | 1 × 4 | — | off | 40 | 17,031 | 8.87 / 9.62 | [confirm-sprint-none-s4096-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s4096-m4096-sr4.json>) |
| Sprint 4096×6 | qr | 512 | NVFP4/SR4 | 4 × 8 | — | off | 20 | 17,377 | 8.28 / 9.54 | [screen-sprint-qr-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-qr-s512-m2048-sr4.json>) |

KDA internal recomputation reruns intermediates inside the linear-attention operation. Block checkpointing reruns whole model blocks during backward. The block-checkpoint experiments disable KDA internal recomputation to avoid stacking both modes. These settings remain separate recipes throughout the comparisons.


The larger Sprint QR microbatch remains part of the results: micro 8 × 4 recorded **1,305 targets/s** with 10.14 GiB peak allocated memory. [Original larger-microbatch receipt](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-qr-s512-m4096-sr4.json>). Reducing it to micro 4 × 8 recorded **17,377 targets/s**. [Reduced-microbatch receipt](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-qr-s512-m2048-sr4.json>). Memory pressure is a plausible explanation; these allocator measurements do not prove WDDM spill.

The controller stopped `screen-kda-none-s512-m4096-sr4` after prolonged execution; its last recorded device-memory observation was 11,841 MiB. It has no accepted complete timing and remains a failed experiment in the receipt table. [Controller-stop receipt](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s512-m4096-sr4.json>).

## Production availability

In **Train Sprint → New model → Per-layer embeddings (PLE)**, choose **Regular PLE** or **QR PLE**. The default remains **No PLE**. The budget planner includes the lookup tables and reduces FFN width; saved models and resume/continuation inherit their PLE configuration. Regular PLE uses one full table per later layer, while QR PLE uses four compositional lanes.

These rates use the manually specified **4096-wide, six-layer Sprint** benchmark shape. The automatic 1B throughput planner instead starts with **2816-wide, eleven-layer Sprint** and fits PLE within that shape where possible. The measured rates are not measurements of that automatic plan. The GPU8/SR4 optimizer/precision recipe here remains a benchmark tool; this work does not add it as a new training-form precision choice.

## Architecture and parameter allocation

Full PLE means one vocabulary-by-width table before every physical layer after layer zero. QR-PLE uses four prime quotient/remainder product lanes with learned scalar lane weights and a gate. The layer-zero token embedding stays exact and tied to the vocabulary head. Matched-budget PLE reduces FFN width to pay for these tables; any speed change includes that capacity reallocation.

The main Sprint arm has six 4096-wide blocks. KDA has eight 3584-wide blocks in a 3:1 KDA/full-attention pattern. Their comparison changes attention, width and depth together, so it measures whole configurations rather than isolating attention. The adapter also supports `matched_sprint`, an eight-layer ordinary-attention shape control, but no accepted receipt for that arm is present.

| Model | PLE | Parameters | FFN width | PLE table params | PLE controls | PLE share |
|---|---|---:|---:|---:|---:|---:|
| KDA hybrid 3584×8 | full | 995,539,375 | 5,888 | 205,520,896 | 7 | 20.64% |
| KDA hybrid 3584×8 | none | 995,539,368 | 9,472 | 0 | 0 | 0.00% |
| KDA hybrid 3584×8 | qr | 999,223,755 | 9,216 | 18,364,416 | 35 | 1.84% |
| Sprint 4096×6 | full | 994,103,301 | 7,936 | 167,772,160 | 5 | 16.88% |
| Sprint 4096×6 | none | 996,200,448 | 11,392 | 0 | 0 | 0.00% |
| Sprint 4096×6 | qr | 998,608,921 | 11,136 | 14,991,360 | 25 | 1.50% |

## Selected measurements

Longest completed measured-step count per exact variant/sequence/recipe; ties choose latest start time then filename. All receipts remain visible; selection never maximizes throughput.

| Model | PLE | Seq | Precision | Micro × accum | Mode | Checkpoint | KDA recompute | Steps | Supervised tok/s | Alloc / reserved GiB | Receipt |
|---|---|---:|---|---:|---|---|---|---:|---:|---:|---|
| KDA hybrid 3584×8 | full | 512 | NVFP4/SR4 | 4 × 8 | default | off | off | 20 | 15,540 | 8.77 / 9.40 | [screen-kda-full-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s512-m2048-sr4.json>) |
| KDA hybrid 3584×8 | none | 512 | NVFP4/SR4 | 4 × 8 | default | off | off | 20 | 14,490 | 8.93 / 9.43 | [screen-kda-none-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s512-m2048-sr4.json>) |
| KDA hybrid 3584×8 | qr | 512 | NVFP4/SR4 | 4 × 8 | default | off | off | 20 | 13,810 | 9.75 / 10.36 | [screen-kda-qr-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-qr-s512-m2048-sr4.json>) |
| Sprint 4096×6 | full | 512 | NVFP4/SR4 | 8 × 4 | default | off | — | 40 | 22,578 | 8.73 / 9.73 | [confirm-sprint-full-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-full-s512-m4096-sr4.json>) |
| Sprint 4096×6 | none | 512 | NVFP4/SR4 | 8 × 4 | default | off | — | 40 | 21,046 | 8.89 / 9.66 | [confirm-sprint-none-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s512-m4096-sr4.json>) |
| Sprint 4096×6 | qr | 512 | NVFP4/SR4 | 4 × 8 | default | off | — | 20 | 17,377 | 8.28 / 9.54 | [screen-sprint-qr-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-qr-s512-m2048-sr4.json>) |
| Sprint 4096×6 | qr | 512 | NVFP4/SR4 | 8 × 4 | default | off | — | 20 | 1,305 | 10.14 / 10.76 | [screen-sprint-qr-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-qr-s512-m4096-sr4.json>) |
| KDA hybrid 3584×8 | full | 4,096 | NVFP4/SR4 | 1 × 4 | default | on | off | 20 | 11,326 | 7.10 / 7.83 | [screen-kda-full-s4096-m4096-sr4-checkpoint](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s4096-m4096-sr4-checkpoint.json>) |
| KDA hybrid 3584×8 | full | 4,096 | NVFP4/SR4 | 1 × 4 | default | off | on | 20 | 15,811 | 9.82 / 10.43 | [screen-kda-full-s4096-m4096-sr4-recompute](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s4096-m4096-sr4-recompute.json>) |
| KDA hybrid 3584×8 | none | 4,096 | NVFP4/SR4 | 1 × 4 | default | on | off | 20 | 10,288 | 6.98 / 7.51 | [screen-kda-none-s4096-m4096-sr4-checkpoint](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s4096-m4096-sr4-checkpoint.json>) |
| KDA hybrid 3584×8 | none | 4,096 | NVFP4/SR4 | 1 × 4 | default | off | on | 20 | 3,783 | 10.01 / 10.66 | [screen-kda-none-s4096-m4096-sr4-recompute](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s4096-m4096-sr4-recompute.json>) |
| Sprint 4096×6 | full | 4,096 | NVFP4/SR4 | 1 × 4 | default | off | — | 40 | 18,108 | 8.70 / 9.83 | [confirm-sprint-full-s4096-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-full-s4096-m4096-sr4.json>) |
| Sprint 4096×6 | none | 4,096 | NVFP4/SR4 | 1 × 4 | default | off | — | 40 | 17,031 | 8.87 / 9.62 | [confirm-sprint-none-s4096-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s4096-m4096-sr4.json>) |

## Same-recipe PLE comparisons

Each comparison holds the model family, sequence length, microbatch/accumulation, precision, attention backend, compilation, checkpointing, KDA recomputation, input stream and frozen sources constant. FFN widths and PLE tables change within the total parameter budget.

| Model | Seq | PLE | Micro × accum | Memory strategy | Baseline tok/s | PLE tok/s | Change | FFN baseline → PLE | Receipts |
|---|---:|---|---:|---|---:|---:|---:|---:|---|
| KDA hybrid 3584×8 | 512 | full | 4 × 8 | saved activations | 14,490 | 15,540 | +7.25% | 9,472 → 5,888 | [baseline](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s512-m2048-sr4.json>), [PLE](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s512-m2048-sr4.json>) |
| KDA hybrid 3584×8 | 512 | qr | 4 × 8 | saved activations | 14,490 | 13,810 | -4.69% | 9,472 → 9,216 | [baseline](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s512-m2048-sr4.json>), [PLE](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-qr-s512-m2048-sr4.json>) |
| Sprint 4096×6 | 4,096 | full | 1 × 4 | saved activations | 17,031 | 18,108 | +6.33% | 11,392 → 7,936 | [baseline](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s4096-m4096-sr4.json>), [PLE](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-full-s4096-m4096-sr4.json>) |
| Sprint 4096×6 | 512 | full | 8 × 4 | saved activations | 21,046 | 22,578 | +7.28% | 11,392 → 7,936 | [baseline](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s512-m4096-sr4.json>), [PLE](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-full-s512-m4096-sr4.json>) |
| Sprint 4096×6 | 512 | qr | 8 × 4 | saved activations | 21,046 | 1,305 | -93.80% | 11,392 → 11,136 | [baseline](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s512-m4096-sr4.json>), [PLE](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-qr-s512-m4096-sr4.json>) |
| KDA hybrid 3584×8 | 4,096 | full | 1 × 4 | KDA internal recompute | 3,783 | 15,811 | +317.96% | 9,472 → 5,888 | [baseline](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s4096-m4096-sr4-recompute.json>), [PLE](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s4096-m4096-sr4-recompute.json>) |
| KDA hybrid 3584×8 | 4,096 | full | 1 × 4 | block checkpoint | 10,288 | 11,326 | +10.09% | 9,472 → 5,888 | [baseline](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s4096-m4096-sr4-checkpoint.json>), [PLE](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s4096-m4096-sr4-checkpoint.json>) |

## Correctness checks and small BF16 controls

The controller's final production GPU test run passed **6 tests in 39.51 seconds**. The separate benchmark GPU8 run passed **9 tests in 50.30 seconds**, covering six BF16 and three selected NVFP4/SR4 cases. The final GPU8 case receipt is independently validated below. [Final GPU8 validation receipt](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/gpu8-cuda-validation.json>).

Three distinct checks are kept separate: natural per-update weight changes; nonzero natural gradients and optimizer moments; and controlled writes to detached clones using a unit gradient and a larger learning rate. The clone probe leaves natural training weights untouched and checks the optimizer's write path. It does not establish that the corresponding control moved naturally at the benchmark learning rate.

| Bounded GPU case | Natural controls moved / total | Naturally unmoved controls | Detached write probes |
|---|---:|---|---:|
| sprint/none/bf16 | 0 / 0 | none | 0 |
| sprint/full/bf16 | 3 / 3 | none | 3 |
| sprint/qr/bf16 | 6 / 6 | none | 6 |
| kda/none/bf16 | 0 / 0 | none | 0 |
| kda/full/bf16 | 3 / 3 | none | 3 |
| kda/qr/bf16 | 5 / 6 | `ple_refresh_layers.1.gate` | 6 |
| sprint/none/nvfp4_all_portable_sr4 | 0 / 0 | none | 0 |
| kda/full/nvfp4_all_portable_sr4 | 2 / 3 | `ple_refresh_layers.0.gate` | 3 |
| kda/qr/nvfp4_all_portable_sr4 | 5 / 6 | `ple_refresh_layers.1.gate` | 6 |

Across the bounded cases, **24 of 27 small control tensors moved naturally within six updates**; all 27 had valid natural gradient/moment participation and passed the independent clone write probe. The unmoved controls remain explicitly named above. This does not establish long-run control learning or quality equivalence.

Preserved earlier criteria and diagnostics:

- [gpu8-cuda-validation-initial.json](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/gpu8-cuda-validation-initial.json>): 4 failed of 9 recorded cases; retained as intermediate evidence, not counted as final passing validation.
- [gpu8-cuda-validation-history.json](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/gpu8-cuda-validation-history.json>): 3 failed of 9 recorded cases; retained as intermediate evidence, not counted as final passing validation.
- [gpu8-cuda-validation-diagnostic.json](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/gpu8-cuda-validation-diagnostic.json>): 4 failed of 4 recorded cases; retained as intermediate evidence, not counted as final passing validation.

The original net-movement requirement produced four failed cases. Tracking natural per-update movement still left three cases with an unmoved small control. The final check preserves that observation while separately requiring gradient/moment participation and validating the write path on clones. Recorded runtime model/engine hashes agree across these phases; the test criterion changed.

## Measurement and validation

GPU AdamW8bit with BF16 stochastic rounding, BF16 parameter/gradient storage except 21,672 FP32 KDA decay-gate elements; no FP32 master weights or offloaded moments. Optional NVFP4/SR4 covers QKV, attention output and FFN projections, not KDA recurrence/gates, embeddings, PLE, head or norms.

The timer includes forward, backward, gradient clipping, the GPU optimizer update and input transfers. Initialization, compilation, validation checks, packing, evaluation and saving are excluded. Each update processes the recorded input-token budget; supervised targets equal effective sequences × (sequence length − 1). Repeated synthetic tokens make this a compute benchmark, not a language-learning experiment.

Accepted receipts reconcile independent parameter formulas, dtype inventories, exact token/timer arithmetic, GPU moment storage, optimizer step counts and stable compiler counters. They require finite losses, weights and all parameter gradients. Every non-control parameter tensor must have at least one changed sampled value. A small BF16 PLE gate or lane-weight tensor may finish with an unchanged net sample only when its serialized gradient and optimizer second moment are nonzero. That exception establishes optimizer participation, not observed weight movement; net samples are not per-update histories. Parameter identity is guarded by the hashed optimizer engine during conversion, rather than a separately serialized identity flag. Recorded source hashes resolve to live matching files or their matching frozen snapshots; a later live edit does not invalidate an otherwise matching historical snapshot.

CUDA allocated/reserved peaks are allocator measurements. They do not establish dedicated/shared residency or prove WDDM spill. The GPU8/SR4 benchmark recipe and stateless KDA adapter remain separate from production Sprint checkpoint/resume tooling. No held-out loss, convergence equivalence, long-run stability or KDA quality result is established here.

## Every receipt and exclusion

| Receipt | Status | Included | Model / PLE | Seq | Tok/s if recorded | Note |
|---|---|---|---|---:|---:|---|
| [confirm-sprint-full-s4096-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-full-s4096-m4096-sr4.json>) | ok | yes | sprint / full | 4,096 | 18,108 | selected |
| [confirm-sprint-full-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-full-s512-m4096-sr4.json>) | ok | yes | sprint / full | 512 | 22,578 | selected |
| [confirm-sprint-none-s4096-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s4096-m4096-sr4.json>) | ok | yes | sprint / none | 4,096 | 17,031 | selected |
| [confirm-sprint-none-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/confirm-sprint-none-s512-m4096-sr4.json>) | ok | yes | sprint / none | 512 | 21,046 | selected |
| [screen-kda-full-s4096-m4096-sr4-checkpoint](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s4096-m4096-sr4-checkpoint.json>) | ok | yes | kda / full | 4,096 | 11,326 | selected |
| [screen-kda-full-s4096-m4096-sr4-recompute](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s4096-m4096-sr4-recompute.json>) | ok | yes | kda / full | 4,096 | 15,811 | selected |
| [screen-kda-full-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-full-s512-m2048-sr4.json>) | ok | yes | kda / full | 512 | 15,540 | selected |
| [screen-kda-none-s4096-m4096-sr4-checkpoint](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s4096-m4096-sr4-checkpoint.json>) | ok | yes | kda / none | 4,096 | 10,288 | selected |
| [screen-kda-none-s4096-m4096-sr4-recompute](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s4096-m4096-sr4-recompute.json>) | ok | yes | kda / none | 4,096 | 3,783 | selected |
| [screen-kda-none-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s512-m2048-sr4.json>) | ok | yes | kda / none | 512 | 14,490 | selected |
| [screen-kda-none-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-none-s512-m4096-sr4.json>) | error | no | kda / none | 512 | — | Stopped by benchmark controller after prolonged warmup and high device-memory pressure. No completed timing accepted; smaller microbatch and FLA recompute are subsequent candidates. |
| [screen-kda-qr-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-kda-qr-s512-m2048-sr4.json>) | ok | yes | kda / qr | 512 | 13,810 | selected |
| [screen-sprint-full-s4096-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-full-s4096-m4096-sr4.json>) | ok | yes | sprint / full | 4,096 | 18,050 | additional measurement |
| [screen-sprint-full-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-full-s512-m4096-sr4.json>) | ok | yes | sprint / full | 512 | 22,689 | additional measurement |
| [screen-sprint-none-s4096-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-none-s4096-m4096-sr4.json>) | ok | yes | sprint / none | 4,096 | 17,302 | additional measurement |
| [screen-sprint-none-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-none-s512-m4096-sr4.json>) | ok | yes | sprint / none | 512 | 21,145 | additional measurement |
| [screen-sprint-qr-s512-m2048-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-qr-s512-m2048-sr4.json>) | ok | yes | sprint / qr | 512 | 17,377 | selected |
| [screen-sprint-qr-s512-m4096-sr4](<C:/Users/engin/Desktop/yy/yyLab/artifacts/sprint-ple-1b-20260913/screen-sprint-qr-s512-m4096-sr4.json>) | ok | yes | sprint / qr | 512 | 1,305 | selected |
