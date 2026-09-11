# Sprint speed experiments, round 2 — 11 September 2026

Completed screening runs show that wider, shallower dense shapes can improve throughput substantially on this RTX 5070. Smaller FFNs alone did not reliably help. Loss implementation, CPU optimizer dispatch and compile mode also matter, especially around 10M parameters. These are training-step speed measurements, with no language-quality or convergence comparison.

The longer confirmations reached 602k, 182k, 24.6k and 12.1k supervised targets/s at 10M, 50M, 500M and 1B respectively with effective batch 32. This round contains 39 accepted measurements and one explicit failed candidate.

## Measurement and comparison controls

Hardware/runtime: NVIDIA GeForce RTX 5070 on Windows, 20 logical CPU cores; PyTorch 2.11.0+cu128, TorchAO 0.18.0, Transformers 5.16.1 and Triton Windows 3.7.1.post27. Each case runs in a fresh process with seed 42, vocabulary 8,192 and sequence length 512. The main comparison fixes effective batch 32; microbatch and accumulation are recorded separately. Unless specified, runs use 20 CPU threads, cuDNN attention, default torch.compile, no activation checkpointing and five warmup steps.

Every candidate retains BF16 trainable weights and gradients, with FP32 CPU master weights and two FP32 Adam moments. CPU optimizer state occupies 12 bytes per actual parameter, plus transfer/scratch buffers. Precision names describe decoder projection matrix multiplications; embeddings, the tied output head, attention and loss retain BF16/FP32 operations. BitNet-style QAT changes the training recipe and adds norms; it is not packed native ternary training.

Rates count **supervised shifted targets**, not all forwarded tokens. At sequence 512, each example contributes 511 targets. One effective-batch-32 update processes 16,352 targets; 200 measured steps process 3,270,400 targets. Timers include synchronized forward, backward, gradient clipping, FP32 CPU Adam and transfers. Initialization, compilation/warmup, validation, input preparation, evaluation and checkpoint writes are excluded. Very short screening timings are not confidence intervals or complete wall-clock training estimates.

Successful receipts require finite loss and gradient-norm metrics and observed changes in sampled decoder matrices. The assembler checks counts, rate denominators, step timing and recipe consistency, preserves source hashes/runtime versions, and retains failed, incomplete and malformed receipts. The raw receipts and logs live in `artifacts/sprint-speed-round2-20260911`; `docs/sprint-speed-round2-2026-09-11.json` is the consolidated snapshot. Its `comparison_eligible` flag identifies accepted completed rows. Source changes across experiment phases are explicit; the initial source copies remain under `source-before`.

## Selected confirmation results

All seven selected recipes completed their longer confirmation runs and passed receipt validation. The main table fixes effective batch 32; every row uses the fast NVFP4-forward/MXFP8-backward path. These are the best confirmed settings in this round, not an exhaustive global optimum.

| Target | Actual parameters | Confirmed targets/s | Earlier measured best | Change | Micro x accumulation | Measured steps | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10M | 9,964,032 | **601,924** | 371,575 | +62.0% | 32 x 1 | 600 | 1.016 |
| 50M | 49,946,112 | **181,506** | 162,433 | +11.7% | 32 x 1 | 400 | 2.177 |
| 500M | 499,159,040 | **24,611** | 22,970 | +7.1% | 16 x 2 | 80 | 6.578 |
| 1B | 999,346,176 | **12,107** | 11,582 | +4.5% | 8 x 4 | 48 | 8.605 |

The earlier 10M best is this round's original six-layer NVFP4/BF16 baseline. The 50M–1B earlier bests are the prior precision round's confirmed settings. These comparisons combine shape, loss and execution changes; they do not isolate one optimization or establish equivalent model quality. Peak allocation covers initialization, warmup and measured steps.

The selected shapes (width / layers / FFN width) are **512 / 2 / 1792**, **1536 / 2 / 3008**, **4096 / 4 / 6016**, and **4096 / 6 / 11456**. They allocate 36.83%, 37.00%, 39.49% and 56.35% of total parameters to dense FFNs. The 10M/50M recipes use full Torch loss; 500M/1B use auto/Liger. Only 10M combines reduce-overhead compilation with a 262,144-element CPU Adam cutoff. Other rows use default compile mode and the 1,048,576 cutoff.

The matched-shape 10M BF16 confirmation reached **503,613 targets/s** over 600 steps with Torch loss, default compilation and the default cutoff. The final low-bit 10M result also changes those last two settings, so their ratio is a combined-recipe comparison.

The larger-batch confirmations use the same selected shape and fast precision as the main table, but perform four times as much token work per optimizer update:

| Target | Effective batch | Confirmed targets/s | Versus same-recipe B32 | Micro x accumulation | Measured steps | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500M | 128 | **34,294** | +39.3% | 16 x 8 | 24 | 6.578 |
| 1B | 128 | **16,910** | +39.7% | 8 x 16 | 24 | 8.605 |

Higher batch throughput does not establish equal learning per token. All cases reuse a seeded synthetic batch, so their very low final losses are not evidence of language-model quality. Full configurations are saved in [the seven benchmark recipes](recipes/sprint-speed-round2/README.md); the [consolidated receipts](sprint-speed-round2-2026-09-11.json) retain every accepted and failed experiment.

## Original 10M shape: five precision modes

All five core cases use width 320, six layers, FFN width 1,280 and five attention heads of dimension 64. The dense model has 9,998,400 parameters. BitNet adds 9,600 norm parameters, reaching 10,008,000 (+0.096%). Loss selection is `auto`, which resolves to Liger fused linear cross-entropy for this CUDA/vocabulary combination. All cases use microbatch 32, accumulation 1 and 200 measured steps.

| Mode | Supervised targets/s | Versus BF16 | Peak allocated GiB |
| --- | --- | --- | --- |
| BF16 | 364,321 | +0.00% | 1.203 |
| NVFP4 forward / BF16 backward | 371,575 | +1.99% | 1.204 |
| NVFP4 forward / MXFP8 backward | 354,004 | -2.83% | 1.103 |
| MXFP8 forward / backward | 349,506 | -4.07% | 1.106 |
| BitNet-style QAT / dense BF16 GEMMs | 333,296 | -8.52% | 1.290 |

The additional fused-transpose preparation variant, `nvfp4_mxfp8_fast`, reached 349,462 targets/s on this original shape: -1.28% versus standard NVFP4/MXFP8 preparation. BitNet's native INT8-forward variant reached 334,031 targets/s, +0.22% versus dense BitNet, with 0.928 GiB peak allocation. Its backward remains the BF16 straight-through-estimator recipe. Low-bit GEMMs therefore did not provide a universal speed advantage at 10M.

## Dense shape and FFN-share screening

FFN share means unique parameters in all feed-forward matrices divided by the entire model, including embeddings, attention and norms. For these GELU models it is `2 * layers * width * FFN_width / total_parameters`. It is a model allocation ratio, not an active-expert percentage: every FFN is dense and every token uses it.

All rows below fix effective batch 32. Shapes are `width / layers / FFN_width`. The original 10M shape uses head dimension 64; the other listed shapes use 128. The 10M/50M rows use Torch loss and `nvfp4_mxfp8_fast`; 500M uses fast preparation with auto/Liger loss; the 1B row uses standard `nvfp4_mxfp8` preparation and auto/Liger loss. CPU cutoff remains 1,048,576 elements. Exact parameter counts disclose the small differences between nominal budget matches.

| Screening case | Parameters | Shape | FFN share | Targets/s | Micro x accumulation | Steps |
| --- | --- | --- | --- | --- | --- | --- |
| 10M original | 9,998,400 | 320 / 6 / 1280 | 49.16% | 369,118 | 32 x 1 | 200 |
| 10M narrow / 8 layers | 9,965,824 | 256 / 8 / 1408 | 57.87% | 351,003 | 32 x 1 | 200 |
| 10M wide / 2 layers | 9,964,032 | 512 / 2 / 1792 | 36.83% | 547,417 | 32 x 1 | 200 |
| 10M large FFN / 2 layers | 9,930,624 | 384 / 2 / 3648 | 56.42% | 561,692 | 32 x 1 | 200 |
| 10M large FFN / 4 layers | 9,963,776 | 256 / 4 / 3328 | 68.41% | 427,901 | 32 x 1 | 200 |
| 50M original | 49,816,576 | 1024 / 4 / 3008 | 49.46% | 165,834 | 32 x 1 | 200 |
| 50M thin FFN | 49,818,880 | 1280 / 4 / 1280 | 26.31% | 159,824 | 32 x 1 | 200 |
| 50M large FFN | 49,945,344 | 768 / 4 / 5568 | 68.49% | 164,769 | 32 x 1 | 200 |
| 50M wide / 2 layers | 49,946,112 | 1536 / 2 / 3008 | 37.00% | 181,475 | 32 x 1 | 200 |
| 500M wide / 4 layers | 499,159,040 | 4096 / 4 / 6016 | 39.49% | 24,472 | 16 x 2 | 40 |
| 1B wide / 6 layers | 999,346,176 | 4096 / 6 / 11456 | 56.35% | 11,838 | 8 x 4 | 24 |

The 50M two-layer shape is 9.43% faster than the four-layer shape in this screen. At 10M, the 36.83%-FFN two-layer shape substantially outperforms the 49.16%-FFN six-layer shape, but a different two-layer shape with a 56.42% FFN share is faster still. At 50M, moving to a 26.31% FFN share is slower than the original 49.46% share. The evidence favors measuring complete aligned shapes rather than treating smaller FFNs, a fixed FFN percentage or smaller individual matrices as a speed guarantee. These architecture changes may affect quality.

A second 10M wide-two-layer BF16 run measured 500,664 targets/s, against 547,417 for the fast NVFP4/MXFP8 path on the same shape. At 1B, switching the displayed shape to fast transpose preparation measured 12,160 targets/s (+2.73%, 24 steps). At 500M, microbatch 32 with one accumulation step measured 24,445 targets/s versus 24,472 at microbatch 16 with two accumulations; the larger microbatch increased peak allocation from 5.300 to 7.580 GiB without an observed speed benefit.

The omitted `shape-10m-aligned4` candidate (width 384, four layers, FFN 1,408; FFN share 43.99%) aborted and has no accepted throughput. Its failure is documented below.

## Loss, optimizer cutoff and compile-mode screens

Explicit Torch linear-plus-cross-entropy improved these small-model screens versus auto/Liger, at higher peak allocation. The 50M baselines are the previous round's `repeat-50m-bf16` and `repeat-50m-best`; shape, source, workload and optimizer controls match those loss-only comparisons.

| Shape / precision | Auto/Liger targets/s | Torch targets/s | Change | Peak GiB before -> after |
| --- | --- | --- | --- | --- |
| 10M / BF16 | 364,321 | 383,970 | +5.39% | 1.203 -> 1.449 |
| 10M / NVFP4/MXFP8 fast | 349,462 | 369,754 | +5.81% | 1.239 -> 1.518 |
| 50M / BF16 | 121,341 | 123,207 | +1.54% | 2.316 -> 2.521 |
| 50M / NVFP4/MXFP8 fast | 162,433 | 166,070 | +2.24% | 2.148 -> 2.384 |

The single-microbatch training-step simplification was also screened separately: BF16 10M changed from 383,970 to 387,887 targets/s (+1.02%), while fast NVFP4/MXFP8 changed -0.17% at 10M and -0.14% at 50M. Those small differences do not establish a general speedup. Equivalence tests cover the retained change.

For CPU Adam, reducing the compiled-update cutoff from 1,048,576 to 262,144 elements produced 384,147 versus 369,118 targets/s on the original 10M shape (+4.07%), and 578,731 versus 547,417 on the wide-two-layer shape (+5.72%). Eligibility for the compiled path rose to 93.81% and 99.97% of parameters respectively. These historical measurements used `benchmark_optimizer_threshold.py`; the official precision benchmark now accepts the same explicit experimental setting. It remains outside the production training/resume contract. Default training behavior is unchanged.

On the wide-two-layer fast-precision shape, `reduce-overhead` compile mode measured 568,487 targets/s versus 547,417 with default mode (+3.85%). This records the requested compiler mode, not a claim that the entire CPU-offloaded optimizer step was captured in a CUDA graph. Reducing CPU threads from 20 to 8 measured 550,179 (+0.50%); that screen does not establish a meaningful advantage. The combined cutoff/compile setting was subsequently measured in the longer confirmation above.

## Larger effective batches: a separate training comparison

The completed exploratory batch-128 runs measured:

| Shape / precision | Effective batch 32 targets/s | Effective batch 128 targets/s | Microbatch | Accumulation at B128 | B128 measured steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500M wide-4 / NVFP4-MXFP8 fast | 24,472 | 34,280 | 16 | 8 | 12 |
| 1B wide-6 / NVFP4-MXFP8 standard | 11,838 | 16,339 | 8 | 16 | 12 |

Those are +40.08% and +38.02% throughput observations, with identical shapes/precision within each row. Batch 128 processes 65,408 supervised targets per update, four times batch 32. For a fixed token budget it performs one quarter as many optimizer updates, which can amortize CPU optimizer work but changes the optimization trajectory, clipping behavior and learning-rate schedule measured in steps. They cannot be used as equal-learning-per-token speedups. The short 12-step screens also cover different numbers of tokens than the B32 screens. The separate final B128 table above confirms both sizes using fast precision.

## Why MoE was not assumed to be natively faster

Smaller experts and a fixed active fraction are plausible separate architecture experiments. They do not make the dense FFN-share ratio above equivalent to conditional computation. MoE introduces routing and irregular per-expert token batches; efficient implementations use specialized grouped or block-sparse computation and must handle load balance. PyTorch's training account explains both the small-matrix utilization tradeoff and the specialized kernels that avoid padding or token-dropping costs. On this single GPU, inter-device communication is not the issue, but routing and kernel utilization still require measurement. This is an inference about the local workload, not a measured MoE result. [PyTorch: Training MoEs at Scale](https://pytorch.org/blog/training-moes/).

TorchAO offers a native MXFP8 MoE training prototype, with quantized grouped GEMMs in forward and backward. Its published examples emphasize large token dimensions and B200/GB200 workloads. That is evidence that a specialized route exists, not evidence that these 10M–1B shapes on this RTX 5070 would win. A native dense low-bit GEMM passing locally does not validate a grouped-MoE training backend. No MoE speed or quality claim is made in this round. [TorchAO MoE training prototype](https://github.com/pytorch/ao/blob/main/torchao/prototype/moe_training/README.md).

The installed versions impose concrete constraints: PyTorch 2.11's optimized CUTLASS grouped BF16 implementation is excluded on Windows, and TorchAO 0.18's native grouped MXFP8 quantization path requires SM 10.x. This RTX 5070 is SM 12.0. These specific paths therefore do not supply a verified native expert implementation for the current setup; other kernels would require separate implementation and measurement. This is why this round measures dense FFN budgets without claiming an expert speedup. [PyTorch 2.11 GroupMM source](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/GroupMM.cu), [TorchAO 0.18 grouped quantization source](https://github.com/pytorch/ao/blob/v0.18.0/torchao/prototype/moe_training/kernels/mxfp8/quant.py).

## Retained tooling changes

The throughput planner now starts its wider, shallower profile at 10M, with the two-layer floor confined to 10M–<50M. Existing 50M+ planner behavior, the balanced profile and saved checkpoint dimensions remain unchanged. The ordinary trainer removes redundant loss accumulation work for a single microbatch, and new-model setup exposes the loss implementation used by both training and its benchmark. The new arithmetic shape explorer accepts FFN fractions of total parameters. The experimental precision benchmark accepts an explicit compiled CPU Adam cutoff and records its actual bucket coverage. Native precision adapters and the cutoff remain disposable benchmark experiments; the recorded peak rates are not automatic defaults for ordinary training or resume.

## Rejected experiments and validation

The MXFP8 tensor-lifetime candidate was rejected after an unexplained gradient mismatch: six of 2,923 padded dgrad elements differed in the `(37, 79, 113)` case, with maximum absolute difference 9.3125. An isolated rerun, 60 diagnostic repetitions and the ten-test suite subsequently passed, but that did not explain the initial failure. The exact pre-experiment `mxfp8_fast.py` source was restored before the post/shape screens. The rejected candidate and logs remain in the artifact directory, and `lifetime-experiment.json` records `retained_in_runtime: false`. No tolerance was weakened.

Separately, `shape-10m-aligned4` exited with code 3 and a CUDA illegal instruction reported during cleanup. Its receipt remained `running` with no completed cases because the process aborted. `process_exit.json` preserves the failure and log hash; the consolidated report labels it `execution_error` and accepts no timing. The cause has not been established, and the crash is not attributed to the rejected lifetime candidate. Later cases ran in fresh processes.

The final broad suite logged 337 passed and one failed assertion in the new planner coverage test. The assertion incorrectly required every vocabulary/architecture combination to fit within 2%: 128K vocabulary with untied embeddings at a 10M target cannot do so under the planner's constraints. The corrected test proves that bound and checks the explicit balanced fallback; the 49 planner cases then passed. Across the broad suite and corrected rerun, 338 unique cases were verified, including all precision GPU tests. The original failure remains visible in `final-tests.log`; `planner-recheck.log` records the rerun. Ruff passed on all changed production files and related tests, and a separate read-only review found no actionable regression in the planner, shape explorer or loss UI. The consolidated receipt embeds the validation record and final source hashes. This is numerical/runtime validation, not language-quality validation.

## Reproduction

Use the official benchmark from the repository root, one case per fresh process and a fresh output directory. The threshold wrapper is historical only. This example runs the saved wide-two-layer recipe with the explicit experimental CPU cutoff:

```powershell
.venv/Scripts/python.exe -m yylab.train.benchmark_sprint_precision --config artifacts/sprint-speed-round2-20260911/shape-10m-wide2.json --cpu-optimizer-min-elements 262144 --output artifacts/sprint-threshold-reproduction
```

The JSON recipe must select `cpu_optimizer_kernel: "compiled"`; the explicit CLI cutoff overrides its JSON cutoff. With `--config`, ordinary architecture/workload settings come from that JSON, so change the extracted recipe to alter batch size, loss, compile mode or precision. The experimental field is `experimental_cpu_adam_min_elements`. Unsupported or unprepared compiled CPU kernels fail explicitly.

Extract the full recorded recipe from the consolidated JSON rather than relying on planner defaults that may change:

```python
import json
from pathlib import Path

report = json.loads(Path("docs/sprint-speed-round2-2026-09-11.json").read_text(encoding="utf-8"))
name = "confirm-10m-best"
recipe = report["receipts"][name]["receipt"]["recipe"]
Path("artifacts/reproduce-sprint.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
```

```powershell
.venv/Scripts/python.exe -m yylab.train.benchmark_sprint_precision --config artifacts/reproduce-sprint.json --output artifacts/sprint-reproduction
```

Each result embeds source hashes and dependency versions. Exact historical-source reproduction requires those matching sources; extracting a recipe alone does not roll the code back. Historical threshold receipts also record the wrapper hash; current official-CLI runs correctly have different benchmark provenance.

The arithmetic-only shape explorer generates budget-fitting dense candidates with a chosen FFN share and head/alignment constraints:

```powershell
.venv/Scripts/python.exe -m yylab.train.sprint_shapes --target-params 50000000 --vocab-size 8192 --ffn-fractions 0.25,0.375,0.5,0.625,0.70 --depths 2,4,6,8 --head-dim 128 --width-alignment 128 --ffn-alignment 64 --output artifacts/sprint-shape-candidates.json
```

Its tied-embedding candidates must satisfy the requested parameter and FFN-share tolerances. It does not allocate training models, benchmark them, guarantee that every legal shape is enumerated, or rank candidates by speed. Copy a selected candidate's `model_config` into a full benchmark recipe and measure it. After every benchmark process has finished, rerun `artifacts/sprint-speed-round2-20260911/assemble.py` to refresh the consolidated snapshot; do not replace an unfinished row with an assumed success.
