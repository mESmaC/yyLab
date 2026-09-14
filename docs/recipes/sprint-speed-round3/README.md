# Selected Sprint 1B speed recipes

These byte-identical selected recipes accompany the [round 3 report](../../SPRINT_SPEED_ROUND3_2026-09-11.md). [manifest.json](manifest.json) records their original paths and SHA-256 hashes. The [consolidated evidence](../../sprint-speed-round3-2026-09-11.json) preserves all available benchmark receipts, including rejected, stopped and unrun cases.

| Completed case | Exact recipe | Micro x accumulation | Measured updates | Targets/s |
| --- | --- | ---: | ---: | ---: |
| B32 FP32-moment baseline | [baseline-b32.json](baseline-b32.json) | 8 x 4 | 24 | 12,095 |
| B32 BF16-moment confirmation | [confirm-bf16-moments-micro8-b32.json](confirm-bf16-moments-micro8-b32.json) | 8 x 4 | 64 | **13,181** |
| B128 FP32-moment baseline | [baseline-b128.json](baseline-b128.json) | 8 x 16 | 16 | 17,027 |
| B128 BF16-moment confirmation | [confirm-bf16-moments-micro8-b128.json](confirm-bf16-moments-micro8-b128.json) | 8 x 16 | 32 | **17,454** |

All four use the same 999,346,176-parameter architecture: width 4,096, six layers, FFN 11,456, tied vocabulary 8,192 and sequence length 512. Standard optimizer runs retain BF16 trainable parameters/gradients and FP32 CPU masters/moments. Precision names describe decoder-projection matrix operations. Effective batch 128 changes optimizer-update frequency and must be compared with its own baseline. Architecture and BF16-moment experiments change the model or optimizer recipe and remain separate from same-model, same-optimizer comparisons.

The confirmed BF16-moment recipe is **8.97% faster at B32** than its fresh FP32-moment baseline, using 64 versus 24 measured updates, and **2.51% faster at B128**, using 32 versus 16 updates. It reduces CPU master/moment storage from 11,992,154,112 to 8,002,322,432 bytes. It deliberately changes Adam's persisted moments and does not establish equivalent convergence or quality. The short microbatch-16 gains remain unconfirmed after interrupted longer attempts.

These are disposable synthetic training-step benchmarks, not trained checkpoints or ordinary trainer/resume configurations. Preserve the exact model dimensions, precision, loss, CPU settings, accumulation, warmup and measured-step counts. Reproduction requires the runtime/source versions in the associated receipt; the report supplies scope and limitations.

For either FP32-moment baseline, run the official benchmark from the repository root in a fresh process and use a new output directory:

```powershell
.venv/Scripts/python.exe -m yylab.train.benchmark_sprint_precision --config docs/recipes/sprint-speed-round3/baseline-b32.json --output artifacts/reproduce-round3-baseline-b32
```

The BF16-moment recipe requires the [archived experimental optimizer](experiments/README.md), restored to its original location because its repository-root discovery depends on that path. Passing its flag to the ordinary benchmark does not activate the implementation. In a clean experiment workspace:

```powershell
New-Item -ItemType Directory -Path artifacts/sprint-1b-speed-20260911 -Force | Out-Null
Copy-Item -LiteralPath docs/recipes/sprint-speed-round3/experiments/bf16_moments_candidate.py -Destination artifacts/sprint-1b-speed-20260911/bf16_moments_candidate.py
.venv/Scripts/python.exe artifacts/sprint-1b-speed-20260911/bf16_moments_candidate.py --config docs/recipes/sprint-speed-round3/confirm-bf16-moments-micro8-b32.json --output artifacts/reproduce-round3-bf16-moments
```

For the B128 confirmation, use `confirm-bf16-moments-micro8-b128.json` with the same wrapper and a different output directory.

The BF16-moment experiment has no production checkpoint/resume support. The [failed cache attempt](experiments/cache-rejected/README.md), [first-warmup memory diagnostic](experiments/startup-memory/README.md) and [allocator trials](experiments/allocator/README.md) are archived separately. No throughput is accepted for their failed, interrupted or unrun cases.
