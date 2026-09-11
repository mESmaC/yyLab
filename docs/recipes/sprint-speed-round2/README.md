# Confirmed Sprint benchmark recipes

These exact recipes produced the longer runs in [the round 2 report](../../SPRINT_SPEED_ROUND2_2026-09-11.md). All use vocabulary 8,192, sequence 512, BF16 trainable weights and FP32 CPU Adam state. The experimental precision names describe decoder matrix operations. They are disposable benchmark recipes, not trained checkpoints or ordinary trainer/resume configurations.

From the repository root, use a fresh output directory and one process per case:

```powershell
.venv/Scripts/python.exe -m yylab.train.benchmark_sprint_precision --config docs/recipes/sprint-speed-round2/confirm-10m-best.json --output artifacts/reproduce-10m-best
```

The four `confirm-*-best` size recipes use effective batch 32. The two `confirm-batch128-*` recipes use effective batch 128, changing optimizer update frequency. `confirm-10m-bf16` is an additional BF16 control. The 10M best recipe explicitly requests the experimental 262,144-element CPU Adam cutoff and reduce-overhead compile mode; the others retain the default cutoff and compiler mode. Reproduction needs the runtime/source versions recorded in the consolidated receipt.
