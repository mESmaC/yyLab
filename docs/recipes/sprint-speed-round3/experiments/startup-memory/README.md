# First-step memory diagnostic

This archive preserves the exact diagnostic wrapper, final receipt, and flushed step trace from `diagnose-micro16-b32`. All three copies match their source SHA-256 hashes in [manifest.json](manifest.json); the wrapper also matches the source hash recorded by the diagnostic. This is **DIAGNOSTIC_NOT_THROUGHPUT** and is excluded from accepted speed comparisons.

The recipe requests the 999,346,176-parameter dense Sprint model, `nvfp4_mxfp8_fast`, microbatch 16, accumulation 2, sequence 512, five warmup steps, and 64 measured steps. The diagnostic stopped after **step 0, the first warmup/compilation step**, when reserved memory exceeded its 11 GiB guard. No measured-training step ran.

| First completed step | Observation |
| --- | ---: |
| Synchronized elapsed time, including compile warmup | 50.1156 seconds |
| Allocated memory at step end | 6.402 GB |
| Reserved memory at step end | 13.392 GB |
| Peak allocated memory | 12.542 GB |
| Inactive split memory at step end | 1.043 GB |
| Allocator retries / allocator OOMs | 0 / 0 |

GB values above are decimal; the raw byte counters are retained in [diagnostic.json](diagnostic.json). The 11 GiB guard is 11,811,160,064 bytes. The five-second step-time guard excludes compile warmup, so memory triggered this stop.

The observation establishes high startup/warmup allocation and reservation pressure in this instrumented run. It does **not** establish allocation growth across training steps: only one step completed. It also does not isolate compiler workspace from first-step execution allocations or measure Windows shared-GPU-memory residency. Synchronization and logging can change overlap, so these timings are not a substitute for the official benchmark.

The final receipt says `guard_stopped`, and the trace records `reserved_memory_limit` followed by optimizer cleanup. The caller reported execution-tool exit code 1, while the archived wrapper's guarded exit path requests 3. That outer-code discrepancy remains unresolved and is recorded in the manifest; no status was rewritten as success.

Restore the wrapper to its original path before execution, because repository-root discovery depends on that location. Run from the repository root, with the GPU otherwise idle:

```powershell
New-Item -ItemType Directory -Path artifacts/sprint-1b-speed-20260911 -Force | Out-Null
Copy-Item -LiteralPath docs/recipes/sprint-speed-round3/experiments/startup-memory/diagnostic_wrapper.py -Destination artifacts/sprint-1b-speed-20260911/diagnostic_wrapper.py
$receipt = Get-Content -LiteralPath docs/recipes/sprint-speed-round3/experiments/startup-memory/diagnostic.json -Raw | ConvertFrom-Json
$receipt.input_recipe | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath artifacts/sprint-1b-speed-20260911/reproduce-startup-recipe.json -Encoding utf8
.venv/Scripts/python.exe artifacts/sprint-1b-speed-20260911/diagnostic_wrapper.py --config artifacts/sprint-1b-speed-20260911/reproduce-startup-recipe.json --output artifacts/sprint-1b-speed-20260911/reproduce-startup-memory
```

Use a new output directory and the runtime/source versions recorded in the receipt. The guard acts only after a synchronized completed step; it cannot interrupt a blocked GPU call. The archive does not modify allocator settings or production code.
