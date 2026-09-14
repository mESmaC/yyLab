# SPALMER dependency snapshot

yyLab uses an editable sibling SPALMER checkout. The yyLab work committed on
2026-09-14 calls dependency APIs that were still uncommitted in that checkout.
This directory preserves those changes as an explicit patch; it does not alter
the installed dependency automatically or publish the separate SPALMER repository.

- Patch: [spalmer-training-2026-09-14.patch](spalmer-training-2026-09-14.patch)
- Exact SPALMER base commit: `9f07a302f06d61e3ef371f2c4a8cf942da9f0666`
- Patch SHA256: `2fe41e5621fe749c2c75bb6eb6559abef74874c61f80916d0932d9a907e9786f`
- Size: 260,067 bytes; 46 files.
- [Manifest](spalmer-training-2026-09-14.manifest.json): complete included-path
  list, per-file hashes, scope, and validation results.

The patch includes prose preprocessing and text-index support; sampling fixes;
runtime, recurrent-cache, attention, expert packing and FP32-master precision
changes; depth evaluation and BF16 stochastic rounding; focused regression
tests; and the SM120 adaptation's attribution and license. The separate ternary
experiment, tools and report tests, generated reports, hardware work, and unrelated
documentation are excluded.

## Applying the snapshot

Use a **clean SPALMER checkout at the exact base commit above**. Do not apply it
on top of the already modified local dependency. If using another revision,
review the differences and resolve compatibility separately. From the yyLab
directory in PowerShell:

```powershell
$dependencyPatch = (Resolve-Path 'docs/patches/spalmer-training-2026-09-14.patch').Path
Get-FileHash -Algorithm SHA256 -LiteralPath $dependencyPatch
git -C ../spalmer rev-parse HEAD
git -C ../spalmer status --short
git -C ../spalmer apply --check $dependencyPatch
```

Verify the printed hash and base commit against this document, and confirm that
status is empty. After the check succeeds, apply it:

```powershell
git -C ../spalmer apply $dependencyPatch
git -C ../spalmer apply --reverse --check $dependencyPatch
```

Use yyLab's normal environment setup with that sibling checkout. This patch is
a dependency-source backup, not a package lock, checkpoint, dataset, or replacement
for dependency installation.

## Validation and limits

An exact-base archive was reconstructed under yyLab's ignored local
`artifacts/commit-checks-20260914` directory. `git apply --check` and actual
application both passed. Every included file then matched the Git-normalized LF
bytes of the selected dependency working file. All included Python files parsed,
and all absolute `spalmer.*` imports in included source modules resolved to files
in the reconstructed package. The original sibling's status, index bytes, and
included working files remained unchanged.

This validation did **not** execute dependency runtime tests, install a fresh
environment, or validate CUDA kernels. yyLab test results from the installed
environment are separate evidence. External package versions and supported GPU
capabilities still matter, particularly for prototype TorchAO/FLA interfaces.
The patch itself is stored without Git line-ending conversion so its recorded
SHA256 survives checkout; applied source hashes describe LF-normalized contents.
