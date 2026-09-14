# Rejected per-update NVFP4 weight cache

**This candidate is excluded from accepted speed results. No throughput measurement is accepted for it.** The forward cache preserved BF16 trainable parameters and the existing MXFP8 backward implementation, but strict full-model parity failed in both eager and compiled model modes. No tolerance was relaxed and no corrective numerical setting was validated.

The combined GPU/CPU validation run completed in 71.62 seconds with 9 passing and 2 failing checks. Six passes belong to the separate BF16-moment optimizer candidate. For this cache, both compiled payload checks and the scope/invalidation check passed; both full-model checks failed. The first reported eager loss mismatch was expected `5.555180549621582`, actual `5.554201126098633`. `cache-parity-attempt1/validation.json` is the preserved transcribed summary, not a full pytest log; that directory also preserves the failed source and tests.

The later 13.89-second tiny diagnostic (`cache-first-difference.json`) isolates the observed divergence:

- Initial model parameters are identical.
- Eager cache refresh exactly matches eager provider payloads, every captured layer output, and both microbatch losses.
- Compiled refresh changes 45 of 24,576 packed QKV bytes and 5 of 16,384 packed up-projection bytes relative to eager packing. Block scales and global scales match.
- The first differing activation is QKV with identical inputs. Flattening the cached blocked scales does not change the result, so the two-dimensional scale reconstruction is not responsible.
- The original payload tests compare two compiled packing functions. Passing those tests did not establish equality with eager provider packing. Inductor division rounding is a possible mechanism, not a demonstrated fix.

This archive retains the candidate, strict tests, diagnostic, exact failed-attempt snapshot, requested recipe, and diagnostic results. `manifest.json` records SHA256 values and the original repository-relative location of every copied file; all copied files were checked byte-for-byte against their original artifacts. Restore files to those `original_path` locations before attempting execution: the Python scripts depend on that directory depth and adjacent imports. The recipe alone does not activate caching in the production trainer or official benchmark; its dedicated wrapper is required.

The original locations are under `artifacts/sprint-1b-speed-20260911/`. The current design and measured diagnostic are described in `weight-cache-notes.md`. This archive is a record of a rejected experiment, not a supported training/checkpoint format or a validated speed recommendation.
