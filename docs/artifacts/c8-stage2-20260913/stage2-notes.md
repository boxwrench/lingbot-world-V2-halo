# Stage-2 procedures and mishap log (2026-09-13)

## A (static, read-only)

- Listed `torch/lib/aotriton.images/amd-gfx115x/flash/attn_fwd/` (396 files).
- AKS2 = `AKS2` magic + xz stream at offset 16; decompressed with
  `lzma.LZMADecompressor` on `raw[16:]`; read AMDGPU-note msgpack ints
  (vgpr/sgpr/spill/scratch/workgroup-size) with a minimal int decoder,
  skipping the NUL-terminated name table.
- `strings libaotriton_v2.so.0.12.0` for env knobs: none (only
  `optune_op_attn_fwd__Trivial*` internal selection symbols).

## B (rocprofv3)

- System `/usr/bin/rocprofv3` fails against this stack:
  `libamdhip64.so.7: undefined symbol: hsa_amd_vmem_export_fabric_handle`.
- Stack venv's own `rocprofv3` works:
  `rocprofv3 --kernel-trace -d out -- <venv-python> one-call.py`
  (1 warm + 5 timed exact-shape `attention()` calls); read
  `rocpd_kernel_dispatch_*` joined to `rocpd_info_kernel_symbol_*`.
- Raw DB kept out-of-tree (`/tmp/c8-rocprof/out/`); geometry numbers above.

## C (image isolation) + mishap

- Backed up the 3 bf16 F_F images + SHA256 to `/tmp/c8-imgbak/`.
- FIRST ATTEMPT (shell globs, `mv *pat* stagedir/hidden-$v`) mis-staged:
  moved files kept original names, `unhide` matched nothing, two states
  ran with zero images (FAIL) and the tree was left without the 3 files.
  Caught by the count check; restored with `cp` from backup; verified
  396 files + hashes before proceeding.
- SECOND ATTEMPT (python `shutil.move` by exact `＊`-inclusive name,
  assert-present==[only] before each run, move-back + assert after):
  clean (see table in README).
- Final state: 396 files, all three bf16 hashes match backup, staging empty.

## Lesson (binding)

Filesystem-level config experiments require exact-name moves with
assert-before/after each state — never globs — plus hash-verified backup
and restore. This procedure is now the template.
