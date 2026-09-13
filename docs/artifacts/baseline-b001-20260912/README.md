# Accepted baseline: B001 closure (2026-09-12)

Batch `B001-c1r-through-c7-20260912` closed as `AUDIT_PASS_WITH_ACTIONS`
with all CLOSURE_REQUIRED actions resolved and externally reviewed.
This record is the mainline baseline for Plan 2 (P2.0 reconstructs
measurements from this state; numbers below are pointers, not new claims).

## Identity

- Merge SHA (C7 integration tip): `f476440`
- Baseline tag: `lingruntime-gfx1151-b001`
- Previous main: `add8a69` (RC1 Phase 0 baseline, C6B absent)

## Accepted runtime change (C6B only)

`kv_write_only` clean-KV dead-tail skip: patch
`patches/0005-kv-write-only-clean-forward.patch` + `commit_clean_kv` flag
+ pure-block mirror (`scripts/pure_compile_helpers.py`,
`scripts/run_interactive.py`), wiring test
(`tests/test_c6b_patch_wiring.py`).

## Accepted claim

- Accepted latents bitwise identical (screen-A1 vs committed main-path A1,
  16/16 chunks, max_abs 0.0).
- Presentation path unchanged.
- No base-RGB improvement claimed; base-RGB path unchanged.
- ~6 ms mean saving on the rolled clean-KV / next-ready path.
- 14/15 individual clean-forward comparisons favorable (action 5 loses by
  +0.63 ms); rolled actions 12–15 are 4/4 favorable.
- `HUMAN_VISUAL_GATE = NOT REQUIRED` for this merge (bitwise-identical
  accepted latents, unchanged presentation).

## Negative result preserved (not merged)

C6A compiled norm islands: REJECTED (Inductor-CUDA divergence ~1e-2 on
gfx1151). Implementation and evidence remain on unmerged branch
`experiment/c6a-helper-islands`; verdict + probe JSON in
`docs/artifacts/c6a-20260912/` on that branch.

## Verification on the merged state

- `main` == `experiment/c7-integration` at merge (fast-forward, no C6A files
  in diff).
- pytest `tests/`: 11 passed (project venv, torch 2.13+rocm, upstream
  pinned `45fa406`, `PYTHONPATH=.upstream/lingbot-world-v2:scripts`).

## Evidence locations (branch-qualified)

- C6B evidence: `experiment/c6b-clean-tail`, `docs/artifacts/c6b-20260912/`
  (benchmark, latents compare, accepted latents, README).
- C7 verdict: `docs/artifacts/c7-20260912/VERDICT.md` (on main).
- Batch audit: `orchestration/reconcile-c1`,
  `docs/audits/B001-c1r-through-c7-20260912/`.
- Semantic contract (frozen RC1 point): `orchestration/reconcile-c1`,
  `docs/lingruntime/SEMANTIC_CONTRACT.md`; field 4 scoped to the tested
  warmed-Y scene/configuration, universal 3-over-4 direction pending A5.
- C2 quality screen: `experiment/c2-quality-screen`,
  `docs/artifacts/c2-screen-20260912/` (A5 blinded-rater source PNGs).
- C0B latency envelope: `docs/artifacts/phase0-20260912/` (pre-C6B;
  next-ready figures there predate the ~6 ms clean-KV saving).

## Environment

Strix Halo / gfx1151, ROCm (torch 2.13.0+rocm7.15), upstream
`lingbot-world-v2` pinned `45fa406`, TAEHV `taew2_1.pth` FP16,
12-frame/6-sink window, 3-drop-957 schedule, 384x672.
