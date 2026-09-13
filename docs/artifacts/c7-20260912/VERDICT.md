# C7 verdict: integrated, ranking confirmed (2026-09-12)

Integration: `experiment/c7-integration` = `main` + merge of accepted
`experiment/c6b-clean-tail` (merge commit `ae5ce1a`). C6A stays out
(rejected). Verified in-tree: `py_compile`, C6B wiring tests 3/3,
`prepare_upstream.sh` green end-to-end (0001→0005 chain).

## Reprofile (cites the C6B-eval combined capture, same code)

- C1-equiv: bitwise 0.0 vs committed main-path latents (16/16).
- Improvement: clean-forward beats in-run denoise-3 in 14/15 actions
  (action 7 forward-tied; correction 2026-09-12, B001 A1 — original text
  said 15/15; immutable evidence commit 50dc857 message preserved as-is)
  (−4..−8 ms rolled, mean ≈ −6 ms 1:1 off next-ready).
- Combined medians from that capture: base 989.6 / ready 1315.4, with
  the recorded cross-day drift caveat (busy host; C0B envelope stands).

## New ranking: unchanged, by construction

C6B trims only the clean-KV tail (~6 ms off the 196 ms non-SDPA slice);
no other forward changes. Order preserved: denoise non-SDPA, denoise
self-SDPA, clean-KV non-SDPA (~190), clean-KV self-SDPA. No new C5x
spawned: no new information, and remaining residuals are tuned-GEMM
floor plus healthy SDPA with all optimization avenues adjudicated
(C6A rejected, 12→10 declined, conditioning-hoist declined).

## Incidental hygiene fix (whitespace-only, zero behavior)

The shared upstream working copies had drifted by 3 blank-line/
whitespace spots vs canonical HEAD+0001-0004 (one double-spaced
comment, two blank-line placements), which broke `patch` context for
0005 in fresh trees. Normalized the primary copy (future worktrees
clone from it) and the C7 copy to canonical; verified byte-identical
against a rebuilt reference. The C6B eval copy was verified
byte-identical to canonical+0005, so its measurements stand.
