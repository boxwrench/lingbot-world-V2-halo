# C6B verdict: ACCEPTED (2026-09-12)

Package: `kv_write_only` dead-tail skip (patch 0005 + `commit_clean_kv`
flag + pure-block mirror). Branch `experiment/c6b-clean-tail`, unmerged
 awaiting C7 integration.

## Gate 1 — C1-equiv correctness: PASS, bitwise exact

`screen-A1` rerun on this branch vs committed main-path C2 A1 latents:
worst max_abs **0.0 over all 16 chunks** (`latents_compare.json`).
The skip preserves every K/V store bit-identically, as designed.
All actions finite.

## Gate 2 — measured improvement: PASS, modest and consistent

Absolute medians are drift-confounded across the day (939.9/1258.9 C0B →
969.9/1297.1 C4 → 989.6/1315.4 here; busy host, not config effects).
The robust read is within-run, drift-cancelling: clean-forward vs the
third denoise forward in the same action (`benchmark.json`
forward_records). Clean wins in **14/15 actions** (action 5 loses by
+0.63 ms — [correction 2026-09-12, external review: an interim correction
said "action 7 tied", which re-derivation from `benchmark.json`
refutes; original text said 15/15; the immutable evidence commit
50dc857 message is preserved as-is and remains the authoritative
record of that error]; rolled actions
(12–15): −4.1, −5.9, −8.0, −5.5 ms (mean ≈ −6 ms off next-ready 1:1,
clean sits fully on the next-ready path).

Estimate was 6–12 ms; observed lands at the low end. Real, consistent,
zero-risk (bitwise-preserving). Scope: clean-KV / next-ready path only —
no claim on base-RGB wall latency (external review 2026-09-12). The `--host-kv-cursor` adder (measured
−10 ms earlier, still uncombined) is recorded as a future C7 adder,
not part of this verdict.

## Disposition

ACCEPTED into the C7 integration set. Correctness gates unbroken;
measured improvement confirmed within-run.
