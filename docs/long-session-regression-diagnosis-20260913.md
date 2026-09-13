# Long-session regression diagnosis (2026-09-13)

Durable record of the product-regression investigation. Method, numbers, and
preserved evidence — not a re-derivation. Raw per-action tables and key logs
are in [`artifacts/long-session-20260913/`](artifacts/long-session-20260913/).
Full frame sets (~20 MB JPEGs) and clips lived in volatile `/tmp` during the
investigation; six representative frames are preserved in the artifact dir.

## Accepted baseline at investigation start

`main@9cc986c` == `origin/main`; tag `lingruntime-gfx1151-b001` resolves to
`9cc986c`. C8 at `C8_STOP_NEGATIVE` (terminal audit pending). Play observations
under test: ~2.1 s action latency (vs ~0.85 s fresh), visual roughening over a
mature rollout (turns worst, key `k` noted), ~5 min startup. B001 closed; no
kernel work permitted.

## R1 — startup: the ~5 min impression is real

One clean production-equivalent startup (clean `wt-play` code at detached
`9cc986c`, port/output-path diffs only): wall 288 s, process→ready 268.6 s.
Waterfall: TunableOp 0.04 s → model init 37.5 s → prewarm 2.1 s →
**session preparation 226.9 s (84%)** → TAEHV 0.8 s → bootstrap 1.2 s.
GPU 0% through preparation: host-side cost (T5 + full-path VAE encode +
camera/plucker), not compile/autotune. Replicated: 295 s wall on a third
process. Preparation scales linearly with path length (779 s @ 961 frames).

## R2 — latency-vs-age: flat to 240 actions

240 scripted `w` actions (`examples/00`, accepted runtime, existing telemetry
only): denoise ramps 580–616 ms (action 1) → ~955 ms (action 12) as the local
window fills, then holds flat (base-RGB 984–1006, next-ready 1311–1336; worst
single action 1013.7/1342.3). `local_end_index` pins at exactly 12096 from
action 12; global counter keeps counting (expected). **No compute regression
with session age.** The 2.1 s figure was not reproduced serially (2.1× above
the worst observed). Design ceiling found: one process supports at most
(poses−1)/4 actions (67 on `examples/03`, 240 on `examples/00`), then
`maximum configured browser actions reached`. Client action counters are
per-page-lifetime and can exceed any single-process ceiling across
restarts/reloads — the quoted "Frame 457" cannot be a single-process id under
any shipped path (max 240).

## R3 — turning is not slower

Two mature 8-action blocks `w,w,j,j,w,k,w,l` (actions 151–158, 229–236): all
turn keys within ±8 ms of neighboring straight actions, including `k`.
Per-forward, clean-KV, prepare, TAEHV all flat.

## R4 — reset discriminator

No in-process session reset exists in the browser server (only a
decoder-stream reset); reset = fresh process. Three independent processes
replay the identical cache-fill ramp (±15 ms). No process-lifetime
accumulation ≤240 actions.

## R5 — thermal/clocks/memory: clean

70–74 °C driving, sclk at max, power flat ~81–85 W; host memory flat during
driving and back to baseline after shutdown; dropped tails 0; all forwards
finite. No throttling, leak, or allocator signal.

## R6 — visual quality: real drift, onset 15–30, collapse by 45–60

Matched production path (`examples/03`): action 15 clean → 30 heavy mosaic
artifacts → 45 scene dissolving → 60 total static. Mismatched path collapses
identically (path mismatch excluded). Onset coincides with post-eviction
rollout (eviction starts action 12–13), far beyond the validated 16-action
horizon. Turn frames are equally collapsed at matched maturity — turns
*reveal* drift (frame deltas 2–3× higher) rather than causing it. Metric-only
QA would miss this; Laplacian-sharpness trends were content-confounded.
Latency stays flat throughout: quality and latency are decoupled.

## R7 — coupling verdict: separate mechanisms (Hypothesis D)

- Latency limb: bounded cache-fill cost + **input-mailbox queueing** (below).
- Quality limb: causal rollout drift beyond the validated horizon (model/state
  recurrence; KV plateau verified, so not a capacity bug).
- Turns: neither cause (rejected on both limbs).

### The 2.1 s, reproduced

Follow-up experiment (serial 1–4, then 700 ms cadence 5–12): queue waits
189–681 ms, two intermediate presses silently replaced (latest-wins depth-1
mailbox), completed actions inflated to 1084–1610 ms **with denoise unchanged**
on its normal ramp. Human arithmetic: 2126.9 − 986 ≈ 1141 ms ≈ one queued
cycle; 2449.9 − 2126.9 = 323 ms ≈ clean-KV + tails. It presents as "maturity"
because play tempo rises with engagement. Dropped tails stay 0 throughout,
matching the human report.

## Next actions (not started)

1. Latency presentation: split HUD action→RGB into queued
   (`input_received→input_selected`, already in ledger) vs generating;
   acceptance on the pipelined run. Deeper mailbox semantics need design review.
2. Quality: rollout drift beyond the 16-action horizon is model-side; any
   re-grounding/session-cycling experiment changes semantics — plan first.
3. Startup: sub-mark session preparation (T5 vs VAE vs plucker) before
   optimizing.

## Provenance notes / limitations

- Work was done from clean worktrees at detached `9cc986c` (play path) with
  `PRIMARY` env for venv/models/upstream; labeled diffs only (ports, output
  dirs, scripted keys via the same submit path, `examples/00` trajectory for
  the 240-run — identical compute shapes).
- The registered main worktree was observed dirty (staged B001-artifact
  deletions, unstaged C6B unwiring); play/diagnosis runs did not use it.
  Reproduce its state with `git diff HEAD` there; it was left untouched.
- Per-action RSS reads in early drivers matched the wrapper pid and are
  invalid; memory verdicts rest on host `free` trends + post-shutdown baselines.
- Only lead (f0) frames were saved per action (tails arrive after
  `action_record`); clips are f0 sequences, labeled as such.
