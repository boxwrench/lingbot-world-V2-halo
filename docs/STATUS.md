# LingBot World V2 (Strix Halo) — canonical status

**Read this first.** One page: what runs, what was accepted, what failed, what is open, where the evidence lives.

- Accepted production state: `main@9cc986c`, tag `lingruntime-gfx1151-b001`
  (annotated; resolves to `9cc986c`). Verified: `main == origin/main == 9cc986c`.
- Machine-readable research DAG (history through C8/P2):
  `orchestration/reconcile-c1:docs/lingruntime/RESEARCH_AND_EXPERIMENT_DAG.md`
  and `lingruntime-dag.json` in the same directory.
- Artifact catalog: [`docs/artifacts/INDEX.md`](artifacts/INDEX.md).
- Latest product diagnosis (2026-09-13):
  [`docs/long-session-regression-diagnosis-20260913.md`](long-session-regression-diagnosis-20260913.md).

> Scope note: `main` holds history through B001 closure. Later campaign
> records (P2, C8, A5, long-session diagnosis) live on experiment branches
> and in `docs/artifacts/` on this cleanup branch — linked below, not
> duplicated. Do not re-derive accepted numbers with ad-hoc microbenchmarks;
> see "Measurement discipline" below.

## Accepted production configuration

LingBot World V2 1.3B `causal-fast`, chunk 1, local attention 12, sink 6,
384×672 (480×832 @ 264192 max-area px), denoise `999 → 899 → 702`
(`3-drop-957`), BF16 DiT, exact deferred clean-KV, TAEHV `taew2_1` FP16,
pinned TunableOp results, warmed-Y/reset contract, normal production/browser
path. Source of truth:
`orchestration/reconcile-c1:docs/lingruntime/p2-baseline-static-20260912.md`.
Launch: `scripts/run_browser.sh` (defaults encode the accepted config).

## Current measured performance (mature, production runtime)

| Metric (P50) | Value | Source |
|---|---|---|
| action → base RGB | ~847 ms fresh strings; ~986 ms filled-window plateau | P2.0 (`experiment/p2-baseline:docs/artifacts/p20-baseline-20260913/`); long-session runs actions 1–240 |
| next ready | ~1136 ms fresh; ~1311 ms plateau | same |
| denoise total (3 forwards) | ~816 ms fresh; ~955 ms plateau | same |
| clean-KV (next-ready only) | ~305 ms plateau | same |
| startup, process → interactive-ready | ~269 s (≈5 min), 84% session preparation | `/tmp` logs summarized in diagnosis doc; artifact dir `long-session-20260913/` |

Warmed-Y/reset behavior: chunk 0 / first post-reset prepares are never
quality or timing signals; steady-state warmed comparisons only;
16-action quality horizon. There is **no in-process session reset** in the
browser server; reset = fresh process, which replays the identical
cache-fill ramp (±15 ms across three processes).

Human visual gate: **A5 VISUAL_PASS (scoped)** — 3-step acceptable on tested
mature rollouts (2 scenes + 2 reseed realizations); universal superiority not
established. Evidence: `experiment/a5-visual-3v4:docs/artifacts/a5-visual-3v4-20260912/VERDICT.md`.

## Known product limitations (accepted, not bugs in the runtime)

1. **Long-horizon causal drift.** Quality degrades from ~15–30 actions and can
   collapse structurally by ~45–60 actions on tested trajectories, far beyond
   the validated 16-action horizon. Latency is unaffected (flat to 240).
   Reserved investigation — do not start without approval.
2. **Single-process action ceiling.** The browser server supports at most
   (poses−1)/4 actions per process (67 on `examples/03`, 240 on `examples/00`);
   beyond that it raises `maximum configured browser actions reached` and
   stops. Client-side action counters are cumulative per page lifetime and can
   exceed any single-process ceiling across restarts/reloads.
3. **Input-mailbox queueing.** The action mailbox holds one pending action
   (latest wins, rest replaced silently). Keys pressed faster than the
   ~1.3 s generation cycle inflate measured action latency by ~one cycle
   (~2.1 s observed) and drop intermediate presses. Dropped tails stay 0.
4. **Startup latency.** ~5 min dominated by session preparation, scaling
   linearly with camera-path length. Identified target, not yet optimized.
5. **External runtime dependency.** `.venv` is a thin layer; torch/ROCm
   resolve via `strix_halo_rocm_base.pth` into
   `/home/keith/ciru-ling-runtime/.venv` (outside this repo), and
   `models/` (84 GB) plus `.upstream/` (1.2 GB) are unversioned. Reproduction
   requires that host state, not just this checkout.

## Campaign dispositions

| Campaign | Disposition | Canonical evidence |
|---|---|---|
| Phase 0 (RC1 preflight, C0A/C0B) | ACCEPTED | `docs/artifacts/phase0-20260912/` + `docs/phase0-20260912-independent-crosscheck.md` |
| C1 state/cache | SUPERSEDED by C1R (qualified PASS) | `experiment/state-cache-validation` |
| C2 quality causality | ACCEPTED (PASS) | branch `experiment/c2-quality-screen` |
| C3 semantic contract | ACCEPTED (frozen) | `orchestration/reconcile-c1:docs/lingruntime/SEMANTIC_CONTRACT.md` |
| C4 profile | ACCEPTED (ranking) | branch `experiment/c4-profile` |
| C5A/C5B research | ACCEPTED (mechanisms) | DAG `lingruntime-dag.json` |
| C6A helper islands | REJECTED (failed, preserved) | branch `experiment/c6a-helper-islands` |
| C6B kv_write_only | ACCEPTED (in B001) | `docs/artifacts/c6b-20260912/` (on `main`) |
| C7 integration | ACCEPTED (in B001) | `docs/artifacts/c7-20260912/VERDICT.md` (on `main`) |
| B001 closure | ACCEPTED, tag `lingruntime-gfx1151-b001` | `docs/artifacts/baseline-b001-20260912/` (on `main`) |
| A5 visual gate | ACCEPTED (scoped VISUAL_PASS) | `experiment/a5-visual-3v4:docs/artifacts/a5-visual-3v4-20260912/VERDICT.md` |
| Plan 2 (P2.0–P2.9) | DONE, no mainline change | `experiment/p2-baseline:docs/artifacts/p20-baseline-20260913/` |
| C8 attention feasibility | NEGATIVE (`C8_STOP_NEGATIVE`), **AUDIT_PENDING** | branch `experiment/c8-attention`; no terminal audit in `dsh-research-team` |
| Long-session regression diagnosis | DONE (this branch) | `docs/long-session-regression-diagnosis-20260913.md`, `docs/artifacts/long-session-20260913/` |

## Measurement discipline (binding)

- Timing comparisons must use the production runtime environment (pinned
  TunableOp CSV, accepted stack/config/presentation path, warm state).
- Unmatched microbenchmarks cannot support product-latency claims (Plan 2
  precedent: ~70 ms microbenchmark vs ~12–14 ms production value).
- Order-swapped replication for small effects.
- **Distinguish queue wait (`input_received→input_selected`) from model
  execution** before claiming any latency regression.
- Temporal/stateful quality requires mature-rollout video review, never
  still-only QA; do not infer long-horizon behavior from ~16-action validation.
- Human visual gates must match the product goal.

## Open items (do not start without approval)

1. C8 Harness-v1 terminal audit (`AUDIT_PENDING`).
2. Long-horizon stability investigation (reserved hybrid-team workload).
3. Startup/session-prep investigation (reserved hybrid-team workload).
4. Ciru hybrid-team benchmark (see `docs/agent-routing/`).
