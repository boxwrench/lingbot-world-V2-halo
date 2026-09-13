# Artifact index (`docs/artifacts/`)

Per-directory provenance. Paths are stable — do not move directories; add
correction notes here instead of rewriting history. Branch-local records
(P2/C8/A5) are **not** in this tree; see STATUS.md for their locations.

| Directory | Campaign/node | Commit | Status | Standing | Summary file | Raw evidence | Limitations |
|---|---|---|---|---|---|---|---|
| `phase0-20260912/` | C0B Phase 0 | `add8a69` | historical | Primary Phase-0 record | `README.md` | yes (`timings.jsonl`, `waterfall.json`, env) | `next_ready_ms` alias caveat, see cross-check doc |
| `tunable-op-20260910/` | TunableOp pin | pre-B001 | **current** (production CSV) | `metrics-summary.json` | yes (`tunableop_results.csv` = pinned file) | none known |
| `compile-20260910/` | pure-helper compile | pre-B001 | historical | `probe-block-1-compile-metrics.json` | partial (probe metrics) | single-block probe, not full run |
| `prewarm-20260910/` | prewarm | pre-B001 | historical | `summary.json` | summary only | — |
| `kv-cursor-20260910/` | C1 cache cursors | pre-B001 | historical (superseded by C1R) | `summary.json` | summary only | — |
| `sink-budget-20260910/` | sink sizing | pre-B001 | historical | `summary.json` | summary only | — |
| `browser-serving-20260910/` | browser path | pre-B001 | historical | `summary.json` | summary only | 40-action serial client; does not stress mailbox (its own words) |
| `span-upscale-20260910/`, `spatial-upscale-20260910/` | upscale screens | pre-B001 | historical | `provenance.json` each | provenance only | — |
| `rc1-benchmark-20260911/` | RC1 benchmark | `f70e4e0` era | historical | `README.md` | yes (`actions.jsonl`, env) | predates C6B (~−6 ms next-ready) |
| `baseline-b001-20260912/` | **B001 closure** | `9cc986c` (main) | **current** | `README.md` | n/a (record doc) | line 57 says "pending A5" — A5 later reached scoped VISUAL_PASS (correction note, history preserved) |
| `c6b-20260912/` | C6B accept | `9cc986c` (main) | **current** | `README.md` | yes (`benchmark.json`, latents, `.pt`) | — |
| `c7-20260912/` | C7 integration | `9cc986c` (main) | **current** | `VERDICT.md` | verdict only | corrected per external review (see git log `f476440`) |
| `long-session-20260913/` | long-session diagnosis | this branch | **current** | `README.md` | yes (CSV, 4 server reports, logs, drivers, 6 frames) | f0-only frames; full sets volatile in `/tmp` (see its README) |

## Branch-local records (not in this tree)

- P2.0 baseline: `experiment/p2-baseline:docs/artifacts/p20-baseline-20260913/`
- C8 probes/stage2: `experiment/c8-attention:docs/artifacts/c8-probe01-20260913/`, `c8-stage2-20260913/`
- A5 visual gate: `experiment/a5-visual-3v4:docs/artifacts/a5-visual-3v4-20260912/VERDICT.md`
- DAG + contract + P2 static: `orchestration/reconcile-c1:docs/lingruntime/`, audits in `docs/audits/`

## Link/provenance notes (checked 2026-09-13, scripted link walk)

- Artifact READMEs/VERDICTs: all relative links resolve within their
  directories. No stale RUNNING/BLOCKED status language (only historical
  "pending A5" phrasing noted above, plus descriptive
  `max_model_pending_actions` / `pending_input_policy` field names).
- Sept 9–10 docs (`bringup`, `FINDINGS`, `taehv`, `window-*`,
  `action-response`, `pending-input`, `vae-repro`, solver audits) link into
  `../results/raw/...`, which is **intentionally git-ignored** (see
  `.gitignore`: weights, generated media, `results/raw/` not versioned).
  Those links are local-only by design, not regressions — but the underlying
  run outputs were never preserved in-repo, so those docs' figures cannot be
  re-derived from the repository. Left as immutable history; do not "fix" by
  inventing paths.
- `/tmp` references inside `long-session-20260913/` drivers point at the
  volatile run locations documented in its README; the preserved subset is
  listed there.
- Absolute paths: drivers and logs contain machine-local absolute paths
  (`/home/keith/...`) by nature of run transcripts; portable derived data is
  in `per-action-latency.csv`.
