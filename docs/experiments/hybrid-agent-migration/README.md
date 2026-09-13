# Hybrid-agent migration experiment (workspace reserved, NOT RUN)

Objective: benchmark whether routing agent workloads across Ciru Apodex
local workers (execution), Muse Code (supervision), and DeepSeek (diversity
review), under a deterministic benchmark judge, beats the current default —
without changing any provider defaults.

## Required before running

1. **Inspect the current Magpie workflow first.** No Magpie checkout exists in
   this workspace yet (searched 2026-09-13); the experiment plan must name the
   target workflow and its current routing before proposing alternatives.
2. **Billing provenance for Muse Code.** Every supervised run must record
   model, tokens, and cost source so cost/quality tradeoffs are auditable.
3. **Deterministic judge rule.** The judge must be scripted and re-runnable
   (fixed tasks, fixed scoring, pinned inputs) — no model-graded vibes as the
   migration gate.
4. **Proposed backends** (to fill in): local-worker stack spec, supervisor
   configuration, diversity-reviewer configuration.

## Verdict states

`BENCHMARKED_KEEP_DEFAULT` | `BENCHMARKED_MIGRATE` | `BENCHMARKED_HYBRID` |
`INCONCLUSIVE` (plus `NOT_RUN`, the current state).

## Results

None. Do not pre-fill benchmark results or assumed routing percentages.
