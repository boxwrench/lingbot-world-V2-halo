# Agent routing (placeholder — Ciru Apodex migration not yet benchmarked)

This directory reserves a home for the planned hybrid-worker migration:

```text
Ciru Apodex local workers
→ Muse Code supervisor
→ DeepSeek diversity reviewer
→ deterministic benchmark judge
```

Status: **BENCHMARKED MIGRATION — NOT YET DEFAULT.**

Rules until the benchmark lands:

- Do not implement provider-routing changes.
- Do not pre-fill benchmark results or assumed routing percentages.
- The migration verdict states are: `BENCHMARKED_KEEP_DEFAULT`,
  `BENCHMARKED_MIGRATE`, `BENCHMARKED_HYBRID`, `INCONCLUSIVE`.
- Billing provenance for Muse Code usage and a deterministic judge rule are
  required inputs to the verdict; see
  `../experiments/hybrid-agent-migration/README.md`.
- Inspect the current Magpie workflow first — there is no Magpie checkout in
  this workspace yet (searched 2026-09-13); the experiment must name its
  target workflow before running.
