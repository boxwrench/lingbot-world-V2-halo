# C5 deliberation provenance (B001 A6, 2026-09-12)

Why this file exists: the C5A/C5B research verdicts (accepted / declined /
refuted candidates) were recorded in the DAG only as outcomes. The
deliberation behind them — mechanism, audited-and-rejected alternatives,
estimate derivations — is preserved here as far as the existing evidence
allows. Nothing below re-decides anything; dispositions stand as committed.

## C5A (denoise phase) — full report preserved

Verbatim worker report: `c5a-deliberation-report-20260912.md` (this
directory), recovered from the run transcript `/tmp/c5a-report.txt`.
Read-only source research, no GPU used.

Adjudication trail (outcome → downstream record):

- Candidate 1 (12→10 K-reduction, ~60/80 ms): DECLINED for C6 — breaks the
  frozen C3 contract. Recorded as future C2-branch screen candidate.
- Candidate 2 (modulation-split hoist, 5–12 ms): ACCEPTED for C6A trial,
  then REFUTED before implementation (per-block modulation parameters,
  upstream `model_fast.py:288`; hoist impossible). See C6A verdict.
- Candidate 3 (compiled norm islands, 12–22 ms): ACCEPTED for C6A trial,
  then REJECTED on gfx1151 (Inductor-CUDA divergence ~1e-2, C6A verdict
  `docs/artifacts/c6a-20260912/VERDICT.md` on branch
  `experiment/c6a-helper-islands`).
- Audited and NOT proposed: cross-attn non-SDPA, VAE-Conv3d, host KV-cursor
  (already measured below gate), sink reduction at equal K (measured slower),
  full-block compile and C8-class work (excluded per pre-existing evidence).
- Net: no acceptable denoise-phase candidate remains; phase declared at
  floor (healthy K-linear flash attention + tuned GEMMs).

## C5B (clean-KV transaction) — reconstructed from record

The verbatim C5B worker report file is LOST (`/tmp` rotation; gaps recorded
in audit B001 A6). What follows is reconstructed from surviving records and
is marked as such — it is not a verbatim transcript:

- Mechanism (per DAG C5B result + audit): clean-KV is a full fourth DiT
  forward; ~85%+ of non-SDPA is load-bearing (downstream K/V dependency)
  already-tuned GEMMs; one true waste identified: post-store tail of
  block 29 + head/unpatchify feeding the discarded return (early-exit
  point verified on pinned upstream 45fa406: stores precede the SDPA read).
- ACCEPTED for C6B: dead-tail skip (est. 6–12 ms) + host-KV-cursor adder
  (measured −10 ms earlier, uncombined). C6B outcome: dead-tail ACCEPTED
  (bitwise 0.0, ~−6 ms off next-ready, 14/15 actions); cursor adder still
  parked for later, never combined.
- DECLINED: conditioning hoist (invasive, 2–6 ms); PF2 gate corrected to
  partially actionable (~16–28 ms package).
- Both C5 reports' stale C1R-gate passages (written before C1R closure)
  are SUPERSEDED by the committed C1R closure (f6ec5a0, rollout 16/16).

Provenance limit: candidate estimate derivations for C5B survive only via
the C6B measured outcome and the DAG/audit summaries. Any future challenge
to the C5B mechanism must re-derive from C4 profile artifacts, not from
this reconstruction.
