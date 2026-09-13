# Audit B001-c1r-through-c7-20260912 — independent batch audit

Date: 2026-09-12 (UTC 2026-09-13) | Auditor: independent batch auditor (not the decision-maker/implementer)
Scope: C1R closure → C2 → C3 → C4 → C5A/C5B → C6A/C6B → C7, all 2026-09-12, gfx1151 Strix Halo
Workspace (read-only except this dir): `orchestration/reconcile-c1` @ `92f79cd`
Method: every major claim re-derived from git objects, committed JSON evidence, and raw
`/tmp` artifacts on CPU. No GPU runs. No prior session summary trusted. All seven
worktrees have clean status (no uncommitted evidence anywhere). All experiment
branches verified in sync with `origin`.

## Verdict: AUDIT_PASS_WITH_ACTIONS

No material evidence gap capable of changing the accepted runtime or project
direction. The accepted runtime delta (C6B `kv_write_only` dead-tail skip,
integrated via merge `ae5ce1a`) is correctly accepted; the C6A rejection is
correctly rejected; C8 deferral is justified. One factual correction (14/15, not
15/15), evidence-hygiene actions, and one bounded quality follow-up are listed
as actions — none blocks the batch.

## Challenge points (each with explicit verdict)

### 1. Quality-history contradiction (F25 vs C2) — RESOLVED, both observations preserved

- Earlier record: commit `4109834` (F25, 2026-09-09, `docs/FINDINGS.md`): 3-step
  frames show "somewhat more texture/water variation and softer detail than the
  four-step reference". Conditions: 18+6 window, canonical FP16 Wan VAE, lake/tree
  scene, 81-frame run, and critically an **unmatched-state** comparison (the 3-step
  run "independently evolved its own persistent state"; 4-step was a separate reference).
- C2 (2026-09-12): 4-step visibly softer than 3-step. Conditions: 12+6 window,
  TAEHV presentation decoder, Great Wall scene (`examples/03`), **matched-state**
  warmed-Y comparison (identical through chunk 11, same seed/actions), single chunk17 frame pair.
- Verdict: no overwrite in either direction. Differing baseline state (unmatched vs
  matched trajectories), window (18+6 vs 12+6), decoder (canonical VAE vs TAEHV),
  scene, and sampled frames fully explain how two honest single-rater observations can
  point opposite ways. F25 already scoped itself ("qualitative observation, not a formal
  image-quality score… broader human quality evaluation would still be appropriate").
  C2's README likewise leaves "broader-scene confirmation still open". The correct
  reading: 3-step is quality-supported **on the tested scene/configuration**, not
  universally proven superior. Action A5 covers the residual.

### 2. Is single-rater C2 review sufficient to freeze C3? — SUFFICIENT for a bounded PASS, with one named follow-up

- The freeze does not rest on the rater alone: A/A bitwise 0.0 floor (re-derived
  exactly, §3), eviction-onset divergence with bounded RGB MAD ≤0.030 (re-derived
  from committed PNGs: chunk05 0.0, chunk17 0.030101), denoise deltas an order of
  magnitude larger (x0 2.1–3.5, RGB MAD to 0.107), decoder MAD 0.0237 with
  matching temporal stats, all runs finite. The rater only resolves direction
  (which config is softer), which numbers cannot.
- Smallest additional evidence required (not blocking): a blinded second-rater
  A/B on the already-committed PNG pairs (`A1` vs `C` chunk17, `A1` vs `B` chunk17)
  plus one repeat of the 3-vs-4 comparison on a second scene. Recorded as action A5.

### 3. C2 A/A zero floor + canonical warmed-Y contract — VERIFIED by re-derivation

- Re-ran the committed comparison logic on CPU against committed
  `docs/artifacts/c2-screen-20260912/{A1,A2,B,C}/accepted_latents.pt`
  (branch `experiment/c2-quality-screen` @ `889500f`): A1==A2 bitwise over all 16
  chunks (max-abs 0.0 ×16), identical to the committed `c2_screen_analysis.json`.
  Committed files are bitwise identical to `/tmp/c2-screen-20260912/` sources.
- All four configs score chunks 2..17 only (warmup 0–1 hashed, not scored;
  warmup hashes corroborate the C1R X,Y steady-state pattern: A1/A2/B share
  chunk-0 `b3aee5…`, C differs as expected under a different schedule).
- (Audit note: stream layout is `[chunks, channels, h, w]` chunked along dim 1 —
  a naive dim-0 diff disagrees; the committed `c2_screen_compare.py` split was
  followed exactly and reproduces the analysis file bit-for-bit in effect.)

### 4. 18-vs-12 conclusion — VERIFIED numerically; "no-viewed-degradation" correctly scoped

- Trajectory `x0_sha256` hashes: chunks 2–11 identical, first mismatch at chunk 12;
  per-chunk max-abs re-derived: ten 0.0s then 0.164/0.288/0.618/1.180/1.431/1.640.
- Cache positions confirm mechanism: A1 `local_end` hits capacity 12096 exactly at
  chunk 11 and rolls at chunk 12; B (capacity 18144) still filling at chunk 12.
  Onset-at-first-eviction is mechanism-backed, not coincidental.
- Numerical divergence (x0 to 1.64, RGB MAD to 0.030, 14% pixels >0.05 by chunk 17)
  is distinguished from visible degradation: the viewed chunk17 pair shows the same
  coherent scene. The claim is scoped to viewed frames (chunk05 + chunk17 contact
  sheets committed; full sets in /tmp); it does not overclaim beyond the sample,
  and the README says so. Longer-horizon (>16-action) benefit of 18+6 remains
  untested — ranked as possible miss M2 (low-medium impact).

### 5. C3 eight-field contract vs C2 evidence — ACCEPTED with two flagged fields

| # | Field | Support |
|---|-------|---------|
| 1 | chunk 1 | Live path fixed + C2 ran chunk-1 — direct |
| 2 | local 12 | 18+6 diverges post-eviction without proven benefit — direct (rejection is cost/benefit, not quality-inferiority; correctly stated) |
| 3 | sink 6 | **FLAGGED: no direct C2 support** — C2 never varied sink; rests on the 2026-09-10 sink-budget prior. Carried, not proven, in this batch |
| 4 | 3-drop-957 | Direct on tested scene (§1 caveat applies) + C0B cost |
| 5 | 384x672 | **FLAGGED as provenance, not quality evidence** — single geometry tested; "exact-geometry asserted" is a control, not a comparison |
| 6 | TAEHV FP16, MAD 0.0237 noted-not-equivalence | Direct; honestly bounded |
| 7 | responsiveness envelope | C0B-quoted with C4 re-derivation discipline stated — direct with lineage |
| 8 | 16-action horizon | Matches C2/C1R rollout lengths; longer horizons explicitly require recurrence evidence (branch not triggered — consistent with stable 16-action runs) |

No field invents support it does not have; fields 3 and 5 are carried priors,
labeled as such. No HOLD: neither affects the accepted runtime.

### 6. C4: (a) ranking reproducibility VERIFIED, (b) absolute latency NOT reproduced — correctly separated

- (a) Attribution arithmetic re-derived from committed `profile-summary.json`
  (`experiment/c4-profile` @ `549a3d0`): denoise non-SDPA = wall 961.928 −
  self-SDPA 379.432 = **582.496** ✓; clean-KV non-SDPA = 323.073 − 127.063 =
  **196.009** ✓. Definitional note: cross-SDPA (~19/6 ms) sits inside the
  "non-SDPA" bucket (wall-minus-self). Ranking order stands regardless.
- (b) Absolute medians (969.9/1297.1, n=8 pooled) sit +30/+38 ms above C0B's
  envelope (939.9/1258.9, re-derived from committed `metrics-summary.json`),
  outside C0B ranges. The environmental-shift reading (busy host, load ~11.5,
  58 °C vs 40 °C idle) is plausible and the decision consequence is correct
  (C0B envelope retained; C6 must use within-run metrics) — but it is **not**
  proven by a controlled quiet rerun. Ranked as possible miss M3. No algorithmic
  change is claimed from the shift; the two are never conflated in the verdict.

### 7. C5A/C5B derived from C4, not parked hypotheses — VERIFIED with one note

- Phase assignment follows the measured split: C5A ← denoise ranks 1+2, C5B ←
  clean-KV ranks 3+4. C5A's pursued candidates (norm islands vs measured 48 ms
  eager norms; modulation-hoist, refuted) target the measured rank-1 non-SDPA
  remainder, not PF1's self-SDPA slice; C5B's dead-tail targets the measured
  rank-3 slice. PF1/PF2 activation went through the documented C3/C4 gate.
- Notes: (i) no C5 report is committed (`evidence: []`, no `docs/*c5*` file) —
  C5 conclusions survive only as DAG result strings, mitigated by downstream
  re-verification (modulation per-block parameter confirmed at
  `model_fast.py` ~l.288; early-exit validated by C6B's bitwise result). Action A6.
  (ii) PF1's denoise self-SDPA (379 ms) never got a candidate — correctly left
  unaddressed (healthy flash SDPA, no mechanism), recorded as miss M5.

### 8. C6A rejection — CORRECT, evidence-backed negative result

- Branch `experiment/c6a-helper-islands` @ `922e5d0`, unmerged. CPU parity held
  (2/2 pytest ≤2e-6); gfx1151 GPU probe diverges (norm-affine ~1e-2-scale,
  rmsnorm 0.0145) while every norm output feeds the bitwise-0.0 K/V cache gate —
  rejection follows the cache-exactness requirement, not implementation difficulty
  (implementation was complete). The modulation-hoist sub-claim is independently
  refuted (per-block `modulation` parameter). Counts in favor of the batch.
- Discrepancy (negligible): VERDICT cites 0.057 for norm-affine; the preserved
  `/tmp/c6a-eval-20260912/norm_check.json` records 0.0328/0.0145. Both exceed the
  2e-6 gate by four orders of magnitude; conclusion unaffected. Action A2.

### 9. C6B acceptance — ACCEPTED with one factual correction

- Latent equivalence re-derived from raw `/tmp/c6b-eval-20260912/screen-A1/
  accepted_latents.pt` (matched 16-action A1 config: 12+6, 3-drop-957, seed 42)
  vs committed C2 A1: max-abs **0.0 over all 16 chunks, bitwise True**. Claim holds.
- Semantics: patch 0005 threads `kv_write_only` through block/dit forward,
  returning early after K/V stores on the clean-commit path only
  (`commit_clean_kv` sets it; denoise path untouched). `prepare_upstream.sh`
  wiring asserted by 3/3 tests — re-run on CPU: **3 passed** (must run from
  `scripts/` for the `pure_compile_helpers` import; invocation-sensitive, not broken).
- CORRECTION: the "15/15 actions" within-run win claim is wrong. Recomputed from
  committed `benchmark.json` forward_records: **14/15** — action 5 clean loses by
  +0.63 ms (242.96 → 243.59). Rolled actions are 4/4 wins: −4.09/−5.90/−8.01/−5.53
  (mean −5.88 ≈ −6 ms 1:1 off next-ready). The correction changes no decision
  (consistent, zero-risk, bitwise-preserving saving at the low end of the 6–12 ms
  estimate) but the claim must be fixed everywhere it appears (commit message
  `50dc857`, C6B README, DAG C6B result, C7 verdict). Action A1.
- Next-ready vs base-RGB correctly distinguished throughout: the clean forward sits
  fully on the next-ready path; base-RGB is unaffected. Absolute medians
  (989.6/1315.4) are properly disclaimed as drift-confounded, not used for acceptance.
- C7-integration tree verified byte-identical to the C6B side for
  `scripts/ patches/ tests/` (`git diff 50dc857..27f6011` empty there) — the
  "same code" basis for citing the C6B capture holds — but no post-merge GPU run
  exists. Ranked as miss M6 (low: merge adds no code delta on the measured path).

### 10. C7 — VERIFIED integrated; reprofile-by-construction is arithmetically sound

- Merge `ae5ce1a` (parents `add8a69` + `50dc857`) delivers exactly the C6B package:
  patch 0005, `prepare_upstream.sh` wiring, `pure_compile_helpers` mirror,
  `commit_clean_kv` flag, wiring tests, evidence JSONs. C6A correctly excluded.
- "Ranking unchanged" is deductive and checked: trimming ~6 ms off the 196 ms
  rank-3 slice cannot reorder ranks (gap to rank 4 is ~69 ms). No new C5x is
  correct (no new information; remaining residuals are tuned-GEMM floor + healthy
  SDPA with all avenues adjudicated). C8 deferral follows its entry criteria
  (no measured residual with a candidate at complexity-justifying scale). The
  whitespace-hygiene note in the C7 verdict is consistent with the observed patch
  context lines. C7 tests correspond to the integrated tree (re-run green, §9).

### 11. DAG/branch/artifact reconciliation — COMMITS ALL EXIST; three traceability weaknesses

- All 24 cited commits verified present (`c4f9656` … `27f6011`, `add8a69`,
  `2bf97c3`, C4 aggregator chain). All six experiment branches plus
  `orchestration/reconcile-c1` are in sync with `origin`. No uncommitted changes
  in any worktree. All `/tmp` artifact dirs and all seven `run-*.sh` runners present.
- W1: C2/C4/C6B/C7 evidence files live only on experiment branches; the
  orchestration branch lacks `docs/artifacts/{c2-screen,c4-profile,c6b,c7}-*` dirs
  while the DAG cites them as bare paths, sometimes without branch. Commits exist;
  paths resolve only with branch context. Action A3.
- W2: `recommendedQueue.blocked` still lists `[C6A, C6B, C7]` although all are
  terminal; the markdown "Running — C1R" section still describes C1R as running
  with C2 blocked. Stale terminal-state hygiene. Action A4.
- W3: C5A/C5B/C6A/C6B `evidence: []` in machine DAG (verdicts live on branches).
  Action A3/A6.

### 12. Adversarial sweep — no hidden blockers; misses ranked below

Searched for: uncommitted evidence (none — seven clean trees); stale status (W2
above); changed baselines (C0B envelope intact; C4 shift disclaimed, M3);
missing negatives (none — C6A preserved, declined options recorded with reasons:
12→10 contract-breaking, conditioning-hoist invasive 2–6 ms, host-cursor adder
−10 ms deferred uncombined); stronger-than-experiment conclusions (the 14/15
correction; "no-viewed-degradation" scoping holds); weaker-than-claimed configs
(C6B benchmark ran 15 actions vs C2's 16 — separate timing vehicle, disclosed
absolute numbers disclaimed; screen-A1 correctness vehicle is matched 16-action).

## Ranked possible misses (likelihood × impact → rank)

| # | Miss | L × I | What would close it |
|---|------|-------|---------------------|
| M1 | Single-scene/single-rater direction call (F25 points the other way on another scene) | M × M = **HIGH** | Action A5: blinded second rater on committed PNGs + one second-scene 3-vs-4 screen |
| M2 | 18+6 benefit beyond 16-action horizon untested | L × M = MED | One 32+-action matched screen if a long-horizon use case appears |
| M3 | +30 ms C4 shift attributed to environment without controlled quiet rerun | L × M = MED | One quiet-condition waterfall rerun; until then keep C0B as reference (already done) |
| M4 | Host-cursor adder (−10 ms, measured, uncombined) left on the table | certain × L = LOW | Combine-and-measure as a future C6-class candidate |
| M5 | Denoise self-SDPA 379 ms has no candidate (PF1 slice untouched) | certain × L-M = LOW-MED | None available without a new mechanism; correctly not invented |
| M6 | C6B saving measured pre-merge only (same-code argument, no post-merge GPU run) | certain × L = LOW | One post-merge browser run when GPU time is cheap |
| M7 | C6B raw latents + C2 full PNG/RGB sets live only in volatile /tmp | certain × L = LOW | Action A2: commit the .pt (4 MB) + sha-list of PNGs |
| M8 | "15/15" overclaim repeated in four places | certain × negligible | Action A1 correction |
| M9 | C6A 0.057-vs-0.0328 number inconsistency | certain × negligible | Action A2 reconciliation note |
| M10 | Stale queue + stale md Running section | certain × negligible | Action A4 |
| M11 | Qualified-PASS rests on superseding blanket bitwise C==A (one-shot X-reversion characterized, RECOVERED_IMMEDIATELY) | L × L = NEGLIGIBLE for this batch | Revisit only if future work touches the prepare/bootstrap path |
| M12 | C2 RGB rows for 14/16 chunks re-derivable only from /tmp (committed PNGs cover chunks 05/17) | certain × negligible | Covered by A2/A5; chunk17 (max-divergence) pair is committed and re-derived exact |

## Actions (for the orchestrator; this audit edits nothing outside its own dir)

- A1 (factual correction): 15/15 → 14/15 with the action-5 +0.63 ms exception;
  rolled 4/4 −4.1…−8.0 ms mean ≈ −6 ms stands. Touchpoints: C6B README,
  commit-message record, DAG C6B result/Changelog, C7 verdict wording.
- A2 (evidence preservation): commit C6B `screen-A1/accepted_latents.pt`
  (4 MB, or sha + regeneration recipe) and the C6A probe's full numbers;
  reconcile the 0.057/0.0328 pair in one line.
- A3 (traceability): qualify DAG evidence paths with owning branches (or mirror
  artifact dirs); fill C5/C6 machine-DAG `evidence` with branch-qualified pointers.
- A4 (hygiene): refresh `recommendedQueue` terminal state and the markdown
  Running-C1R section to past tense.
- A5 (quality follow-up, only MED-bound item): blinded second-rater check on
  committed pairs + one second-scene 3-vs-4 screen before universalizing field 4.
- A6 (optional): attach the C5 deliberation notes as artifacts or mark the nodes
  deliberation-only with downstream re-verification pointers (already true in prose).

## Re-derivation log (CPU, this audit)

- C2 A/A 16×0.0, window onset-at-10th-index + values, denoise everywhere — exact match to committed analysis JSON.
- C2 RGB chunk05 0.0 / chunk17 0.030101 (+p95 0.137255) from committed PNGs — exact match.
- C2 warmup-hash pattern + trajectory cache-eviction positions — confirm warmed-Y contract and chunk-12 first eviction.
- C0B medians 939.877/1258.887 — match DAG envelope.
- C4 waterfall medians 969.856/1297.137 + attribution arithmetic 582.496/196.009 — match.
- C1R rerun: gate 7/7 true, rollout PASS, C==A false-once (condition/x0/K/V), C1==C2==A recovery true — match closure narrative.
- C6B bitwise 16×0.0 vs C2 A1 from raw — holds; within-run 14/15 (action-5 exception) — corrects record.
- C7 merge content + empty code diff 50dc857→27f6011 on measured path; wiring tests 3 passed (from `scripts/`).
- Per-block `modulation` parameter (refutes hoist); all branches origin-synced; seven clean trees.
