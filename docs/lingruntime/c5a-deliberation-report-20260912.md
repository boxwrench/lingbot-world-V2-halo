C5A DENNOISE-PHASE HOTSPOT MECHANISM RESEARCH — READ-ONLY REPORT

Scope note: docs/artifacts/c4-profile-20260912/ranking.json does not exist in the tree (docs/artifacts/ contains browser-serving, compile, kv-cursor, phase0, prewarm, rc1, sink-budget, span/spatial-upscale, state-cache, tunable-op only). All measured numbers below come from docs/artifacts/phase0-20260912/profile-summary.json and docs/artifacts/phase0-20260912/metrics-summary.json, cross-checked against docs/profile-live-context-20260910.md, docs/window-18-vs-12-20260910.md and docs/window-12-vs-14-20260910.md. No files written, no GPU used.

================================================================
0. MEASURED BASELINE (load-bearing numbers)
================================================================

- Pooled rolled product timing (no profiling): denoise median 909.8 ms (min 899.5 / max 926.5), action-to-base-RGB median 939.9 ms, clean-KV median 298.2 ms (docs/artifacts/phase0-20260912/metrics-summary.json:20-34).
- Intrusive attribution of one denoise phase (chunk 13, rolled): intrusive wall 944.75 ms, module-exclusive total 931.88 ms; classes: CausalWanSelfAttention 446.09, Linear 327.68, CausalWanAttentionBlock 86.72, WanRMSNorm 36.20, WanCrossAttention 20.61, WanLayerNorm 11.83, Conv3d 1.67, rest <1 ms each; self-SDPA 369.33 ms over 90 calls, Q(1,12,1008,128) x K(1,12,12096,128), BF16, mask absent; kernel attn_fwd.kd on gfx1151 (docs/artifacts/phase0-20260912/profile-summary.json:5-41,81-87).
- "Non-SDPA ~582 ms" reconciliation: 944.75 − 369.33 = 575.4 ms on intrusive-wall basis (562.6 ms on module-exclusive basis). Consistent with ~582 ms within run variance; I use 575.4/562.6 below and flag which basis.
- Model config: dim 1536, 12 heads, 30 layers, head-dim 128, 1008 tokens/frame at 384x672 (docs/architecture-notes.md:16-21).

================================================================
1. WHY SELF-ATTENTION COSTS WHAT IT DOES
================================================================

1a. Rectangular shape origin (sink + recent rollover).
- Physical KV cache is local_attn_size frames x 1008 tokens. At the phase0 operating point (12-frame window, sink 6): 6048 sink + 5040 recent + 1008 current = 12096 K tokens, exactly the observed K shape (docs/sink-budget-20260910.md:36-39; docs/artifacts/phase0-20260912/profile-summary.json:29-31).
- "18+6 is not 24 attended frames": sink is a retention policy INSIDE the physical capacity, logical before first eviction, physically front-packed after rollover (docs/profile-live-context-20260910.md:190-203; docs/window-18-vs-12-20260910.md:57-71).
- The attention op receives the already-windowed cache; no mask, is_causal=false — rolling causal semantics are represented by the explicit cache slice, so SDPA sees dense rectangular cached attention, Q=1008 new-frame queries against K=12096 history (docs/profile-live-context-20260910.md:94-119; docs/architecture-notes.md:36-42).

1b. 90 calls = 30 blocks x 3 forwards.
- 30 layers (docs/architecture-notes.md:16-21) x 3 denoise evaluations (999→899→702 schedule; 4th point 957 dropped per F24/F25, docs/FINDINGS.md:720-751,756-803) = 90 self-SDPA calls; clean-KV pass adds 30 more with the same rectangular shape (docs/artifacts/phase0-20260912/profile-summary.json:24-31,61-68; docs/profile-live-context-20260910.md:94-96,182-188).

1c. Cost scaling is K-linear fused flash arithmetic, not overhead.
- Full−Early growth test: self-SDPA +482.8 ms vs total denoise +478.7 ms (~100.9% of growth); QKV/MLP/cross flat (docs/profile-live-context-20260910.md:165-180).
- Dispatcher overhead is ~0.4% (21.9 ms over 450 calls at 18-frame scale ⇒ ~0.05 ms/call ⇒ ~4-5 ms of the 90-call denoise total); single-op trace ~12.2 ms for Q(1,12,1508,128)xK(1,12,27144,128) on attn_fwd.kd, shapes matching the app path (docs/FINDINGS.md:620-649).
- Per-call at phase0: 369.33/90 ≈ 4.10 ms for K=12096. Cross-check vs 18-frame: 552.6/90 ≈ 6.14 ms at K=18144. Ratio 6.14/4.10 = 1.50 = exactly K ratio 18144/12096. So self-SDPA ≈ 0.339 ms per 1024 K-tokens per call (≈30.5 ms per 1024 K per 90-call denoise phase).
- Per-forward split is even (~180-187 ms each at Full/Rolled), so all 3 forwards are equal-cost rectangular evaluations (docs/profile-live-context-20260910.md:167-174).

1d. Rollover shift is real but small.
- First rolled forward +30-40 ms unprofiled; self-attention module body 617.3→651.1 ms while SDPA itself flat — eviction/shift work, not the Early→Full mechanism (docs/profile-live-context-20260910.md:205-210).

================================================================
2. AUDIT OF THE NON-SELF-SDPA ~575 ms (intrusive-wall basis)
================================================================

All figures are the 3-forward denoise total from docs/artifacts/phase0-20260912/profile-summary.json:9-21 unless noted.

a. Linear 327.68 ms — COVERED (TunableOp). Tuned lane: 3-forward linear 402.5→322.8 ms ordinary→tuned; time_projection.1 alone 78.9→11.0 ms; SDPA unchanged 357.3→358.3 ms, proving GEMM-only mechanism (docs/tunable-op-20260910.md:158-172). Captured signatures include the BF16 1536↔8960 MLP pair, 1536-wide QKV/O shapes, 512/4096 conditioning shapes, and float 9216x1008x1536 time projection (docs/tunable-op-20260910.md:78-88; docs/artifacts/tunable-op-20260910/tunableop_results.csv:6-21; docs/artifacts/tunable-op-20260910/tunableop_untuned0.csv:1-16). Residual: tuned GEMM time, not an uncovered target.

b. Self-attention non-SDPA body 446.09 − 369.33 = 76.76 ms — PARTIALLY CHARACTERIZED, no accepted cover. Components: dispatch (~4-5 ms per F21 scaling above), Q/K/V reshape+transpose views, RoPE application, KV-cache slice/index + writeback, qk-norm (qk_norm=True, docs/architecture-notes.md:16-21). Pure-compile explicitly leaves all of it eager (scripts/pure_compile_helpers.py:251-267 — "LayerNorm, attention, cache indexing, and KV writes remain eager"). Direct full-block compile was REJECTED: 23 graph breaks from local_end_index.item()/rollover branches, non-finite output after bootstrap, TunableOp preservation not established (docs/compile-20260910.md:53-65,67-84,91-100). Host-cursor variant proved the .item() syncs are NOT the cost: 360 model-side .item reads/action removed (401→41 aten::item, 389→29 syncs) yet only −8.3 ms first-visible / −10.0 ms next-ready, below release threshold, rejected as default (docs/kv-cursor-20260910.md:3-13,64-84,86-115).

c. CausalWanAttentionBlock non-child 86.72 ms — PARTIALLY COVERED (pure-helper islands). Matched profile: block remainder 112.04→87.02 ms (~25 ms saved), GELU module 10.28→0.04 ms (moved into helper), while Linear and SDPA unchanged (docs/pure-helper-compile-20260910.md:163-180). Compiled set is exactly {modulation, affine, scaled_residual, add, silu, gelu_tanh, camera_update}; norms, attention, ALL linears, KV/cache/rollover stay eager (docs/pure-helper-compile-20260910.md:44-64; scripts/pure_compile_helpers.py:1-6,226-306). Whole-path gain −28.2 ms denoise / −27.6 ms first-visible, 8 stable graphs, 0 breaks, 0 recompiles (docs/pure-helper-compile-20260910.md:75-97,129-153). Residual uncovered block work ≈ 60-87 ms (mostly per-block modulation chunk/affine dispatch + camera-path norms — see proposal 3).

d. Norms WanRMSNorm 36.20 + WanLayerNorm 11.83 = 48.02 ms — UNCOVERED. make_pure_block_forward calls self.norm1/norm3/norm2 EAGER and only compiles the surrounding affine (scripts/pure_compile_helpers.py:252-256,286-300). No TunableOp entry covers norms (all 16 signatures are GEMM, docs/artifacts/tunable-op-20260910/tunableop_untuned0.csv:1-16). Entire 48 ms is eligible remainder.

e. Cross-attention non-SDPA 20.61 − 18.72 = ~1.9 ms — negligible, no action.

f. Conv3d 1.67 + CausalHead 0.48 + Sequential 0.40 + SiLU 0.17 + GELU 0.04 ≈ 2.8 ms — negligible in DiT denoise. scripts/conv3d_micro.py is VAE-Conv3d-only (matched Wan-VAE shapes, docs/FINDINGS.md:8-32; scripts/conv3d_micro.py:1-7) — NOT relevant to this hotspot.

Summary of uncovered remainder (module-exclusive basis, sums to 562.6): self-attn body ~76.8 + block remainder ~60-87 + norms ~48.0 + cross ~1.9 + tail ~2.8 = ~190-215 ms addressable-in-principle; the other ~350 ms is tuned GEMM + fused SDPA at their measured floor.

================================================================
3. CANDIDATE INTERVENTIONS (concrete, C8-excluded)
================================================================

Gates context: C1 fixtures require cache bitwise 0.0, consumer 1e-6 (host) / 0.02 (gfx1151), and C1 currently PASSES incl. stale-state/broken-KV controls (docs/artifacts/state-cache-validation-20260912/summary.json:9-39,40-69). C1R exact-RC1 run: 15/16 PASS; the one FAIL (fresh_reset) was resolved by the discriminator + encode-probe to first-encode warmup nondeterminism (pattern X,Y,Y; steady Y universal), NOT a rollout leak — remedy is a throwaway conditioning warmup, production run_browser.py untouched (docs/artifacts/state-cache-c1r-20260912/README.md:6-11,39-64,71-95). Any intervention touching cache layout must re-run C1/C1R; interventions that only remove redundant compute or re-implement pointwise math keep cache code byte-identical.

--- Candidate 1 (largest): bounded K-reduction A/B, local 12 → 10 frames, sink 6 preserved ---
- Mechanism: same code path, smaller physical capacity (10080 vs 12096 K). SDPA cost is measured-linear in K (section 1c). No kernel/compiler change; one CLI flag.
- Expected ms (derived, not wished): slope from two independent A/Bs: (i) 18→12 rolled: ΔK=−6048, ΔSDPA=−183.2 ms ⇒ 30.3 ms per 1024 K per 3-forward denoise (docs/window-18-vs-12-20260910.md:81-98); (ii) 12→14 rolled: ΔK=+2016, ΔSDPA=+60.3 ms ⇒ 29.9 ms per 1024 K (docs/window-12-vs-14-20260910.md:57-74). Converged slope ≈ 30 ms per 1024 K. 12→10 (ΔK=−2016): denoise SDPA −59 to −61 ms; clean-KV scales same slope over 30 calls ⇒ −19 to −22 ms (check: 18→12 clean −65.4 × 2016/6048 = −21.8, docs/window-18-vs-12-20260910.md:81-98); first-visible ≈ −60 ms, next-ready ≈ −80 ms. This brackets parked PF1 (37-92 ms) and upgrades it from medium-low to measured-slope confidence. 12→8 would be ~2x (≈−120 denoise / −160 next-ready) but doubles history loss — recommend testing 10 first.
- Correctness risk vs C1/C1R: LOW-MEDIUM structurally (cache code unchanged, capacity parameter only; sink-6 retention layout 6048+3024+1008 re-validated by existing KV-invariant checks), but QUALITY risk is real: 12-frame is already ACCEPT-INTERACTIVE minimum-latency with "detectable scene/texture variation" vs 18 (docs/window-18-vs-12-20260910.md:115-144,163-176), and sink-2 proved history loss causes late anchoring failure at EQUAL K (docs/sink-budget-20260910.md:71-104,126-134) — so this A/B must carry the same 40-action traversal + contact-sheet quality bar, not just finite-output. C1/C1R must re-run at the new capacity (tolerances 0.0/0.02 unchanged).
- Sketch: bash scripts/run_live.sh --local-attn-size 10 (same 40-action string, docs/window-18-vs-12-20260910.md:34-49); verify K=10080 and layout 6048/3024/1008 in live metrics; no source change. Optional companion: --local-attn-size 8 only if 10 holds quality.

--- Candidate 2 (small, near-zero risk): hoist per-forward timestep-modulation split out of the 30-block loop ---
- Mechanism: within one forward, timestep embedding e is identical for all 30 blocks, yet make_pure_block_forward calls islands.modulation(self.modulation, e) → .chunk(6, dim=2) per block per forward = 90 chunk-dispatches per denoise action for 3 distinct e values (scripts/pure_compile_helpers.py:229-256,296-304; _modulation at scripts/pure_compile_helpers.py:19-20 operates on [1,1008,6,1536] f32 ≈ 9.3M elems). Precompute the 6-way split once per forward (3x/action) and pass parts in; math is an exact split hoist, zero FLOP-value change.
- Expected ms: modulation/chunk/affine-dispatch lives inside the ~60-87 ms block remainder (section 2c); 90 → 3 dispatches removes 87 Helsinki-dispatch-size ops on a ~37 MB f32 tensor plus 87 compiled-graph launches. Calibrated against the measured pure-helper saving (25 ms block remainder from comparable pointwise/dispatch elimination, docs/pure-helper-compile-20260910.md:163-180): this subsumed fraction is roughly one-third of that class ⇒ estimate 5-12 ms per 3-forward denoise (≈2-4 ms clean-KV). Deliberately NOT claimed as a primary lever.
- Correctness risk vs C1/C1R: NEGLIGIBLE. No cache/attention/Linear touched; chunk is exact (no rounding — split, not recompute); cache bitwise-0.0 gate unaffected. Only risk is plumbing error (wrong e parts to wrong block), caught by the existing C1 consumer check (1e-6/0.02) and finite-output gate.
- Sketch: in scripts/pure_compile_helpers.py make_pure_block_forward (scripts/pure_compile_helpers.py:226-306), add an optional e_parts parameter computed once per forward at the model-loop level (the DiT forward that iterates model.blocks), replacing the per-block islands.modulation call at scripts/pure_compile_helpers.py:248; keep eager fallback when absent. Same-class follow-up (only if this lands): cache the affine scale/shift squeezes per forward — same derivation, a few ms more.

--- Candidate 3 (medium): extend the ACCEPTED pure-island pattern to the eager norms (RMSNorm + LayerNorm), norms stay out of GEMM/SDPA paths ---
- Mechanism: 48.02 ms of norms are the largest single UNCOVERED remainder (section 2d) and are state-free pointwise+reduction math of the exact class already accepted (8 stable graphs, 0 breaks, docs/pure-helper-compile-20260910.md:75-105). Add _rmsnorm/_layernorm compiled helpers (fullgraph=True, Inductor default — same contract as scripts/pure_compile_helpers.py:106-131) and route the three per-block norm calls (scripts/pure_compile_helpers.py:252-256,289-300) through them; weights/bias read-only, no state.
- Expected ms: the accepted islands saved ~25 ms out of ~112 ms block remainder (~22%) plus GELU 10.3 ms by removing dispatch + fusing pointwise chains (docs/pure-helper-compile-20260910.md:163-180). Norms are reduction-fusion-friendly (single-pass Welford vs multi-kernel eager) but memory-bound at [1,1008,1536]; assume a conservative 25-45% of 48.02 ms ⇒ 12-22 ms per 3-forward denoise (≈4-7 ms clean-KV). Floor check: cannot exceed 48 ms; do not claim above 25 ms without measurement.
- Correctness risk vs C1/C1R: LOW-MEDIUM. No cache semantics touched (norms precede attention/cache writes; KV code byte-identical), so the 0.0 cache gate is structurally safe; the exposure is BF16 reduction-order rounding at the 1e-6 fixture / 0.02 gfx1151 consumer tolerance (docs/artifacts/state-cache-validation-20260912/summary.json:20-23,32-38) — same class as the accepted camera_update BF16 order difference (max 0.0625 at range −16.9..14.5, accepted as rounding, docs/pure-helper-compile-20260910.md:107-127). Requires: numerical helper check on real 1008/1536 BF16 shapes (pattern of scripts/check_pure_compile_helpers.py per docs/pure-helper-compile-20260910.md:107-112) + 40-action finite/KV-invariant run before any latency claim.
- Sketch: new helpers in scripts/pure_compile_helpers.py next to _affine (scripts/pure_compile_helpers.py:23-24), installed via PureTensorIslands.compile (scripts/pure_compile_helpers.py:106-131), called from make_pure_block_forward (scripts/pure_compile_helpers.py:226-306); keep all Linear/attention/cache lines exactly as-is. Explicitly NOT full-block compile (rejected, docs/compile-20260910.md:115-127).

--- Audited and NOT proposed ---
- Cross-attention non-SDPA (~1.9 ms), Conv3d/patch/head (~2.8 ms): below noise; conv3d_micro.py is VAE-only and irrelevant here.
- Host KV-cursor: already measured (−8.3 ms, below 15-20 ms gate, rejected as default, docs/kv-cursor-20260910.md:3-13).
- Sink reduction at equal K: measured SLOWER (+18.8 ms) with quality loss, REJECTED (docs/sink-budget-20260910.md:47-69,126-134) — do not revisit; Candidate 1 cuts recent frames while keeping all 6 sink frames for this reason.
- Full-block/regional torch.compile, attention-Library swap, custom kernels, capture/AOT/fusion/arena: excluded per task (C8 class) and per the full-block rejection evidence (docs/compile-20260910.md:115-127; docs/FINDINGS.md:643-649).

Bottom line: the denoise hotspot is healthy dense rectangular flash attention (K-linear, ~30 ms per 1024 K per 3-forward phase) plus tuned GEMMs at their floor; the only measured-scale lever left is attending to less history (Candidate 1, ~60/80 ms at 12→10). Candidates 2+3 together are worth ~17-34 ms and are the correct second step, both staying inside the already-accepted pure-helper envelope with cache code untouched.
