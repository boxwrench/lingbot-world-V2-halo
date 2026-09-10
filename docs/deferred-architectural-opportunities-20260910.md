# Deferred architectural opportunities to revisit

> Status note (2026-09-10): This document preserves an external-agent architectural brief for the next decision point. It is **not authorization to start new implementation branches**. By the time this snapshot was committed, the Strix TAEHV/`taew2_1` evaluation and isolated three-denoise-step A/B had already completed successfully elsewhere in the chronology. Preserve the intent and ordering below when revisiting the remaining ideas; reconcile against the latest accepted baseline and `docs/FINDINGS.md` before executing anything.

## Purpose

Preserve these findings for the next decision point. **Do not interrupt current experiments or open new implementation branches yet.**

## Finish the current work first

- TAEHV / `taew2_1` offline and presentation-only decoder evaluation.
- Isolated three-denoising-step A/B.
- R9700 actual SDPA dispatch identification.
- Any already-running matched tests.

Keep the accepted paths and protected scripts intact. Do not combine changes until their individual effects are understood.

---

## 1. Treat the application as three components

```text
CONTROLS
input → camera/action conditioning

WORLD SIMULATOR
persistent DiT state → accepted latent → exact clean-KV commit

RENDERER
accepted latent → displayed RGB
```

The canonical decoder need not necessarily be the interactive renderer.

If decoded RGB and canonical VAE features do not feed back into generation, a compatible approximate decoder can change presentation without changing the accepted latent world or DiT KV.

**Verify that dependency rather than assuming it.**

### Renderer decision

Prefer evaluating:

```text
accepted latent → fast decoder only
```

before:

```text
accepted latent → fast preview → canonical decode afterward
```

The former can remove work and improve cadence. The latter still pays canonical decode cost and may introduce a visible replacement transition.

Keep conditioning encode unchanged. Switching back to a canonical decoder after skipping its updates requires reconstruction of valid decoder state or a fresh session.

---

## 2. Next simple untested lever: actual historical attention length

Persistent memory is required. **The current quantity of full-spatial-resolution historical KV may not be.**

Before proposing a shorter window, document:

```text
allocated KV capacity
actual attended K length
sink slots
recent-history slots
current-frame slots
whether sink is included in the reported local capacity
```

At 384×672:

```text
1008 tokens/frame
18144-token reported local capacity
18144 / 1008 = 18 frames
```

This does not by itself establish whether “18+6” means 18 or 24 total attended frames.

### Potential later experiment — not authorized yet

One shorter-window A/B, preserving sink semantics and all other settings.

For example, reducing **actual attended frame equivalents** from 18 to 12 would reduce dense attention QK/AV arithmetic by roughly one-third at fixed Q. It would not reduce MLP, most projections, or VAE work.

If measured context-dependent attention time is `A`:

```text
idealized saving ≈ A × (1 − K_new / K_old)
```

Actual kernels may scale differently.

Validate camera response, geometry, revisits, and drift over many window turnovers. Finite output is insufficient.

**Do not begin custom pooled, quantized, sparse, or multiscale KV before evaluating a simpler supported shorter window.**

---

## 3. Borrow components, not entire pipelines

Existing projects are most useful as sources of specific compatible pieces:

| Need | Lead | What to investigate |
|---|---|---|
| Fast display decoding | TAEHV and existing Wan integrations | Weights, latent normalization, streaming/frame scheduling |
| Transformer execution | Regional/block compilation and projection fusion | Exact isolated block improvements |
| AMD GEMM selection | PyTorch TunableOp | Better kernels for recurring actual shapes |
| Few-step causal sampling | Relevant Self-Forcing/CausVid implementations | Update equations and cache-commit accounting |
| Persistent attention | Streaming/local-attention implementations | Cache layout and bounded-history handling |

These are leads, **not verified drop-in replacements**.

Avoid whole-framework migration unless a specific demonstrated benefit outweighs the risk to persistent-state correctness.

---

## 4. Small implementation audit to retain for later

Answer from the code/trace; do not assume problems exist:

- Is classifier-free guidance duplicating computation unexpectedly?
- Are invariant conditioning projections recomputed?
- Does clean encoding compute a discarded final-layer tail/output head?
- Are there redundant latent casts, copies, or cache repacks?
- Is inference consistently free of autograd retention?
- Does presentation wait for a four-frame packet, encoding flush, or playback queue?

Exact clean KV still generally requires the preceding transformer blocks. Omitting an unused final tail is a bounded optimization, not a shortcut through the entire clean pass.

---

## 5. Measure useful responsiveness, not just output arrival

Keep these separate:

```text
keypress → first genuinely new RGB presented
keypress → first materially action-responsive RGB
keypress → exact KV committed
keypress → next action permitted
sustained action cadence
```

A first decoded frame may be new but strongly dominated by historical state. A matched-state comparison with different camera actions can reveal when the new command materially affects the output.

Also measure immediate-next-input behavior: a keypress after presentation may still wait for deferred clean commit.

For controls, avoid an unbounded queue of stale movement commands. Any queue-policy improvement is a usability change—not an inference speedup.

---

## Decision order after current results arrive

1. Select renderer based on measured latency and visual acceptance.
2. Evaluate the isolated three-step result.
3. Combine only independently accepted changes and revalidate.
4. Consider one shorter actual-attention-window test.
5. Pursue remaining kernel work only where profiling supports a meaningful gain.

**Working hypothesis:** The largest remaining gains may come from a cheaper compatible renderer, an appropriate spatial/context budget, and fewer safe sampling evaluations—not another generic “optimized Wan” pipeline.

**No new work is requested by this brief until the current experiments are reviewed.**
