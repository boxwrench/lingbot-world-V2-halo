# LingRuntime research and experiment DAG

Date created: 2026-09-12
Machine-readable source of truth: [`lingruntime-dag.json`](lingruntime-dag.json)

This is the first authoritative LingRuntime research/experiment DAG for
Strix Halo. It shows completed work, currently running work, blocked work,
dependencies, independent/parallel work, GPU-exclusive work, conditional
branches, and the recommended execution order. It will be updated as
experiment results arrive; see "Update protocol" below.

This document explains the DAG. It does not duplicate the evidence behind
each node — every node in `lingruntime-dag.json` links to the actual
artifact (a file already in this repository, or a report already in
`dsh-research-team`) rather than restating its contents here.

## Location of the machine-readable file

The task that requested this DAG suggested `campaign/lingruntime-dag.json`
as the machine-readable location. That path was **not** used. `campaign/`
(singular) sits one edit-distance away from `dsh-research-team/campaigns/`
(plural), which is that repository's own directory for DSH campaign runs —
introducing a similarly-named top-level directory here would blur exactly
the ownership boundary [`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md)
exists to keep clear (DSH owns `campaigns/`; this repository owns its own
project state under its own naming). No other top-level `campaign/`-like
directory exists in this repository today, so nothing on disk technically
conflicted, but the name itself would have created ambiguity for a future
reader. Instead, the DAG's JSON lives next to its own human-readable
explanation, under the `docs/lingruntime/` directory already established
for LingRuntime project-specific documentation. This avoids introducing a
new top-level directory for one file.

## Node schema

Each node in `lingruntime-dag.json` has this shape (see the file for the
authoritative field list):

```json
{
  "id": "C1",
  "name": "State/cache correctness",
  "type": "experiment",
  "status": "running",
  "priority": 10,
  "resource": "gpu-exclusive",
  "dependsOn": ["C0A", "C0B"],
  "blocks": ["C2"],
  "canRunParallelWith": [],
  "question": "...",
  "entryCriteria": [],
  "outputs": [],
  "exitCriteria": [],
  "nextIf": [],
  "evidence": [],
  "result": null,
  "notes": ""
}
```

Allowed `status` values: `blocked`, `ready`, `running`, `done`, `failed`,
`inconclusive`, `deferred`.

Allowed `resource` values (see `resourcesAllowed` in the JSON for the exact
wording used to classify each node):

| Resource | Meaning |
| --- | --- |
| `source-cpu` | reading/writing source, docs, or DAG state; no GPU |
| `api-research` | a DSH/LLM-backed research campaign; no local GPU |
| `gpu-light` | brief, non-sustained GPU use (e.g. a device/env check) that does not require exclusive ownership |
| `gpu-exclusive` | sustained and/or timing-sensitive GPU work; only one such node may run at a time on the Radeon 8060S |
| `implementation` | code/config authoring, parallelizable in isolated worktrees; its GPU evaluation step is called out separately as gpu-exclusive |

## Current DAG

```text
H0 DSH research harness                         DONE
        |
        v
P0 LingRuntime preflight                         DONE
        |
        v
C0A environment/gfx1151 gate                     DONE
        |
        v
C0B RC1 Phase 0 waterfall                        DONE  (commit add8a69)
        |
        v
C1 state/cache correctness                       RUNNING  (owned by Codex C)
        |
        v
C2 quality causality                             BLOCKED  (waits on C1 PASS)
   |-- bounded chunk/context screen (first, mandatory branch)
   +-- conditional, evidence-triggered only:
       context-sensitive / chunk-boundary-sensitive / decoder-specific /
       precision / recurrence-stress / deeper-limitation branches
        |
        v
C3 freeze quality-approved semantic contract     BLOCKED
        |
        v
C4 profile the quality-approved runtime          BLOCKED  (distinct from C0B)
        |
     +--+--+
     |     |
    C5A   C5B
 hotspot#1 hotspot#2 research (unlabeled until C4 ranks them)  BLOCKED, parallelizable
     |     |
    C6A   C6B
 candidate implementations (isolated worktrees; GPU eval serialized)  BLOCKED
     \     /
      \   /
       C7  integration + reprofile                BLOCKED
        |
        +----> new bottleneck -> spawns new C5x node(s); the DAG iterates

C8 advanced runtime escalation                    DEFERRED / CONDITIONAL
```

### Completed nodes (verified against repository evidence, not asserted from memory)

- **H0** — DSH research harness validated. Evidence:
  `dsh-research-team/reports/live-validation.md` (S7/S8/S9 checks: 4 passed,
  0 failed) and `dsh-research-team/docs/smoke-tests.md`.
- **P0** — LingRuntime preflight research complete. Evidence:
  `dsh-research-team/reports/lingruntime-preflight.md` and
  `lingruntime-preflight-live.md`.
- **C0A** — environment/gfx1151 gate. Evidence:
  [`docs/artifacts/phase0-20260912/environment.json`](../artifacts/phase0-20260912/environment.json)
  (`"gate": "PASS"`, single unambiguous Radeon 8060S / gfx1151 device).
- **C0B** — warmed RC1 Phase 0 waterfall, commit `add8a69`. Evidence:
  [`docs/artifacts/phase0-20260912/README.md`](../artifacts/phase0-20260912/README.md),
  `waterfall.json`, `metrics-summary.json`. Verified headline numbers:
  base-RGB median 939.9 ms (927.6-954.5 ms range), next-ready median
  1258.9 ms (1244.8-1272.3 ms range), reproducing the 2026-09-11 historical
  RC1 P50 (938.4 ms / 1255.3 ms) within measured A/A noise. Denoise
  (~910 ms, ~97% of base-RGB) dominates base-RGB latency; the exact
  clean-KV forward (~298-300 ms, ~93% of the added next-ready delta)
  dominates most of the additional next-ready delay. No optimization was
  implemented.

### Running

- **C1** — state/cache semantic correctness. Owned by another agent (Codex
  C) as of 2026-09-12. Validates sink/recent cache semantics,
  ordering/positions, eviction, wrap, reset, recurrence, and clean-KV
  transaction behavior. This document does **not** invent or assume its
  result; `lingruntime-dag.json`'s `C1.result` stays `null` until that
  agent reports a verdict.

### Blocked

C2 through C8, in the chain shown above. C2 specifically is blocked on C1
returning `PASS` — not merely on C1 finishing, since a `failed` or
`inconclusive` C1 result should keep C2 blocked and require this DAG to be
updated with that outcome before anything downstream proceeds.

## C2 — quality causality: bounded, not a sweep

C2's first and only currently-mandatory branch is a bounded chunk/context
screen. The six follow-up branches listed in `lingruntime-dag.json`
(context-sensitive, chunk-boundary-sensitive, decoder-specific, precision,
recurrence/action-stress, deeper model/checkpoint limitation) are recorded
as `nextIf` conditions, not as separate nodes with their own status. They
are only instantiated as real nodes once the chunk/context screen's own
evidence points at one of them. Running all six regardless of evidence
would be blind sweeping, which this DAG deliberately does not schedule.

## C3 — the semantic contract C4 profiles

C3's eventual output is a fixed operating point: chunk size, local
attention size, sink size, denoise schedule, resolution, decoder contract,
responsiveness requirement, and quality horizon. C4 profiles exactly that
point — which may or may not equal the RC1 configuration C0B already
measured. Do not reuse C0B's hotspot ranking for optimization decisions
once C3 exists; re-derive it from C4.

## C5/C6/C7 — the iterative optimization loop

C5A and C5B are deliberately unlabeled ("measured hotspot #1/#2 research")
rather than pre-named as attention, KV, graph capture, decoder, fusion, or
PM4 work — C4's measured ranking determines what they actually are, and it
may justify a different count than two. They are normally independent and
safely parallelizable as source/API research. C6A/C6B may be *implemented*
concurrently in separate isolated worktrees, but their **GPU evaluation
steps are gpu-exclusive and must be serialized** against each other and
against every other gpu-exclusive node (see the Parallelism section). C7
integrates whichever candidates were accepted and reprofiles; a new
bottleneck there spawns new C5x nodes rather than closing the loop — this
DAG models an iterative process, not a one-shot pipeline.

## C8 — advanced escalation stays conditional

Graph capture, AOT specialization, large-scale fusion, a fixed memory
arena, retained PM4, custom ROCr/HIP work, or removing major PyTorch layers
are recorded as `deferred`, not as an inevitable future phase. Entry
requires a specific measured residual opportunity (from C4 or a later
reprofile in C7) whose scale justifies that complexity — there is no
standing authorization to start any of these.

## Parked Phase 0 findings

Two hypotheses came out of C0B's measured opportunity ranking. They are
recorded so they are not lost, and so they are not re-discovered from
scratch later — but they are **not active optimization campaigns**:

- **PF1** — rolled rectangular self-SDPA in the three denoise passes:
  plausibly 37-92 ms of base-RGB latency (medium-low confidence).
- **PF2** — the non-SDPA portion of the exact clean-KV recomputation:
  plausibly 44-89 ms of next-ready latency (low confidence).

Both are gated: they only become candidate input to a C5x node if C3/C4
confirm the hotspot they describe is still present and relevant under
whatever configuration C3 freezes. PF2 additionally requires C1 to have
returned `PASS`, since it targets the exact clean-KV transaction that C1 is
currently validating for correctness.

## Parallelism

Read-only inspection of either repository, or of this checkout, may happen
from multiple agents at once. Write-capable work must not collide in the
same working tree. As of this DAG:

```text
Codex B:
DAG / documentation (this branch)
source-cpu
NO GPU

Codex C:
C1 state/cache correctness
GPU owner (gpu-exclusive)
```

These two are independent and are correctly running concurrently right
now. General rules this DAG follows going forward:

- DSH/API research (`api-research`) can run while a GPU experiment
  (`gpu-exclusive`) executes elsewhere — they don't touch the same
  resource.
- Two source-analysis jobs (`source-cpu`) can run concurrently.
- C5A and C5B research can run concurrently (both `source-cpu`).
- C6A and C6B *implementation* can run concurrently in different isolated
  worktrees (both `implementation`).
- Two `gpu-exclusive` nodes cannot run concurrently — this includes C6A's
  and C6B's GPU evaluation steps, even though their coding steps can
  overlap.
- Two agents must not write into the same working tree concurrently; a
  write-capable agent uses its own isolated git worktree when another
  agent may be active in the checkout.

## Recommended queue

```text
RUNNING
C1 — state/cache correctness (owned by Codex C)

BLOCKED
C2 — quality causality, waiting on C1 PASS
C3, C4, C5A, C5B, C6A, C6B, C7 — waiting on their upstream dependency
C8 — deferred pending a measured residual opportunity from the C4-C7 loop

INDEPENDENT WORK AVAILABLE NOW
repository/DAG maintenance (this branch)
source-only research that does not presume C1's result

NOT YET VALID
performance optimization
C5/C6 work
advanced runtime work (C8)
```

No node was added merely to fill this queue; the two "independent work"
entries are the only work this DAG currently identifies as both valid and
unblocked besides C1 itself.

## Update protocol

After each accepted result, in this order:

1. attach evidence path/commit;
2. record a concise result;
3. update node status;
4. evaluate branch conditions (`nextIf`);
5. unblock newly valid nodes;
6. mark new independent work;
7. rerank the recommended queue;
8. preserve failed and inconclusive results.

`lingruntime-dag.json` has a `changelog` array for exactly this purpose.
**Never rewrite a prior node's result or status to make history look
cleaner** — append a new changelog entry and update the node's current
`status`/`result` fields; do not delete or silently overwrite what a past
entry recorded. A `failed` or `inconclusive` result is left as such in the
changelog even after a later attempt succeeds.

## Ambiguities carried into this DAG

- C0B's evidence (`docs/artifacts/phase0-20260912/`) is committed at
  `add8a69`. A second, independently-derived cross-check of the same raw
  data was reported to exist in the primary LingBot working tree on
  2026-09-12, but it is not committed on any branch as of this DAG and is
  therefore not cited as evidence here. If it is committed later, add it to
  `C0B.evidence` in `lingruntime-dag.json` rather than treating this
  markdown file as needing a rewrite.
- `add8a69` is on local `main` but was not on `origin/main` as of this
  DAG's creation (only reachable via the separately-pushed
  `docs/research-orchestration` branch). If `main` is later rebased,
  amended, or diverges before being pushed, `C0B.commit` in the JSON should
  be re-verified rather than assumed still accurate.
- C1's actual exit criteria/verdict format has not been negotiated with
  Codex C; this DAG guesses a PASS/FAIL/inconclusive shape consistent with
  the rest of the schema. If Codex C's campaign reports something
  differently structured, reconcile the node's `exitCriteria`/`result`
  shape rather than forcing Codex C's report into this DAG's guess.
