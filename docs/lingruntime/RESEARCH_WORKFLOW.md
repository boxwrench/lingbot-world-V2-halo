# LingRuntime research workflow

Date: 2026-09-12

This document is this repository's side of the contract with
`dsh-research-team` (the reusable DeepSeek Harness research composition at
`/home/keith/Desktop/dsh-research-team`). See that repository's
`docs/PROJECT_INTEGRATION.md` for the other side of the same contract.

## What this repository owns

`lingbot-world-V2-halo` is the authoritative scientific/engineering history
for LingRuntime on Strix Halo. It owns:

- the LingRuntime research and experiment DAG:
  [`RESEARCH_AND_EXPERIMENT_DAG.md`](RESEARCH_AND_EXPERIMENT_DAG.md) (human-
  readable) and [`lingruntime-dag.json`](lingruntime-dag.json)
  (machine-readable, the current source of truth for node status);
- experiment status, the current recommended work queue, and dependency
  relationships between experiments;
- source findings that materially define LingBot behavior;
- Phase 0 evidence (`docs/artifacts/phase0-20260912/README.md` and the rest
  of that directory), quality experiments, cache/state validation, and
  benchmark results (`docs/rc1-benchmark-20260911.md`,
  `docs/artifacts/rc1-benchmark-20260911/`, and the other dated `docs/*.md`
  reports);
- accepted and rejected hypotheses (see `docs/FINDINGS.md` and
  `docs/interactive-latency-history-20260910.md` for the negative-result
  ledger);
- implementation and optimization branches, and the project-specific
  scripts/fixtures under `scripts/` and `tests/` that produced the evidence
  above;
- links/references to relevant DSH campaign artifacts, where DSH performed
  research relevant to a decision made here.

A future developer should be able to clone this repository alone and
understand what has been measured, what has been established, what failed,
what remains uncertain, what should happen next, and why — without needing
`dsh-research-team`.

## What DSH is, and is not, here

`dsh-research-team` is reusable research **substrate**: it owns how a
research campaign is run (DeepSeek lead, Muse worker roles, campaign
persistence, evidence export), and it stays usable for projects other than
this one. It is not this project's state database.

When a DSH campaign researches a LingRuntime question, its raw/persisted
campaign artifact and any `reports/*.md` synthesis stay in
`dsh-research-team` — that is where the research act happened. What belongs
here is the **accepted conclusion** plus a reference to that artifact, for
example:

```text
LingRuntime-gfx1151 pre-implementation preflight
  conclusion: Phase 0 (env gate + one warmed RC1 waterfall) is the
    highest-value next action before any DiT/KV/attention intervention.
  source: dsh-research-team reports/lingruntime-preflight-live.md
    (campaign 20260912T160926Z-analyze-boxwrench-lingbot-world-v2-halo-and-the-)
```

Do not copy a full DSH campaign transcript into this repository. Summarize
the decision and link back to the source campaign/report; the raw DSH
session state remains authoritative for the research process itself.

DSH campaigns are read-only researchers (see `dsh-research-team/AGENTS.md`):
they inspect this checkout but do not modify it, run GPU experiments, or
implement anything here. All LingBot-side experiments, benchmarks, and
implementation work happen in this repository, by whatever agent is
assigned to it.

## Multi-agent / worktree rules

Multiple agents (Codex instances, Hermes, humans) may work against this
repository concurrently:

- **Read-only work** (inspecting docs, source, or results) may happen from
  multiple agents at once.
- **Write-capable work**: two agents must not make unrelated writes in the
  same working tree concurrently. Before writing, run
  `git status --short --branch`, `git log --oneline -10`, and
  `git worktree list` to check whether another agent already has
  uncommitted changes, a running process, or an active worktree here. If
  so, create a separate git worktree on a clearly named branch (for example
  `docs/research-orchestration`) rather than writing into the primary
  working directory, and do not disturb the other agent's branch,
  uncommitted changes, running processes, or experiment artifacts.
- **GPU ownership**: only one timing-sensitive process may own the Radeon
  8060S/gfx1151 GPU at a time. Confirm no other LingBot process
  (`run_browser.py`, `run_live.py`, `run_interactive.py`, etc.) is using the
  GPU before starting a timing-sensitive run; a documentation-only or
  repository-organization task should not touch the GPU at all.
- This repository's git metadata lives at `.git-experiment/.git` with the
  repository root as the work tree (not the usual co-located `.git/`
  layout); commands run outside that root, or in a script, need
  `--git-dir`/`--work-tree` (or `GIT_DIR`/`GIT_WORK_TREE`) set explicitly.

## Status

The LingRuntime research/experiment DAG now exists:
[`RESEARCH_AND_EXPERIMENT_DAG.md`](RESEARCH_AND_EXPERIMENT_DAG.md) explains
it, and [`lingruntime-dag.json`](lingruntime-dag.json) is its current,
authoritative machine-readable state (node status, dependencies, evidence,
recommended queue). This document defines the ownership boundary and
workflow rules that the DAG operates under; it does not itself track
node-by-node status — read the DAG files for that.
