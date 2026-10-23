# stacker-bot (Big Two AI, v2) — CLAUDE.md

## Key documents
- **Architecture plan (the plan):** `~/personal-os/projects/big-two-ai/architecture-plan.md` — scope, home rules, rules of evidence, process, milestones, starting hypotheses
- **Research manifest:** [`docs/research-manifest.md`](docs/research-manifest.md) — frozen rules, paths, slug map
- **Decision log / lab notebook:** `docs/decision-log.md`, `docs/lab-notebook.md`
- **Stage plans:** `docs/stage-plans/<slug>.md`, one per stage, written when its milestone is reached
- **Project index:** `~/os/projects/big-two-ai/README.md`
- **Process standard:** `~/os/knowledge/frameworks/research-standards.md`; code style `research-code-standards.md`

## Who writes what
Nate writes the core code: engine, legal-move generation, the net, the DMC loop, eval statistics,
and the first test of each module. Claude explains, reviews, writes the remaining tests, scripts
and plots, and enforces the plan. Reviews run as fresh-context subagents. No `dt-*` workers.
One piece (one function) at a time, planned before written, checkpointed after.

## Structure
```
stacker-bot/
├── src/stacker/          # the package; modules land per milestone:
│                         #   engine → agents, harness (baselines) → net, dmc (specialist) → search (belief_search)
├── tests/                # pytest; the rules suite is the engine check
├── scripts/              # rerunnable producers of every reported number
├── data/sealed-deals/    # per-config sealed deal sets + hashes (baselines), tracked
├── results/              # tables and plots that get reviewed, tracked
├── runs/                 # checkpoints and training logs, gitignored
├── scratch/              # one-off exploration, gitignored
└── docs/                 # manifest, decision log, notebook, stage plans; gitignored
```
`dashboard/` (GitHub Pages, ONNX) is created at the `dashboard` milestone, not before.

## Rules that block
- **Rules of evidence** (plan §3) are frozen: sealed deals, test seeds 0–2 only in confirm runs,
  the ladder gate, duplicate-deal paired eval with 95% bootstrap CI and ≥3 seeds, pre-register
  before confirm, negative results kept. Reopening one needs a decision-log entry.
- **Budget rule:** every run estimate = training + eval, each from a 2-minute probe on the target
  box, re-estimated before launch.
- **Stage size:** one sitting of Nate's code, ending in a check that fails if the code is wrong.
- **Naming:** nothing tracked in git carries a bare milestone number (`m0_`, `stage1`). Slugs only,
  from the manifest's slug map.
- **ML verification gate:** any change to model, loss, or data-pipeline code passes the
  `/ml-engineer` gates before a real training run or a recorded result. Config-only changes exempt.
- **Blocking questions:** a decision the plan and decision log do not cover → stop and ask.
- v1 numbers (plan §7–8) are priors, never evidence.

## Stack and compute
Python, PyTorch, pytest; ONNX export at the `dashboard` milestone. Laptop for development; teapot
(HMC CS, `/teapot/cs152/individual/nseluga/`, world-readable, nothing sensitive) for training.

## Session startup
1. `/research-partner big-two-ai` for any build session; it reads the manifest, then the plan.
2. Check the project index for the current milestone and next step.
3. Progress ticks live in the tracker artifact named in the manifest; Nate ticks, agents read.
