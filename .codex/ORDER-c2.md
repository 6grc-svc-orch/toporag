# Conference Implementation Order

## Goal

Align this repository with the conference paper **A RAG-Assisted DRL Framework for Microservices Deployment in 6G Vehicular Networks**, using `.codex/c2.pdf` as the implementation specification and `.codex/j2.pdf` only to identify later journal extensions that must be removed.

Build a compact, runnable research implementation that reproduces only the conference evaluation: RAG-DRL, DQN Scheduler, DRP, and measured Figures 3--6.

## Required Order of Work

1. Read both papers completely and audit the journal implementation against the conference system model, RAG planner, PPO training path, baselines, and result figures.
2. Preserve useful deterministic simulation, graph, allocation, checkpoint, and logging logic while removing journal-only Metaverse, multimodal, TopoRAG, MAPPO, trust, ablation, resilience, and deployment-overhead paths.
3. Implement the vehicular edge-cloud simulator with service DAGs, exact replicas, placement, fractional routing, CPU/memory and bandwidth constraints, Poisson/Pareto arrivals, mobility, latency, resource use, and SLA measurement.
4. Implement the conference graph memory, three-layer GAT, Qwen2-7B-Instruct planner, offline behavior cloning, perturbed-environment meta-training, and online PPO/GAE scheduler.
5. Implement DQN Scheduler and DRP on the same seeded scenarios and objective.
6. Generate Figures 3--6 only from measured experiment records. Never hard-code or digitize the publication curves.
7. Reduce documentation and dependencies, remove obsolete files and every test artifact, then validate the direct Python entrypoints.

## Project Constraints

- Keep sufficient conference-compatible code unchanged and make additions only when they are justified by the paper or required to complete its measured pipeline. Prefer removing journal-only code over introducing replacement abstractions.
- Keep code function-oriented. Use classes only for neural-network modules required by PyTorch.
- Keep one required configuration at `configs/paper.json`; access required values directly and fail clearly when they are absent. Avoid optional defaults, broad exception handling, and silent fallbacks unless correctness requires them.
- Keep every runtime prompt in `prompts/` as Markdown with explicit placeholders.
- Use direct Python files without argument parsers, console frameworks, package entrypoints, or `python -m` workflows.
- Keep all measured figure inputs and rendered outputs under `figures/`.
- Use narrow validation around malformed model output and expected external-data failures; do not add broad exception handling or silent fallback behavior.
- Keep the README concise and affirmative. Describe the repository and paper without equations, long examples, negative comparisons, or statements about what the repository does not implement.
- Remove every test file and test script after validation; retain only the direct runnable research files.

## Reproduction Decisions

- Produce figures from measured runs only.
- Use deterministic paper-scale synthetic data by default, with strict optional adapters for Shanghai Telecom base stations, Shanghai taxi trajectories, and Alibaba microservice traces.
- Use one self-contained SQLite deployment-memory store.
- Use the simulation setup's 16 edge controllers managing 3,233 ESPs and treat 4,328 as the taxi-trace count.
- Use a parameter-shared PPO scheduler rather than the journal's MAPPO design.
- Use Qwen's ranked placement, utility, and reward output as a soft placement prior; let PPO produce the fractional path-routing action.
- Compute the paper objective once per complete DAG scheduling slot, and propagate it across the sequential replica decisions with slot-aware GAE discounts.
- Refresh cloud plans on the paper's 30-minute cadence or anomalies, push/reuse them at the edge every 500 ms, and evict them when their next placement is infeasible.
- Keep Monte Carlo trials independent by resetting transient planner state while retaining the loaded policy parameters.
- Plot Figure 3 with the bounded measured score `1 / (1 + mean request cost)` and measure Figure 6 as mean occupied CPU/memory fraction across available ESPs.
- Limit evaluation artifacts to conference Figures 3--6 and the methods RAG-DRL, DQN Scheduler, and DRP.
