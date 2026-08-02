# Orchestration planner

Create {candidate_count} plans for window {window_id}, time {sim_time_s}, controller {controller_id}, region
{region_id}, slice {slice_tag}, policy {policy_id}. Respect residual resources and slice-feasible links.

Image left: circle/server, square/controller, diamond/core, green-to-red load, X failure, C/R/L/Q pressure.
Image right: exact source-row/destination-column DAG; C/M/G/N/R heatmaps follow text task order.

Infrastructure: {topology_json}

DAGs: {service_dags_json}

Telemetry: {telemetry_json}

QoS intent: {qos_intent_json}

Placements: {current_placement_json}

Retrieved traces: {retrieved_traces_json}

Return only JSON:

- `reward_weights`: `latency`, `resource_utilization`, `load_balance`, `reliability`; [0,1], sum 1.
- `penalty_weights`: independent `resource`, `temporal`, `robustness`; each [0,1].
- `cooperation_bias`: finite scalar logit.
- `candidates`: {candidate_count} objects with `candidate_id`, `node_preferences`, `placement_strategy`,
  `predicted_utility`.

`node_preferences` has 1-8 distinct active node IDs then `"<eop>"`. `placement_strategy` is `latency_first`,
`capacity_first`, `reliability_first`, `spread`, or `consolidate`. The runtime expands this guidance into all replica
assignments. `predicted_utility` is in [-1,1]; candidate strategy/preference combinations must be unique.
