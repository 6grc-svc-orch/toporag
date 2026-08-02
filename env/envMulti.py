"""Function-oriented deterministic cloud-edge orchestration environment."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from env.constants import (
    NODE_TIERS,
    RADIO_PROFILES,
    RESOURCE_NAMES,
    SCENARIO_OVERRIDE_KEYS,
    SLICE_NAMES,
    WORKLOAD_CLASSES,
    validate_config,
)
from env.helper import (
    bandwidth_utilization,
    candidate_paths,
    controller_utilization,
    current_violation_components,
    load_balance_index,
    placement_mask,
    region_node_ids,
    request_latency_metrics,
    reward_components,
    service_for_request,
    utilization_matrix,
    validate_semantic_action,
)
from env.multiProducer import generate_request_trace, update_request_locations
from template import (
    generate_service_catalog,
    generate_topology,
    json_ready,
    load_wsdot_events,
    validate_external_sources,
)

NODE_FEATURE_DIM = 22
SERVICE_FEATURE_DIM = 12
PATH_FEATURE_DIM = 6


def observation_spec(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the fixed tensor contract used by all edge policies."""

    max_local_nodes = int(config["simulation"]["max_local_nodes"])
    max_microservices = int(config["training"]["max_microservices"])
    max_paths = int(config["training"]["max_paths"])
    return {
        "infrastructure_nodes": (max_local_nodes, NODE_FEATURE_DIM),
        "infrastructure_adjacency": (max_local_nodes, max_local_nodes),
        "infrastructure_mask": (max_local_nodes,),
        "service_nodes": (max_microservices, SERVICE_FEATURE_DIM),
        "service_adjacency": (max_microservices, max_microservices),
        "service_mask": (max_microservices,),
        "path_features": (max_microservices, max_local_nodes, max_paths, PATH_FEATURE_DIM),
        "path_mask": (max_microservices, max_local_nodes, max_paths),
        "incoming_edge_mask": (max_microservices,),
        "incoming_edge_ids": (max_microservices,),
        "context": (int(config["training"]["context_dim"]),),
        "placement_mask": (max_local_nodes,),
    }


def _validate_overrides(overrides: dict[str, Any]) -> None:
    unknown = set(overrides) - SCENARIO_OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"unknown scenario overrides: {sorted(unknown)}")
    for key in ("arrival_rate", "duration_s"):
        if key in overrides and (isinstance(overrides[key], bool) or float(overrides[key]) <= 0):
            raise ValueError(f"scenario override {key} must be positive")
    for key in ("microservices", "replicas"):
        if key in overrides and (
            isinstance(overrides[key], bool)
            or not isinstance(overrides[key], (int, np.integer))
            or int(overrides[key]) < 1
        ):
            raise ValueError(f"scenario override {key} must be a positive integer")
    if "failure_rate" in overrides and not 0 <= float(overrides["failure_rate"]) <= 100:
        raise ValueError("scenario override failure_rate must be in [0,100]")
    if "burst" in overrides and not isinstance(overrides["burst"], bool):
        raise TypeError("scenario override burst must be boolean")


def prepare_scenario(
    config: dict[str, Any],
    seed: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducible topology, DAG catalog, and request trace."""

    validate_config(config)
    selected_seed = int(config["simulation"]["seed"] if seed is None else seed)
    selected_overrides = {} if overrides is None else dict(overrides)
    _validate_overrides(selected_overrides)
    rng = np.random.default_rng(selected_seed)
    external_events = None
    if config["simulation"]["source"] == "external":
        external_paths = validate_external_sources(config)
        external_events = load_wsdot_events(config, external_paths)
    topology = generate_topology(config, rng, selected_overrides)
    services = generate_service_catalog(config, rng, selected_overrides)
    requests = generate_request_trace(config, topology, services, rng, selected_overrides, external_events)
    digest_input = {
        "seed": selected_seed,
        "source": config["simulation"]["source"],
        "overrides": selected_overrides,
        "nodes": topology["coordinates_km"].round(8).tolist(),
        "requests": [(item["service_id"], item["arrival_time_s"]) for item in requests],
    }
    scenario_hash = hashlib.sha256(json.dumps(digest_input, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "paper": "AI-native cloud-edge orchestration for 6G metaverse networks",
        "source": config["simulation"]["source"],
        "seed": selected_seed,
        "overrides": selected_overrides,
        "scenario_hash": scenario_hash,
        "config": config,
        "topology": topology,
        "services": services,
        "requests": requests,
        "trace_schedule": {
            "kind": "spatial_poisson_pareto",
            "duration_s": float(
                selected_overrides["duration_s"]
                if "duration_s" in selected_overrides
                else config["simulation"]["duration_s"]
            ),
            "request_count": len(requests),
            "mean_rate_rps": len(requests)
            / float(
                selected_overrides["duration_s"]
                if "duration_s" in selected_overrides
                else config["simulation"]["duration_s"]
            ),
        },
    }


def _scheduled_scenario(
    scenario: dict[str, Any],
    arrival_times: np.ndarray,
    schedule: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    source_requests = scenario["requests"]
    if not source_requests:
        raise ValueError("cannot schedule an empty request trace")
    requests = [
        copy.deepcopy(source_requests[index % len(source_requests)])
        for index in range(len(arrival_times))
    ]
    if not np.isfinite(arrival_times).all() or np.any(np.diff(arrival_times) < 0):
        raise ValueError("scheduled arrivals must be finite and nondecreasing")
    if arrival_times.size and (arrival_times[0] < 0 or arrival_times[-1] > duration_s + 1e-9):
        raise ValueError("scheduled arrivals fall outside the trace duration")
    for request_id, (request, arrival_time) in enumerate(
        zip(requests, arrival_times, strict=True)
    ):
        request["request_id"] = request_id
        request["arrival_time_s"] = float(arrival_time)

    config = copy.deepcopy(scenario["config"])
    config["simulation"]["duration_s"] = float(duration_s)
    overrides = dict(scenario["overrides"])
    overrides["duration_s"] = float(duration_s)
    digest = hashlib.sha256()
    digest.update(str(scenario["scenario_hash"]).encode("ascii"))
    digest.update(json.dumps(schedule, sort_keys=True).encode("utf-8"))
    for request in requests:
        digest.update(
            f"{request['request_id']}:{request['service_id']}:{request['arrival_time_s']:.12g};".encode(
                "ascii"
            )
        )
    return {
        **scenario,
        "config": config,
        "overrides": overrides,
        "requests": requests,
        "trace_schedule": schedule,
        "scenario_hash": digest.hexdigest(),
    }


def schedule_burst_trace(
    scenario: dict[str, Any],
    start_s: float,
    end_s: float,
    intensity_multiplier: float,
) -> dict[str, Any]:
    """Clone a scenario and concentrate arrivals in an explicit burst window."""

    duration_s = float(
        scenario["overrides"]["duration_s"]
        if "duration_s" in scenario["overrides"]
        else scenario["config"]["simulation"]["duration_s"]
    )
    start = float(start_s)
    end = float(end_s)
    multiplier = float(intensity_multiplier)
    if not 0 <= start < end <= duration_s:
        raise ValueError("burst window must satisfy 0 <= start_s < end_s <= duration_s")
    if multiplier <= 1:
        raise ValueError("intensity_multiplier must be greater than one")
    baseline_rate = len(scenario["requests"]) / duration_s
    weighted_duration_s = duration_s + (multiplier - 1.0) * (end - start)
    request_count = max(1, round(baseline_rate * weighted_duration_s))
    quantiles = (np.arange(request_count, dtype=np.float64) + 0.5) / request_count
    burst_mass_start = start
    burst_mass_end = burst_mass_start + multiplier * (end - start)
    total_mass = burst_mass_end + duration_s - end
    targets = quantiles * total_mass
    arrivals = np.empty(request_count, dtype=np.float64)
    before = targets < burst_mass_start
    during = (targets >= burst_mass_start) & (targets < burst_mass_end)
    arrivals[before] = targets[before]
    arrivals[during] = start + (targets[during] - burst_mass_start) / multiplier
    arrivals[~(before | during)] = end + targets[~(before | during)] - burst_mass_end
    schedule = {
        "kind": "burst",
        "start_s": start,
        "end_s": end,
        "intensity_multiplier": multiplier,
        "baseline_rate_rps": baseline_rate,
        "request_count": request_count,
    }
    return _scheduled_scenario(scenario, arrivals, schedule, duration_s)


def schedule_diurnal_trace(
    scenario: dict[str, Any],
    hourly_rates: list[float] | np.ndarray,
    accelerated_duration_s: float,
) -> dict[str, Any]:
    """Clone a scenario and map its requests onto an accelerated 24-hour profile."""

    rates = np.asarray(hourly_rates, dtype=np.float64)
    duration_s = float(accelerated_duration_s)
    if rates.shape != (24,) or not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("hourly_rates must contain exactly 24 positive finite values")
    if duration_s <= 0:
        raise ValueError("accelerated_duration_s must be positive")
    bin_duration_s = duration_s / 24.0
    hourly_mass = rates * bin_duration_s
    cumulative_mass = np.concatenate(([0.0], np.cumsum(hourly_mass)))
    request_count = max(1, round(float(cumulative_mass[-1])))
    quantiles = (np.arange(request_count, dtype=np.float64) + 0.5) / request_count
    targets = quantiles * cumulative_mass[-1]
    hour_ids = np.searchsorted(cumulative_mass[1:], targets, side="right")
    hour_ids = np.minimum(hour_ids, 23)
    arrivals = hour_ids * bin_duration_s + (
        targets - cumulative_mass[hour_ids]
    ) / rates[hour_ids]
    schedule = {
        "kind": "diurnal_24h",
        "hourly_rates": rates.tolist(),
        "accelerated_duration_s": duration_s,
        "request_count": request_count,
    }
    return _scheduled_scenario(scenario, arrivals, schedule, duration_s)


def _request_tasks(
    state: dict[str, Any],
    request: dict[str, Any],
    service: dict[str, Any],
) -> list[dict[str, int]]:
    tasks = []
    origin = int(request["controller_id"])
    collaborator = int(request["collaborating_controller_id"])
    overrides = state["scenario"]["overrides"]
    distribute_stress_workflow = "microservices" in overrides or "replicas" in overrides
    controller_count = int(state["config"]["simulation"]["edge_controllers"])
    for microservice_id in service["microservice_ids"]:
        replica_count = int(request["replica_counts"][microservice_id])
        participants = [
            (origin + offset) % controller_count
            for offset in range(min(replica_count, controller_count))
        ]
        for replica_id in range(replica_count):
            if distribute_stress_workflow:
                controller_id = participants[replica_id % len(participants)]
            else:
                controller_id = origin
                if collaborator != origin and (int(microservice_id) + replica_id) % 2:
                    controller_id = collaborator
            tasks.append(
                {
                    "request_id": int(request["request_id"]),
                    "microservice_id": int(microservice_id),
                    "replica_id": replica_id,
                    "controller_id": controller_id,
                }
            )
    return tasks


def _activate_arrivals(state: dict[str, Any]) -> list[int]:
    activated = []
    requests = state["scenario"]["requests"]
    while state["next_request_index"] < len(requests):
        request = requests[state["next_request_index"]]
        if request["arrival_time_s"] > state["time_s"] + 1e-12:
            break
        service = service_for_request(state, request)
        allocation = {
            "request": request,
            "tasks": _request_tasks(state, request, service),
            "task_index": 0,
            "placements": {},
            "routes": [],
            "node_claims": [],
            "link_claims": [],
            "status": "pending",
            "expires_at_s": None,
            "disruption_started_s": None,
            "failure_count": 0,
            "last_recovery_time_s": 0.0,
        }
        request_id = int(request["request_id"])
        controller_id = int(allocation["tasks"][0]["controller_id"])
        state["allocations"][request_id] = allocation
        state["controller_queues"][controller_id].append(request_id)
        state["next_request_index"] += 1
        activated.append(request_id)
    return activated


def _advance_clock(state: dict[str, Any], target_time_s: float) -> None:
    target = float(target_time_s)
    if target < state["time_s"] - 1e-12:
        raise ValueError("simulation time cannot move backwards")
    state["time_s"] = max(float(state["time_s"]), target)
    mobility_interval = float(state["config"]["simulation"]["mobility_update_s"])
    if state["time_s"] - state["last_mobility_update_s"] >= mobility_interval:
        mobility_elapsed = state["time_s"] - state["last_mobility_update_s"]
        update_request_locations(
            state["scenario"]["requests"],
            mobility_elapsed,
            state["config"]["simulation"]["area_km"],
        )
        state["last_mobility_update_s"] = state["time_s"]


def _advance_until_decision(state: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Advance through expiry events until the next request decision is ready."""

    activated: list[int] = []
    expired: list[int] = []
    requests = state["scenario"]["requests"]
    while not any(state["controller_queues"].values()):
        next_arrival_s = (
            float(requests[state["next_request_index"]]["arrival_time_s"])
            if state["next_request_index"] < len(requests)
            else np.inf
        )
        active_expiries = [
            float(state["allocations"][request_id]["expires_at_s"])
            for request_id in state["active_request_ids"]
        ]
        next_expiry_s = min(active_expiries) if active_expiries else np.inf
        next_event_s = min(next_arrival_s, next_expiry_s)
        if not np.isfinite(next_event_s):
            break
        _advance_clock(state, next_event_s)
        expired.extend(_release_expired(state))
        activated.extend(_activate_arrivals(state))
    return activated, expired


def reset_environment(scenario: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    """Create mutable simulation state from an immutable generated scenario."""

    selected_seed = int(scenario["seed"] if seed is None else seed)
    topology = {
        **scenario["topology"],
        "node_active": scenario["topology"]["node_active"].copy(),
    }
    runtime_scenario = {
        **scenario,
        "requests": copy.deepcopy(scenario["requests"]),
        "topology": topology,
    }
    controller_count = int(scenario["config"]["simulation"]["edge_controllers"])
    semantic_dim = int(scenario["config"]["training"]["semantic_context_dim"])
    state = {
        "scenario": runtime_scenario,
        "config": scenario["config"],
        "topology": topology,
        "seed": selected_seed,
        "rng": np.random.default_rng(selected_seed),
        "time_s": 0.0,
        "step_index": 0,
        "next_request_index": 0,
        "node_available": topology["capacity"].copy(),
        "link_available_mbps": topology["link_capacity_mbps"].copy(),
        "node_used_by_slice": np.zeros(
            (len(topology["node_ids"]), len(SLICE_NAMES), len(RESOURCE_NAMES)), dtype=np.float64
        ),
        "controller_queues": {controller_id: [] for controller_id in range(controller_count)},
        "allocations": {},
        "active_request_ids": [],
        "completed_requests": [],
        "controller_completed_metrics": {
            controller_id: [] for controller_id in range(controller_count)
        },
        "rejected_requests": [],
        "released_request_ids": [],
        "failure_events": [],
        "recovery_events": [],
        "semantic_context": np.zeros((controller_count, semantic_dim), dtype=np.float64),
        "recent_metrics": {
            "mean_latency_ms": 0.0,
            "sla_violation_rate_percent": 0.0,
            "reliability": 1.0,
        },
        "last_mobility_update_s": 0.0,
        "_path_cache": {},
        "done": False,
    }
    _activate_arrivals(state)
    _advance_until_decision(state)
    return state


def _pending_allocation(state: dict[str, Any], controller_id: int) -> dict[str, Any] | None:
    queue = state["controller_queues"][controller_id]
    if not queue:
        return None
    allocation = state["allocations"][queue[0]]
    if allocation["status"] != "pending":
        raise RuntimeError("controller queue references a non-pending request")
    return allocation


def _pending_task(state: dict[str, Any], controller_id: int) -> tuple[dict[str, Any], dict[str, int]] | None:
    allocation = _pending_allocation(state, controller_id)
    if allocation is None:
        return None
    return allocation, allocation["tasks"][allocation["task_index"]]


def _incoming_edges(service: dict[str, Any], microservice_id: int) -> list[tuple[int, int, int]]:
    return [
        (edge_id, int(source), int(destination))
        for edge_id, (source, destination) in enumerate(service["edges"])
        if int(destination) == microservice_id
    ]


def _source_replica_node(
    allocation: dict[str, Any],
    request: dict[str, Any],
    source_microservice_id: int,
    replica_id: int,
) -> int:
    source_replicas = int(request["replica_counts"][source_microservice_id])
    source_replica_id = replica_id % source_replicas
    return int(allocation["placements"][(source_microservice_id, source_replica_id)])


def _feasible_for_task(
    state: dict[str, Any],
    controller_id: int,
    allocation: dict[str, Any],
    task: dict[str, int],
    path_count: int,
) -> dict[str, Any]:
    request = allocation["request"]
    service = service_for_request(state, request)
    demand = service["resource_demands"][task["microservice_id"]]
    local_nodes, mask = placement_mask(state, controller_id, demand)
    incoming = _incoming_edges(service, task["microservice_id"])
    max_microservices = int(state["config"]["training"]["max_microservices"])
    max_local_nodes = int(state["config"]["simulation"]["max_local_nodes"])
    path_features = np.zeros(
        (max_microservices, max_local_nodes, path_count, PATH_FEATURE_DIM), dtype=np.float32
    )
    path_mask = np.zeros((max_microservices, max_local_nodes, path_count), dtype=np.float32)
    incoming_edge_mask = np.zeros(max_microservices, dtype=np.float32)
    incoming_edge_ids = np.full(max_microservices, -1, dtype=np.int64)
    candidates: dict[int, dict[int, list[dict[str, Any]]]] = {
        int(node_id): {} for node_id in local_nodes
    }
    for incoming_index, (edge_id, source_microservice_id, _) in enumerate(incoming):
        incoming_edge_mask[incoming_index] = 1.0
        incoming_edge_ids[incoming_index] = edge_id
        source_node = _source_replica_node(
            allocation, request, source_microservice_id, task["replica_id"]
        )
        for local_index, node_id in enumerate(local_nodes):
            if not mask[local_index]:
                candidates[int(node_id)][edge_id] = []
                continue
            paths = candidate_paths(
                state, source_node, int(node_id), int(request["slice_id"]), path_count
            )
            candidates[int(node_id)][edge_id] = paths
            if not paths:
                mask[local_index] = 0.0
                continue
            for rank, path in enumerate(paths):
                bottleneck = (
                    state["topology"]["link_total_mbps"][path["link_indices"]].min()
                    if path["link_indices"].size
                    else 1e9
                )
                serialization_ms = (
                    float(service["edge_payload_mb"][edge_id])
                    * float(request["payload_multiplier"])
                    * 8.0
                    / max(float(bottleneck), 1e-9)
                    * 1000.0
                )
                path_features[incoming_index, local_index, rank] = [
                    (path["delay_ms"] + serialization_ms) / 100.0,
                    path["available_fraction"],
                    path["reliability"],
                    (len(path["nodes"]) - 1) / max(len(state["topology"]["node_ids"]), 1),
                    float(path["cross_region"]),
                    int(request["slice_id"]) / max(len(SLICE_NAMES) - 1, 1),
                ]
                path_mask[incoming_index, local_index, rank] = 1.0
    forced_rejection = not bool(np.any(mask))
    if forced_rejection:
        # Preserve a valid categorical distribution while the selected gateway
        # action deterministically produces an infeasibility penalty.
        mask[-1] = 1.0
        active_edges = np.flatnonzero(incoming_edge_mask)
        path_mask[active_edges, len(local_nodes) - 1, 0] = 1.0
    return {
        "local_node_ids": local_nodes,
        "placement_mask": mask,
        "candidate_paths": candidates,
        "path_features": path_features,
        "path_mask": path_mask,
        "incoming_edge_mask": incoming_edge_mask,
        "incoming_edge_ids": incoming_edge_ids,
        "forced_rejection": forced_rejection,
    }


def feasible_action_data(
    state: dict[str, Any],
    request: int | dict[str, Any] | None = None,
    candidate_path_count: int | None = None,
    controller_id: int | None = None,
) -> dict[str, Any]:
    """Describe feasible local placements and ranked routes for pending decisions."""

    path_count = int(state["config"]["simulation"]["candidate_paths"] if candidate_path_count is None else candidate_path_count)
    if request is None and controller_id is None:
        return {
            candidate_controller: feasible_action_data(
                state,
                controller_id=candidate_controller,
                candidate_path_count=path_count,
            )
            for candidate_controller in range(state["config"]["simulation"]["edge_controllers"])
            if _pending_task(state, candidate_controller) is not None
        }
    if request is not None:
        request_id = int(request["request_id"] if isinstance(request, dict) else request)
        allocation = state["allocations"][request_id]
        controller_id = int(allocation["request"]["controller_id"])
        task = allocation["tasks"][allocation["task_index"]]
    else:
        pending = _pending_task(state, int(controller_id))
        if pending is None:
            raise ValueError(f"controller {controller_id} has no pending decision")
        allocation, task = pending
    return _feasible_for_task(state, int(controller_id), allocation, task, path_count)


def _empty_observation(config: dict[str, Any], controller_id: int) -> dict[str, Any]:
    observation = {key: np.zeros(shape, dtype=np.float32) for key, shape in observation_spec(config).items()}
    observation["incoming_edge_ids"] = np.full(
        observation_spec(config)["incoming_edge_ids"], -1, dtype=np.int64
    )
    observation.update(
        {
            "controller_id": int(controller_id),
            "request_id": -1,
            "microservice_id": -1,
            "replica_id": -1,
        }
    )
    return observation


def controller_observation(
    state: dict[str, Any],
    controller_id: int,
    _feasible: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fixed-shape regional partial observation for one edge controller."""

    controller_count = int(state["config"]["simulation"]["edge_controllers"])
    if not 0 <= controller_id < controller_count:
        raise IndexError("controller_id is outside the configured controller set")
    pending = _pending_task(state, controller_id)
    if pending is None:
        return _empty_observation(state["config"], controller_id)
    allocation, task = pending
    request = allocation["request"]
    service = service_for_request(state, request)
    feasible = _feasible if _feasible is not None else _feasible_for_task(
        state,
        controller_id,
        allocation,
        task,
        int(state["config"]["simulation"]["candidate_paths"]),
    )
    spec = observation_spec(state["config"])
    observation = _empty_observation(state["config"], controller_id)
    local_nodes = feasible["local_node_ids"]
    local_count = len(local_nodes)
    topology = state["topology"]
    capacity = topology["capacity"][local_nodes]
    residual = state["node_available"][local_nodes]
    residual_ratio = np.divide(residual, capacity, out=np.zeros_like(residual), where=capacity > 0)
    capacity_scale = np.maximum(topology["capacity"].max(axis=0), 1e-9)
    local_utilization = utilization_matrix(state)[local_nodes]
    incident_availability = np.zeros((local_count, len(SLICE_NAMES)), dtype=np.float64)
    for local_index, node_id in enumerate(local_nodes):
        incident = (topology["link_sources"] == node_id) | (
            topology["link_destinations"] == node_id
        )
        incident_capacity = topology["link_capacity_mbps"][incident].sum(axis=0)
        incident_available = state["link_available_mbps"][incident].sum(axis=0)
        incident_availability[local_index] = np.divide(
            incident_available,
            incident_capacity,
            out=np.zeros(len(SLICE_NAMES), dtype=np.float64),
            where=incident_capacity > 0,
        )
    incident_utilization = 1.0 - incident_availability
    node_features = np.column_stack(
        [
            residual_ratio,
            capacity / capacity_scale,
            local_utilization.mean(axis=1),
            topology["latency_ms"][local_nodes] / np.array([5.0, 5.0, 5.0]),
            topology["radio_profiles"][local_nodes],
            topology["node_tiers"][local_nodes] / 2.0,
            incident_availability,
            incident_utilization,
        ]
    )
    if node_features.shape[1] != NODE_FEATURE_DIM:
        raise RuntimeError("infrastructure node feature contract changed")
    observation["infrastructure_nodes"][:local_count] = node_features
    observation["infrastructure_adjacency"][:local_count, :local_count] = topology["adjacency"][np.ix_(local_nodes, local_nodes)]
    observation["infrastructure_mask"][:local_count] = 1.0
    observation["placement_mask"][:local_count] = feasible["placement_mask"]

    microservice_count = len(service["microservice_ids"])
    max_resource = np.maximum(topology["capacity"].max(axis=0), 1e-9)
    qos_vector = np.array(
        [
            service["qos"]["latency_ms"] / 30.0,
            service["qos"]["jitter_ms"] / 5.0,
            service["qos"]["sync_ms"] / 30.0,
            service["qos"]["min_reliability"],
        ],
        dtype=np.float64,
    )
    max_replicas = max(item["replicas"] for item in state["config"]["evaluation"]["complexity"])
    current_task = np.zeros(microservice_count, dtype=np.float64)
    current_task[int(task["microservice_id"])] = 1.0
    local_node_set = {int(node_id) for node_id in local_nodes}
    local_placements = np.zeros(microservice_count, dtype=np.float64)
    for (microservice_id, _), node_id in allocation["placements"].items():
        if int(node_id) in local_node_set:
            local_placements[int(microservice_id)] += 1.0
    local_placement_fraction = np.divide(
        local_placements,
        request["replica_counts"],
        out=np.zeros_like(local_placements),
        where=request["replica_counts"] > 0,
    )
    service_nodes = np.column_stack(
        [
            service["resource_demands"] / max_resource,
            service["workloads_mi"] / state["config"]["workloads"]["cpu_workload_mi"][1],
            request["replica_counts"] / max_replicas,
            np.tile(qos_vector, (microservice_count, 1)),
            current_task,
            local_placement_fraction,
        ]
    )
    if service_nodes.shape[1] != SERVICE_FEATURE_DIM:
        raise RuntimeError("service node feature contract changed")
    observation["service_nodes"][:microservice_count] = service_nodes
    for source, destination in service["edges"]:
        observation["service_adjacency"][source, destination] = 1.0
    observation["service_mask"][:microservice_count] = 1.0
    observation["path_features"] = feasible["path_features"]
    observation["path_mask"] = feasible["path_mask"]
    observation["incoming_edge_mask"] = feasible["incoming_edge_mask"]
    observation["incoming_edge_ids"] = feasible["incoming_edge_ids"]

    utilization = utilization_matrix(state)
    controller_queues = state["controller_queues"][controller_id]
    active_local = sum(
        1
        for request_id in state["active_request_ids"]
        if any(
            int(node_id) in local_node_set
            for node_id in state["allocations"][request_id]["placements"].values()
        )
    )
    incident_links = np.array(
        [
            int(source) in local_node_set or int(destination) in local_node_set
            for source, destination in zip(
                topology["link_sources"], topology["link_destinations"]
            )
        ],
        dtype=bool,
    )
    regional_bandwidth = bandwidth_utilization(state)[incident_links]
    local_recent = state["controller_completed_metrics"][controller_id][-100:]
    if local_recent:
        local_latency = float(np.mean([item["latency_ms"] for item in local_recent]))
        local_violation_rate = float(np.mean([item["sla_violation"] for item in local_recent]))
        local_reliability = float(np.mean([item["reliability"] for item in local_recent]))
    else:
        local_latency = 0.0
        local_violation_rate = 0.0
        local_reliability = 1.0
    class_index = WORKLOAD_CLASSES.index(request["class"])
    base_context = np.array(
        [
            state["time_s"] / state["config"]["simulation"]["duration_s"],
            len(controller_queues) / len(state["scenario"]["requests"]),
            active_local / len(state["scenario"]["requests"]),
            utilization[local_nodes].mean(),
            utilization[local_nodes].max(),
            load_balance_index(state, local_nodes),
            regional_bandwidth.mean() if regional_bandwidth.size else 0.0,
            local_latency / 30.0,
            local_violation_rate,
            local_reliability,
            class_index / max(len(WORKLOAD_CLASSES) - 1, 1),
            float(request["controller_id"] != request["collaborating_controller_id"]),
        ],
        dtype=np.float32,
    )
    semantic = state["semantic_context"][controller_id].astype(np.float32)
    context = np.concatenate([base_context, semantic])
    if context.shape != spec["context"]:
        raise RuntimeError("context_dim must equal 12 + training.semantic_context_dim")
    observation["context"] = context
    observation.update(
        {
            "controller_id": int(controller_id),
            "request_id": int(task["request_id"]),
            "microservice_id": int(task["microservice_id"]),
            "replica_id": int(task["replica_id"]),
        }
    )
    return observation


def pending_controller_decisions(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return one dependency-safe decision for each ready regional controller."""

    decisions = {}
    controller_count = int(state["config"]["simulation"]["edge_controllers"])
    for controller_id in range(controller_count):
        pending = _pending_task(state, controller_id)
        if pending is None:
            continue
        _, task = pending
        feasible = feasible_action_data(state, controller_id=controller_id)
        decisions[controller_id] = {
            **task,
            "controller_id": controller_id,
            "local_node_ids": feasible["local_node_ids"].copy(),
            "candidate_paths": copy.deepcopy(feasible["candidate_paths"]),
            "incoming_edge_mask": feasible["incoming_edge_mask"].copy(),
            "incoming_edge_ids": feasible["incoming_edge_ids"].copy(),
            "forced_rejection": bool(feasible["forced_rejection"]),
            "observation": controller_observation(state, controller_id, _feasible=feasible),
        }
    return decisions


def global_state_vector(state: dict[str, Any]) -> np.ndarray:
    """Compress infrastructure, service, and mapping state for the critic."""

    resource = utilization_matrix(state)
    link = bandwidth_utilization(state)
    controller_values = controller_utilization(state)
    queue_values = np.array(
        [len(queue) for queue in state["controller_queues"].values()], dtype=np.float64
    ) / float(len(state["scenario"]["requests"]))
    controller_summary = np.array(
        [
            controller_values.mean(),
            controller_values.std(),
            controller_values.max(),
            controller_values.min(),
            queue_values.mean(),
            queue_values.std(),
            queue_values.max(),
            queue_values.sum(),
        ],
        dtype=np.float64,
    )

    live_allocations = [
        allocation
        for allocation in state["allocations"].values()
        if allocation["status"] in {"pending", "active"}
    ]
    service_summary = np.zeros(10, dtype=np.float64)
    if live_allocations:
        services = [service_for_request(state, allocation["request"]) for allocation in live_allocations]
        max_microservices = float(state["config"]["training"]["max_microservices"])
        max_replicas = float(
            max(item["replicas"] for item in state["config"]["evaluation"]["complexity"])
        )
        class_fractions = np.array(
            [
                np.mean([allocation["request"]["class"] == name for allocation in live_allocations])
                for name in WORKLOAD_CLASSES
            ],
            dtype=np.float64,
        )
        service_summary = np.concatenate(
            [
                np.array(
                    [
                        np.mean([len(service["microservice_ids"]) for service in services])
                        / max_microservices,
                        np.mean([len(service["edges"]) for service in services])
                        / max(max_microservices * (max_microservices - 1.0) / 2.0, 1.0),
                        np.mean(
                            [
                                np.mean(allocation["request"]["replica_counts"])
                                for allocation in live_allocations
                            ]
                        )
                        / max_replicas,
                    ],
                    dtype=np.float64,
                ),
                class_fractions,
                np.array(
                    [
                        np.mean([service["qos"]["latency_ms"] for service in services]) / 30.0,
                        np.mean([service["qos"]["jitter_ms"] for service in services]) / 5.0,
                        np.mean([service["qos"]["sync_ms"] for service in services]) / 30.0,
                        np.mean([service["qos"]["min_reliability"] for service in services]),
                    ],
                    dtype=np.float64,
                ),
            ]
        )

    placements = [
        int(node_id)
        for allocation in live_allocations
        for node_id in allocation["placements"].values()
    ]
    routes = [
        route for allocation in live_allocations for route in allocation["routes"]
    ]
    requested_placements = sum(
        len(allocation["tasks"]) for allocation in live_allocations
    )
    node_count = len(state["topology"]["node_ids"])
    if placements:
        placement_counts = np.bincount(placements, minlength=node_count).astype(np.float64)
        placement_probabilities = placement_counts[placement_counts > 0] / len(placements)
        placement_entropy = -float(
            np.dot(placement_probabilities, np.log(placement_probabilities))
        ) / max(np.log(node_count), 1.0)
        utilized_node_fraction = float(np.count_nonzero(placement_counts) / node_count)
    else:
        placement_entropy = 0.0
        utilized_node_fraction = 0.0
    if routes:
        cross_region = []
        route_entropies = []
        for route in routes:
            path_cross_region = False
            fractions = []
            for path in route["paths"]:
                controller_ids = state["topology"]["node_controller_ids"][path["nodes"]]
                path_cross_region |= np.unique(controller_ids[controller_ids >= 0]).size > 1
                fractions.append(float(path["fraction"]))
            cross_region.append(path_cross_region)
            positive = np.asarray([value for value in fractions if value > 0.0])
            route_entropies.append(
                -float(np.dot(positive, np.log(positive)))
                / max(np.log(state["config"]["simulation"]["candidate_paths"]), 1.0)
            )
        cross_region_fraction = float(np.mean(cross_region))
        path_entropy = float(np.mean(route_entropies))
        route_reliability = float(np.mean([route["weighted_reliability"] for route in routes]))
        route_delay = float(np.mean([route["weighted_delay_ms"] for route in routes]) / 100.0)
    else:
        cross_region_fraction = 0.0
        path_entropy = 0.0
        route_reliability = 1.0
        route_delay = 0.0
    capacity = state["topology"]["capacity"][:, None, :]
    normalized_slice_use = np.divide(
        state["node_used_by_slice"],
        capacity,
        out=np.zeros_like(state["node_used_by_slice"]),
        where=capacity > 0,
    ).mean(axis=(0, 2))
    mapping_summary = np.concatenate(
        [
            np.array(
                [
                    len(placements) / max(requested_placements, 1),
                    utilized_node_fraction,
                    placement_entropy,
                    cross_region_fraction,
                    path_entropy,
                    route_reliability,
                    route_delay,
                ],
                dtype=np.float64,
            ),
            normalized_slice_use,
        ]
    )
    values = np.concatenate(
        [
            np.array(
                [
                    state["time_s"] / state["config"]["simulation"]["duration_s"],
                    state["next_request_index"] / len(state["scenario"]["requests"]),
                    len(state["active_request_ids"]) / len(state["scenario"]["requests"]),
                    len(state["completed_requests"]) / len(state["scenario"]["requests"]),
                    len(state["rejected_requests"]) / len(state["scenario"]["requests"]),
                    state["recent_metrics"]["mean_latency_ms"] / 30.0,
                    state["recent_metrics"]["sla_violation_rate_percent"] / 100.0,
                    state["recent_metrics"]["reliability"],
                ]
            ),
            resource.mean(axis=0),
            resource.max(axis=0),
            resource.std(axis=0),
            link.mean(axis=0),
            link.max(axis=0),
            link.std(axis=0),
            controller_summary,
            service_summary,
            mapping_summary,
        ]
    )
    required = int(state["config"]["training"]["global_state_dim"])
    if values.size > required:
        raise ValueError(f"training.global_state_dim={required} is smaller than required {values.size}")
    return np.pad(values, (0, required - values.size)).astype(np.float32)


def set_semantic_context(state: dict[str, Any], controller_id: int, context: np.ndarray) -> None:
    expected = int(state["config"]["training"]["semantic_context_dim"])
    value = np.asarray(context, dtype=np.float64)
    if value.shape != (expected,):
        raise ValueError(f"semantic context must have shape ({expected},)")
    state["semantic_context"][controller_id] = value


def _release_claims(state: dict[str, Any], allocation: dict[str, Any]) -> None:
    request = allocation["request"]
    slice_id = int(request["slice_id"])
    for node_id, demand in allocation["node_claims"]:
        state["node_available"][node_id] += demand
        state["node_used_by_slice"][node_id, slice_id] -= demand
    for link_index, claimed_slice_id, bandwidth in allocation["link_claims"]:
        state["link_available_mbps"][link_index, claimed_slice_id] += bandwidth
    capacity = state["topology"]["capacity"]
    link_capacity = state["topology"]["link_capacity_mbps"]
    if np.any(state["node_available"] < -1e-8) or np.any(state["node_available"] > capacity + 1e-8):
        raise RuntimeError("node resource release violated capacity bounds")
    if np.any(state["link_available_mbps"] < -1e-8) or np.any(state["link_available_mbps"] > link_capacity + 1e-8):
        raise RuntimeError("link release violated bandwidth bounds")
    state["node_available"] = np.minimum(state["node_available"], capacity)
    state["link_available_mbps"] = np.minimum(state["link_available_mbps"], link_capacity)
    state["node_used_by_slice"] = np.maximum(state["node_used_by_slice"], 0.0)
    allocation["node_claims"] = []
    allocation["link_claims"] = []


def _reject_allocation(
    state: dict[str, Any],
    allocation: dict[str, Any],
    reason: str,
    controller_id: int,
) -> None:
    _release_claims(state, allocation)
    request_id = int(allocation["request"]["request_id"])
    allocation["status"] = "rejected"
    allocation["reason"] = reason
    queue = state["controller_queues"][controller_id]
    if not queue or queue[0] != request_id:
        raise RuntimeError("request rejection order is inconsistent")
    queue.pop(0)
    state["rejected_requests"].append({"request_id": request_id, "reason": reason, "time_s": state["time_s"]})


def _complete_allocation(
    state: dict[str, Any],
    allocation: dict[str, Any],
    controller_id: int,
) -> dict[str, Any]:
    request = allocation["request"]
    request_id = int(request["request_id"])
    metrics = request_latency_metrics(state, allocation)
    disruption_started_s = allocation["disruption_started_s"]
    recovery_time_s = (
        0.0
        if disruption_started_s is None
        else float(state["time_s"] - disruption_started_s)
    )
    metrics["recovery_time_s"] = recovery_time_s
    allocation["status"] = "active"
    allocation["expires_at_s"] = state["time_s"] + float(request["lifetime_s"])
    allocation["disruption_started_s"] = None
    allocation["last_recovery_time_s"] = recovery_time_s
    queue = state["controller_queues"][controller_id]
    if not queue or queue[0] != request_id:
        raise RuntimeError("request completion order is inconsistent")
    queue.pop(0)
    state["active_request_ids"].append(request_id)
    record = {"request_id": request_id, "controller_id": controller_id, "time_s": state["time_s"], **metrics}
    state["completed_requests"].append(record)
    if disruption_started_s is not None:
        state["recovery_events"].append(
            {
                "request_id": request_id,
                "failure_count": int(allocation["failure_count"]),
                "disruption_started_s": float(disruption_started_s),
                "recovered_at_s": float(state["time_s"]),
                "recovery_time_s": recovery_time_s,
            }
        )
    participant_controllers = {
        int(task["controller_id"]) for task in allocation["tasks"]
    }
    for participant in participant_controllers:
        state["controller_completed_metrics"][participant].append(record.copy())
    recent = state["completed_requests"][-100:]
    state["recent_metrics"] = {
        "mean_latency_ms": float(np.mean([item["latency_ms"] for item in recent])),
        "sla_violation_rate_percent": float(100.0 * np.mean([item["sla_violation"] for item in recent])),
        "reliability": float(np.mean([item["reliability"] for item in recent])),
    }
    return metrics


def _route_claims(
    state: dict[str, Any],
    allocation: dict[str, Any],
    task: dict[str, int],
    destination_node: int,
    routing: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float], float]:
    request = allocation["request"]
    service = service_for_request(state, request)
    incoming = _incoming_edges(service, task["microservice_id"])
    route_records = []
    claims: dict[tuple[int, int], float] = {}
    connectivity_violation = 0.0
    for incoming_index, (edge_id, source_microservice_id, destination_microservice_id) in enumerate(incoming):
        edge_routing = routing[incoming_index]
        source_replica_id = int(task["replica_id"]) % int(
            request["replica_counts"][source_microservice_id]
        )
        source_node = _source_replica_node(allocation, request, source_microservice_id, task["replica_id"])
        paths = candidate_paths(
            state,
            source_node,
            destination_node,
            int(request["slice_id"]),
            int(state["config"]["simulation"]["candidate_paths"]),
        )
        if not paths or np.any(edge_routing[len(paths) :] > 1e-8):
            connectivity_violation += 1.0
            continue
        edge_bandwidth_mbps = (
            float(service["edge_payload_mb"][edge_id])
            * float(request["payload_multiplier"])
            * 8.0
            / max(float(request["lifetime_s"]), state["config"]["simulation"]["decision_interval_s"])
        )
        weighted_delay = 0.0
        weighted_reliability = 0.0
        path_records = []
        for rank, fraction in enumerate(edge_routing[: len(paths)]):
            if fraction <= 0:
                continue
            path = paths[rank]
            claimed_bandwidth = edge_bandwidth_mbps * float(fraction)
            effective_delay_ms = float(
                path["delay_ms"]
                + float(service["edge_payload_mb"][edge_id])
                * float(request["payload_multiplier"])
                * 8.0
                / max(
                    state["topology"]["link_total_mbps"][path["link_indices"]].min() if path["link_indices"].size else 1e9,
                    1e-9,
                ) * 1000.0
            )
            weighted_delay += float(fraction) * effective_delay_ms
            weighted_reliability += float(fraction) * path["reliability"]
            for link_index in path["link_indices"]:
                key = (int(link_index), int(request["slice_id"]))
                claims[key] = claims.get(key, 0.0) + claimed_bandwidth
            path_records.append(
                {
                    "path_id": path["path_id"],
                    "fraction": float(fraction),
                    "delay_ms": effective_delay_ms,
                    "reliability": float(path["reliability"]),
                    "nodes": path["nodes"].copy(),
                    "link_indices": path["link_indices"].copy(),
                }
            )
        route_records.append(
            {
                "edge_id": edge_id,
                "source_microservice_id": source_microservice_id,
                "source_replica_id": source_replica_id,
                "destination_microservice_id": destination_microservice_id,
                "destination_replica_id": int(task["replica_id"]),
                "weighted_delay_ms": weighted_delay,
                "weighted_reliability": weighted_reliability,
                "paths": path_records,
            }
        )
    return route_records, claims, connectivity_violation


def _provisional_metrics(
    state: dict[str, Any],
    allocation: dict[str, Any],
    task: dict[str, int],
    node_id: int,
    routes: list[dict[str, Any]],
) -> dict[str, Any]:
    request = allocation["request"]
    service = service_for_request(state, request)
    microservice_id = int(task["microservice_id"])
    processing = service["workloads_mi"][microservice_id] / (
        service["resource_demands"][microservice_id, 0]
        * state["config"]["simulation"]["mips_per_vcpu"]
    ) * 1000.0 + state["topology"]["latency_ms"][node_id, 0]
    route_delays = [record["weighted_delay_ms"] for record in routes]
    route_variances = []
    for route in routes:
        fractions = np.asarray(
            [path["fraction"] for path in route["paths"]], dtype=np.float64
        )
        delays = np.asarray(
            [path["delay_ms"] for path in route["paths"]], dtype=np.float64
        )
        route_variances.append(
            float(np.dot(fractions, np.square(delays - route["weighted_delay_ms"])))
        )
    reliability = float(
        np.prod([record["weighted_reliability"] for record in routes])
    ) if routes else 1.0
    access = 0.0
    if not routes:
        distance = float(
            np.linalg.norm(
                request["source_location_km"]
                - state["topology"]["coordinates_km"][node_id]
            )
        )
        access = float(state["topology"]["latency_ms"][node_id, 1] + distance / 0.2)
    return {
        "request": request,
        "latency_ms": float(access + processing + sum(route_delays)),
        "jitter_ms": float(np.sqrt(sum(route_variances))),
        "synchronization_ms": 0.0,
        "reliability": float(reliability),
        "sla_violation": False,
        "recovery_time_s": 0.0,
    }


def _apply_action(
    state: dict[str, Any],
    controller_id: int,
    action: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    allocation, task = _pending_task(state, controller_id)  # type: ignore[misc]
    request = allocation["request"]
    service = service_for_request(state, request)
    for key in ("controller_id", "request_id", "microservice_id", "replica_id"):
        if int(action[key]) != int(controller_id if key == "controller_id" else task[key]):
            raise ValueError(f"action.{key} does not match the pending decision")
    node_id = int(action["node_id"])
    routing = action["routing_fractions"]
    expected_paths = int(state["config"]["simulation"]["candidate_paths"])
    max_microservices = int(state["config"]["training"]["max_microservices"])
    if routing.shape != (max_microservices, expected_paths):
        raise ValueError(
            "action.routing_fractions must have shape "
            f"({max_microservices},{expected_paths})"
        )
    incoming_count = len(_incoming_edges(service, task["microservice_id"]))
    row_sums = routing.sum(axis=1)
    if incoming_count and not np.allclose(row_sums[:incoming_count], 1.0):
        raise ValueError("active incoming-edge routing rows must each sum to one")
    if np.any(np.abs(routing[incoming_count:]) > 1e-10):
        raise ValueError("inactive incoming-edge routing rows must be zero")
    demand = service["resource_demands"][task["microservice_id"]]
    local_nodes, mask = placement_mask(state, controller_id, demand)
    local_matches = np.flatnonzero(local_nodes == node_id)
    capacity_violation = 0.0
    connectivity_violation = 0.0
    if local_matches.size != 1 or mask[int(local_matches[0])] == 0:
        if 0 <= node_id < len(state["topology"]["node_ids"]):
            capacity_violation = float(
                np.maximum(demand - state["node_available"][node_id], 0.0).sum()
                / max(state["topology"]["capacity"][node_id].sum(), 1e-9)
            )
        else:
            capacity_violation = 1.0
        metrics = _provisional_metrics(state, allocation, task, int(local_nodes[0]), [])
        metrics["sla_violation"] = True
        involved_nodes = [int(local_nodes[0])]
        _reject_allocation(state, allocation, "capacity", controller_id)
        accepted = False
        completed = False
    else:
        routes, claims, connectivity_violation = _route_claims(state, allocation, task, node_id, routing)
        bandwidth_overuse = 0.0
        for (link_index, slice_id), bandwidth in claims.items():
            available = state["link_available_mbps"][link_index, slice_id]
            bandwidth_overuse += max(0.0, bandwidth - available) / max(
                state["topology"]["link_capacity_mbps"][link_index, slice_id], 1e-9
            )
        capacity_violation += bandwidth_overuse
        if connectivity_violation or bandwidth_overuse:
            metrics = _provisional_metrics(state, allocation, task, node_id, routes)
            metrics["sla_violation"] = True
            involved_nodes = [node_id]
            reason = "connectivity" if connectivity_violation else "bandwidth"
            _reject_allocation(state, allocation, reason, controller_id)
            accepted = False
            completed = False
        else:
            slice_id = int(request["slice_id"])
            state["node_available"][node_id] -= demand
            state["node_used_by_slice"][node_id, slice_id] += demand
            allocation["node_claims"].append((node_id, demand.copy()))
            allocation["placements"][(task["microservice_id"], task["replica_id"])] = node_id
            for (link_index, claimed_slice_id), bandwidth in claims.items():
                state["link_available_mbps"][link_index, claimed_slice_id] -= bandwidth
                allocation["link_claims"].append((link_index, claimed_slice_id, bandwidth))
            allocation["routes"].extend(routes)
            allocation["task_index"] += 1
            involved_nodes = [node for node, _ in allocation["node_claims"]]
            completed = allocation["task_index"] == len(allocation["tasks"])
            if completed:
                completed_metrics = _complete_allocation(state, allocation, controller_id)
                metrics = {"request": request, **completed_metrics}
            else:
                next_controller = int(allocation["tasks"][allocation["task_index"]]["controller_id"])
                if next_controller != controller_id:
                    queue = state["controller_queues"][controller_id]
                    request_id = int(request["request_id"])
                    if not queue or queue[0] != request_id:
                        raise RuntimeError("cross-region request handoff order is inconsistent")
                    queue.pop(0)
                    state["controller_queues"][next_controller].append(request_id)
                metrics = _provisional_metrics(state, allocation, task, node_id, routes)
            accepted = True

    violations = current_violation_components(
        state,
        request,
        metrics,
        capacity_violation,
        connectivity_violation,
        involved_nodes,
    )
    semantic = {
        "reward_weights": action["reward_weights"],
        "penalty_weights": action["penalty_weights"],
        "cooperation": action["cooperation"],
    }
    reward_context = {
        "controller_id": controller_id,
        "metrics": metrics,
        "semantic": semantic,
        "violations": violations,
    }
    event = {
        "controller_id": controller_id,
        "request_id": int(request["request_id"]),
        "microservice_id": int(task["microservice_id"]),
        "replica_id": int(task["replica_id"]),
        "node_id": node_id,
        "accepted": accepted,
        "request_completed": completed,
        "sla_violation": bool(metrics["sla_violation"]),
        "latency_ms": float(metrics["latency_ms"]),
        "jitter_ms": float(metrics["jitter_ms"]),
        "synchronization_ms": float(metrics["synchronization_ms"]),
        "reliability": float(metrics["reliability"]),
        "recovery_time_s": float(metrics["recovery_time_s"]),
    }
    return reward_context, event


def _release_expired(state: dict[str, Any]) -> list[int]:
    expired = []
    for request_id in list(state["active_request_ids"]):
        allocation = state["allocations"][request_id]
        if float(allocation["expires_at_s"]) <= state["time_s"] + 1e-12:
            _release_claims(state, allocation)
            allocation["status"] = "released"
            state["active_request_ids"].remove(request_id)
            state["released_request_ids"].append(request_id)
            expired.append(request_id)
    return expired


def sample_environment(state: dict[str, Any]) -> dict[str, Any]:
    """Return raw time-series measurements from the current simulator state.

    Availability is the active share of the 200 paper edge servers. Per-node
    utilization is the mean across CPU, memory, GPU, and NPU capacity.
    """

    time_s = float(state["time_s"])
    completed_in_window = sum(
        time_s - 1.0 < float(record["time_s"]) <= time_s + 1e-12
        for record in state["completed_requests"]
    )
    utilization = utilization_matrix(state)
    edge_server_count = int(state["config"]["simulation"]["edge_servers"])
    active_edge_servers = state["topology"]["node_active"][:edge_server_count]
    return {
        "time_s": time_s,
        "throughput_rps": float(completed_in_window),
        "availability_percent": float(100.0 * np.mean(active_edge_servers)),
        "load_balance_index": float(load_balance_index(state)),
        "compute_utilization_percent": float(100.0 * np.mean(utilization[:, 0])),
        "bandwidth_utilization_percent": float(
            100.0 * np.mean(bandwidth_utilization(state))
        ),
        "per_node_utilization_percent": 100.0 * np.mean(utilization, axis=1),
    }


def inject_node_failures(
    state: dict[str, Any],
    node_ids: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Deactivate edge servers and requeue every hosted allocation for recovery."""

    requested_ids = list(node_ids)
    if not requested_ids:
        raise ValueError("node_ids must contain at least one edge-server ID")
    if any(
        isinstance(node_id, bool) or not isinstance(node_id, (int, np.integer))
        for node_id in requested_ids
    ):
        raise TypeError("node_ids must contain integers")
    failed_ids = [int(node_id) for node_id in requested_ids]
    if len(set(failed_ids)) != len(failed_ids):
        raise ValueError("node_ids must be unique")
    edge_server_count = int(state["config"]["simulation"]["edge_servers"])
    if any(node_id < 0 or node_id >= edge_server_count for node_id in failed_ids):
        raise ValueError(f"node failures must target edge servers in [0,{edge_server_count - 1}]")
    inactive_ids = [
        node_id for node_id in failed_ids if not state["topology"]["node_active"][node_id]
    ]
    if inactive_ids:
        raise ValueError(f"edge servers are already inactive: {inactive_ids}")

    failed_set = set(failed_ids)
    state["topology"]["node_active"][failed_ids] = False
    impacted_request_ids = []
    recovery_queues: dict[int, list[int]] = {
        controller_id: [] for controller_id in state["controller_queues"]
    }
    for request_id, allocation in state["allocations"].items():
        if allocation["status"] not in {"pending", "active"}:
            continue
        hosted_nodes = {int(node_id) for node_id in allocation["placements"].values()}
        if not hosted_nodes.intersection(failed_set):
            continue
        queue_occurrences = 0
        for queue in state["controller_queues"].values():
            if request_id in queue:
                queue.remove(request_id)
                queue_occurrences += 1
        if allocation["status"] == "pending" and queue_occurrences != 1:
            raise RuntimeError("pending allocation did not occur in exactly one controller queue")
        if request_id in state["active_request_ids"]:
            state["active_request_ids"].remove(request_id)
        _release_claims(state, allocation)
        allocation["task_index"] = 0
        allocation["placements"] = {}
        allocation["routes"] = []
        allocation["status"] = "pending"
        allocation["expires_at_s"] = None
        allocation["disruption_started_s"] = float(state["time_s"])
        allocation["failure_count"] += 1
        first_controller = int(allocation["tasks"][0]["controller_id"])
        recovery_queues[first_controller].append(int(request_id))
        impacted_request_ids.append(int(request_id))

    for controller_id, request_ids in recovery_queues.items():
        if request_ids:
            state["controller_queues"][controller_id][:0] = request_ids
    state["_path_cache"] = {}
    if impacted_request_ids:
        state["done"] = False
    event = {
        "time_s": float(state["time_s"]),
        "node_ids": failed_ids,
        "impacted_request_ids": impacted_request_ids,
        "requeue_policy": "recovery_first",
    }
    state["failure_events"].append(event)
    return copy.deepcopy(event)


def step_environment(
    state: dict[str, Any],
    actions: dict[int, dict[str, Any]] | list[dict[str, Any]],
    dt_s: float | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], bool, dict[str, Any]]:
    """Apply one placement/routing decision per ready controller and advance simulation time."""

    ready = pending_controller_decisions(state)
    if isinstance(actions, list):
        action_map = {int(action["controller_id"]): action for action in actions}
    else:
        action_map = {int(controller_id): action for controller_id, action in actions.items()}
    if set(action_map) != set(ready):
        raise ValueError(f"actions must cover ready controllers exactly: expected {sorted(ready)}, got {sorted(action_map)}")
    validated = {controller_id: validate_semantic_action(action) for controller_id, action in action_map.items()}
    rewards: dict[int, dict[str, Any]] = {}
    reward_contexts: dict[int, dict[str, Any]] = {}
    events = []
    for controller_id in sorted(validated):
        reward_context, event = _apply_action(state, controller_id, validated[controller_id])
        reward_contexts[controller_id] = reward_context
        events.append(event)
    global_step_metrics = [
        reward_contexts[controller_id]["metrics"]
        for controller_id in sorted(reward_contexts)
    ]
    for event in events:
        controller_id = int(event["controller_id"])
        context = reward_contexts[controller_id]
        reward = reward_components(
            state,
            controller_id,
            context["metrics"],
            context["semantic"],
            context["violations"],
            global_step_metrics,
        )
        rewards[controller_id] = reward
        event["reward"] = reward

    elapsed = float(state["config"]["simulation"]["decision_interval_s"] if dt_s is None else dt_s)
    if elapsed <= 0:
        raise ValueError("dt_s must be positive")
    _advance_clock(state, state["time_s"] + elapsed)
    state["step_index"] += 1
    expired = _release_expired(state)
    activated = _activate_arrivals(state)
    later_activated, later_expired = _advance_until_decision(state)
    activated.extend(later_activated)
    expired.extend(later_expired)
    done = (
        state["next_request_index"] == len(state["scenario"]["requests"])
        and not any(state["controller_queues"].values())
        and not state["active_request_ids"]
    )
    state["done"] = done
    compute_utilization_percent = float(100.0 * np.mean(utilization_matrix(state)[:, 0]))
    bandwidth_utilization_percent = float(100.0 * np.mean(bandwidth_utilization(state)))
    normalized = float(np.mean([event["reward"]["normalized_reward"] for event in events])) if events else 0.0
    info = {
        "time_s": state["time_s"],
        "controller_events": events,
        "accepted": sum(event["accepted"] for event in events),
        "rejected": sum(not event["accepted"] for event in events),
        "activated_request_ids": activated,
        "released_request_ids": expired,
        "normalized_reward": normalized,
        "sla_violation": bool(any(event["sla_violation"] for event in events)),
        "mean_latency_ms": float(state["recent_metrics"]["mean_latency_ms"]),
        "load_balance_index": load_balance_index(state),
        "compute_utilization_percent": compute_utilization_percent,
        "bandwidth_utilization_percent": bandwidth_utilization_percent,
        "reliability": float(state["recent_metrics"]["reliability"]),
        "global_cost_vectors": {controller_id: reward["global_cost_vector"].copy() for controller_id, reward in rewards.items()},
        "local_cost_vectors": {controller_id: reward["local_cost_vector"].copy() for controller_id, reward in rewards.items()},
        "violation_vectors": {controller_id: reward["violation_vector"].copy() for controller_id, reward in rewards.items()},
    }
    return state, rewards, done, info


def snapshot_environment(state: dict[str, Any]) -> dict[str, Any]:
    """Create a deep deterministic checkpoint of simulator state."""

    snapshot = copy.deepcopy(state)
    snapshot["_path_cache"] = {}
    return snapshot


def restore_environment(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Restore a simulator checkpoint and validate resource shapes and bounds."""

    state = copy.deepcopy(snapshot)
    topology = state["topology"]
    if state["node_available"].shape != topology["capacity"].shape:
        raise ValueError("snapshot node resource shape is invalid")
    if state["link_available_mbps"].shape != topology["link_capacity_mbps"].shape:
        raise ValueError("snapshot link resource shape is invalid")
    if np.any(state["node_available"] < -1e-8) or np.any(state["node_available"] > topology["capacity"] + 1e-8):
        raise ValueError("snapshot contains out-of-range node resources")
    if np.any(state["link_available_mbps"] < -1e-8) or np.any(state["link_available_mbps"] > topology["link_capacity_mbps"] + 1e-8):
        raise ValueError("snapshot contains out-of-range link resources")
    state["_path_cache"] = {}
    return state


def planner_snapshot(state: dict[str, Any], controller_id: int | None = None) -> dict[str, Any]:
    """Expose live topology, service DAGs, telemetry, and placements to TopoRAG."""

    topology = state["topology"]
    if controller_id is None:
        request_ids = sorted(
            request_id
            for queue in state["controller_queues"].values()
            for request_id in queue
        )
        selected_node_ids = topology["node_ids"]
    else:
        request_ids = list(state["controller_queues"][controller_id][:1])
        selected_node_ids = region_node_ids(state, controller_id)
    selected_node_set = {int(node_id) for node_id in selected_node_ids}
    service_dags = []
    for request_id in request_ids:
        allocation = state["allocations"][request_id]
        request = allocation["request"]
        service = service_for_request(state, request)
        regional_replica_ids = {
            int(microservice_id): [
                int(task["replica_id"])
                for task in allocation["tasks"]
                if int(task["microservice_id"]) == int(microservice_id)
                and int(task["controller_id"]) == controller_id
            ]
            for microservice_id in service["microservice_ids"]
        }
        service_dags.append(
            {
                "request_id": request_id,
                "service_id": int(service["service_id"]),
                "class": service["class"],
                "slice_tag": SLICE_NAMES[int(service["slice_id"])],
                "arrival_time_s": float(request["arrival_time_s"]),
                "source_location_km": request["source_location_km"].tolist(),
                "microservices": [
                    {
                        "microservice_id": int(microservice_id),
                        "resource_demand": service["resource_demands"][microservice_id].tolist(),
                        "workload_mi": float(service["workloads_mi"][microservice_id]),
                        "replica_limit": int(service["replica_limits"][microservice_id]),
                        "replicas": (
                            len(regional_replica_ids[int(microservice_id)])
                            if controller_id is not None
                            else int(request["replica_counts"][microservice_id])
                        ),
                        "replica_ids": (
                            regional_replica_ids[int(microservice_id)]
                            if controller_id is not None
                            else list(range(int(request["replica_counts"][microservice_id])))
                        ),
                    }
                    for microservice_id in service["microservice_ids"]
                ],
                "edges": [
                    {
                        "edge_id": edge_id,
                        "source": int(source),
                        "destination": int(destination),
                        "payload_mb": float(service["edge_payload_mb"][edge_id]),
                    }
                    for edge_id, (source, destination) in enumerate(service["edges"])
                ],
                "qos": dict(service["qos"]),
            }
        )
    links = [
        {
            "source": int(source),
            "destination": int(destination),
            "capacity_mbps": topology["link_capacity_mbps"][index].tolist(),
            "available_mbps": state["link_available_mbps"][index].tolist(),
            "delay_ms": float(topology["link_delay_ms"][index]),
            "reliability": float(topology["link_reliability"][index]),
            "slice_tags": list(SLICE_NAMES),
        }
        for index, (source, destination) in enumerate(zip(topology["link_sources"], topology["link_destinations"]))
        if int(source) in selected_node_set and int(destination) in selected_node_set
    ]
    current_placement = [
        {
            "request_id": request_id,
            "microservice_id": microservice_id,
            "replica_id": replica_id,
            "node_id": int(node_id),
        }
        for request_id, allocation in state["allocations"].items()
        if controller_id is None or request_id in request_ids
        for (microservice_id, replica_id), node_id in allocation["placements"].items()
    ]
    candidate_context = {}
    selected_controllers = (
        range(state["config"]["simulation"]["edge_controllers"])
        if controller_id is None
        else [controller_id]
    )
    all_decisions = pending_controller_decisions(state)
    for candidate_controller in selected_controllers:
        if candidate_controller not in all_decisions:
            continue
        decision = all_decisions[candidate_controller]
        candidate_context[str(candidate_controller)] = {
            "request_id": decision["request_id"],
            "microservice_id": decision["microservice_id"],
            "replica_id": decision["replica_id"],
            "local_node_ids": decision["local_node_ids"].tolist(),
            "placement_mask": decision["observation"]["placement_mask"].tolist(),
            "incoming_edge_ids": decision["incoming_edge_ids"][
                decision["incoming_edge_mask"].astype(bool)
            ].tolist(),
            "feasible_path_counts": decision["observation"]["path_mask"].sum(axis=(1, 2))[
                decision["incoming_edge_mask"].astype(bool)
            ].tolist(),
        }
    telemetry = {
        "controller_utilization": controller_utilization(state).tolist(),
        "global_utilization": utilization_matrix(state).mean(axis=0).tolist(),
        "link_utilization": bandwidth_utilization(state).mean(axis=0).tolist(),
        "queue_depths": [len(queue) for queue in state["controller_queues"].values()],
        "recent_qos": dict(state["recent_metrics"]),
        "rollout_summary": {
            "completed": len(state["completed_requests"]),
            "rejected": len(state["rejected_requests"]),
            "active": len(state["active_request_ids"]),
        },
        "candidate_feasibility_context": candidate_context,
    }
    return {
        "service_dags": service_dags,
        "topology": {
            "node_ids": topology["node_ids"][selected_node_ids].tolist(),
            "node_tiers": [NODE_TIERS[int(value)] for value in topology["node_tiers"][selected_node_ids]],
            "controller_ids": topology["node_controller_ids"][selected_node_ids].tolist(),
            "coordinates_km": topology["coordinates_km"][selected_node_ids].tolist(),
            "capacity": topology["capacity"][selected_node_ids].tolist(),
            "residual_capacity": np.where(
                topology["node_active"][selected_node_ids, None],
                state["node_available"][selected_node_ids],
                0.0,
            ).tolist(),
            "radio_profiles": [
                [RADIO_PROFILES[index] for index, enabled in enumerate(row) if enabled]
                for row in topology["radio_profiles"][selected_node_ids]
            ],
            "latency_ms": topology["latency_ms"][selected_node_ids].tolist(),
            "node_active": topology["node_active"][selected_node_ids].tolist(),
            "links": links,
            "resource_names": list(RESOURCE_NAMES),
        },
        "telemetry": telemetry,
        "current_placement": current_placement,
    }


def planner_query(state: dict[str, Any], controller_id: int) -> dict[str, Any]:
    """Build the exact eleven-field cloud-planner query contract."""

    snapshot = planner_snapshot(state, controller_id)
    pending = _pending_task(state, controller_id)
    if pending is None:
        qos_intent: dict[str, Any] = {}
        slice_tag = "none"
    else:
        allocation, _ = pending
        request = allocation["request"]
        service = service_for_request(state, request)
        qos_intent = dict(service["qos"])
        slice_tag = SLICE_NAMES[int(request["slice_id"])]
    interval = int(state["config"]["planner"]["planning_interval_s"])
    return {
        "window_id": int(state["time_s"] // interval),
        "controller_id": int(controller_id),
        "region_id": int(controller_id),
        "sim_time_s": float(state["time_s"]),
        "service_dags": snapshot["service_dags"],
        "topology": snapshot["topology"],
        "telemetry": snapshot["telemetry"],
        "qos_intent": qos_intent,
        "slice_tag": slice_tag,
        "policy_id": "semantic_mappo",
        "current_placement": snapshot["current_placement"],
    }


def run_evaluation_matrix(
    config: dict[str, Any],
    evaluate_method: Any,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Run Figure 5--10 only with explicitly instrumented measurements.

    Aggregate rollout summaries cannot establish recovery time, temporal burst
    adaptation, semantic drift, planner feasibility, hyperparameter
    sensitivity, diurnal traces, or spatial utilization.  The evaluation
    callback must therefore advertise and return the dedicated measurements
    below.  This preflight prevents plausible-looking proxy curves from being
    written as experimental evidence.
    """

    validate_config(config)
    required_capabilities = {
        "arrival_rate_sweep",
        "complexity_sweep",
        "failure_recovery_timeline",
        "burst_latency_timeline",
        "diurnal_utilization_timeline",
        "component_ablation_runs",
        "hyperparameter_sensitivity_runs",
        "planner_feasibility_trials",
        "semantic_drift_windows",
        "spatial_node_utilization",
    }
    capabilities = set(getattr(evaluate_method, "measurement_capabilities", ()))
    missing = sorted(required_capabilities - capabilities)
    if missing:
        raise RuntimeError(
            "Measured Figure 5--10 evaluation requires dedicated callback instrumentation; "
            f"missing capabilities: {missing}. No proxy result rows were generated."
        )
    result = evaluate_method(config=config, measurement_suite="figures_5_10")
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("instrumented evaluation callback must return (result_rows, spatial_arrays)")
    rows, spatial = result
    if not isinstance(rows, list) or not rows:
        raise ValueError("instrumented evaluation returned no measured result rows")
    required_spatial = {
        "x",
        "y",
        "utilization_before",
        "utilization_after",
        "node_x",
        "node_y",
    }
    if set(spatial) != required_spatial:
        raise ValueError(
            "instrumented spatial arrays must be exactly "
            f"{sorted(required_spatial)}; received {sorted(spatial)}"
        )
    for name, values in spatial.items():
        array = np.asarray(values)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"instrumented spatial array {name!r} is empty or non-finite")
    return rows, {name: np.asarray(values) for name, values in spatial.items()}
def scenario_json(state_or_scenario: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-ready scenario or planner/debug state without private caches."""

    return json_ready(state_or_scenario)
