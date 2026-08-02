"""Graph, feasibility, QoS, and reward helpers for the simulator."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import networkx as nx
import numpy as np


def service_for_request(state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    return state["scenario"]["services"][int(request["service_id"])]


def region_node_ids(state: dict[str, Any], controller_id: int) -> np.ndarray:
    topology = state["topology"]
    edge_nodes = topology["controller_regions"][controller_id]
    controller_node = np.array([topology["controller_node_ids"][controller_id]], dtype=np.int64)
    return np.concatenate([edge_nodes, controller_node, topology["core_node_ids"]])


def _network_graph(state: dict[str, Any], slice_id: int) -> nx.Graph:
    topology = state["topology"]
    graph = nx.Graph()
    active = topology["node_active"]
    graph.add_nodes_from(int(node_id) for node_id in topology["node_ids"][active])
    for link_index, (source, destination) in enumerate(
        zip(topology["link_sources"], topology["link_destinations"])
    ):
        if active[source] and active[destination] and state["link_available_mbps"][link_index, slice_id] > 1e-9:
            graph.add_edge(
                int(source),
                int(destination),
                link_index=link_index,
                weight=float(topology["link_delay_ms"][link_index]),
            )
    return graph


def candidate_paths(
    state: dict[str, Any],
    source: int,
    destination: int,
    slice_id: int,
    count: int,
) -> list[dict[str, Any]]:
    """Return up to ``count`` deterministic delay-shortest feasible paths."""

    if source == destination:
        return [
            {
                "path_id": f"{source}:{destination}:{rank}:{slice_id}",
                "nodes": np.array([source], dtype=np.int64),
                "link_indices": np.empty(0, dtype=np.int64),
                "delay_ms": 0.0,
                "available_fraction": 1.0,
                "reliability": 1.0,
                "cross_region": False,
            }
            for rank in range(count)
        ]
    cache_key = (int(source), int(destination), int(slice_id))
    cached = state["_path_cache"].get(cache_key)
    if cached is not None:
        paths = cached
    else:
        graph = _network_graph(state, slice_id)
        if source not in graph or destination not in graph:
            return []
        try:
            generator = nx.shortest_simple_paths(graph, int(source), int(destination), weight="weight")
            paths = []
            for nodes in generator:
                paths.append(np.asarray(nodes, dtype=np.int64))
                if len(paths) == count:
                    break
        except nx.NetworkXNoPath:
            paths = []
        state["_path_cache"][cache_key] = paths

    topology = state["topology"]
    results: list[dict[str, Any]] = []
    for rank, nodes in enumerate(paths[:count]):
        link_indices = []
        for source_node, destination_node in pairwise(nodes):
            matches = np.flatnonzero(
                ((topology["link_sources"] == source_node) & (topology["link_destinations"] == destination_node))
                | ((topology["link_sources"] == destination_node) & (topology["link_destinations"] == source_node))
            )
            if matches.size != 1:
                raise RuntimeError("topology edge lookup is inconsistent")
            link_indices.append(int(matches[0]))
        link_array = np.asarray(link_indices, dtype=np.int64)
        if link_array.size:
            fractions = state["link_available_mbps"][link_array, slice_id] / topology["link_capacity_mbps"][link_array, slice_id]
            available_fraction = float(np.min(fractions))
            reliability = float(np.prod(topology["link_reliability"][link_array]))
            delay_ms = float(np.sum(topology["link_delay_ms"][link_array]))
        else:
            available_fraction = 1.0
            reliability = 1.0
            delay_ms = 0.0
        controller_ids = topology["node_controller_ids"][nodes]
        cross_region = bool(np.unique(controller_ids[controller_ids >= 0]).size > 1)
        results.append(
            {
                "path_id": f"{source}:{destination}:{rank}:{slice_id}",
                "nodes": nodes.copy(),
                "link_indices": link_array,
                "delay_ms": delay_ms,
                "available_fraction": available_fraction,
                "reliability": reliability,
                "cross_region": cross_region,
            }
        )
    return results


def utilization_matrix(state: dict[str, Any]) -> np.ndarray:
    capacity = state["topology"]["capacity"]
    used = capacity - state["node_available"]
    return np.divide(used, capacity, out=np.zeros_like(used), where=capacity > 0)


def controller_utilization(state: dict[str, Any]) -> np.ndarray:
    values = []
    utilization = utilization_matrix(state)
    for controller_id in range(state["config"]["simulation"]["edge_controllers"]):
        values.append(float(np.mean(utilization[region_node_ids(state, controller_id)])))
    return np.asarray(values, dtype=np.float64)


def load_balance_index(state: dict[str, Any], node_ids: np.ndarray | None = None) -> float:
    utilization = utilization_matrix(state)
    if node_ids is not None:
        utilization = utilization[node_ids]
    return float(np.var(utilization, axis=0).sum())


def bandwidth_utilization(state: dict[str, Any]) -> np.ndarray:
    capacity = state["topology"]["link_capacity_mbps"]
    used = capacity - state["link_available_mbps"]
    return np.divide(used, capacity, out=np.zeros_like(used), where=capacity > 0)


def placement_mask(state: dict[str, Any], controller_id: int, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local_nodes = region_node_ids(state, controller_id)
    feasible = np.all(state["node_available"][local_nodes] + 1e-10 >= demand[None, :], axis=1)
    feasible &= state["topology"]["node_active"][local_nodes]
    return local_nodes, feasible.astype(np.float32)


def request_latency_metrics(state: dict[str, Any], allocation: dict[str, Any]) -> dict[str, float | bool]:
    """Compute replica-aware critical-path QoS for a completed service DAG.

    Packet-delay jitter is the path-mixture dispersion along the selected
    critical execution path.  Logical-edge reliability combines redundant
    destination replicas and then composes the DAG dependencies in series.
    Synchronization is reported as the largest join skew on that execution
    path; it is already represented by the critical-path finish time and is
    therefore not added to latency a second time.
    """

    request = allocation["request"]
    service = service_for_request(state, request)
    topology = state["topology"]
    mips_per_vcpu = float(state["config"]["simulation"]["mips_per_vcpu"])
    microservice_count = len(service["microservice_ids"])
    utilization = utilization_matrix(state)
    incoming: dict[int, list[int]] = {microservice_id: [] for microservice_id in range(microservice_count)}
    outgoing: dict[int, list[int]] = {microservice_id: [] for microservice_id in range(microservice_count)}
    for source, destination in service["edges"]:
        incoming[int(destination)].append(int(source))
        outgoing[int(source)].append(int(destination))

    route_by_replica: dict[tuple[int, int, int], dict[str, Any]] = {}
    edge_replica_reliability: dict[tuple[int, int], list[float]] = {}
    for route in allocation["routes"]:
        source = int(route["source_microservice_id"])
        destination = int(route["destination_microservice_id"])
        destination_replica = int(route["destination_replica_id"])
        key = (source, destination, destination_replica)
        if key in route_by_replica:
            raise RuntimeError(f"duplicate route record for dependency replica {key}")
        route_by_replica[key] = route
        edge_replica_reliability.setdefault((source, destination), []).append(
            float(route["weighted_reliability"])
        )

    # Each tuple stores total finish, access, processing, queueing, transfer,
    # accumulated path jitter, and the largest synchronization skew.
    replica_timing: dict[tuple[int, int], tuple[float, float, float, float, float, float, float]] = {}
    for microservice_id in range(microservice_count):
        replica_ids = sorted(
            replica_id
            for candidate_ms, replica_id in allocation["placements"]
            if int(candidate_ms) == microservice_id
        )
        if not replica_ids:
            raise RuntimeError(f"completed allocation is missing microservice {microservice_id}")
        cpu_demand = float(service["resource_demands"][microservice_id, 0])
        if cpu_demand <= 0.0:
            raise ValueError("microservice CPU demand must be positive for latency calculation")
        base_processing_ms = float(
            service["workloads_mi"][microservice_id]
            / (cpu_demand * mips_per_vcpu)
            * 1000.0
        )
        for replica_id in replica_ids:
            node_id = int(allocation["placements"][(microservice_id, replica_id)])
            processing_ms = base_processing_ms + float(topology["latency_ms"][node_id, 0])
            cpu_utilization = min(float(utilization[node_id, 0]), 0.99)
            queueing_ms = base_processing_ms * cpu_utilization / max(1.0 - cpu_utilization, 0.01)
            if not incoming[microservice_id]:
                distance = float(
                    np.linalg.norm(
                        request["source_location_km"] - topology["coordinates_km"][node_id]
                    )
                )
                access_ms = float(topology["latency_ms"][node_id, 1] + distance / 0.2)
                replica_timing[(microservice_id, replica_id)] = (
                    access_ms + processing_ms + queueing_ms,
                    access_ms,
                    processing_ms,
                    queueing_ms,
                    0.0,
                    0.0,
                    0.0,
                )
                continue

            arrivals = []
            for source in incoming[microservice_id]:
                route = route_by_replica[(source, microservice_id, replica_id)]
                source_replica = int(route["source_replica_id"])
                parent = replica_timing[(source, source_replica)]
                transfer_ms = float(route["weighted_delay_ms"])
                fractions = np.asarray(
                    [path["fraction"] for path in route["paths"]], dtype=np.float64
                )
                delays = np.asarray(
                    [path["delay_ms"] for path in route["paths"]], dtype=np.float64
                )
                route_jitter_ms = float(
                    np.sqrt(np.dot(fractions, np.square(delays - transfer_ms)))
                )
                arrivals.append((parent[0] + transfer_ms, parent, transfer_ms, route_jitter_ms))
            arrival_times = [item[0] for item in arrivals]
            join_skew_ms = max(arrival_times) - min(arrival_times)
            longest_arrival, parent, transfer_ms, route_jitter_ms = max(
                arrivals, key=lambda item: item[0]
            )
            replica_timing[(microservice_id, replica_id)] = (
                longest_arrival + processing_ms + queueing_ms,
                parent[1],
                parent[2] + processing_ms,
                parent[3] + queueing_ms,
                parent[4] + transfer_ms,
                float(np.hypot(parent[5], route_jitter_ms)),
                max(parent[6], join_skew_ms),
            )

    sinks = [microservice_id for microservice_id in range(microservice_count) if not outgoing[microservice_id]]
    selected_sinks = []
    for microservice_id in sinks:
        candidates = [
            timing
            for (candidate_ms, _), timing in replica_timing.items()
            if candidate_ms == microservice_id
        ]
        selected_sinks.append(min(candidates, key=lambda item: item[0]))
    critical = max(selected_sinks, key=lambda item: item[0])
    total_ms, access_ms, processing_ms, queueing_ms, transfer_ms, jitter_ms, synchronization_ms = critical

    logical_edge_reliability = []
    for source, destination in service["edges"]:
        replica_values = edge_replica_reliability[(int(source), int(destination))]
        # A dependency remains available when at least one independently routed
        # destination replica receives it successfully.
        logical_edge_reliability.append(
            1.0 - float(np.prod(1.0 - np.clip(replica_values, 0.0, 1.0)))
        )
    reliability = float(np.prod(logical_edge_reliability)) if logical_edge_reliability else 1.0
    qos = service["qos"]
    sla_violation = bool(
        total_ms > qos["latency_ms"]
        or jitter_ms > qos["jitter_ms"]
        or synchronization_ms > qos["sync_ms"]
        or reliability < qos["min_reliability"]
    )
    return {
        "access_latency_ms": access_ms,
        "processing_latency_ms": float(processing_ms),
        "queueing_latency_ms": float(queueing_ms),
        "transfer_latency_ms": float(transfer_ms),
        "synchronization_ms": float(synchronization_ms),
        "jitter_ms": float(jitter_ms),
        "latency_ms": float(total_ms),
        "reliability": reliability,
        "sla_violation": sla_violation,
    }


def reward_components(
    state: dict[str, Any],
    controller_id: int,
    metrics: dict[str, Any],
    semantic: dict[str, Any],
    violations: dict[str, float],
    global_step_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return paper objective costs and signed trust-conditioned reward."""

    utilization = utilization_matrix(state)
    global_resource = float(np.mean(utilization))
    local_nodes = region_node_ids(state, controller_id)
    local_resource = float(np.mean(utilization[local_nodes]))
    global_lb = load_balance_index(state)
    local_lb = load_balance_index(state, local_nodes)

    def performance_costs(
        records: list[dict[str, Any]], current_metrics: list[dict[str, Any]]
    ) -> tuple[float, float]:
        selected = list(records[-100:])
        latency_values = []
        reliability_values = []
        included_request_ids = set()
        for record in selected:
            record_request_id = int(record["request_id"])
            included_request_ids.add(record_request_id)
            record_request = state["allocations"][record_request_id]["request"]
            qos = service_for_request(state, record_request)["qos"]
            latency_values.append(float(record["latency_ms"]) / float(qos["latency_ms"]))
            reliability_values.append(float(record["reliability"]))
        for current in current_metrics:
            current_request = current["request"]
            current_request_id = int(current_request["request_id"])
            if current_request_id in included_request_ids:
                continue
            qos = service_for_request(state, current_request)["qos"]
            latency_values.append(float(current["latency_ms"]) / float(qos["latency_ms"]))
            reliability_values.append(float(current["reliability"]))
            included_request_ids.add(current_request_id)
        if not latency_values:
            raise RuntimeError("reward aggregation requires an observed request metric")
        return float(np.mean(latency_values)), float(1.0 - np.mean(reliability_values))

    global_latency, global_reliability = performance_costs(
        state["completed_requests"], global_step_metrics
    )
    local_latency, local_reliability = performance_costs(
        state["controller_completed_metrics"][controller_id], [metrics]
    )
    global_costs = np.array(
        [global_latency, global_resource, global_lb, global_reliability], dtype=np.float64
    )
    local_costs = np.array(
        [local_latency, local_resource, local_lb, local_reliability], dtype=np.float64
    )
    global_weights = np.asarray(
        [
            state["config"]["reward"]["global_weights"][name]
            for name in ("latency", "resource", "load_balance", "reliability")
        ],
        dtype=np.float64,
    )
    local_weights = np.asarray(semantic["reward_weights"], dtype=np.float64)
    penalty_weights = np.asarray(semantic["penalty_weights"], dtype=np.float64)
    cooperation = float(semantic["cooperation"])
    global_reward = -float(np.dot(global_weights, global_costs))
    local_reward = -float(np.dot(local_weights, local_costs))
    grouped_violations = np.array(
        [
            violations["capacity"] + violations["slice_isolation"],
            violations["temporal"],
            violations["stability"] + violations["reliability"] + violations["connectivity"],
        ],
        dtype=np.float64,
    )
    penalty_cost = float(np.dot(penalty_weights, grouped_violations))
    total = cooperation * global_reward + (1.0 - cooperation) * local_reward - penalty_cost
    combined_cost = cooperation * -global_reward + (1.0 - cooperation) * -local_reward + penalty_cost
    return {
        "total": total,
        "global": global_reward,
        "local": local_reward,
        "penalty": -penalty_cost,
        "normalized_reward": float(1.0 / (1.0 + max(combined_cost, 0.0))),
        "global_cost_vector": global_costs,
        "local_cost_vector": local_costs,
        "violation_vector": grouped_violations,
        "violations": violations,
    }


def current_violation_components(
    state: dict[str, Any],
    request: dict[str, Any],
    metrics: dict[str, Any],
    capacity: float,
    connectivity: float,
    involved_nodes: list[int],
) -> dict[str, float]:
    service = service_for_request(state, request)
    qos = service["qos"]
    temporal = (
        max(0.0, metrics["latency_ms"] / qos["latency_ms"] - 1.0)
        + max(0.0, metrics["jitter_ms"] / qos["jitter_ms"] - 1.0)
        + max(0.0, metrics["synchronization_ms"] / qos["sync_ms"] - 1.0)
    )
    reliability = max(0.0, qos["min_reliability"] / max(metrics["reliability"], 1e-12) - 1.0)
    utilization = utilization_matrix(state)
    maximum_utilization = float(np.max(utilization[involved_nodes])) if involved_nodes else float(np.max(utilization))
    stability_limit = float(state["config"]["reward"]["stability_utilization"])
    stability = max(0.0, maximum_utilization / stability_limit - 1.0)
    slice_used = state["node_used_by_slice"]
    slice_id = int(request["slice_id"])
    other_slice_use = slice_used.sum(axis=1) - slice_used[:, slice_id]
    topology_capacity = state["topology"]["capacity"]
    other_ratio = np.divide(other_slice_use, topology_capacity, out=np.zeros_like(other_slice_use), where=topology_capacity > 0)
    relevant = involved_nodes if involved_nodes else list(range(topology_capacity.shape[0]))
    isolation_limit = float(state["config"]["reward"]["slice_isolation_utilization"])
    slice_isolation = max(0.0, float(np.max(other_ratio[relevant])) / isolation_limit - 1.0)
    return {
        "capacity": float(capacity),
        "temporal": float(temporal),
        "stability": float(stability),
        "reliability": float(reliability),
        "connectivity": float(connectivity),
        "slice_isolation": float(slice_isolation),
    }


def validate_semantic_action(action: dict[str, Any]) -> dict[str, Any]:
    required = {
        "controller_id", "request_id", "microservice_id", "replica_id", "node_id", "routing_fractions",
        "reward_weights", "penalty_weights", "cooperation",
    }
    missing = required - set(action)
    unknown = set(action) - required
    if missing or unknown:
        raise ValueError(f"action keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    reward_weights = np.asarray(action["reward_weights"], dtype=np.float64)
    penalty_weights = np.asarray(action["penalty_weights"], dtype=np.float64)
    routing = np.asarray(action["routing_fractions"], dtype=np.float64)
    if reward_weights.shape != (4,) or np.any(reward_weights < 0) or not np.isclose(reward_weights.sum(), 1.0):
        raise ValueError("action.reward_weights must be four non-negative values summing to 1")
    if penalty_weights.shape != (3,) or np.any(penalty_weights < 0):
        raise ValueError("action.penalty_weights must contain three non-negative values")
    cooperation = float(action["cooperation"])
    if not 0 <= cooperation <= 1:
        raise ValueError("action.cooperation must be in [0,1]")
    if routing.ndim != 2 or np.any(routing < 0):
        raise ValueError("action.routing_fractions must be a non-negative edge-by-path matrix")
    row_sums = routing.sum(axis=1)
    if np.any(~(np.isclose(row_sums, 0.0) | np.isclose(row_sums, 1.0))):
        raise ValueError("each action.routing_fractions row must sum to zero or one")
    return {
        **action,
        "routing_fractions": routing,
        "reward_weights": reward_weights,
        "penalty_weights": penalty_weights,
        "cooperation": cooperation,
    }
