"""Conference-paper simulator for joint microservice placement and routing.

The implementation keeps the paper's decision variables explicit: every step
places one exact replica and routes each incoming DAG dependency fractionally
over three paths.  State is held in dictionaries so the learning and planning
code can consume it without an environment framework.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _require(mapping: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise KeyError(f"{where} is missing required keys: {missing}")


def _range(
    values: list[float], where: str, positive: bool = True
) -> tuple[float, float]:
    if not isinstance(values, list) or len(values) != 2:
        raise TypeError(f"{where} must be a two-number list")
    low, high = float(values[0]), float(values[1])
    if low > high or (positive and low <= 0.0):
        raise ValueError(f"invalid range at {where}: {values}")
    return low, high


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the single strict conference configuration."""

    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    _require(
        config,
        {
            "seed",
            "data",
            "network",
            "traffic",
            "service",
            "objective",
            "rag",
            "llm",
            "ppo",
            "baseline",
            "experiment",
            "paths",
        },
        "config",
    )
    _require(
        config["data"],
        {
            "source",
            "scenario",
            "base_stations_csv",
            "taxi_trajectories_csv",
            "alibaba_trace_csv",
        },
        "data",
    )
    _require(
        config["network"],
        {
            "esps",
            "controllers",
            "clouds",
            "candidate_paths",
            "esp_cpu",
            "esp_memory_gb",
            "link_bandwidth_mbps",
            "link_delay_ms",
        },
        "network",
    )
    _require(
        config["traffic"],
        {
            "taxis",
            "duration_s",
            "decision_interval_ms",
            "request_rate_s",
            "pareto_shape",
            "mobility_speed",
        },
        "traffic",
    )
    _require(
        config["service"],
        {"cpu", "memory_gb", "traffic_mbps", "lifetime_s", "sla_ms"},
        "service",
    )
    _require(config["objective"], {"latency_weight", "resource_weight"}, "objective")
    if config["data"]["source"] not in {"synthetic", "external"}:
        raise ValueError("data.source must be 'synthetic' or 'external'")
    paper_counts = (
        int(config["network"]["esps"]),
        int(config["network"]["controllers"]),
        int(config["network"]["clouds"]),
    )
    if paper_counts != (3233, 16, 1):
        raise ValueError(
            "the conference setup requires 3,233 ESPs, 16 controllers, and one cloud"
        )
    if int(config["traffic"]["taxis"]) != 4328:
        raise ValueError("the conference setup requires 4,328 taxi traces")
    if int(config["network"]["candidate_paths"]) != 3:
        raise ValueError("the conference formulation uses three candidate paths")
    slots = float(config["traffic"]["duration_s"]) / (
        float(config["traffic"]["decision_interval_ms"]) / 1000.0
    )
    if not np.isclose(slots, 100.0):
        raise ValueError(
            "traffic duration and decision interval must define the paper's 100-slot horizon"
        )
    if [float(value) for value in config["traffic"]["request_rate_s"]] != [5.0, 300.0]:
        raise ValueError(
            "traffic.request_rate_s must preserve the paper's 5-300 requests/s range"
        )
    if [float(value) for value in config["traffic"]["mobility_speed"]] != [0.0, 2.0]:
        raise ValueError("traffic.mobility_speed must preserve the paper's 0x-2x range")
    for section, key in (
        ("network", "esp_cpu"),
        ("network", "esp_memory_gb"),
        ("network", "link_bandwidth_mbps"),
        ("network", "link_delay_ms"),
        ("service", "cpu"),
        ("service", "memory_gb"),
        ("service", "traffic_mbps"),
        ("service", "lifetime_s"),
    ):
        _range(config[section][key], f"{section}.{key}")
    if (
        float(config["objective"]["latency_weight"]) <= 0
        or float(config["objective"]["resource_weight"]) <= 0
    ):
        raise ValueError("the two objective weights must be positive")
    return config


def _normalized_coordinates(values: np.ndarray) -> np.ndarray:
    span = values.max(axis=0) - values.min(axis=0)
    if np.any(span <= 0.0):
        raise ValueError("coordinates must span both dimensions")
    return (values - values.min(axis=0)) / span


def _read_rows(path: str | Path, columns: set[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != columns:
            raise ValueError(
                f"{path} must contain exactly these columns: {sorted(columns)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _external_base_stations(path: str | Path, count: int) -> np.ndarray:
    rows = _read_rows(path, {"base_station_id", "longitude", "latitude"})
    if len(rows) != count or len({row["base_station_id"] for row in rows}) != count:
        raise ValueError(f"{path} must contain exactly {count} unique base stations")
    coordinates = np.asarray(
        [[float(row["longitude"]), float(row["latitude"])] for row in rows],
        dtype=np.float64,
    )
    return _normalized_coordinates(coordinates)


def _external_taxis(
    path: str | Path, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_rows(path, {"taxi_id", "time_s", "longitude", "latitude"})
    samples: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        samples.setdefault(row["taxi_id"], []).append(
            (float(row["time_s"]), float(row["longitude"]), float(row["latitude"]))
        )
    if len(samples) != count or any(len(points) < 2 for points in samples.values()):
        raise ValueError(
            f"{path} must contain at least two samples for each of exactly {count} taxis"
        )
    ordered = [sorted(samples[taxi_id]) for taxi_id in sorted(samples)]
    endpoints = np.asarray(
        [[[row[1], row[2]] for row in (trace[0], trace[-1])] for trace in ordered],
        dtype=np.float64,
    )
    flattened = endpoints.reshape(-1, 2)
    normalized = _normalized_coordinates(flattened).reshape(count, 2, 2)
    elapsed = np.asarray(
        [trace[-1][0] - trace[0][0] for trace in ordered], dtype=np.float64
    )
    if np.any(elapsed <= 0.0):
        raise ValueError("taxi traces must advance in time")
    velocity = (normalized[:, 1] - normalized[:, 0]) / elapsed[:, None]
    magnitude = np.linalg.norm(velocity, axis=1)
    speed = np.zeros(count, dtype=np.float64)
    if magnitude.max() > 0.0:
        speed = 2.0 * magnitude / magnitude.max()
    direction = np.zeros_like(velocity)
    moving = magnitude > 0.0
    direction[moving] = velocity[moving] / magnitude[moving, None]
    direction[~moving, 0] = 1.0
    return normalized[:, 0], direction, speed


def _external_service(
    path: str | Path, maximum_microservices: int
) -> dict[str, np.ndarray]:
    rows = _read_rows(
        path, {"microservice_id", "parent_id", "cpu", "memory_gb", "traffic_mbps"}
    )
    indexed = sorted(rows, key=lambda row: int(row["microservice_id"]))
    identifiers = [int(row["microservice_id"]) for row in indexed]
    if identifiers[:maximum_microservices] != list(range(maximum_microservices)):
        raise ValueError(
            f"{path} must provide contiguous microservice_id values from 0 through {maximum_microservices - 1}"
        )
    indexed = indexed[:maximum_microservices]
    parents = np.asarray(
        [int(row["parent_id"]) if row["parent_id"] else -1 for row in indexed],
        dtype=np.int64,
    )
    if parents[0] != -1 or np.any(parents[1:] >= np.arange(1, maximum_microservices)):
        raise ValueError(
            "Alibaba adapter requires a topological row order and a blank parent for the root"
        )
    return {
        "external_cpu": np.asarray([float(row["cpu"]) for row in indexed]),
        "external_memory": np.asarray([float(row["memory_gb"]) for row in indexed]),
        "external_traffic": np.asarray([float(row["traffic_mbps"]) for row in indexed]),
        "external_parent": parents,
    }


def _controller_partition(
    coordinates: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    controller = np.empty(len(coordinates), dtype=np.int64)
    for controller_id, region in enumerate(np.array_split(order, count)):
        controller[region] = controller_id
    centers = np.vstack(
        [coordinates[controller == index].mean(axis=0) for index in range(count)]
    )
    return controller, centers


def prepare_scenario(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build deterministic infrastructure, mobility, and optional trace arrays."""

    rng = np.random.default_rng(int(config["seed"]))
    network, traffic, data = config["network"], config["traffic"], config["data"]
    esp_count, controller_count = int(network["esps"]), int(network["controllers"])
    if data["source"] == "external":
        esp_coordinates = _external_base_stations(data["base_stations_csv"], esp_count)
        taxi_coordinates, taxi_directions, taxi_speed = _external_taxis(
            data["taxi_trajectories_csv"], int(traffic["taxis"])
        )
    else:
        esp_coordinates = rng.random((esp_count, 2))
        taxi_coordinates = rng.random((int(traffic["taxis"]), 2))
        angles = rng.uniform(0.0, 2.0 * np.pi, int(traffic["taxis"]))
        taxi_directions = np.column_stack((np.cos(angles), np.sin(angles)))
        speed_low, speed_high = _range(
            traffic["mobility_speed"], "traffic.mobility_speed", positive=False
        )
        taxi_speed = rng.uniform(speed_low, speed_high, int(traffic["taxis"]))
    controller_ids, controller_coordinates = _controller_partition(
        esp_coordinates, controller_count
    )
    cpu_low, cpu_high = _range(network["esp_cpu"], "network.esp_cpu")
    memory_low, memory_high = _range(network["esp_memory_gb"], "network.esp_memory_gb")
    capacity = np.column_stack(
        (
            rng.uniform(cpu_low, cpu_high, esp_count),
            rng.uniform(memory_low, memory_high, esp_count),
        )
    )
    cloud_id = esp_count + controller_count
    link_sources = np.concatenate(
        (
            np.arange(esp_count),
            esp_count + np.arange(controller_count),
            esp_count + np.arange(controller_count),
        )
    ).astype(np.int64)
    link_destinations = np.concatenate(
        (
            esp_count + controller_ids,
            esp_count + (np.arange(controller_count) + 1) % controller_count,
            np.full(controller_count, cloud_id),
        )
    ).astype(np.int64)
    bandwidth_low, bandwidth_high = _range(
        network["link_bandwidth_mbps"], "network.link_bandwidth_mbps"
    )
    delay_low, delay_high = _range(network["link_delay_ms"], "network.link_delay_ms")
    scenario: dict[str, np.ndarray] = {
        "esp_coordinates": esp_coordinates,
        "controller_ids": controller_ids,
        "controller_coordinates": controller_coordinates,
        "capacity": capacity,
        "link_sources": link_sources,
        "link_destinations": link_destinations,
        "link_capacity": rng.uniform(bandwidth_low, bandwidth_high, len(link_sources)),
        "link_delay": rng.uniform(delay_low, delay_high, len(link_sources)),
        "taxi_coordinates": taxi_coordinates,
        "taxi_directions": taxi_directions,
        "taxi_speed": taxi_speed,
        "node_available": np.ones(esp_count, dtype=bool),
        "service_feature_dim": np.asarray(5, dtype=np.int64),
    }
    if data["source"] == "external":
        maximum = max(
            int(scale["microservices"])
            for scale in config["experiment"]["utilization_scales"]
        )
        scenario.update(_external_service(data["alibaba_trace_csv"], maximum))
    return scenario


def save_scenario(scenario: dict[str, np.ndarray], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **scenario)
    return output


def load_scenario(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        scenario = {name: archive[name] for name in archive.files}
    _require(
        scenario,
        {
            "esp_coordinates",
            "controller_ids",
            "controller_coordinates",
            "capacity",
            "link_sources",
            "link_destinations",
            "link_capacity",
            "link_delay",
            "taxi_coordinates",
            "taxi_directions",
            "taxi_speed",
            "node_available",
            "service_feature_dim",
        },
        "scenario",
    )
    return scenario


def perturb_scenario(
    scenario: dict[str, np.ndarray], kind: str, seed: int
) -> dict[str, np.ndarray]:
    """Return one copied topology with a paper-stated meta-training perturbation."""

    if kind not in {"node_failure", "bandwidth", "node_churn"}:
        raise ValueError("kind must be 'node_failure', 'bandwidth', or 'node_churn'")
    changed = {name: value.copy() for name, value in scenario.items()}
    rng = np.random.default_rng(int(seed))
    esp_count = len(changed["esp_coordinates"])
    if kind == "node_failure":
        changed["node_available"][int(rng.integers(esp_count))] = False
    elif kind == "bandwidth":
        changed["link_capacity"][int(rng.integers(len(changed["link_capacity"])))] *= (
            0.5
        )
    else:
        node = int(rng.integers(esp_count))
        controllers = len(changed["controller_coordinates"])
        controller = (int(changed["controller_ids"][node]) + 1) % controllers
        changed["controller_ids"][node] = controller
        changed["link_destinations"][node] = esp_count + controller
    return changed


def generate_arrivals(
    config: dict[str, Any], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Generate a nonstationary Poisson process with Pareto-distributed peaks."""

    duration = float(config["traffic"]["duration_s"])
    interval = float(config["traffic"]["decision_interval_ms"]) / 1000.0
    low, high = _range(config["traffic"]["request_rate_s"], "traffic.request_rate_s")
    shape = float(config["traffic"]["pareto_shape"])
    if shape <= 0.0:
        raise ValueError("traffic.pareto_shape must be positive")
    arrivals: list[np.ndarray] = []
    start = 0.0
    while start < duration:
        width = min(interval, duration - start)
        rate = min(high, low * (1.0 + float(rng.pareto(shape))))
        count = int(rng.poisson(rate * width))
        if count:
            arrivals.append(start + rng.uniform(0.0, width, count))
        start += width
    times = (
        np.sort(np.concatenate(arrivals)) if arrivals else np.empty(0, dtype=np.float64)
    )
    lifetime_low, lifetime_high = _range(
        config["service"]["lifetime_s"], "service.lifetime_s"
    )
    return {
        "arrival_time": times,
        "taxi_id": rng.integers(
            0, int(config["traffic"]["taxis"]), len(times), dtype=np.int64
        ),
        "lifetime": rng.uniform(lifetime_low, lifetime_high, len(times)),
    }


def _service(
    config: dict[str, Any],
    scenario: dict[str, np.ndarray],
    scale: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, np.ndarray | int | float]:
    microservices, replicas = int(scale["microservices"]), int(scale["replicas"])
    if microservices < 1 or replicas < 1:
        raise ValueError("microservices and replicas must be positive")
    if "external_cpu" in scenario:
        cpu = scenario["external_cpu"][:microservices].copy()
        memory = scenario["external_memory"][:microservices].copy()
        parents = scenario["external_parent"][:microservices].copy()
        incoming_traffic = scenario["external_traffic"][:microservices].copy()
    else:
        cpu = rng.uniform(
            *_range(config["service"]["cpu"], "service.cpu"), microservices
        )
        memory = rng.uniform(
            *_range(config["service"]["memory_gb"], "service.memory_gb"), microservices
        )
        parents = np.concatenate(
            (np.asarray([-1], dtype=np.int64), (np.arange(1, microservices) - 1) // 2)
        )
        incoming_traffic = rng.uniform(
            *_range(config["service"]["traffic_mbps"], "service.traffic_mbps"),
            microservices,
        )
        incoming_traffic[0] = 0.0
    adjacency = np.zeros((microservices, microservices), dtype=np.float32)
    traffic = np.zeros((microservices, microservices), dtype=np.float64)
    for child, parent in enumerate(parents):
        if parent >= 0:
            adjacency[parent, child] = 1.0
            traffic[parent, child] = incoming_traffic[child]
    return {
        "microservices": microservices,
        "replicas": replicas,
        "cpu": cpu,
        "memory": memory,
        "adjacency": adjacency,
        "traffic": traffic,
        "sla_ms": float(config["service"]["sla_ms"]),
    }


def _taxi_position(state: dict[str, Any], taxi_id: int) -> np.ndarray:
    scenario = state["scenario"]
    displacement = (
        scenario["taxi_directions"][taxi_id]
        * scenario["taxi_speed"][taxi_id]
        * state["time_s"]
    )
    return np.mod(scenario["taxi_coordinates"][taxi_id] + displacement, 1.0)


def _release_until(state: dict[str, Any], time_s: float) -> None:
    live = []
    for allocation in state["active"]:
        if float(allocation["release_time"]) <= time_s:
            for node, cpu, memory in allocation["nodes"]:
                state["residual_nodes"][node] += (cpu, memory)
            for link, bandwidth in allocation["links"]:
                state["residual_links"][link] += bandwidth
        else:
            live.append(allocation)
    state["active"] = live
    state["observation_cache"] = None


def _node_utilization(state: dict[str, Any]) -> np.ndarray:
    """Return mean occupied CPU and memory fractions across the ESPs."""

    available = np.asarray(state["scenario"]["node_available"], dtype=bool)
    if not np.any(available):
        return np.zeros(2, dtype=np.float64)
    return np.mean(
        1.0
        - state["residual_nodes"][available] / state["scenario"]["capacity"][available],
        axis=0,
    )


def _advance(state: dict[str, Any], time_s: float) -> None:
    _release_until(state, time_s)
    state["time_s"] = time_s
    arrivals = state["requests"]["arrival_time"]
    while (
        state["next_request"] < len(arrivals)
        and arrivals[state["next_request"]] <= time_s + 1e-12
    ):
        state["queue"].append(state["next_request"])
        state["next_request"] += 1


def _start_request(state: dict[str, Any]) -> None:
    if state["current_request"] is not None:
        return
    if not state["queue"] and state["next_request"] < len(
        state["requests"]["arrival_time"]
    ):
        _advance(state, float(state["requests"]["arrival_time"][state["next_request"]]))
    if not state["queue"]:
        state["done"] = True
        return
    request_id = state["queue"].pop(0)
    taxi_id = int(state["requests"]["taxi_id"][request_id])
    taxi = _taxi_position(state, taxi_id)
    controller = int(
        np.argmin(
            np.linalg.norm(state["scenario"]["controller_coordinates"] - taxi, axis=1)
        )
    )
    microservices, replicas = (
        int(state["service"]["microservices"]),
        int(state["service"]["replicas"]),
    )
    state["current_request"] = request_id
    state["current_taxi"] = taxi_id
    state["current_controller"] = controller
    state["microservice_index"] = 0
    state["replica_index"] = 0
    state["placements"] = np.full((microservices, replicas), -1, dtype=np.int64)
    state["current_nodes"] = []
    state["current_links"] = []
    state["access_latency"] = 0.0
    state["task_latency"] = np.zeros((microservices, replicas, 2), dtype=np.float64)
    state["edge_latency"] = np.zeros(
        (microservices, microservices, replicas), dtype=np.float64
    )
    state["observation_cache"] = None


def reset_environment(
    config: dict[str, Any],
    scenario: dict[str, np.ndarray],
    scale: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """Create a deterministic episode state for one reported paper scale."""

    rng = np.random.default_rng(int(seed))
    service_rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(config["seed"]),
                int(scale["microservices"]),
                int(scale["replicas"]),
            ]
        )
    )
    state: dict[str, Any] = {
        "config": config,
        "scenario": scenario,
        "service": _service(config, scenario, scale, service_rng),
        "requests": generate_arrivals(config, rng),
        "rng": rng,
        "time_s": 0.0,
        "residual_nodes": scenario["capacity"].copy(),
        "residual_links": scenario["link_capacity"].copy(),
        "active": [],
        "queue": [],
        "next_request": 0,
        "current_request": None,
        "completed": [],
        "cumulative_reward": 0.0,
        "cumulative_cost": 0.0,
        "step_count": 0,
        "placement_steps": 0,
        "anomaly": False,
        "max_steps": round(
            float(config["traffic"]["duration_s"])
            / (float(config["traffic"]["decision_interval_ms"]) / 1000.0)
        ),
        "done": False,
        "observation_cache": None,
    }
    state["residual_nodes"][~scenario["node_available"]] = 0.0
    _advance(state, 0.0)
    _start_request(state)
    return state


def _region_nodes(state: dict[str, Any]) -> tuple[np.ndarray, int]:
    controller_ids = state["scenario"]["controller_ids"]
    maximum = max(
        int(np.sum(controller_ids == index))
        for index in range(int(state["config"]["network"]["controllers"]))
    )
    if state["current_request"] is None:
        return np.empty(0, dtype=np.int64), maximum
    return np.flatnonzero(controller_ids == int(state["current_controller"])), maximum


def _ring_links(
    source: int, target: int, controller_count: int, clockwise: bool
) -> list[int]:
    links: list[int] = []
    current = source
    while current != target:
        if clockwise:
            links.append(current)
            current = (current + 1) % controller_count
        else:
            previous = (current - 1) % controller_count
            links.append(previous)
            current = previous
    return links


def _candidate_paths(
    state: dict[str, Any], source: int, target: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    esp_count = int(state["config"]["network"]["esps"])
    controllers = int(state["config"]["network"]["controllers"])
    source_controller = int(state["scenario"]["controller_ids"][source])
    target_controller = int(state["scenario"]["controller_ids"][target])
    ring_start, cloud_start = esp_count, esp_count + controllers
    if source_controller == target_controller:
        ring_cycle = [
            ring_start + (source_controller + offset) % controllers
            for offset in range(controllers)
        ]
        paths = [
            [] if source == target else [source, target],
            [source, *ring_cycle, target],
            [
                source,
                cloud_start + source_controller,
                cloud_start + source_controller,
                target,
            ],
        ]
    else:
        clockwise = [
            ring_start + link
            for link in _ring_links(
                source_controller, target_controller, controllers, True
            )
        ]
        counter = [
            ring_start + link
            for link in _ring_links(
                source_controller, target_controller, controllers, False
            )
        ]
        cloud = [cloud_start + source_controller, cloud_start + target_controller]
        paths = [
            [source, *clockwise, target],
            [source, *counter, target],
            [source, *cloud, target],
        ]
    features = np.zeros((3, 4), dtype=np.float32)
    feasible = np.zeros(3, dtype=bool)
    demand = float(
        state["service"]["traffic"][
            state["incoming_parent"], state["microservice_index"]
        ]
    ) / int(state["service"]["replicas"])
    for index, links in enumerate(paths):
        if not links:
            features[index] = (
                0.0,
                1.0,
                0.0,
                float(source_controller != target_controller),
            )
            feasible[index] = True
            continue
        unique, multiplicity = np.unique(links, return_counts=True)
        available = state["residual_links"][unique] / multiplicity
        capacity = state["scenario"]["link_capacity"][unique] / multiplicity
        features[index] = (
            float(
                state["scenario"]["link_delay"][links].sum()
                / float(state["service"]["sla_ms"])
            ),
            float(np.min(available / capacity)),
            float(len(links) / (controllers + 4)),
            float(source_controller != target_controller),
        )
        feasible[index] = bool(np.all(available + 1e-9 >= demand))
    return np.asarray(paths, dtype=object), features, feasible


def _observation_data(state: dict[str, Any]) -> dict[str, np.ndarray | int]:
    if state["observation_cache"] is not None:
        return state["observation_cache"]
    service = state["service"]
    microservices, replicas = int(service["microservices"]), int(service["replicas"])
    region, maximum = _region_nodes(state)
    node_features = np.zeros((maximum, 6), dtype=np.float32)
    infrastructure_mask = np.zeros(maximum, dtype=bool)
    placement_mask = np.zeros(maximum, dtype=bool)
    path_features = np.zeros((microservices, maximum, 3, 4), dtype=np.float32)
    path_mask = np.zeros((microservices, maximum, 3), dtype=bool)
    incoming_mask = np.zeros(microservices, dtype=bool)
    candidate_costs = np.full(maximum, np.inf, dtype=np.float32)
    candidate_ids = np.full(maximum, -1, dtype=np.int64)
    service_nodes = np.zeros((microservices, 5), dtype=np.float32)
    capacity = state["scenario"]["capacity"]
    service_nodes[:, 0] = np.asarray(service["cpu"]) / capacity[:, 0].max()
    service_nodes[:, 1] = np.asarray(service["memory"]) / capacity[:, 1].max()
    service_nodes[:, 2] = np.asarray(service["traffic"]).sum(axis=0) / max(
        1.0, np.asarray(service["traffic"]).max()
    )
    service_nodes[:, 3] = np.asarray(service["traffic"]).sum(axis=1) / max(
        1.0, np.asarray(service["traffic"]).max()
    )
    if state["current_request"] is not None:
        service_nodes[:, 4] = np.sum(state["placements"] >= 0, axis=1) / replicas
        current = int(state["microservice_index"])
        incoming = np.flatnonzero(np.asarray(service["adjacency"])[:, current])
        incoming_mask[incoming] = True
        taxi = _taxi_position(state, int(state["current_taxi"]))
        count = len(region)
        candidate_ids[:count] = region
        infrastructure_mask[:count] = True
        residual = state["residual_nodes"][region]
        cap = capacity[region]
        node_features[:count, 0:2] = residual / cap
        node_features[:count, 2:4] = state["scenario"]["esp_coordinates"][region]
        node_features[:count, 4] = np.linalg.norm(
            state["scenario"]["esp_coordinates"][region] - taxi, axis=1
        )
        node_features[:count, 5] = int(state["current_controller"]) / max(
            1, int(state["config"]["network"]["controllers"]) - 1
        )
        resource_ok = np.all(
            residual + 1e-9
            >= (float(service["cpu"][current]), float(service["memory"][current])),
            axis=1,
        )
        resource_ok &= state["scenario"]["node_available"][region]
        placement_mask[:count] = resource_ok
        route_delay = np.zeros(count, dtype=np.float64)
        for parent in incoming:
            source = int(
                state["placements"][parent, int(state["replica_index"]) % replicas]
            )
            state["incoming_parent"] = int(parent)
            for local, target in enumerate(region):
                _, features, feasible = _candidate_paths(state, source, int(target))
                path_features[parent, local] = features
                path_mask[parent, local] = feasible
                placement_mask[local] &= bool(feasible.any())
                route_delay[local] += (
                    float(np.min(features[feasible, 0])) * float(service["sla_ms"])
                    if feasible.any()
                    else np.inf
                )
        utilization = 1.0 - residual[:, :2] / cap[:, :2]
        processing = (
            float(state["config"]["traffic"]["decision_interval_ms"])
            * float(service["cpu"][current])
            / cap[:, 0]
        )
        queueing = (
            float(state["config"]["traffic"]["decision_interval_ms"])
            * utilization[:, 0]
            / np.maximum(1e-3, 1.0 - utilization[:, 0])
        )
        latency_cost = (processing + queueing + route_delay) / float(service["sla_ms"])
        resource_after = np.mean(utilization, axis=1) + 0.5 * (
            float(service["cpu"][current]) / cap[:, 0]
            + float(service["memory"][current]) / cap[:, 1]
        )
        weights = state["config"]["objective"]
        candidate_costs[:count] = (
            float(weights["latency_weight"]) * latency_cost
            + float(weights["resource_weight"]) * resource_after
        )
        candidate_costs[~placement_mask] = np.inf
    service_mask = np.ones(microservices, dtype=bool)
    node_utilization = _node_utilization(state)
    bandwidth_utilization = (
        1.0 - state["residual_links"].sum() / state["scenario"]["link_capacity"].sum()
    )
    request_count = max(1, len(state["requests"]["arrival_time"]))
    context = np.asarray(
        [
            state["time_s"] / float(state["config"]["traffic"]["duration_s"]),
            len(state["queue"]) / request_count,
            len(state["active"]) / request_count,
            int(state["microservice_index"]) / microservices
            if state["current_request"] is not None
            else 1.0,
            int(state["replica_index"]) / replicas
            if state["current_request"] is not None
            else 1.0,
            node_utilization[0],
            node_utilization[1],
            bandwidth_utilization,
            state["scenario"]["taxi_speed"][int(state["current_taxi"])] / 2.0
            if state["current_request"] is not None
            else 0.0,
            float(service["sla_ms"]) / 100.0,
        ],
        dtype=np.float32,
    )
    result = {
        "service_nodes": service_nodes,
        "service_adjacency": np.asarray(service["adjacency"], dtype=np.float32),
        "service_mask": service_mask,
        "infrastructure_nodes": node_features,
        "infrastructure_mask": infrastructure_mask,
        "context": context,
        "placement_mask": placement_mask,
        "path_features": path_features,
        "path_mask": path_mask,
        "incoming_edge_mask": incoming_mask,
        "candidate_costs": candidate_costs,
        "candidate_node_ids": candidate_ids,
        "microservice_index": int(state["microservice_index"])
        if state["current_request"] is not None
        else -1,
        "replica_index": int(state["replica_index"])
        if state["current_request"] is not None
        else -1,
    }
    state["observation_cache"] = result
    return result


def observation(state: dict[str, Any]) -> dict[str, np.ndarray | int]:
    """Return the graph, candidate, path, and mask arrays used by the agents."""

    current = _observation_data(state)
    while state["current_request"] is not None and not np.any(
        current["placement_mask"]
    ):
        _reject_infeasible_request(state)
        _advance_request_slot(state)
        current = _observation_data(state)
    return current


def feasible_action(state: dict[str, Any]) -> dict[str, Any]:
    """Choose the least-cost feasible placement and one feasible path per dependency."""

    current = observation(state)
    feasible = np.flatnonzero(current["placement_mask"])
    if not len(feasible):
        raise RuntimeError("no feasible placement remains")
    placement = int(feasible[np.argmin(current["candidate_costs"][feasible])])
    routing = np.zeros((int(state["service"]["microservices"]), 3), dtype=np.float64)
    for parent in np.flatnonzero(current["incoming_edge_mask"]):
        paths = np.flatnonzero(current["path_mask"][parent, placement])
        routing[
            parent,
            paths[np.argmin(current["path_features"][parent, placement, paths, 0])],
        ] = 1.0
    return {
        "placement_index": placement,
        "node_id": int(current["candidate_node_ids"][placement]),
        "routing_fractions": routing,
    }


def _reserve_routes(state: dict[str, Any], node: int, routing: np.ndarray) -> float:
    service = state["service"]
    current, replica = int(state["microservice_index"]), int(state["replica_index"])
    transmission = 0.0
    for parent in np.flatnonzero(np.asarray(service["adjacency"])[:, current]):
        parent_latency = 0.0
        fractions = np.asarray(routing[parent], dtype=np.float64)
        if (
            np.any(~np.isfinite(fractions))
            or np.any(fractions < 0.0)
            or not np.isclose(fractions.sum(), 1.0)
        ):
            raise ValueError(
                f"routing fractions for incoming microservice {parent} must be nonnegative and sum to one"
            )
        source = int(state["placements"][parent, replica % int(service["replicas"])])
        state["incoming_parent"] = int(parent)
        paths, _, path_allowed = _candidate_paths(state, source, node)
        if np.any((fractions > 1e-12) & ~path_allowed):
            raise ValueError("routing uses a path without enough residual bandwidth")
        demand = float(service["traffic"][parent, current]) / int(service["replicas"])
        for fraction, links in zip(fractions, paths, strict=True):
            if fraction <= 0.0:
                continue
            amount = demand * float(fraction)
            unique, multiplicity = np.unique(
                np.asarray(links, dtype=np.int64), return_counts=True
            )
            usage = amount * multiplicity
            if np.any(state["residual_links"][unique] + 1e-9 < usage):
                raise ValueError("routing fractions exceed residual link bandwidth")
            state["residual_links"][unique] -= usage
            state["current_links"].extend(
                (int(link), float(value))
                for link, value in zip(unique, usage, strict=True)
            )
            if len(links):
                delay = float(
                    state["scenario"]["link_delay"][
                        np.asarray(links, dtype=np.int64)
                    ].sum()
                )
                bottleneck = float(
                    np.min(state["scenario"]["link_capacity"][unique] / multiplicity)
                )
                path_latency = float(fraction) * (
                    delay
                    + float(state["config"]["traffic"]["decision_interval_ms"])
                    * demand
                    / bottleneck
                )
                transmission += path_latency
                parent_latency += path_latency
        state["edge_latency"][parent, current, replica] = parent_latency
    return transmission


def _finish_request(state: dict[str, Any]) -> dict[str, float]:
    service = state["service"]
    microservices, replicas = int(service["microservices"]), int(service["replicas"])
    critical_components = []
    for replica in range(replicas):
        paths = np.zeros((microservices, 3), dtype=np.float64)
        for microservice in range(microservices):
            parents = np.flatnonzero(np.asarray(service["adjacency"])[:, microservice])
            if len(parents):
                candidates = paths[parents].copy()
                candidates[:, 2] += state["edge_latency"][
                    parents, microservice, replica
                ]
                predecessor = candidates[np.argmax(candidates.sum(axis=1))]
                paths[microservice] = predecessor
            paths[microservice, :2] += state["task_latency"][microservice, replica]
        critical_components.append(paths[np.argmax(paths.sum(axis=1))])
    processing, queueing, transmission = np.mean(critical_components, axis=0)
    latency = {
        "access": float(state["access_latency"]),
        "processing": float(processing),
        "queueing": float(queueing),
        "transmission": float(transmission),
    }
    total = float(sum(latency.values()))
    utilization = _node_utilization(state)
    metrics = {
        "latency_ms": total,
        "sla_violation": float(total > float(state["service"]["sla_ms"])),
        "resource_utilization": float(np.mean(utilization)),
        "deployed": 1.0,
        "observed": 1.0,
        **{f"{name}_ms": float(value) for name, value in latency.items()},
    }
    state["completed"].append(metrics)
    request_id = int(state["current_request"])
    state["active"].append(
        {
            "release_time": state["time_s"]
            + float(state["requests"]["lifetime"][request_id]),
            "nodes": state["current_nodes"],
            "links": state["current_links"],
        }
    )
    state["current_request"] = None
    return metrics


def _reject_infeasible_request(state: dict[str, Any]) -> None:
    """Reject an unschedulable request, release partial claims, and record its SLA failure."""

    for node, cpu, memory in state["current_nodes"]:
        state["residual_nodes"][node] += (cpu, memory)
    for link, bandwidth in state["current_links"]:
        state["residual_links"][link] += bandwidth
    utilization = _node_utilization(state)
    sla = float(state["service"]["sla_ms"])
    failed_latency = sla + float(state["config"]["traffic"]["decision_interval_ms"])
    metrics = {
        "latency_ms": failed_latency,
        "sla_violation": 1.0,
        "resource_utilization": float(np.mean(utilization)),
        "deployed": 0.0,
        "observed": 1.0,
        "access_ms": float(state["access_latency"]),
        "processing_ms": 0.0,
        "queueing_ms": 0.0,
        "transmission_ms": 0.0,
    }
    state["completed"].append(metrics)
    reward = request_objective(state["config"], metrics)["reward"]
    state["cumulative_reward"] += reward
    state["cumulative_cost"] -= reward
    state["current_request"] = None
    state["anomaly"] = True
    state["observation_cache"] = None


def _finish_horizon(state: dict[str, Any]) -> None:
    """Reject work that remains when the paper's 100-slot horizon closes."""

    if state["current_request"] is not None:
        _reject_infeasible_request(state)
    remaining = (
        len(state["queue"])
        + len(state["requests"]["arrival_time"])
        - int(state["next_request"])
    )
    utilization = _node_utilization(state)
    sla = float(state["service"]["sla_ms"])
    failed_latency = sla + float(state["config"]["traffic"]["decision_interval_ms"])
    for _ in range(remaining):
        metrics = {
            "latency_ms": failed_latency,
            "sla_violation": 1.0,
            "resource_utilization": float(np.mean(utilization)),
            "deployed": 0.0,
            "observed": 0.0,
            "access_ms": 0.0,
            "processing_ms": 0.0,
            "queueing_ms": 0.0,
            "transmission_ms": 0.0,
        }
        state["completed"].append(metrics)
        reward = request_objective(state["config"], metrics)["reward"]
        state["cumulative_reward"] += reward
        state["cumulative_cost"] -= reward
    state["queue"].clear()
    state["next_request"] = len(state["requests"]["arrival_time"])
    state["done"] = True
    state["observation_cache"] = None


def _advance_request_slot(state: dict[str, Any]) -> None:
    interval = float(state["config"]["traffic"]["decision_interval_ms"]) / 1000.0
    state["step_count"] += 1
    horizon = float(state["config"]["traffic"]["duration_s"])
    _advance(state, min(horizon, state["time_s"] + interval))
    if state["step_count"] >= state["max_steps"] or state["time_s"] >= horizon:
        _finish_horizon(state)
    else:
        _start_request(state)


def step_environment(
    state: dict[str, Any], action: dict[str, Any]
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    """Apply one sequential replica placement and its fractional routes."""

    if state["done"] or state["current_request"] is None:
        raise RuntimeError("cannot step a completed environment")
    cumulative_before = float(state["cumulative_reward"])
    completed_before = len(state["completed"])
    slot_before = int(state["step_count"])
    current_observation = _observation_data(state)
    if "placement_index" in action:
        placement_index = int(action["placement_index"])
        if placement_index < 0 or placement_index >= len(
            current_observation["placement_mask"]
        ):
            raise IndexError("placement_index is outside the regional candidate array")
        if "node_id" in action and int(action["node_id"]) != int(
            current_observation["candidate_node_ids"][placement_index]
        ):
            raise ValueError("placement_index and node_id select different ESPs")
    elif "node_id" in action:
        matches = np.flatnonzero(
            current_observation["candidate_node_ids"] == int(action["node_id"])
        )
        if not len(matches):
            raise ValueError("node_id is not in the current regional candidates")
        placement_index = int(matches[0])
    else:
        raise KeyError("action requires placement_index or node_id")
    if not bool(current_observation["placement_mask"][placement_index]):
        raise ValueError("selected placement is infeasible")
    routing = np.asarray(action["routing_fractions"], dtype=np.float64)
    microservices = int(state["service"]["microservices"])
    incoming = np.flatnonzero(current_observation["incoming_edge_mask"])
    if routing.shape == (len(incoming), 3):
        expanded = np.zeros((microservices, 3), dtype=np.float64)
        expanded[incoming] = routing
        routing = expanded
    elif routing.shape != (microservices, 3):
        raise ValueError(
            f"routing_fractions must have shape ({microservices}, 3) or ({len(incoming)}, 3)"
        )
    node = int(current_observation["candidate_node_ids"][placement_index])
    service = state["service"]
    current, replica = int(state["microservice_index"]), int(state["replica_index"])
    demand = np.asarray(
        (service["cpu"][current], service["memory"][current]), dtype=np.float64
    )
    utilization_before = (
        1.0 - state["residual_nodes"][node] / state["scenario"]["capacity"][node]
    )
    state["residual_nodes"][node] -= demand
    state["current_nodes"].append((node, float(demand[0]), float(demand[1])))
    state["placements"][current, replica] = node
    state["observation_cache"] = None
    processing = (
        float(state["config"]["traffic"]["decision_interval_ms"])
        * demand[0]
        / state["scenario"]["capacity"][node, 0]
    )
    queueing = (
        float(state["config"]["traffic"]["decision_interval_ms"])
        * utilization_before[0]
        / max(1e-3, 1.0 - utilization_before[0])
    )
    transmission = _reserve_routes(state, node, routing)
    if current == 0 and replica == 0:
        taxi = _taxi_position(state, int(state["current_taxi"]))
        distance = float(
            np.linalg.norm(state["scenario"]["esp_coordinates"][node] - taxi)
        ) / np.sqrt(2.0)
        state["access_latency"] = 1.0 + 4.0 * distance
    state["task_latency"][current, replica] = (processing, queueing)
    total_utilization = _node_utilization(state)
    access = state["access_latency"] if current == 0 and replica == 0 else 0.0
    incremental_latency = access + processing + queueing + transmission
    state["placement_steps"] += 1
    replica += 1
    completed_metrics: dict[str, float] | None = None
    completed_placement: np.ndarray | None = None
    if replica == int(service["replicas"]):
        replica = 0
        current += 1
    if current == microservices:
        completed_placement = state["placements"].copy()
        completed_metrics = _finish_request(state)
        reward = request_objective(state["config"], completed_metrics)["reward"]
        state["cumulative_reward"] += reward
        state["cumulative_cost"] -= reward
    else:
        state["microservice_index"], state["replica_index"] = current, replica
    if completed_metrics is not None:
        _advance_request_slot(state)
    resource_use = float(np.mean(total_utilization))
    latency_ms = (
        float(completed_metrics["latency_ms"])
        if completed_metrics is not None
        else float(incremental_latency)
    )
    sla_violation = (
        float(completed_metrics["sla_violation"])
        if completed_metrics is not None
        else 0.0
    )
    state["anomaly"] = bool(state["anomaly"] or sla_violation > 0.0)
    next_observation = observation(state)
    reward = float(state["cumulative_reward"]) - cumulative_before
    request_rewards = [
        request_objective(state["config"], metrics)["reward"]
        for metrics in state["completed"][completed_before:]
    ]
    info = {
        "completed_request": completed_metrics,
        "completed_placement": completed_placement,
        "time_s": state["time_s"],
        "latency_ms": latency_ms,
        "resource_use": resource_use,
        "sla_violation": sla_violation,
        "request_rewards": request_rewards,
        "slot_advance": int(state["step_count"]) - slot_before,
        "score": float(1.0 / (1.0 - reward)),
        "anomaly": state["anomaly"],
    }
    return next_observation, reward, bool(state["done"]), info


def episode_metrics(state: dict[str, Any]) -> dict[str, float]:
    completed = state["completed"]
    mean_cost = float(state["cumulative_cost"]) / max(1, len(completed))
    score = 1.0 / (1.0 + mean_cost)
    if not completed:
        return {
            "cumulative_reward": float(state["cumulative_reward"]),
            "reward": float(state["cumulative_reward"]),
            "score": score,
            "average_latency_ms": 0.0,
            "end_to_end_latency_ms": 0.0,
            "sla_violation_rate": 0.0,
            "sla_violation_rate_percent": 0.0,
            "resource_utilization": 0.0,
            "resource_utilization_percent": 0.0,
            "resource_use": 0.0,
            "completed_requests": 0.0,
        }
    reward = float(state["cumulative_reward"])
    deployed = [item for item in completed if bool(item["deployed"])]
    latency = (
        float(np.mean([item["latency_ms"] for item in deployed])) if deployed else 0.0
    )
    violations = float(np.mean([item["sla_violation"] for item in completed]))
    observed = [item for item in completed if bool(item["observed"])]
    utilization = (
        float(np.mean([item["resource_utilization"] for item in observed]))
        if observed
        else 0.0
    )
    return {
        "cumulative_reward": reward,
        "reward": reward,
        "score": score,
        "average_latency_ms": latency,
        "end_to_end_latency_ms": latency,
        "sla_violation_rate": violations,
        "sla_violation_rate_percent": 100.0 * violations,
        "resource_utilization": utilization,
        "resource_utilization_percent": 100.0 * utilization,
        "resource_use": utilization,
        "completed_requests": float(len(completed)),
    }


def request_objective(
    config: dict[str, Any], metrics: dict[str, float]
) -> dict[str, float]:
    """Return the planner utility and reward for one completed request."""

    cost = float(config["objective"]["latency_weight"]) * float(
        metrics["latency_ms"]
    ) / float(config["service"]["sla_ms"]) + float(
        config["objective"]["resource_weight"]
    ) * float(metrics["resource_utilization"])
    return {"utility": 1.0 / (1.0 + cost), "reward": -cost}


def planner_query(state: dict[str, Any], embedding: np.ndarray) -> dict[str, Any]:
    """Expose semantic-planner inputs without leaking simulator internals."""

    current = observation(state)
    if state["current_request"] is None:
        raise RuntimeError("a terminal state has no deployment plan")
    service = state["service"]
    replicas = int(service["replicas"])
    microservices = int(service["microservices"])
    start = int(state["microservice_index"]) * replicas + int(state["replica_index"])
    region, _ = _region_nodes(state)
    residual = state["residual_nodes"][region]
    available = state["scenario"]["node_available"][region]
    assignments = []
    assignment_demands = []
    feasible_nodes = []
    feasible_indices = []
    for flat_index in range(start, microservices * replicas):
        microservice, replica = divmod(flat_index, replicas)
        demand = np.asarray(
            (service["cpu"][microservice], service["memory"][microservice]),
            dtype=np.float64,
        )
        mask = available & np.all(residual + 1e-9 >= demand, axis=1)
        if flat_index == start:
            mask &= np.asarray(current["placement_mask"][: len(region)], dtype=bool)
        assignments.append({"microservice": microservice, "replica": replica})
        assignment_demands.append(demand.tolist())
        feasible_nodes.append(region[mask].tolist())
        feasible_indices.append(np.flatnonzero(mask).tolist())
    edges = np.argwhere(np.asarray(state["service"]["adjacency"]) > 0).tolist()
    node_ratio = state["residual_nodes"] / state["scenario"]["capacity"]
    link_ratio = state["residual_links"] / state["scenario"]["link_capacity"]
    return {
        "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
        "telemetry": {
            "time_s": float(state["time_s"]),
            "queue_depth": len(state["queue"]),
            "active_requests": len(state["active"]),
            "node_residual_mean": node_ratio.mean(axis=0).tolist(),
            "node_residual_min": node_ratio.min(axis=0).tolist(),
            "link_residual_mean": float(link_ratio.mean()),
            "link_residual_min": float(link_ratio.min()),
            "placement": state["placements"].tolist()
            if state["current_request"] is not None
            else [],
        },
        "intent": {
            "microservice": int(current["microservice_index"]),
            "replica": int(current["replica_index"]),
            "sla_ms": float(state["service"]["sla_ms"]),
        },
        "dag": {"nodes": int(state["service"]["microservices"]), "edges": edges},
        "placement": state["placements"].tolist()
        if state["current_request"] is not None
        else [],
        "assignments": assignments,
        "assignment_demands": assignment_demands,
        "feasible_nodes": feasible_nodes,
        "feasible_indices": feasible_indices,
        "resources": {
            "node_ids": region.tolist(),
            "residual": residual.tolist(),
        },
    }


def reset(
    config: dict[str, Any], scenario: dict[str, np.ndarray], scale: Any, seed: int
) -> dict[str, Any]:
    """Reset from a scale mapping or a ``(microservices, replicas)`` pair."""

    if isinstance(scale, dict):
        selected = scale
    elif isinstance(scale, (tuple, list)) and len(scale) == 2:
        selected = {"microservices": int(scale[0]), "replicas": int(scale[1])}
    else:
        raise TypeError(
            "scale must be a paper scale mapping or a (microservices, replicas) pair"
        )
    return reset_environment(config, scenario, selected, seed)


# Short aliases retained for direct, framework-free use.
observe = observation
step = step_environment
