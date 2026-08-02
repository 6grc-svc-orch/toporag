"""Deterministic paper-aligned request and mobility generation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from env.constants import WORKLOAD_CLASSES


def _lognormal_parameters(mean: float, std: float) -> tuple[float, float]:
    variance = std * std
    sigma2 = math.log(1.0 + variance / (mean * mean))
    return math.log(mean) - sigma2 / 2.0, math.sqrt(sigma2)


def _weighted_choice(rng: np.random.Generator, weights: np.ndarray) -> int:
    cumulative = np.cumsum(weights / weights.sum())
    return int(np.searchsorted(cumulative, rng.random(), side="right"))


def _controller_density(topology: dict[str, Any], controller_id: int) -> str:
    edge_ids = topology["controller_regions"][controller_id]
    center = topology["coordinates_km"][edge_ids].mean(axis=0)
    area_center = topology["coordinates_km"][: topology["edge_server_count"]].mean(axis=0)
    diagonal = np.linalg.norm(
        np.ptp(topology["coordinates_km"][: topology["edge_server_count"]], axis=0)
    )
    relative_distance = np.linalg.norm(center - area_center) / max(diagonal / 2.0, 1e-9)
    if relative_distance < 0.35:
        return "urban"
    if relative_distance < 0.70:
        return "medium"
    return "suburban"


def _mobility_vector(rng: np.random.Generator) -> tuple[str, np.ndarray]:
    labels = ("static", "walking", "driving", "accelerated")
    speeds_kmh = np.array([0.0, 4.8, 45.0, 90.0], dtype=np.float64)
    label_index = _weighted_choice(rng, np.array([0.20, 0.35, 0.35, 0.10]))
    angle = rng.uniform(0.0, 2.0 * math.pi)
    speed_km_s = speeds_kmh[label_index] / 3600.0
    velocity = speed_km_s * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    return labels[label_index], velocity


def update_request_locations(
    requests: list[dict[str, Any]],
    elapsed_s: float,
    area_km: list[float],
) -> None:
    """Advance request locations deterministically and reflect at city boundaries."""

    limits = np.asarray(area_km, dtype=np.float64)
    for request in requests:
        location = request["source_location_km"] + request["velocity_km_s"] * elapsed_s
        period = 2.0 * limits
        reflected = np.mod(location, period)
        request["source_location_km"] = np.where(reflected <= limits, reflected, period - reflected)


def generate_request_trace(
    config: dict[str, Any],
    topology: dict[str, Any],
    services: list[dict[str, Any]],
    rng: np.random.Generator,
    overrides: dict[str, Any],
    external_events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate arrivals across the complete configured observation horizon."""

    simulation = config["simulation"]
    workload = config["workloads"]
    if simulation["source"] == "external":
        if external_events is None:
            raise ValueError("external scenario requires parsed WSDOT events")
        return _external_request_trace(config, topology, services, rng, overrides, external_events)
    duration_s = float(overrides["duration_s"]) if "duration_s" in overrides else float(simulation["duration_s"])
    controller_count = int(simulation["edge_controllers"])
    density_override = overrides.get("density")
    if density_override is not None and density_override not in {"urban", "medium", "suburban"}:
        raise ValueError("scenario override density must be urban, medium, or suburban")

    densities = [density_override or _controller_density(topology, controller_id) for controller_id in range(controller_count)]
    rates = np.array(
        [np.mean(workload["arrival_rates"][density]) for density in densities],
        dtype=np.float64,
    )
    if "arrival_rate" in overrides:
        rate = float(overrides["arrival_rate"])
        if rate <= 0:
            raise ValueError("scenario override arrival_rate must be positive")
        rates[:] = rate / controller_count

    pareto = workload["pareto"]
    burst_enabled = bool(overrides["burst"]) if "burst" in overrides else True
    burst_on = burst_enabled
    burst_change_s = 0.0
    current_time = 0.0
    requests: list[dict[str, Any]] = []
    class_weights = np.array([workload["class_mix"][name] for name in WORKLOAD_CLASSES], dtype=np.float64)
    service_by_class = {
        name: [service for service in services if service["class"] == name]
        for name in WORKLOAD_CLASSES
    }
    payload_mu, payload_sigma = _lognormal_parameters(
        float(workload["payload_lognormal_mb"]["mean"]),
        float(workload["payload_lognormal_mb"]["std"]),
    )
    cross_region_range = simulation["cross_region_fraction"]
    cross_region_probability = float(rng.uniform(cross_region_range[0], cross_region_range[1]))

    while current_time <= duration_s:
        if current_time >= burst_change_s:
            if burst_on:
                burst_change_s = current_time + float(pareto["on_scale_s"]) * (rng.pareto(float(pareto["on_shape"])) + 1.0)
            else:
                burst_change_s = current_time + float(pareto["off_scale_s"]) * (rng.pareto(float(pareto["off_shape"])) + 1.0)
            burst_on = not burst_on
        multiplier = float(pareto["burst_multiplier"]) if burst_enabled and burst_on else 1.0
        total_rate = float(rates.sum() * multiplier)
        current_time += float(rng.exponential(1.0 / total_rate))
        if current_time > duration_s:
            break

        controller_id = _weighted_choice(rng, rates)
        region_nodes = topology["controller_regions"][controller_id]
        entry_node = int(region_nodes[int(rng.integers(0, len(region_nodes)))])
        location = topology["coordinates_km"][entry_node] + rng.normal(0.0, 0.08, size=2)
        location = np.clip(location, np.zeros(2), np.asarray(simulation["area_km"], dtype=np.float64))
        workload_class = WORKLOAD_CLASSES[_weighted_choice(rng, class_weights)]
        candidates = service_by_class[workload_class]
        service = candidates[int(rng.integers(0, len(candidates)))]
        mobility_label, velocity = _mobility_vector(rng)
        collaborator = controller_id
        if rng.random() < cross_region_probability:
            offset = int(rng.integers(1, controller_count))
            collaborator = (controller_id + offset) % controller_count

        replica_counts = np.array(service["replica_limits"], dtype=np.int64)
        if "replicas" in overrides:
            replica_count = int(overrides["replicas"])
            if replica_count < 1:
                raise ValueError("scenario override replicas must be positive")
            replica_counts[:] = replica_count
        request = {
            "request_id": len(requests),
            "service_id": int(service["service_id"]),
            "class": workload_class,
            "slice_id": int(service["slice_id"]),
            "arrival_time_s": current_time,
            "lifetime_s": float(rng.uniform(workload["lifetime_s"][0], workload["lifetime_s"][1])),
            "source_location_km": location.astype(np.float64),
            "entry_node": entry_node,
            "controller_id": controller_id,
            "collaborating_controller_id": collaborator,
            "mobility": mobility_label,
            "velocity_km_s": velocity,
            "payload_multiplier": float(rng.lognormal(payload_mu, payload_sigma) / workload["payload_lognormal_mb"]["mean"]),
            "replica_counts": replica_counts,
        }
        requests.append(request)

    if not requests:
        raise RuntimeError("configured trace duration produced no service requests")
    requests[0]["arrival_time_s"] = 0.0
    return requests


def _external_request_trace(
    config: dict[str, Any],
    topology: dict[str, Any],
    services: list[dict[str, Any]],
    rng: np.random.Generator,
    overrides: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn WSDOT observations into mobility-aware semantic requests."""

    simulation = config["simulation"]
    workload = config["workloads"]
    duration_s = float(overrides["duration_s"]) if "duration_s" in overrides else float(simulation["duration_s"])
    selected = events
    count = len(selected)
    timestamps = np.asarray([event["timestamp_s"] for event in selected], dtype=np.float64)
    timestamps -= timestamps[0]
    if "arrival_rate" in overrides:
        timestamps *= count / max(float(overrides["arrival_rate"]) * max(timestamps[-1], 1e-9), 1e-9)
    elif timestamps[-1] > duration_s:
        timestamps *= duration_s / timestamps[-1]
    lat_lon = np.asarray([[event["latitude"], event["longitude"]] for event in selected], dtype=np.float64)
    span = np.maximum(np.ptp(lat_lon, axis=0), 1e-9)
    locations = (lat_lon - lat_lon.min(axis=0)) / span * np.asarray(simulation["area_km"], dtype=np.float64)
    vehicle_counts = np.asarray([event["vehicle_count"] for event in selected], dtype=np.float64)
    mean_vehicle_count = max(float(vehicle_counts.mean()), 1.0)
    edge_coordinates = topology["coordinates_km"][: topology["edge_server_count"]]
    service_by_class = {
        name: [service for service in services if service["class"] == name]
        for name in WORKLOAD_CLASSES
    }
    requests = []
    for index, (event, arrival_time, location) in enumerate(zip(selected, timestamps, locations)):
        label = event["semantic_label"]
        if "holo" in label or "volumetric" in label:
            workload_class = "holographic"
        elif "twin" in label or "sensor" in label or "industrial" in label:
            workload_class = "digital_twin"
        else:
            workload_class = "xr"
        candidates = service_by_class[workload_class]
        service = candidates[index % len(candidates)]
        entry_node = int(np.argmin(np.linalg.norm(edge_coordinates - location, axis=1)))
        controller_id = int(topology["node_controller_ids"][entry_node])
        if index + 1 < len(locations):
            direction = locations[index + 1] - location
        else:
            direction = location - locations[index - 1]
        norm = np.linalg.norm(direction)
        speed_km_s = float(event["speed_kmh"]) / 3600.0
        velocity = np.zeros(2, dtype=np.float64) if norm == 0 else direction / norm * speed_km_s
        replica_counts = np.array(service["replica_limits"], dtype=np.int64)
        if "replicas" in overrides:
            replica_counts[:] = int(overrides["replicas"])
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else duration_s
        lifetime = max(float(next_time - arrival_time), workload["lifetime_s"][0])
        requests.append(
            {
                "request_id": index,
                "service_id": int(service["service_id"]),
                "class": workload_class,
                "slice_id": int(service["slice_id"]),
                "arrival_time_s": float(arrival_time),
                "lifetime_s": float(min(lifetime, workload["lifetime_s"][1])),
                "source_location_km": location.astype(np.float64),
                "entry_node": entry_node,
                "controller_id": controller_id,
                "collaborating_controller_id": int(
                    topology["node_controller_ids"][
                        int(np.argmin(np.linalg.norm(edge_coordinates - (location + velocity * simulation["mobility_update_s"]), axis=1)))
                    ]
                ),
                "mobility": "wsdot_vehicle",
                "velocity_km_s": velocity,
                "payload_multiplier": max(float(event["vehicle_count"]) / mean_vehicle_count, 0.1),
                "replica_counts": replica_counts,
            }
        )
    if not requests:
        raise RuntimeError("WSDOT adapter produced no requests")
    return requests
