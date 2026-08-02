"""Paper-scale topology and general service-DAG construction functions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.constants import RADIO_PROFILES, RESOURCE_NAMES, SLICE_NAMES, WORKLOAD_CLASSES


def validate_external_sources(config: dict[str, Any]) -> dict[str, Path]:
    """Validate explicitly selected OpenCellID, WSDOT, and Alibaba adapters."""

    configured = config["simulation"]["external_data"]
    paths = {name: Path(value) for name, value in configured.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("external scenario selected but required inputs are absent: " + ", ".join(missing))

    with paths["opencellid_csv"].open("r", encoding="utf-8", newline="") as handle:
        fields = set(csv.DictReader(handle).fieldnames or [])
    if not ({"lat", "lon"} <= fields or {"latitude", "longitude"} <= fields):
        raise ValueError("OpenCellID CSV must contain lat/lon or latitude/longitude columns")
    with paths["wsdot_manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    events = manifest["events"] if isinstance(manifest, dict) and "events" in manifest else manifest
    if not isinstance(events, list) or len(events) < 2:
        raise ValueError("WSDOT manifest must contain at least two events")
    required_event = {"timestamp_s", "latitude", "longitude", "speed_kmh", "vehicle_count", "semantic_label"}
    missing_event = required_event - set(events[0]) if isinstance(events[0], dict) else required_event
    if missing_event:
        raise ValueError(
            "WSDOT events require timestamp_s, latitude, longitude, speed_kmh, "
            f"vehicle_count, and semantic_label; missing {sorted(missing_event)}"
        )
    with paths["alibaba_trace"].open("r", encoding="utf-8", newline="") as handle:
        alibaba_fields = set(csv.DictReader(handle).fieldnames or [])
        if not alibaba_fields:
            raise ValueError("Alibaba trace CSV must contain a header")
    adapted_columns = {
        "job_id", "task_id", "parent_task_id", "cpu_vcpu", "memory_gb",
        "gpu_units", "npu_units", "workload_mi", "payload_mb",
    }
    native_columns = {
        "task_name", "instance_num", "job_name", "task_type", "status",
        "start_time", "end_time", "plan_cpu", "plan_mem",
    }
    if not (adapted_columns <= alibaba_fields or native_columns <= alibaba_fields):
        raise ValueError(
            "Alibaba CSV must use either the adapted columns "
            f"{sorted(adapted_columns)} or native batch_task columns {sorted(native_columns)}"
        )
    return paths


def load_wsdot_events(config: dict[str, Any], paths: dict[str, Path] | None = None) -> list[dict[str, Any]]:
    """Parse WSDOT mobility/semantic events used by external request generation."""

    selected_paths = validate_external_sources(config) if paths is None else paths
    with selected_paths["wsdot_manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    events = manifest["events"] if isinstance(manifest, dict) else manifest
    parsed = []
    for index, event in enumerate(events):
        timestamp = float(event["timestamp_s"])
        latitude = float(event["latitude"])
        longitude = float(event["longitude"])
        speed = float(event["speed_kmh"])
        vehicle_count = int(event["vehicle_count"])
        label = str(event["semantic_label"]).strip().lower()
        if timestamp < 0 or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"invalid WSDOT coordinates/time at event {index}")
        if speed < 0 or vehicle_count < 0 or not label:
            raise ValueError(f"invalid WSDOT speed/count/label at event {index}")
        parsed.append(
            {
                "timestamp_s": timestamp,
                "latitude": latitude,
                "longitude": longitude,
                "speed_kmh": speed,
                "vehicle_count": vehicle_count,
                "semantic_label": label,
            }
        )
    parsed.sort(key=lambda event: event["timestamp_s"])
    return parsed


def _alibaba_rows(paths: dict[str, Path]) -> tuple[list[dict[str, str]], bool]:
    with paths["alibaba_trace"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    if not rows:
        raise ValueError("Alibaba trace contains no task records")
    return rows, "job_id" in fields


def _external_base_station_coordinates(
    config: dict[str, Any], paths: dict[str, Path]
) -> np.ndarray:
    base_stations = int(config["simulation"]["base_stations"])
    rows: list[tuple[float, float]] = []
    with paths["opencellid_csv"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        latitude_key = "lat" if "lat" in (reader.fieldnames or []) else "latitude"
        longitude_key = "lon" if "lon" in (reader.fieldnames or []) else "longitude"
        for row in reader:
            latitude = float(row[latitude_key])
            longitude = float(row[longitude_key])
            if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                rows.append((latitude, longitude))
    if len(rows) < base_stations:
        raise ValueError(
            f"OpenCellID adapter needs at least {base_stations} valid tower rows"
        )
    raw = np.unique(np.asarray(rows, dtype=np.float64), axis=0)
    if raw.shape[0] < base_stations:
        raise ValueError(
            f"OpenCellID adapter needs at least {base_stations} unique towers"
        )
    selected = raw[np.linspace(0, raw.shape[0] - 1, base_stations, dtype=np.int64)]
    span = np.maximum(selected.max(axis=0) - selected.min(axis=0), 1e-9)
    return (selected - selected.min(axis=0)) / span * np.asarray(config["simulation"]["area_km"])


def _resource_matrix(rng: np.random.Generator, count: int, ranges: dict[str, list[float]]) -> np.ndarray:
    return np.column_stack(
        [rng.uniform(ranges[name][0], ranges[name][1], size=count) for name in RESOURCE_NAMES]
    ).astype(np.float64)


def _capacity_matrix(rng: np.random.Generator, count: int, ranges: dict[str, list[float]]) -> np.ndarray:
    capacity = _resource_matrix(rng, count, ranges)
    for resource_index in (2, 3):
        resource_name = RESOURCE_NAMES[resource_index]
        lower = int(np.ceil(ranges[resource_name][0]))
        upper = int(np.floor(ranges[resource_name][1]))
        if lower > upper:
            raise ValueError(f"{resource_name} capacity range contains no whole device count")
        capacity[:, resource_index] = rng.integers(lower, upper + 1, size=count)
    return capacity


def _synthetic_base_station_coordinates(
    config: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    count = int(config["simulation"]["base_stations"])
    area = np.asarray(config["simulation"]["area_km"], dtype=np.float64)
    hotspot_count = max(4, int(config["simulation"]["edge_controllers"]) // 2)
    hotspot_centers = rng.uniform(area * 0.15, area * 0.85, size=(hotspot_count, 2))
    coordinates = np.empty((count, 2), dtype=np.float64)
    for node_id in range(count):
        if rng.random() < 0.72:
            center = hotspot_centers[int(rng.integers(0, hotspot_count))]
            coordinates[node_id] = rng.normal(center, area * 0.07)
        else:
            coordinates[node_id] = rng.uniform(np.zeros(2), area)
    return np.clip(coordinates, np.zeros(2), area)


def _cluster_base_stations(
    coordinates: np.ndarray, cluster_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically form the paper's edge-server catchments."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("base-station coordinates must have shape (count, 2)")
    if cluster_count < 1 or len(coordinates) < cluster_count:
        raise ValueError("edge-server cluster count exceeds the base-station count")

    ordering = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    seeds = ordering[
        np.linspace(0, len(ordering) - 1, cluster_count, dtype=np.int64)
    ]
    centroids = coordinates[seeds].copy()
    assignments = np.zeros(len(coordinates), dtype=np.int64)
    for _ in range(30):
        distances = np.sum(
            np.square(coordinates[:, None, :] - centroids[None, :, :]), axis=2
        )
        updated_assignments = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(updated_assignments, assignments):
            break
        assignments = updated_assignments
        nearest_distance = distances[np.arange(len(coordinates)), assignments]
        for cluster_id in range(cluster_count):
            members = coordinates[assignments == cluster_id]
            if len(members):
                centroids[cluster_id] = members.mean(axis=0)
            else:
                replacement = int(np.argmax(nearest_distance))
                centroids[cluster_id] = coordinates[replacement]
                assignments[replacement] = cluster_id
                nearest_distance[replacement] = -1.0
    return centroids, assignments


def _controller_regions(edge_coordinates: np.ndarray, count: int) -> tuple[list[np.ndarray], np.ndarray]:
    ordering = np.lexsort((edge_coordinates[:, 1], edge_coordinates[:, 0]))
    regions = [part.astype(np.int64) for part in np.array_split(ordering, count)]
    controller_coordinates = np.vstack([edge_coordinates[region].mean(axis=0) for region in regions])
    return regions, controller_coordinates


def _add_link(
    links: set[tuple[int, int]],
    source: int,
    destination: int,
) -> None:
    if source != destination:
        links.add((min(source, destination), max(source, destination)))


def generate_topology(
    config: dict[str, Any],
    rng: np.random.Generator,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Construct a connected multi-tier substrate with slice-partitioned links."""

    simulation = config["simulation"]
    edge_count = int(simulation["edge_servers"])
    controller_count = int(simulation["edge_controllers"])
    core_count = int(simulation["core_nodes"])
    if simulation["source"] == "external":
        paths = validate_external_sources(config)
        base_station_coordinates = _external_base_station_coordinates(config, paths)
    else:
        base_station_coordinates = _synthetic_base_station_coordinates(config, rng)
    edge_coordinates, base_station_edge_ids = _cluster_base_stations(
        base_station_coordinates, edge_count
    )
    regions, controller_coordinates = _controller_regions(edge_coordinates, controller_count)
    area = np.asarray(simulation["area_km"], dtype=np.float64)
    core_coordinates = np.tile(area / 2.0, (core_count, 1))
    coordinates = np.vstack([edge_coordinates, controller_coordinates, core_coordinates])
    node_count = coordinates.shape[0]
    controller_node_ids = np.arange(edge_count, edge_count + controller_count, dtype=np.int64)
    core_node_ids = np.arange(edge_count + controller_count, node_count, dtype=np.int64)

    capacity = np.vstack(
        [
            _capacity_matrix(rng, edge_count, simulation["edge_server_resources"]),
            _capacity_matrix(rng, controller_count, simulation["controller_resources"]),
            _capacity_matrix(rng, core_count, simulation["core_resources"]),
        ]
    )
    tiers = np.concatenate(
        [np.zeros(edge_count), np.ones(controller_count), np.full(core_count, 2)]
    ).astype(np.int8)
    node_controllers = np.full(node_count, -1, dtype=np.int64)
    for controller_id, region in enumerate(regions):
        node_controllers[region] = controller_id
        node_controllers[controller_node_ids[controller_id]] = controller_id

    radio_profiles = np.zeros((node_count, len(RADIO_PROFILES)), dtype=np.float64)
    radio_profiles[:edge_count, 0] = 1.0
    radio_profiles[:edge_count, 1] = (rng.random(edge_count) < 0.72).astype(np.float64)
    radio_profiles[:edge_count, 2] = (rng.random(edge_count) < 0.18).astype(np.float64)
    radio_profiles[edge_count : edge_count + controller_count, :2] = 1.0
    radio_profiles[core_node_ids, 0] = 1.0
    latency_ms = np.empty((node_count, 3), dtype=np.float64)
    latency_ms[:edge_count] = np.column_stack(
        [rng.uniform(0.10, 0.80, edge_count), rng.uniform(0.20, 1.20, edge_count), rng.uniform(0.50, 2.50, edge_count)]
    )
    latency_ms[controller_node_ids] = np.column_stack(
        [rng.uniform(0.08, 0.40, controller_count), rng.uniform(0.20, 0.80, controller_count), rng.uniform(0.30, 1.20, controller_count)]
    )
    latency_ms[core_node_ids] = np.column_stack(
        [rng.uniform(0.02, 0.08, core_count), rng.uniform(0.10, 0.30, core_count), rng.uniform(1.00, 3.00, core_count)]
    )

    link_pairs: set[tuple[int, int]] = set()
    neighbors = int(simulation["nearest_neighbors"])
    distances = np.linalg.norm(edge_coordinates[:, None, :] - edge_coordinates[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    for source in range(edge_count):
        for destination in np.argpartition(distances[source], neighbors)[:neighbors]:
            _add_link(link_pairs, source, int(destination))
    for controller_id, region in enumerate(regions):
        controller_node = int(controller_node_ids[controller_id])
        for edge_node in region:
            _add_link(link_pairs, int(edge_node), controller_node)
        next_controller = int(controller_node_ids[(controller_id + 1) % controller_count])
        _add_link(link_pairs, controller_node, next_controller)
        for core_node in core_node_ids:
            _add_link(link_pairs, controller_node, int(core_node))

    ordered_links = sorted(link_pairs)
    link_sources = np.array([pair[0] for pair in ordered_links], dtype=np.int64)
    link_destinations = np.array([pair[1] for pair in ordered_links], dtype=np.int64)
    link_count = len(ordered_links)
    total_bandwidth_mbps = np.empty(link_count, dtype=np.float64)
    delay = np.empty(link_count, dtype=np.float64)
    reliability = np.empty(link_count, dtype=np.float64)
    for index, (source, destination) in enumerate(ordered_links):
        maximum_tier = max(int(tiers[source]), int(tiers[destination]))
        if maximum_tier == 0:
            bandwidth_range = simulation["edge_bandwidth_gbps"]
            processing_delay = 0.15
        elif maximum_tier == 1:
            bandwidth_range = simulation["controller_bandwidth_gbps"]
            processing_delay = 0.25
        else:
            bandwidth_range = simulation["core_bandwidth_gbps"]
            processing_delay = 0.50
        total_bandwidth_mbps[index] = rng.uniform(bandwidth_range[0], bandwidth_range[1]) * 1000.0
        distance_km = np.linalg.norm(coordinates[source] - coordinates[destination])
        delay[index] = processing_delay + distance_km / 200.0
        reliability[index] = rng.uniform(0.9985, 0.999999)
    slice_shares = np.array([simulation["slice_shares"][name] for name in SLICE_NAMES], dtype=np.float64)
    slice_capacity = total_bandwidth_mbps[:, None] * slice_shares[None, :]

    node_active = np.ones(node_count, dtype=bool)
    if "failure_rate" in overrides:
        failure_rate = float(overrides["failure_rate"])
        if not 0 <= failure_rate <= 100:
            raise ValueError("scenario override failure_rate must be a percentage in [0,100]")
        failed_count = round(edge_count * failure_rate / 100.0)
        if failed_count:
            failed = rng.choice(edge_count, failed_count, replace=False)
            node_active[failed] = False

    adjacency = np.zeros((node_count, node_count), dtype=np.float64)
    adjacency[link_sources, link_destinations] = 1.0
    adjacency[link_destinations, link_sources] = 1.0
    return {
        "node_ids": np.arange(node_count, dtype=np.int64),
        "node_tiers": tiers,
        "node_controller_ids": node_controllers,
        "coordinates_km": coordinates,
        "capacity": capacity,
        "radio_profiles": radio_profiles,
        "latency_ms": latency_ms,
        "node_active": node_active,
        "link_sources": link_sources,
        "link_destinations": link_destinations,
        "link_capacity_mbps": slice_capacity,
        "link_total_mbps": total_bandwidth_mbps,
        "link_delay_ms": delay,
        "link_reliability": reliability,
        "adjacency": adjacency,
        "controller_regions": regions,
        "controller_node_ids": controller_node_ids,
        "core_node_ids": core_node_ids,
        "edge_server_count": edge_count,
        "base_station_coordinates_km": base_station_coordinates,
        "base_station_edge_ids": base_station_edge_ids,
    }


def _class_counts(total: int, mix: dict[str, float]) -> dict[str, int]:
    raw = np.array([total * mix[name] for name in WORKLOAD_CLASSES], dtype=np.float64)
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts))[: total - int(counts.sum())]:
        counts[index] += 1
    return {name: int(counts[index]) for index, name in enumerate(WORKLOAD_CLASSES)}


def _general_dag(node_count: int, probability: float, rng: np.random.Generator) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for destination in range(1, node_count):
        parent = int(rng.integers(0, destination))
        edges.add((parent, destination))
    for source in range(node_count - 1):
        for destination in range(source + 1, node_count):
            if destination > source + 1 and rng.random() < probability:
                edges.add((source, destination))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def _external_service_catalog(
    config: dict[str, Any],
    rng: np.random.Generator,
    overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    paths = validate_external_sources(config)
    rows, adapted = _alibaba_rows(paths)
    job_key = "job_id" if adapted else "job_name"
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[job_key], []).append(row)
    usable = [items for items in grouped.values() if len(items) >= 2]
    catalog_size = int(config["workloads"]["catalog_size"])
    if len(usable) < catalog_size:
        raise ValueError(
            f"Alibaba adapter needs {catalog_size} jobs with at least two tasks; found {len(usable)}"
        )
    class_counts = _class_counts(catalog_size, config["workloads"]["class_mix"])
    class_sequence = [name for name in WORKLOAD_CLASSES for _ in range(class_counts[name])]
    slice_for_class = {"xr": 0, "digital_twin": 2, "holographic": 0}
    services = []
    for service_id, (job_rows, class_name) in enumerate(zip(usable[:catalog_size], class_sequence)):
        spec = config["workloads"]["classes"][class_name]
        if adapted:
            ordered = sorted(job_rows, key=lambda row: int(row["task_id"]))
        else:
            ordered = sorted(job_rows, key=lambda row: (float(row["start_time"]), row["task_name"]))
        requested_count = int(overrides["microservices"]) if "microservices" in overrides else len(ordered)
        requested_count = min(requested_count, int(config["training"]["max_microservices"]))
        if requested_count < 2:
            raise ValueError("Alibaba jobs must provide at least two retained microservices")
        if len(ordered) < requested_count:
            raise ValueError(
                f"Alibaba job {service_id} has {len(ordered)} tasks but override requests {requested_count}"
            )
        ordered = ordered[:requested_count]
        demands = np.zeros((requested_count, len(RESOURCE_NAMES)), dtype=np.float64)
        workloads_mi = np.zeros(requested_count, dtype=np.float64)
        payloads_by_task = np.zeros(requested_count, dtype=np.float64)
        for task_index, row in enumerate(ordered):
            if adapted:
                demands[task_index] = [float(row[name]) for name in RESOURCE_NAMES]
                workloads_mi[task_index] = float(row["workload_mi"])
                payloads_by_task[task_index] = float(row["payload_mb"])
            else:
                cpu = max(float(row["plan_cpu"]), 0.01)
                memory = max(float(row["plan_mem"]), 0.001)
                if cpu > config["simulation"]["edge_server_resources"]["cpu_vcpu"][1]:
                    cpu /= 100.0
                if memory <= 1.0:
                    memory *= config["simulation"]["edge_server_resources"]["memory_gb"][1]
                runtime = max(float(row["end_time"]) - float(row["start_time"]), 0.001)
                demands[task_index] = [cpu, memory, 0.0, 0.0]
                workloads_mi[task_index] = cpu * runtime * config["simulation"]["mips_per_vcpu"]
                payloads_by_task[task_index] = max(memory * 0.05, 0.001)
        if np.any(demands < 0) or np.any(workloads_mi <= 0) or np.any(payloads_by_task <= 0):
            raise ValueError(f"Alibaba job {service_id} contains invalid demands/workload/payload")
        if adapted:
            task_to_index = {row["task_id"]: index for index, row in enumerate(ordered)}
            edges = set()
            for destination, row in enumerate(ordered):
                parent = row["parent_task_id"].strip()
                if parent and parent not in {"-1", "none", "null"}:
                    if parent not in task_to_index:
                        raise ValueError(f"Alibaba job {service_id} references missing parent task {parent!r}")
                    source = task_to_index[parent]
                    if source >= destination:
                        raise ValueError(f"Alibaba job {service_id} parent ordering is not acyclic")
                    edges.add((source, destination))
                elif destination:
                    edges.add((destination - 1, destination))
        else:
            # Native batch_task traces expose timing rather than explicit
            # parents.  Start-time ordering and overlapping predecessors form
            # a deterministic trace-derived DAG.
            edges = {(destination - 1, destination) for destination in range(1, requested_count)}
            for destination in range(2, requested_count):
                destination_start = float(ordered[destination]["start_time"])
                candidates = [
                    source
                    for source in range(destination - 1)
                    if float(ordered[source]["end_time"]) <= destination_start
                ]
                if candidates:
                    edges.add((candidates[-1], destination))
        edge_array = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
        replica_count = int(overrides["replicas"]) if "replicas" in overrides else int(spec["replicas"][0])
        replica_limits = np.full(requested_count, replica_count, dtype=np.int64)
        services.append(
            {
                "service_id": service_id,
                "class": class_name,
                "slice_id": slice_for_class[class_name],
                "microservice_ids": np.arange(requested_count, dtype=np.int64),
                "resource_demands": demands,
                "workloads_mi": workloads_mi,
                "replica_limits": replica_limits,
                "edges": edge_array,
                "edge_payload_mb": np.asarray(
                    [payloads_by_task[int(destination)] for _, destination in edge_array], dtype=np.float64
                ),
                "qos": {
                    "latency_ms": float(np.mean(spec["latency_ms"])),
                    "jitter_ms": float(np.mean(spec["jitter_ms"])),
                    "sync_ms": float(np.mean(spec["sync_ms"])),
                    "min_reliability": float(spec["min_reliability"]),
                },
            }
        )
    return services


def generate_service_catalog(
    config: dict[str, Any],
    rng: np.random.Generator,
    overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate representative XR, digital-twin, and holographic DAGs."""

    if config["simulation"]["source"] == "external":
        return _external_service_catalog(config, rng, overrides)
    workload = config["workloads"]
    counts = _class_counts(int(workload["catalog_size"]), workload["class_mix"])
    services: list[dict[str, Any]] = []
    slice_for_class = {"xr": 0, "digital_twin": 2, "holographic": 0}
    payload_mean = float(workload["payload_lognormal_mb"]["mean"])
    payload_std = float(workload["payload_lognormal_mb"]["std"])
    variance = payload_std * payload_std
    sigma2 = np.log(1.0 + variance / (payload_mean * payload_mean))
    log_mu = np.log(payload_mean) - sigma2 / 2.0
    log_sigma = np.sqrt(sigma2)
    for class_name in WORKLOAD_CLASSES:
        spec = workload["classes"][class_name]
        for _ in range(counts[class_name]):
            if "microservices" in overrides:
                microservice_count = int(overrides["microservices"])
                if microservice_count < 2 or microservice_count > int(config["training"]["max_microservices"]):
                    raise ValueError("scenario override microservices is outside supported [2,max_microservices]")
            else:
                microservice_count = int(rng.integers(int(spec["microservices"][0]), int(spec["microservices"][1]) + 1))
            demands = _resource_matrix(rng, microservice_count, spec["resources"])
            # Accelerators are heterogeneous options, not a compulsory fractional
            # claim on every microservice. CPU and memory remain mandatory.
            for resource_index in (2, 3):
                demands[rng.random(microservice_count) < 0.80, resource_index] = 0.0
            workloads_mi = rng.uniform(workload["cpu_workload_mi"][0], workload["cpu_workload_mi"][1], size=microservice_count)
            replica_limits = rng.integers(int(spec["replicas"][0]), int(spec["replicas"][1]) + 1, size=microservice_count, dtype=np.int64)
            if "replicas" in overrides:
                replica_limits[:] = int(overrides["replicas"])
            edges = _general_dag(microservice_count, float(workload["dag_extra_edge_probability"]), rng)
            service = {
                "service_id": len(services),
                "class": class_name,
                "slice_id": slice_for_class[class_name],
                "microservice_ids": np.arange(microservice_count, dtype=np.int64),
                "resource_demands": demands,
                "workloads_mi": workloads_mi.astype(np.float64),
                "replica_limits": replica_limits,
                "edges": edges,
                "edge_payload_mb": rng.lognormal(log_mu, log_sigma, size=edges.shape[0]).astype(np.float64),
                "qos": {
                    "latency_ms": float(rng.uniform(spec["latency_ms"][0], spec["latency_ms"][1])),
                    "jitter_ms": float(rng.uniform(spec["jitter_ms"][0], spec["jitter_ms"][1])),
                    "sync_ms": float(rng.uniform(spec["sync_ms"][0], spec["sync_ms"][1])),
                    "min_reliability": float(spec["min_reliability"]),
                },
            }
            services.append(service)
    return services


def json_ready(value: Any) -> Any:
    """Convert nested NumPy scenario values into JSON-native values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_scenario(scenario: dict[str, Any], path: str | Path) -> Path:
    """Persist a generated scenario with explicit units and provenance."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(scenario), handle, separators=(",", ":"))
    return output
