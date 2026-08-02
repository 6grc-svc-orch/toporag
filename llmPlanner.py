"""Cloud LLM planner for topology-aware 6G service orchestration.

Running this file initializes the configured trace memory and validates any
existing planner snapshot.  Model inference is deliberately exposed through
``run_planning_window`` so merely importing or running the file never downloads
the 7-B model.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from string import Formatter
from typing import Any

from rag import (
    QUERY_FIELDS,
    build_query_embedding,
    hashed_embedding,
    initialize_memory,
    memory_stats,
    new_trace_id,
    prune_memory,
    record_trace,
    retrieve_traces,
    successful_traces,
    validate_query,
)

PLANNER_FIELDS = {
    "backend",
    "model_id",
    "dtype",
    "planning_interval_s",
    "candidate_count",
    "prompt_input_tokens",
    "max_new_tokens",
    "prompt_path",
    "memory_path",
    "latest_plan_path",
    "snapshot_path",
    "retrieval_pool",
    "retrieval_top_k",
    "similarity_weights",
    "soft_prune_threshold",
    "memory_ttl_windows",
    "max_traces",
    "drift_threshold",
    "lora",
}
LORA_FIELDS = {"rank", "alpha", "dropout", "trainable_fraction"}
SIMILARITY_FIELDS = {"latent", "topology", "placement"}
PROMPT_FIELDS = {
    "candidate_count",
    "controller_id",
    "current_placement_json",
    "policy_id",
    "qos_intent_json",
    "region_id",
    "retrieved_traces_json",
    "service_dags_json",
    "sim_time_s",
    "slice_tag",
    "telemetry_json",
    "topology_json",
    "window_id",
}
RAW_PLAN_FIELDS = {"reward_weights", "penalty_weights", "cooperation_bias", "candidates"}
RAW_CANDIDATE_FIELDS = {"candidate_id", "node_preferences", "placement_strategy", "predicted_utility"}
EXPANDED_CANDIDATE_FIELDS = RAW_CANDIDATE_FIELDS | {"assignments"}
DISPATCH_CANDIDATE_FIELDS = EXPANDED_CANDIDATE_FIELDS | {"feasibility"}
PLACEMENT_STRATEGIES = {"latency_first", "capacity_first", "reliability_first", "spread", "consolidate"}
MAX_NODE_PREFERENCES = 8
REWARD_KEYS = {"latency", "resource_utilization", "load_balance", "reliability"}
PENALTY_KEYS = {"resource", "temporal", "robustness"}
FEASIBILITY_FIELDS = {"feasible", "violations"}
DISPATCH_PLAN_FIELDS = {
    "schema_version",
    "plan_id",
    "window_id",
    "controller_id",
    "region_id",
    "created_at",
    "valid_from_s",
    "valid_until_s",
    "backend",
    "model_id",
    "reward_weights",
    "penalty_weights",
    "cooperation_bias",
    "candidates",
    "selected_candidate_id",
    "selected_assignments",
    "placement_prior",
    "predicted_utility",
    "plan_embedding",
    "retrieval_trace_ids",
    "feasibility",
}
PLANNER_VISUAL_SIZE = 224
# Qwen2-VL merges 2x2 14-pixel patches, so one language-model visual
# token spans 28x28 pixels. Two 224px panels use exactly 128 visual
# tokens; 64 additional tokens cover image and chat markers.
PLANNER_VISUAL_WIDTH = 2 * PLANNER_VISUAL_SIZE
PLANNER_VISUAL_HEIGHT = PLANNER_VISUAL_SIZE
PLANNER_VISUAL_PIXELS = PLANNER_VISUAL_WIDTH * PLANNER_VISUAL_HEIGHT
PLANNER_VISUAL_TOKENS = (PLANNER_VISUAL_WIDTH // 28) * (PLANNER_VISUAL_HEIGHT // 28)
VISUAL_TOKEN_RESERVE = PLANNER_VISUAL_TOKENS + 64


def _require_exact_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - value.keys()
    unexpected = value.keys() - required
    if missing or unexpected:
        raise ValueError(f"{label} keys are invalid; missing={sorted(missing)}, unexpected={sorted(unexpected)}")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _replica_ids(microservice: dict[str, Any], replicas: int) -> list[int]:
    replica_ids = microservice["replica_ids"]
    if (
        not isinstance(replica_ids, list)
        or len(replica_ids) != replicas
        or any(
            isinstance(replica_id, bool)
            or not isinstance(replica_id, int)
            or replica_id < 0
            for replica_id in replica_ids
        )
        or len(set(replica_ids)) != replicas
    ):
        raise ValueError(
            "microservice replica_ids must contain distinct non-negative integers matching replicas"
        )
    return replica_ids


def _validate_distribution(weights: dict[str, Any], keys: set[str], label: str) -> dict[str, float]:
    if not isinstance(weights, dict):
        raise TypeError(f"{label} must be an object")
    _require_exact_keys(weights, keys, label)
    normalized = {key: _finite_number(weights[key], f"{label}.{key}") for key in sorted(keys)}
    if any(value < 0.0 or value > 1.0 for value in normalized.values()):
        raise ValueError(f"{label} values must be in [0, 1]")
    if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"{label} values must sum to 1")
    return normalized


def _validate_coefficients(weights: dict[str, Any], keys: set[str], label: str) -> dict[str, float]:
    if not isinstance(weights, dict):
        raise TypeError(f"{label} must be an object")
    _require_exact_keys(weights, keys, label)
    normalized = {key: _finite_number(weights[key], f"{label}.{key}") for key in sorted(keys)}
    if any(value < 0.0 or value > 1.0 for value in normalized.values()):
        raise ValueError(f"{label} values must be in [0, 1]")
    return normalized


def validate_planner_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict) or "planner" not in config:
        raise ValueError("configuration requires a planner section")
    planner = config["planner"]
    if not isinstance(planner, dict):
        raise TypeError("planner configuration must be an object")
    _require_exact_keys(planner, PLANNER_FIELDS, "planner configuration")
    if planner["backend"] != "qwen2_vl":
        raise ValueError("planner.backend must explicitly be qwen2_vl")
    for key in ("model_id", "prompt_path", "memory_path", "latest_plan_path", "snapshot_path"):
        if not isinstance(planner[key], str) or not planner[key]:
            raise ValueError(f"planner.{key} must be a non-empty string")
    if planner["dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("planner.dtype must be float16, bfloat16, or float32")
    planning_interval = _positive_integer(planner["planning_interval_s"], "planner.planning_interval_s")
    if not 60 <= planning_interval <= 3600:
        raise ValueError("planner.planning_interval_s must be in [60, 3600]")
    if _positive_integer(planner["candidate_count"], "planner.candidate_count") != 5:
        raise ValueError("planner.candidate_count must be 5")
    prompt_input_tokens = _positive_integer(planner["prompt_input_tokens"], "planner.prompt_input_tokens")
    if prompt_input_tokens <= VISUAL_TOKEN_RESERVE:
        raise ValueError(f"planner.prompt_input_tokens must exceed the {VISUAL_TOKEN_RESERVE}-token visual reserve")
    _positive_integer(planner["max_new_tokens"], "planner.max_new_tokens")
    if _positive_integer(planner["retrieval_pool"], "planner.retrieval_pool") != 64:
        raise ValueError("planner.retrieval_pool must be 64")
    if _positive_integer(planner["retrieval_top_k"], "planner.retrieval_top_k") != 5:
        raise ValueError("planner.retrieval_top_k must be 5")
    _positive_integer(planner["memory_ttl_windows"], "planner.memory_ttl_windows")
    _positive_integer(planner["max_traces"], "planner.max_traces")

    similarity = planner["similarity_weights"]
    if not isinstance(similarity, dict):
        raise TypeError("planner.similarity_weights must be an object")
    _require_exact_keys(similarity, SIMILARITY_FIELDS, "planner.similarity_weights")
    expected_similarity = {"latent": 0.5, "topology": 0.3, "placement": 0.2}
    for key, expected in expected_similarity.items():
        if not math.isclose(_finite_number(similarity[key], f"planner.similarity_weights.{key}"), expected):
            raise ValueError(f"planner.similarity_weights.{key} must be {expected}")

    threshold = _finite_number(planner["soft_prune_threshold"], "planner.soft_prune_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("planner.soft_prune_threshold must be in [0, 1]")
    drift = _finite_number(planner["drift_threshold"], "planner.drift_threshold")
    if not 0.0 < drift < 1.0:
        raise ValueError("planner.drift_threshold must be in (0, 1)")

    lora = planner["lora"]
    if not isinstance(lora, dict):
        raise TypeError("planner.lora must be an object")
    _require_exact_keys(lora, LORA_FIELDS, "planner.lora")
    lora_rank = _positive_integer(lora["rank"], "planner.lora.rank")
    if not 4 <= lora_rank <= 64:
        raise ValueError("planner.lora.rank must be in [4, 64]")
    if _positive_integer(lora["alpha"], "planner.lora.alpha") != 32:
        raise ValueError("planner.lora.alpha must be 32")
    if not math.isclose(_finite_number(lora["dropout"], "planner.lora.dropout"), 0.05):
        raise ValueError("planner.lora.dropout must be 0.05")
    trainable_fraction = _finite_number(lora["trainable_fraction"], "planner.lora.trainable_fraction")
    if not 0.10 <= trainable_fraction <= 0.50:
        raise ValueError("planner.lora.trainable_fraction must be in [0.10, 0.50]")
    return planner


def load_prompt(prompt_path: str) -> str:
    template = Path(prompt_path).read_text(encoding="utf-8")
    placeholders = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None}
    if placeholders != PROMPT_FIELDS:
        raise ValueError(
            f"planner prompt placeholders are invalid; missing={sorted(PROMPT_FIELDS - placeholders)}, "
            f"unexpected={sorted(placeholders - PROMPT_FIELDS)}"
        )
    return template


def _rounded_for_prompt(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rounded_for_prompt(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded_for_prompt(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def _topology_prompt_view(topology: dict[str, Any]) -> str:
    required = {
        "node_ids",
        "node_tiers",
        "controller_ids",
        "coordinates_km",
        "capacity",
        "residual_capacity",
        "radio_profiles",
        "latency_ms",
        "node_active",
        "links",
        "resource_names",
    }
    _require_exact_keys(topology, required, "planner topology")
    node_ids = topology["node_ids"]
    if not isinstance(node_ids, list):
        raise TypeError("planner topology node_ids must be a list")
    parallel_keys = (
        "node_tiers",
        "controller_ids",
        "coordinates_km",
        "capacity",
        "residual_capacity",
        "radio_profiles",
        "latency_ms",
        "node_active",
    )
    for key in parallel_keys:
        if not isinstance(topology[key], list) or len(topology[key]) != len(node_ids):
            raise ValueError(f"planner topology {key} must align with node_ids")
    if not isinstance(topology["links"], list) or not isinstance(topology["resource_names"], list):
        raise TypeError("planner topology links and resource_names must be lists")
    resource_names = topology["resource_names"]
    if not resource_names or not all(isinstance(name, str) for name in resource_names):
        raise TypeError("planner topology resource_names must contain strings")
    tier_nodes = {"S": [], "E": [], "C": []}
    active_rows = []
    tier_codes = {"edge_server": "S", "edge_controller": "E", "cloud_core": "C"}
    for index, node_id in enumerate(node_ids):
        capacity = topology["capacity"][index]
        residual = topology["residual_capacity"][index]
        if (
            not isinstance(capacity, list)
            or not isinstance(residual, list)
            or len(capacity) != len(resource_names)
            or len(residual) != len(resource_names)
        ):
            raise ValueError("planner topology capacity dimensions must align with resource_names")
        latency = topology["latency_ms"][index]
        if not isinstance(latency, list) or not latency:
            raise ValueError("planner topology latency_ms rows must be non-empty")
        if not topology["node_active"][index]:
            continue
        tier = str(topology["node_tiers"][index])
        if tier not in tier_codes:
            raise ValueError(f"unknown topology tier {tier}")
        capacity_values = [_finite_number(value, f"topology capacity {index}") for value in capacity]
        residual_values = [_finite_number(value, f"topology residual_capacity {index}") for value in residual]
        node_id_text = str(node_id)
        tier_nodes[tier_codes[tier]].append(node_id_text)
        active_rows.append(
            {
                "id": node_id_text,
                "capacity": capacity_values,
                "residual": residual_values,
                "latency": min(_finite_number(value, f"topology latency_ms {index}") for value in latency),
            }
        )
    tier_text = "|".join(f"{tier}:{','.join(node_ids)}" for tier, node_ids in tier_nodes.items())
    headroom = []
    for resource_index, resource_name in enumerate(resource_names):
        ranked = sorted(
            (row for row in active_rows if row["capacity"][resource_index] > 0.0),
            key=lambda row: (-row["residual"][resource_index], row["id"]),
        )[:5]
        headroom.append(f"{resource_name.split('_', 1)[0]}:{','.join(row['id'] for row in ranked)}")
    low_latency = sorted(active_rows, key=lambda row: (row["latency"], row["id"]))[:5]
    return f"tiers={tier_text};headroom_top5={'|'.join(headroom)};low_latency_top5={','.join(row['id'] for row in low_latency)}"


def _service_value(service: dict[str, Any], key: str, label: str) -> Any:
    if key not in service:
        raise ValueError(f"{label} requires {key}")
    return service[key]


def _base36(value: int) -> str:
    if value < 0:
        raise ValueError("base-36 planner values must be non-negative")
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value == 0:
        return "0"
    encoded = ""
    while value:
        value, remainder = divmod(value, 36)
        encoded = alphabet[remainder] + encoded
    return encoded


def _sequence_text(values: list[str]) -> str:
    if values and all(value.isdecimal() for value in values):
        integers = [int(value) for value in values]
        if integers == list(range(integers[0], integers[0] + len(integers))):
            return values[0] if len(values) == 1 else f"{values[0]}-{values[-1]}"
    return ",".join(values)


def _run_length_text(values: list[int]) -> str:
    runs = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return ",".join(str(value) if count == 1 else f"{value}x{count}" for value, count in runs)


def _service_dag_prompt_view(service_dags: list[dict[str, Any]]) -> str:
    views = []
    for service_index, service in enumerate(service_dags):
        if not isinstance(service, dict):
            raise TypeError(f"service_dags[{service_index}] must be an object")
        request_id = _service_value(service, "request_id", f"service_dags[{service_index}]")
        microservices = _service_value(service, "microservices", f"service_dags[{service_index}]")
        edges = _service_value(service, "edges", f"service_dags[{service_index}]")
        if not isinstance(microservices, list) or not isinstance(edges, list):
            raise TypeError("service DAG microservices and edges must be lists")
        if not microservices:
            raise ValueError(f"service_dags[{service_index}] requires at least one microservice")
        microservice_ids = []
        replica_counts = []
        centi_demands = []
        for microservice_index, microservice in enumerate(microservices):
            if not isinstance(microservice, dict):
                raise TypeError("service DAG microservices must contain objects")
            microservice_ids.append(
                str(
                    _service_value(
                        microservice,
                        "microservice_id",
                        f"service_dags[{service_index}].microservices[{microservice_index}]",
                    )
                )
            )
            replicas = _nonnegative_integer(
                _service_value(
                    microservice,
                    "replicas",
                    f"service_dags[{service_index}].microservices[{microservice_index}]",
                ),
                "microservice replicas",
            )
            _replica_ids(microservice, replicas)
            demand = _service_value(
                microservice,
                "resource_demand",
                f"service_dags[{service_index}].microservices[{microservice_index}]",
            )
            demand_values = _visual_values(demand, "microservice resource_demand")
            if any(value < 0.0 for value in demand_values):
                raise ValueError("microservice resource demand must be non-negative")
            demand_centi = [round(100.0 * value) for value in demand_values]
            if centi_demands and len(demand_centi) != len(centi_demands[0]):
                raise ValueError("microservice resource dimensions must align within a service DAG")
            replica_counts.append(replicas)
            centi_demands.append(demand_centi)
        if len(set(microservice_ids)) != len(microservice_ids):
            raise ValueError(f"service_dags[{service_index}] contains duplicate microservice IDs")
        radices = [max(row[index] for row in centi_demands) + 1 for index in range(len(centi_demands[0]))]
        demand_codes = []
        for row in centi_demands:
            packed = 0
            for value, radix in zip(row, radices):
                packed = packed * radix + value
            demand_codes.append(_base36(packed))
        qos = _service_value(service, "qos", f"service_dags[{service_index}]")
        if not isinstance(qos, dict):
            raise TypeError(f"service_dags[{service_index}].qos must be an object")
        qos_fields = ("latency_ms", "jitter_ms", "sync_ms", "min_reliability")
        if set(qos) != set(qos_fields):
            raise ValueError(f"service_dags[{service_index}].qos has an invalid schema")
        qos_values = [
            round(100.0 * _finite_number(qos[field], f"service_dags[{service_index}].qos.{field}"))
            for field in qos_fields[:-1]
        ]
        qos_values.append(
            round(1_000_000.0 * _finite_number(qos["min_reliability"], "service minimum reliability"))
        )
        service_id = _service_value(service, "service_id", f"service_dags[{service_index}]")
        service_class = _service_value(service, "class", f"service_dags[{service_index}]")
        slice_tag = _service_value(service, "slice_tag", f"service_dags[{service_index}]")
        views.append(
            f"{request_id}/{service_id}/{service_class}/{slice_tag}/ids={_sequence_text(microservice_ids)}/"
            f"rep={_run_length_text(replica_counts)}/radix={','.join(map(str, radices))}/"
            f"d36={','.join(demand_codes)}/q={','.join(str(value) for value in qos_values)}"
        )
    return (
        "fields=request/service/class/slice/task_order/replica_RLE/demand_radix/demand36/"
        "qos_latency,jitter,sync_centi_ms,reliability_ppm;demand36 is a base36 integer packing each task's "
        "[cpu,memory,gpu,npu] centi-units; decode by divmod(radix) right-to-left;"
        f"services={'||'.join(views)}"
    )


def _trace_prompt_view(trace: dict[str, Any]) -> dict[str, Any]:
    placement_routing = trace["placement_routing"]
    if "assignments" in placement_routing:
        placements = placement_routing["assignments"]
    elif "placements" in placement_routing:
        placements = placement_routing["placements"]
    else:
        placements = []
    node_counts: dict[str, int] = {}
    for placement in placements:
        if placement == "<eop>":
            continue
        if isinstance(placement, list) and len(placement) == 2:
            node_id = str(placement[1])
        elif isinstance(placement, dict) and "node_id" in placement:
            node_id = str(placement["node_id"])
        else:
            continue
        node_counts[node_id] = node_counts.get(node_id, 0) + 1
    return {
        "id": trace["trace_id"],
        "node_counts": node_counts,
        "qos": _rounded_for_prompt(trace["qos_summary"]),
        "utility": round(float(trace["metadata"]["observed_utility"]), 4),
        "relevance": round(float(trace["scores"]["rerank"]), 4),
    }


def _telemetry_prompt_view(telemetry: dict[str, Any]) -> str:
    required = {
        "controller_utilization",
        "global_utilization",
        "link_utilization",
        "queue_depths",
        "recent_qos",
        "rollout_summary",
        "candidate_feasibility_context",
    }
    _require_exact_keys(telemetry, required, "planner telemetry")
    if not isinstance(telemetry["candidate_feasibility_context"], dict):
        raise TypeError("candidate_feasibility_context must be an object")
    recent_qos = telemetry["recent_qos"]
    rollout = telemetry["rollout_summary"]
    if not isinstance(recent_qos, dict) or not isinstance(rollout, dict):
        raise TypeError("recent_qos and rollout_summary must be objects")
    _require_exact_keys(
        recent_qos,
        {"mean_latency_ms", "sla_violation_rate_percent", "reliability"},
        "recent_qos",
    )
    _require_exact_keys(rollout, {"completed", "rejected", "active"}, "rollout_summary")
    qos_values = (
        round(100.0 * _finite_number(recent_qos["mean_latency_ms"], "recent mean latency")),
        round(100.0 * _finite_number(recent_qos["sla_violation_rate_percent"], "recent SLA violation rate")),
        round(1_000_000.0 * _finite_number(recent_qos["reliability"], "recent reliability")),
    )
    rollout_values = tuple(
        round(_finite_number(rollout[key], f"rollout_summary.{key}"))
        for key in ("completed", "rejected", "active")
    )
    return f"q=latency_centi_ms/sla_centi_pct/reliability_ppm:{'/'.join(map(str, qos_values))};rollout=c/r/a:{'/'.join(map(str, rollout_values))}"


def _current_placement_prompt_view(placements: list[dict[str, Any]], local_node_ids: set[str]) -> str:
    node_counts: dict[str, int] = {}
    required = {"request_id", "microservice_id", "replica_id", "node_id"}
    seen = set()
    for index, placement in enumerate(placements):
        if not isinstance(placement, dict):
            raise TypeError(f"current_placement[{index}] must be an object")
        _require_exact_keys(placement, required, f"current_placement[{index}]")
        node_id = str(placement["node_id"])
        if node_id not in local_node_ids:
            continue
        placement_id = (
            str(placement["request_id"]),
            str(placement["microservice_id"]),
            str(placement["replica_id"]),
        )
        if placement_id in seen:
            raise ValueError(f"current_placement contains duplicate unit {'/'.join(placement_id)}")
        seen.add(placement_id)
        node_counts[node_id] = node_counts.get(node_id, 0) + 1
    if not node_counts:
        return "none"
    return ",".join(f"{node_id}*{count}" for node_id, count in sorted(node_counts.items()))


def _qos_intent_prompt_view(qos_intent: dict[str, Any]) -> str:
    if not isinstance(qos_intent, dict):
        raise TypeError("qos_intent must be an object")
    if not qos_intent:
        return "none"
    fields = ("latency_ms", "jitter_ms", "sync_ms", "min_reliability")
    _require_exact_keys(qos_intent, set(fields), "qos_intent")
    values = [
        round(100.0 * _finite_number(qos_intent[field], f"qos_intent.{field}"))
        for field in fields[:-1]
    ]
    values.append(round(1_000_000.0 * _finite_number(qos_intent["min_reliability"], "qos_intent.min_reliability")))
    return f"latency/jitter/sync_centi_ms,reliability_ppm={','.join(map(str, values))}"


def _visual_values(values: Any, label: str) -> list[float]:
    if not isinstance(values, list):
        raise TypeError(f"{label} must be a list")
    return [_finite_number(value, f"{label}[{index}]") for index, value in enumerate(values)]


def _load_color(utilization: float) -> tuple[int, int, int]:
    bounded = min(1.0, max(0.0, utilization))
    return (round(45 + 200 * bounded), round(170 - 115 * bounded), round(85 - 35 * bounded))


def _service_dag_visual(query: dict[str, Any], image_module: Any, draw_module: Any) -> Any:
    panel = image_module.new("RGB", (PLANNER_VISUAL_SIZE, PLANNER_VISUAL_SIZE), (250, 250, 248))
    draw = draw_module.Draw(panel)
    tasks = []
    dependencies = []
    service_ranges = []
    offset = 0
    for service_index, service in enumerate(query["service_dags"]):
        microservices = service["microservices"]
        local_indices = {str(item["microservice_id"]): index for index, item in enumerate(microservices)}
        if len(local_indices) != len(microservices):
            raise ValueError(f"service_dags[{service_index}] contains duplicate microservice IDs")
        for microservice_index, microservice in enumerate(microservices):
            demand = _visual_values(
                microservice["resource_demand"],
                f"service_dags[{service_index}].microservices[{microservice_index}].resource_demand",
            )
            if len(demand) != 4 or any(value < 0.0 for value in demand):
                raise ValueError("service DAG visual requires four non-negative resource demands per task")
            tasks.append(
                {
                    "id": str(microservice["microservice_id"]),
                    "demand": demand,
                    "replicas": _nonnegative_integer(
                        microservice["replicas"], "microservice replicas"
                    ),
                }
            )
        for edge_index, edge in enumerate(service["edges"]):
            if not isinstance(edge, dict) or "source" not in edge or "destination" not in edge:
                raise ValueError(f"service_dags[{service_index}].edges[{edge_index}] requires source and destination")
            source = str(edge["source"])
            destination = str(edge["destination"])
            if source not in local_indices or destination not in local_indices:
                raise ValueError(f"service_dags[{service_index}] edge references an unknown microservice")
            dependencies.append((offset + local_indices[source], offset + local_indices[destination]))
        service_ranges.append((offset, offset + len(microservices)))
        offset += len(microservices)
    if not tasks:
        raise ValueError("service DAG visual requires at least one task")

    task_count = len(tasks)
    cell = max(1, min(5, 150 // task_count))
    matrix_span = task_count * cell
    matrix_x = 36 + (150 - matrix_span) // 2
    matrix_y = 18
    draw.text((3, 2), "DAG: source row -> destination column", fill=(25, 25, 25))
    draw.rectangle(
        (matrix_x - 1, matrix_y - 1, matrix_x + matrix_span, matrix_y + matrix_span),
        fill=(242, 242, 240),
        outline=(95, 95, 95),
    )
    for source, destination in dependencies:
        left = matrix_x + destination * cell
        top = matrix_y + source * cell
        draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=(30, 105, 190))
    for task_index in range(task_count):
        left = matrix_x + task_index * cell
        top = matrix_y + task_index * cell
        draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=(40, 40, 40))
    for start, end in service_ranges:
        left = matrix_x + start * cell
        top = matrix_y + start * cell
        right = matrix_x + end * cell
        bottom = matrix_y + end * cell
        draw.rectangle((left, top, right, bottom), outline=(170, 95, 45), width=1)
    label_stride = max(1, math.ceil(task_count / 8))
    for task_index in range(0, task_count, label_stride):
        task_id = tasks[task_index]["id"]
        draw.text((2, matrix_y + task_index * cell - 3), task_id, fill=(55, 55, 55))

    heatmap_y = max(174, matrix_y + matrix_span + 8)
    row_height = max(2, min(7, (PLANNER_VISUAL_SIZE - heatmap_y - 3) // 5))
    demand_maxima = [max(task["demand"][index] for task in tasks) for index in range(4)]
    resource_colors = ((190, 65, 45), (45, 110, 195), (125, 65, 180), (30, 155, 115))
    for resource_index, (label, base_color) in enumerate(zip("CMGN", resource_colors)):
        top = heatmap_y + resource_index * row_height
        draw.text((3, top - 2), label, fill=base_color)
        for task_index, task in enumerate(tasks):
            maximum = demand_maxima[resource_index]
            intensity = task["demand"][resource_index] / maximum if maximum > 0.0 else 0.0
            color = tuple(round(245 - intensity * (245 - channel)) for channel in base_color)
            left = matrix_x + task_index * cell
            draw.rectangle((left, top, left + cell - 1, top + row_height - 1), fill=color)
    replica_top = heatmap_y + 4 * row_height
    replica_maximum = max(1, max(task["replicas"] for task in tasks))
    draw.text((3, replica_top - 2), "R", fill=(75, 75, 75))
    for task_index, task in enumerate(tasks):
        intensity = task["replicas"] / replica_maximum
        shade = round(245 - 190 * intensity)
        left = matrix_x + task_index * cell
        draw.rectangle((left, replica_top, left + cell - 1, replica_top + row_height - 1), fill=(shade, shade, shade))
    return panel


def build_planner_visual(query: dict[str, Any]) -> Any:
    """Rasterize measured regional topology and telemetry for Qwen2-VL."""

    validate_query(query)
    topology = query["topology"]
    _topology_prompt_view(topology)
    telemetry = query["telemetry"]
    _telemetry_prompt_view(telemetry)
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("multimodal planner input requires Pillow") from error

    image = Image.new("RGB", (PLANNER_VISUAL_SIZE, PLANNER_VISUAL_SIZE), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    node_ids = [str(node_id) for node_id in topology["node_ids"]]
    coordinates = topology["coordinates_km"]
    if not node_ids:
        raise ValueError("planner topology requires at least one node")
    points = []
    for index, coordinate in enumerate(coordinates):
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise ValueError(f"planner topology coordinate {index} must contain x and y")
        points.append(
            (
                _finite_number(coordinate[0], f"coordinates_km[{index}][0]"),
                _finite_number(coordinate[1], f"coordinates_km[{index}][1]"),
            )
        )
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    span_x = maximum_x - minimum_x
    span_y = maximum_y - minimum_y
    positions = {
        node_id: (
            round(10 + (x - minimum_x) / (span_x if span_x > 0.0 else 1.0) * 204),
            round(10 + (y - minimum_y) / (span_y if span_y > 0.0 else 1.0) * 150),
        )
        for node_id, (x, y) in zip(node_ids, points)
    }
    active = {node_id: topology["node_active"][index] for index, node_id in enumerate(node_ids)}
    slice_tag = str(query["slice_tag"])
    link_fields = {
        "source",
        "destination",
        "capacity_mbps",
        "available_mbps",
        "delay_ms",
        "reliability",
        "slice_tags",
    }
    for link_index, link in enumerate(topology["links"]):
        if not isinstance(link, dict):
            raise TypeError(f"topology link {link_index} must be an object")
        _require_exact_keys(link, link_fields, f"topology link {link_index}")
        source = str(link["source"])
        destination = str(link["destination"])
        if source not in positions or destination not in positions:
            raise ValueError(f"topology link {source}-{destination} references an unknown node")
        slices = link["slice_tags"]
        capacities = _visual_values(link["capacity_mbps"], f"topology link {link_index} capacity_mbps")
        available = _visual_values(link["available_mbps"], f"topology link {link_index} available_mbps")
        if not isinstance(slices, list) or len(slices) != len(capacities) or len(slices) != len(available):
            raise ValueError(f"topology link {link_index} slice arrays must align")
        if any(capacity < 0.0 for capacity in capacities) or any(value < 0.0 for value in available):
            raise ValueError(f"topology link {link_index} bandwidth values must be non-negative")
        selected = [index for index, tag in enumerate(slices) if str(tag) == slice_tag]
        if not selected:
            selected = list(range(len(slices)))
        used_fraction = [
            1.0 - min(available[index], capacities[index]) / capacities[index]
            for index in selected
            if capacities[index] > 0.0
        ]
        utilization = sum(used_fraction) / len(used_fraction) if used_fraction else 0.0
        color = _load_color(utilization)
        if not active[source] or not active[destination]:
            color = (150, 150, 150)
        draw.line((positions[source], positions[destination]), fill=color, width=1 + round(2 * utilization))

    resource_count = len(topology["resource_names"])
    for index, node_id in enumerate(node_ids):
        capacity = _visual_values(topology["capacity"][index], f"topology capacity {index}")
        residual = _visual_values(topology["residual_capacity"][index], f"topology residual_capacity {index}")
        if len(capacity) != resource_count or len(residual) != resource_count:
            raise ValueError("planner topology capacity dimensions must align with resource_names")
        fractions = [
            1.0 - min(remaining, total) / total
            for total, remaining in zip(capacity, residual)
            if total > 0.0
        ]
        utilization = sum(fractions) / len(fractions) if fractions else 0.0
        fill = _load_color(utilization) if active[node_id] else (125, 125, 125)
        x, y = positions[node_id]
        tier = str(topology["node_tiers"][index])
        if tier == "edge_server":
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=fill, outline=(35, 35, 35))
        elif tier == "edge_controller":
            draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=fill, outline=(20, 20, 20))
        elif tier == "cloud_core":
            draw.polygon(((x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)), fill=fill, outline=(20, 20, 20))
        else:
            raise ValueError(f"unknown topology tier {tier}")
        if not active[node_id]:
            draw.line((x - 4, y - 4, x + 4, y + 4), fill=(240, 240, 240), width=1)
            draw.line((x - 4, y + 4, x + 4, y - 4), fill=(240, 240, 240), width=1)
        draw.text((x + 4, y - 5), node_id, fill=(20, 20, 20))

    telemetry_groups = [
        ("C", _visual_values(telemetry["controller_utilization"], "controller_utilization"), (110, 70, 180)),
        ("R", _visual_values(telemetry["global_utilization"], "global_utilization"), (45, 105, 190)),
        ("L", _visual_values(telemetry["link_utilization"], "link_utilization"), (20, 145, 165)),
        ("Q", _visual_values(telemetry["queue_depths"], "queue_depths"), (220, 125, 35)),
    ]
    queue_values = telemetry_groups[-1][1]
    queue_scale = max([1.0, *queue_values])
    normalized_groups = [
        (label, [min(1.0, max(0.0, value)) for value in values], color)
        if label != "Q"
        else (label, [min(1.0, max(0.0, value / queue_scale)) for value in values], color)
        for label, values, color in telemetry_groups
    ]
    bar_count = sum(len(values) for _, values, _ in normalized_groups)
    if bar_count == 0:
        raise ValueError("planner telemetry visual requires utilization or queue measurements")
    draw.line((7, 170, 217, 170), fill=(90, 90, 90), width=1)
    slot = 208.0 / bar_count
    cursor = 8.0
    for label, values, color in normalized_groups:
        draw.text((round(cursor), 173), label, fill=color)
        for value in values:
            left = round(cursor)
            right = max(left + 1, round(cursor + max(1.0, slot - 1.0)))
            height = round(34 * value)
            draw.rectangle((left, 216 - height, right, 216), fill=color)
            cursor += slot
    dag_panel = _service_dag_visual(query, Image, ImageDraw)
    combined = Image.new("RGB", (PLANNER_VISUAL_WIDTH, PLANNER_VISUAL_HEIGHT), (250, 250, 248))
    combined.paste(image, (0, 0))
    combined.paste(dag_panel, (PLANNER_VISUAL_SIZE, 0))
    ImageDraw.Draw(combined).line(
        (PLANNER_VISUAL_SIZE - 1, 0, PLANNER_VISUAL_SIZE - 1, PLANNER_VISUAL_HEIGHT),
        fill=(65, 65, 65),
        width=1,
    )
    return combined


def render_prompt(
    template: str,
    query: dict[str, Any],
    retrieved_traces: list[dict[str, Any]],
    candidate_count: int,
    input_token_budget: int,
) -> str:
    validate_query(query)
    _positive_integer(candidate_count, "candidate_count")
    _positive_integer(input_token_budget, "input_token_budget")
    topology_view = _topology_prompt_view(query["topology"])
    service_view = _service_dag_prompt_view(query["service_dags"])
    trace_views = [_trace_prompt_view(trace) for trace in retrieved_traces]
    local_node_ids = {str(node_id) for node_id in query["topology"]["node_ids"]}
    values = {
        "candidate_count": candidate_count,
        "controller_id": query["controller_id"],
        "current_placement_json": _current_placement_prompt_view(query["current_placement"], local_node_ids),
        "policy_id": query["policy_id"],
        "qos_intent_json": _qos_intent_prompt_view(query["qos_intent"]),
        "region_id": query["region_id"],
        "retrieved_traces_json": "",
        "service_dags_json": service_view,
        "sim_time_s": query["sim_time_s"],
        "slice_tag": query["slice_tag"],
        "telemetry_json": _telemetry_prompt_view(query["telemetry"]),
        "topology_json": topology_view,
        "window_id": query["window_id"],
    }
    while True:
        values["retrieved_traces_json"] = json.dumps(trace_views, separators=(",", ":"))
        prompt = template.format_map(values)
        # Packed numeric orchestration data tokenizes more densely than prose.
        # 2.25 UTF-8 bytes per token is conservative for the pinned Qwen tokenizer;
        # generation still enforces the exact post-processor token count.
        estimated_tokens = math.ceil(len(prompt.encode("utf-8")) / 2.25)
        if estimated_tokens <= input_token_budget:
            return prompt
        if not trace_views:
            raise ValueError(
                f"planner prompt requires approximately {estimated_tokens} tokens, exceeding the {input_token_budget} token budget"
            )
        trace_views.pop()


def _decode_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("planner output does not contain a JSON object")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(stripped[start:])
    except json.JSONDecodeError as error:
        raise ValueError(f"planner output contains malformed JSON: {error.msg}") from error
    if stripped[start + end :].strip():
        raise ValueError("planner output contains content after the JSON object")
    if not isinstance(value, dict):
        raise TypeError("planner output must be a JSON object")
    return value


def parse_assignments(assignments: Any) -> list[Any]:
    if not isinstance(assignments, list) or len(assignments) < 2:
        raise ValueError("candidate assignments must contain assignments followed by <eop>")
    if assignments[-1] != "<eop>":
        raise ValueError("candidate assignments must terminate with <eop>")
    normalized = []
    seen = set()
    for index, assignment in enumerate(assignments[:-1]):
        if not isinstance(assignment, list) or len(assignment) != 2:
            raise TypeError(f"candidate assignment {index} must be a [placement_id, node_id] pair")
        if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in assignment):
            raise TypeError(f"candidate assignment {index} identifiers must be strings or integers")
        pair = [str(assignment[0]), str(assignment[1])]
        if pair[0] in seen:
            raise ValueError(f"candidate contains duplicate placement id {pair[0]}")
        seen.add(pair[0])
        normalized.append(pair)
    normalized.append("<eop>")
    return normalized


def parse_node_preferences(preferences: Any) -> list[Any]:
    if not isinstance(preferences, list) or not 2 <= len(preferences) <= MAX_NODE_PREFERENCES + 1:
        raise ValueError(
            f"candidate node_preferences must contain 1-{MAX_NODE_PREFERENCES} node IDs followed by <eop>"
        )
    if preferences[-1] != "<eop>":
        raise ValueError("candidate node_preferences must terminate with <eop>")
    normalized = []
    seen = set()
    for index, node_id in enumerate(preferences[:-1]):
        if isinstance(node_id, bool) or not isinstance(node_id, (str, int)):
            raise TypeError(f"candidate node preference {index} must be a string or integer")
        normalized_id = str(node_id)
        if normalized_id in seen:
            raise ValueError(f"candidate contains duplicate node preference {normalized_id}")
        seen.add(normalized_id)
        normalized.append(normalized_id)
    normalized.append("<eop>")
    return normalized


def _validate_placement_prior(prior: Any, assignment_ids: set[str]) -> dict[str, dict[str, float]]:
    if not isinstance(prior, dict):
        raise TypeError("candidate placement_prior must be an object")
    if set(map(str, prior.keys())) != assignment_ids:
        raise ValueError("candidate placement_prior keys must exactly match assigned placement ids")
    normalized = {}
    for placement_id, probabilities in prior.items():
        if not isinstance(probabilities, dict) or not probabilities:
            raise TypeError(f"placement prior for {placement_id} must be a non-empty object")
        distribution = {
            str(node_id): _finite_number(probability, f"placement_prior.{placement_id}.{node_id}")
            for node_id, probability in probabilities.items()
        }
        if any(probability < 0.0 or probability > 1.0 for probability in distribution.values()):
            raise ValueError(f"placement prior for {placement_id} must be in [0, 1]")
        if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"placement prior for {placement_id} must sum to 1")
        normalized[str(placement_id)] = distribution
    return normalized


def parse_plan_output(text: str, candidate_count: int, query: dict[str, Any]) -> dict[str, Any]:
    validate_query(query)
    value = _decode_json_object(text)
    _require_exact_keys(value, RAW_PLAN_FIELDS, "planner output")
    reward_weights = _validate_distribution(value["reward_weights"], REWARD_KEYS, "reward_weights")
    penalty_weights = _validate_coefficients(value["penalty_weights"], PENALTY_KEYS, "penalty_weights")
    cooperation_bias = _finite_number(value["cooperation_bias"], "cooperation_bias")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != candidate_count:
        raise ValueError(f"planner output must contain exactly {candidate_count} candidates")
    normalized_candidates = []
    candidate_ids = set()
    candidate_signatures = set()
    _, _, _, active = _topology_inventory(query["topology"])
    active_nodes = {node_id for node_id, is_active in active.items() if is_active}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"candidate {index} must be an object")
        _require_exact_keys(candidate, RAW_CANDIDATE_FIELDS, f"candidate {index}")
        candidate_id = str(candidate["candidate_id"])
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("candidate IDs must be non-empty and unique")
        candidate_ids.add(candidate_id)
        preferences = parse_node_preferences(candidate["node_preferences"])
        unknown_nodes = set(preferences[:-1]) - active_nodes
        if unknown_nodes:
            raise ValueError(f"candidate node_preferences include unknown or inactive nodes: {sorted(unknown_nodes)}")
        strategy = candidate["placement_strategy"]
        if strategy not in PLACEMENT_STRATEGIES:
            raise ValueError(f"candidate placement_strategy must be one of {sorted(PLACEMENT_STRATEGIES)}")
        signature = (strategy, *preferences[:-1])
        if signature in candidate_signatures:
            raise ValueError("planner candidates must have unique strategy/preference combinations")
        candidate_signatures.add(signature)
        predicted_utility = _finite_number(candidate["predicted_utility"], "predicted_utility")
        if not -1.0 <= predicted_utility <= 1.0:
            raise ValueError("predicted_utility must be in [-1, 1]")
        normalized_candidates.append(
            {
                "candidate_id": candidate_id,
                "node_preferences": preferences,
                "placement_strategy": strategy,
                "predicted_utility": predicted_utility,
            }
        )
    return {
        "reward_weights": reward_weights,
        "penalty_weights": penalty_weights,
        "cooperation_bias": cooperation_bias,
        "candidates": normalized_candidates,
    }


def _resource_vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty numeric list")
    return [_finite_number(component, f"{label}[{index}]") for index, component in enumerate(value)]


def _placement_inventory(query: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], list[str]]]:
    inventory = {}
    replicas_by_microservice: dict[tuple[str, str], list[str]] = {}
    for service in query["service_dags"]:
        request_id = str(service["request_id"])
        slice_tag = str(service["slice_tag"])
        for microservice in service["microservices"]:
            microservice_id = str(microservice["microservice_id"])
            replicas = _nonnegative_integer(
                microservice["replicas"], "microservice replicas"
            )
            replica_ids = _replica_ids(microservice, replicas)
            demand = _resource_vector(microservice["resource_demand"], "microservice resource_demand")
            key = (request_id, microservice_id)
            replicas_by_microservice[key] = []
            for replica_id in replica_ids:
                placement_id = f"{request_id}/{microservice_id}/{replica_id}"
                inventory[placement_id] = {
                    "request_id": request_id,
                    "microservice_id": microservice_id,
                    "replica_id": replica_id,
                    "slice_tag": slice_tag,
                    "demand": demand,
                }
                replicas_by_microservice[key].append(placement_id)
    return inventory, replicas_by_microservice


def _topology_inventory(
    topology: dict[str, Any],
) -> tuple[
    dict[str, list[float]],
    dict[str, dict[str, set[str]]],
    dict[str, set[str]],
    dict[str, bool],
]:
    node_ids = [str(node_id) for node_id in topology["node_ids"]]
    residual = topology["residual_capacity"]
    if len(residual) != len(node_ids):
        raise ValueError("topology residual_capacity must align with node_ids")
    capacities = {
        node_id: _resource_vector(residual[index], f"node {node_id} residual_capacity")
        for index, node_id in enumerate(node_ids)
    }
    active_values = topology["node_active"]
    if not isinstance(active_values, list) or len(active_values) != len(node_ids):
        raise ValueError("topology node_active must align with node_ids")
    if not all(isinstance(value, bool) for value in active_values):
        raise TypeError("topology node_active must contain booleans")
    active = {node_id: active_values[index] for index, node_id in enumerate(node_ids)}
    adjacency: dict[str, dict[str, set[str]]] = {node_id: {} for node_id in node_ids}
    allowed_slices = {node_id: set() for node_id in node_ids}
    for link_index, link in enumerate(topology["links"]):
        if not isinstance(link, dict):
            raise TypeError(f"topology link {link_index} must be an object")
        _require_exact_keys(
            link,
            {"source", "destination", "capacity_mbps", "available_mbps", "delay_ms", "reliability", "slice_tags"},
            f"topology link {link_index}",
        )
        source = str(link["source"])
        destination = str(link["destination"])
        if source not in adjacency or destination not in adjacency:
            raise ValueError(f"topology link {source}-{destination} references an unknown node")
        slices = link["slice_tags"]
        if not isinstance(slices, list):
            raise TypeError("topology link slice_tags must be a list")
        available = link["available_mbps"]
        if not isinstance(available, list) or len(available) != len(slices):
            raise ValueError("topology link available_mbps must align with slice_tags")
        active_slices = {
            str(tag)
            for tag, bandwidth in zip(slices, available)
            if _finite_number(bandwidth, "link available bandwidth") > 1e-9
        }
        if not active[source] or not active[destination]:
            active_slices = set()
        adjacency[source][destination] = active_slices
        adjacency[destination][source] = active_slices
        allowed_slices[source].update(active_slices)
        allowed_slices[destination].update(active_slices)
    return capacities, adjacency, allowed_slices, active


def _connected_for_slice(
    source: str,
    destination: str,
    slice_tag: str,
    adjacency: dict[str, dict[str, set[str]]],
    allowed_slices: dict[str, set[str]],
) -> bool:
    if source == destination:
        return True
    frontier = [source]
    visited = {source}
    while frontier:
        node = frontier.pop()
        for neighbor, link_slices in adjacency[node].items():
            if slice_tag not in link_slices:
                continue
            if neighbor == destination:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def _topological_placement_ids(
    query: dict[str, Any],
    replicas_by_microservice: dict[tuple[str, str], list[str]],
) -> tuple[list[str], dict[tuple[str, str], list[str]]]:
    placement_ids = []
    parents_by_microservice: dict[tuple[str, str], list[str]] = {}
    for service_index, service in enumerate(query["service_dags"]):
        request_id = str(service["request_id"])
        microservice_ids = [str(item["microservice_id"]) for item in service["microservices"]]
        if len(set(microservice_ids)) != len(microservice_ids):
            raise ValueError(f"service_dags[{service_index}] contains duplicate microservice IDs")
        order_index = {microservice_id: index for index, microservice_id in enumerate(microservice_ids)}
        children = {microservice_id: [] for microservice_id in microservice_ids}
        parents = {microservice_id: [] for microservice_id in microservice_ids}
        edge_pairs = set()
        for edge_index, edge in enumerate(service["edges"]):
            if not isinstance(edge, dict) or "source" not in edge or "destination" not in edge:
                raise ValueError(f"service_dags[{service_index}].edges[{edge_index}] requires source and destination")
            source = str(edge["source"])
            destination = str(edge["destination"])
            if source not in order_index or destination not in order_index:
                raise ValueError(f"service_dags[{service_index}] edge references an unknown microservice")
            if source == destination:
                raise ValueError(f"service_dags[{service_index}] contains a self-loop")
            edge_pairs.add((source, destination))
        for source, destination in edge_pairs:
            children[source].append(destination)
            parents[destination].append(source)
        indegree = {microservice_id: len(parents[microservice_id]) for microservice_id in microservice_ids}
        ready = [microservice_id for microservice_id in microservice_ids if indegree[microservice_id] == 0]
        ordered_microservices = []
        while ready:
            ready.sort(key=order_index.__getitem__)
            microservice_id = ready.pop(0)
            ordered_microservices.append(microservice_id)
            for child in sorted(children[microservice_id], key=order_index.__getitem__):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered_microservices) != len(microservice_ids):
            raise ValueError(f"service_dags[{service_index}] must be acyclic")
        for microservice_id in ordered_microservices:
            key = (request_id, microservice_id)
            parents_by_microservice[key] = parents[microservice_id]
            placement_ids.extend(replicas_by_microservice[key])
    return placement_ids, parents_by_microservice


def _node_reliability(
    topology: dict[str, Any],
    slice_tag: str,
    active: dict[str, bool],
) -> dict[str, float]:
    samples = {node_id: [] for node_id in active}
    for link_index, link in enumerate(topology["links"]):
        source = str(link["source"])
        destination = str(link["destination"])
        slices = [str(value) for value in link["slice_tags"]]
        available = link["available_mbps"]
        if slice_tag not in slices or not active[source] or not active[destination]:
            continue
        slice_index = slices.index(slice_tag)
        if _finite_number(available[slice_index], f"link {link_index} available bandwidth") <= 1e-9:
            continue
        reliability = _finite_number(link["reliability"], f"link {link_index} reliability")
        if not 0.0 <= reliability <= 1.0:
            raise ValueError(f"link {link_index} reliability must be in [0, 1]")
        samples[source].append(reliability)
        samples[destination].append(reliability)
    return {
        node_id: sum(values) / len(values) if values else 0.0
        for node_id, values in samples.items()
    }


def _strategy_key(
    strategy: str,
    node_id: str,
    preference_rank: dict[str, int],
    residual: dict[str, list[float]],
    initial_capacity: dict[str, list[float]],
    demand: list[float],
    latency: dict[str, float],
    reliability: dict[str, float],
    placements_per_node: dict[str, int],
) -> tuple[Any, ...]:
    ratios = [
        (remaining - amount) / total
        for remaining, total, amount in zip(residual[node_id], initial_capacity[node_id], demand)
        if total > 0.0
    ]
    capacity = min(ratios) if ratios else 0.0
    preferred = -preference_rank.get(node_id, len(preference_rank))
    common = (capacity, reliability[node_id], -latency[node_id])
    if strategy == "latency_first":
        strategy_values = (-latency[node_id], capacity, reliability[node_id], -placements_per_node[node_id])
    elif strategy == "capacity_first":
        strategy_values = (*common, -placements_per_node[node_id])
    elif strategy == "reliability_first":
        strategy_values = (reliability[node_id], capacity, -latency[node_id], -placements_per_node[node_id])
    elif strategy == "spread":
        strategy_values = (-placements_per_node[node_id], *common)
    elif strategy == "consolidate":
        strategy_values = (placements_per_node[node_id], *common)
    else:
        raise ValueError(f"unknown placement strategy {strategy}")
    return (preferred, *strategy_values, node_id)


def _compatible_node(
    node_id: str,
    item: dict[str, Any],
    residual: dict[str, list[float]],
    allowed_slices: dict[str, set[str]],
    parent_nodes: list[str],
    adjacency: dict[str, dict[str, set[str]]],
    capacity_required: bool,
    connectivity_required: bool,
) -> bool:
    if allowed_slices[node_id] and item["slice_tag"] not in allowed_slices[node_id]:
        return False
    if capacity_required and any(
        amount > remaining + 1e-9 for amount, remaining in zip(item["demand"], residual[node_id])
    ):
        return False
    return not connectivity_required or all(
        _connected_for_slice(parent_node, node_id, item["slice_tag"], adjacency, allowed_slices)
        for parent_node in parent_nodes
    )


def expand_candidate_assignments(candidate: dict[str, Any], query: dict[str, Any]) -> list[Any]:
    """Expand compact LLM node guidance into a complete deterministic placement."""

    validate_query(query)
    inventory, replicas_by_microservice = _placement_inventory(query)
    placement_ids, parents_by_microservice = _topological_placement_ids(query, replicas_by_microservice)
    capacities, adjacency, allowed_slices, active = _topology_inventory(query["topology"])
    active_nodes = [node_id for node_id, is_active in active.items() if is_active]
    if not active_nodes:
        raise ValueError("planner topology contains no active placement node")
    preferences = parse_node_preferences(candidate["node_preferences"])
    preference_rank = {node_id: index for index, node_id in enumerate(preferences[:-1])}
    strategy = candidate["placement_strategy"]
    if strategy not in PLACEMENT_STRATEGIES:
        raise ValueError(f"unknown placement strategy {strategy}")
    topology = query["topology"]
    latency_rows = topology["latency_ms"]
    latency = {
        str(node_id): min(_visual_values(latency_rows[index], f"topology latency_ms {index}"))
        for index, node_id in enumerate(topology["node_ids"])
    }
    reliability_by_slice = {
        item["slice_tag"]: _node_reliability(topology, item["slice_tag"], active)
        for item in inventory.values()
    }
    initial_capacity = {node_id: values.copy() for node_id, values in capacities.items()}
    residual = {node_id: values.copy() for node_id, values in capacities.items()}
    placements_per_node = {node_id: 0 for node_id in capacities}
    assigned_by_microservice: dict[tuple[str, str], list[str]] = {
        key: [] for key in replicas_by_microservice
    }
    assignments = []

    for placement_id in placement_ids:
        item = inventory[placement_id]
        key = (item["request_id"], item["microservice_id"])
        parent_nodes = [
            node_id
            for parent_id in parents_by_microservice[key]
            for node_id in assigned_by_microservice[(item["request_id"], parent_id)]
        ]

        choices = [
            node_id
            for node_id in active_nodes
            if _compatible_node(
                node_id,
                item,
                residual,
                allowed_slices,
                parent_nodes,
                adjacency,
                True,
                True,
            )
        ]
        if not choices:
            choices = [
                node_id
                for node_id in active_nodes
                if _compatible_node(
                    node_id,
                    item,
                    residual,
                    allowed_slices,
                    parent_nodes,
                    adjacency,
                    False,
                    True,
                )
            ]
        if not choices:
            choices = [
                node_id
                for node_id in active_nodes
                if _compatible_node(
                    node_id,
                    item,
                    residual,
                    allowed_slices,
                    parent_nodes,
                    adjacency,
                    False,
                    False,
                )
            ]
        if not choices:
            choices = active_nodes
        selected_node = max(
            choices,
            key=lambda node_id: _strategy_key(
                strategy,
                node_id,
                preference_rank,
                residual,
                initial_capacity,
                item["demand"],
                latency,
                reliability_by_slice[item["slice_tag"]],
                placements_per_node,
            ),
        )
        residual[selected_node] = [
            remaining - amount for remaining, amount in zip(residual[selected_node], item["demand"])
        ]
        placements_per_node[selected_node] += 1
        assigned_by_microservice.setdefault(key, []).append(selected_node)
        assignments.append([placement_id, selected_node])
    assignments.append("<eop>")
    return assignments


def validate_candidate_feasibility(candidate: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    validate_query(query)
    if not isinstance(candidate, dict):
        raise TypeError("candidate must be an object")
    if set(candidate.keys()) not in (EXPANDED_CANDIDATE_FIELDS, DISPATCH_CANDIDATE_FIELDS):
        raise ValueError("candidate has an invalid schema")
    assignments = parse_assignments(candidate["assignments"])
    assignment_map = {placement_id: node_id for placement_id, node_id in assignments[:-1]}
    inventory, replicas_by_microservice = _placement_inventory(query)
    capacities, adjacency, allowed_slices, active = _topology_inventory(query["topology"])
    violations = []
    missing = inventory.keys() - assignment_map.keys()
    unknown = assignment_map.keys() - inventory.keys()
    if missing:
        violations.append(f"missing placement units: {sorted(missing)}")
    if unknown:
        violations.append(f"unknown placement units: {sorted(unknown)}")

    usage = {node_id: [0.0] * len(capacity) for node_id, capacity in capacities.items()}
    for placement_id, node_id in assignment_map.items():
        if placement_id not in inventory:
            continue
        if node_id not in capacities:
            violations.append(f"placement unit {placement_id} targets unknown node {node_id}")
            continue
        if not active[node_id]:
            violations.append(f"placement unit {placement_id} targets inactive node {node_id}")
        demand = inventory[placement_id]["demand"]
        if len(demand) != len(capacities[node_id]):
            violations.append(f"resource dimensions differ for placement unit {placement_id} and node {node_id}")
            continue
        for index, amount in enumerate(demand):
            usage[node_id][index] += amount
        slice_tag = inventory[placement_id]["slice_tag"]
        if allowed_slices[node_id] and slice_tag not in allowed_slices[node_id]:
            violations.append(f"node {node_id} has no link access for slice {slice_tag}")

    for node_id, consumed in usage.items():
        for resource_index, (amount, capacity) in enumerate(zip(consumed, capacities[node_id])):
            if amount > capacity + 1e-9:
                violations.append(
                    f"node {node_id} resource {resource_index} exceeds residual capacity ({amount:.6g}>{capacity:.6g})"
                )

    for service in query["service_dags"]:
        request_id = str(service["request_id"])
        slice_tag = str(service["slice_tag"])
        for edge in service["edges"]:
            source_key = (request_id, str(edge["source"]))
            destination_key = (request_id, str(edge["destination"]))
            source_nodes = {
                assignment_map[placement_id]
                for placement_id in replicas_by_microservice[source_key]
                if placement_id in assignment_map and assignment_map[placement_id] in capacities
            }
            destination_nodes = {
                assignment_map[placement_id]
                for placement_id in replicas_by_microservice[destination_key]
                if placement_id in assignment_map and assignment_map[placement_id] in capacities
            }
            if not source_nodes or not destination_nodes:
                continue
            for source_node in source_nodes:
                if not any(
                    _connected_for_slice(source_node, destination_node, slice_tag, adjacency, allowed_slices)
                    for destination_node in destination_nodes
                ):
                    violations.append(
                        f"dependency {edge['source']}->{edge['destination']} has no {slice_tag} path from node {source_node}"
                    )
    return {"feasible": not violations, "violations": violations}


def semantic_drift(predicted_utility: float, observed_utility: float) -> float:
    predicted = _finite_number(predicted_utility, "predicted_utility")
    observed = _finite_number(observed_utility, "observed_utility")
    return abs(predicted - observed)


def _validate_feasibility(value: Any) -> None:
    if not isinstance(value, dict):
        raise TypeError("feasibility must be an object")
    _require_exact_keys(value, FEASIBILITY_FIELDS, "feasibility")
    if not isinstance(value["feasible"], bool):
        raise TypeError("feasibility.feasible must be boolean")
    if not isinstance(value["violations"], list) or not all(isinstance(item, str) for item in value["violations"]):
        raise TypeError("feasibility.violations must be a list of strings")


def validate_dispatch_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise TypeError("dispatch plan must be an object")
    _require_exact_keys(plan, DISPATCH_PLAN_FIELDS, "dispatch plan")
    if plan["schema_version"] != 1:
        raise ValueError("dispatch plan schema_version must be 1")
    for key in ("plan_id", "window_id", "controller_id", "region_id", "backend", "model_id", "selected_candidate_id"):
        if not isinstance(plan[key], (str, int)) or isinstance(plan[key], bool):
            raise TypeError(f"dispatch plan {key} must be a string or integer")
    for key in ("created_at", "valid_from_s", "valid_until_s", "predicted_utility"):
        _finite_number(plan[key], f"dispatch plan {key}")
    if plan["valid_until_s"] <= plan["valid_from_s"]:
        raise ValueError("dispatch plan validity interval is empty")
    _validate_distribution(plan["reward_weights"], REWARD_KEYS, "reward_weights")
    _validate_coefficients(plan["penalty_weights"], PENALTY_KEYS, "penalty_weights")
    _finite_number(plan["cooperation_bias"], "cooperation_bias")
    if not isinstance(plan["candidates"], list) or not plan["candidates"]:
        raise ValueError("dispatch plan candidates must be a non-empty list")
    selected = None
    for candidate in plan["candidates"]:
        _require_exact_keys(candidate, DISPATCH_CANDIDATE_FIELDS, "dispatch candidate")
        parse_node_preferences(candidate["node_preferences"])
        if candidate["placement_strategy"] not in PLACEMENT_STRATEGIES:
            raise ValueError("dispatch candidate has an invalid placement_strategy")
        parse_assignments(candidate["assignments"])
        predicted_utility = _finite_number(candidate["predicted_utility"], "dispatch candidate predicted_utility")
        if not -1.0 <= predicted_utility <= 1.0:
            raise ValueError("dispatch candidate predicted_utility must be in [-1, 1]")
        _validate_feasibility(candidate["feasibility"])
        if str(candidate["candidate_id"]) == str(plan["selected_candidate_id"]):
            selected = candidate
    if selected is None:
        raise ValueError("selected_candidate_id does not identify a candidate")
    if not selected["feasibility"]["feasible"]:
        raise ValueError("dispatch plan selected candidate is infeasible")
    if plan["selected_assignments"] != selected["assignments"]:
        raise ValueError("selected_assignments differ from selected candidate")
    selected_ids = {placement_id for placement_id, _ in selected["assignments"][:-1]}
    _validate_placement_prior(plan["placement_prior"], selected_ids)
    if not isinstance(plan["plan_embedding"], list) or len(plan["plan_embedding"]) != 128:
        raise ValueError("dispatch plan embedding must contain 128 values")
    if not isinstance(plan["retrieval_trace_ids"], list):
        raise TypeError("retrieval_trace_ids must be a list")
    _validate_feasibility(plan["feasibility"])


def build_dispatch_plan(
    config: dict[str, Any],
    query: dict[str, Any],
    decoded: dict[str, Any],
    retrieved_traces: list[dict[str, Any]],
    created_at: float,
) -> dict[str, Any]:
    planner = validate_planner_config(config)
    validate_query(query)
    candidates = []
    for candidate in decoded["candidates"]:
        enriched = dict(candidate)
        enriched["assignments"] = expand_candidate_assignments(candidate, query)
        enriched["feasibility"] = validate_candidate_feasibility(enriched, query)
        candidates.append(enriched)
    feasible = [candidate for candidate in candidates if candidate["feasibility"]["feasible"]]
    if not feasible:
        violations = {candidate["candidate_id"]: candidate["feasibility"]["violations"] for candidate in candidates}
        raise ValueError(f"planner produced no feasible candidate: {violations}")
    selected = max(feasible, key=lambda candidate: candidate["predicted_utility"])
    maximum_utility = max(candidate["predicted_utility"] for candidate in feasible)
    candidate_weights = [math.exp(candidate["predicted_utility"] - maximum_utility) for candidate in feasible]
    placement_prior: dict[str, dict[str, float]] = {}
    for candidate, candidate_weight in zip(feasible, candidate_weights):
        for placement_id, node_id in candidate["assignments"][:-1]:
            node_distribution = placement_prior.setdefault(placement_id, {})
            if node_id in node_distribution:
                node_distribution[node_id] += candidate_weight
            else:
                node_distribution[node_id] = candidate_weight
    for node_distribution in placement_prior.values():
        total = sum(node_distribution.values())
        for node_id in node_distribution:
            node_distribution[node_id] /= total
    plan_core = {
        "reward_weights": decoded["reward_weights"],
        "penalty_weights": decoded["penalty_weights"],
        "cooperation_bias": decoded["cooperation_bias"],
        "selected_candidate": selected,
    }
    valid_from = float(query["sim_time_s"])
    plan = {
        "schema_version": 1,
        "plan_id": uuid.uuid4().hex,
        "window_id": query["window_id"],
        "controller_id": query["controller_id"],
        "region_id": query["region_id"],
        "created_at": _finite_number(created_at, "plan created_at"),
        "valid_from_s": valid_from,
        "valid_until_s": valid_from + planner["planning_interval_s"],
        "backend": planner["backend"],
        "model_id": planner["model_id"],
        "reward_weights": decoded["reward_weights"],
        "penalty_weights": decoded["penalty_weights"],
        "cooperation_bias": decoded["cooperation_bias"],
        "candidates": candidates,
        "selected_candidate_id": selected["candidate_id"],
        "selected_assignments": selected["assignments"],
        "placement_prior": placement_prior,
        "predicted_utility": selected["predicted_utility"],
        "plan_embedding": hashed_embedding(plan_core),
        "retrieval_trace_ids": [trace["trace_id"] for trace in retrieved_traces],
        "feasibility": selected["feasibility"],
    }
    validate_dispatch_plan(plan)
    return plan


def _atomic_write_json(path: str, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = temporary.name
    os.replace(temporary_path, target)


def _validate_exchange(exchange: dict[str, Any]) -> None:
    _require_exact_keys(exchange, {"schema_version", "updated_at", "plans"}, "plan exchange")
    if exchange["schema_version"] != 1 or not isinstance(exchange["plans"], dict):
        raise ValueError("plan exchange has an invalid schema")
    _finite_number(exchange["updated_at"], "plan exchange updated_at")
    for plan in exchange["plans"].values():
        validate_dispatch_plan(plan)


def publish_latest_plan(path: str, plan: dict[str, Any]) -> None:
    validate_dispatch_plan(plan)
    target = Path(path)
    if target.exists():
        exchange = json.loads(target.read_text(encoding="utf-8"))
        _validate_exchange(exchange)
    else:
        exchange = {"schema_version": 1, "updated_at": 0.0, "plans": {}}
    exchange["updated_at"] = time.time()
    exchange["plans"][str(plan["controller_id"])] = plan
    _atomic_write_json(path, exchange)


def read_latest_plan(path: str, controller_id: str | int, sim_time_s: float) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    exchange = json.loads(target.read_text(encoding="utf-8"))
    _validate_exchange(exchange)
    controller_key = str(controller_id)
    if controller_key not in exchange["plans"]:
        return None
    plan = exchange["plans"][controller_key]
    simulation_time = _finite_number(sim_time_s, "sim_time_s")
    if not plan["valid_from_s"] <= simulation_time < plan["valid_until_s"]:
        return None
    return plan


def prepare_planning_window(
    config: dict[str, Any],
    query: dict[str, Any],
    use_toporag: bool,
) -> dict[str, Any]:
    planner = validate_planner_config(config)
    validate_query(query)
    if not isinstance(use_toporag, bool):
        raise TypeError("use_toporag must be boolean")
    now = time.time()
    initialize_memory(planner["memory_path"])
    prune_memory(planner["memory_path"], now, planner["max_traces"])
    if use_toporag:
        retrieved = retrieve_traces(
            planner["memory_path"],
            query,
            planner["retrieval_pool"],
            planner["retrieval_top_k"],
            planner["similarity_weights"],
            planner["soft_prune_threshold"],
            now,
        )
    else:
        retrieved = []
    template = load_prompt(planner["prompt_path"])
    prompt = render_prompt(
        template,
        query,
        retrieved,
        planner["candidate_count"],
        planner["prompt_input_tokens"] - VISUAL_TOKEN_RESERVE,
    )
    return {"prompt": prompt, "retrieved_traces": retrieved, "visual": build_planner_visual(query)}


def _lora_target_modules(model: Any, trainable_fraction: float) -> list[str]:
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    layer_indices = []
    for name, _ in model.named_modules():
        match = layer_pattern.search(name)
        if match:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        raise RuntimeError("Qwen model exposes no transformer layer indices for LoRA")
    layer_count = max(layer_indices) + 1
    first_adapted = math.floor(layer_count * (1.0 - trainable_fraction))
    target_suffixes = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"}
    targets = []
    for name, module in model.named_modules():
        match = layer_pattern.search(name)
        if (
            match
            and int(match.group(1)) >= first_adapted
            and name.rsplit(".", 1)[-1] in target_suffixes
            and hasattr(module, "in_features")
            and hasattr(module, "out_features")
        ):
            targets.append(name)
    if not targets:
        raise RuntimeError("Qwen model exposes no upper-layer attention/feed-forward projections for LoRA")
    return targets


def load_qwen_planner(config: dict[str, Any], trainable: bool, adapter_path: str | None) -> dict[str, Any]:
    """Lazily load Qwen2-VL and optionally construct/load upper-layer QLoRA."""

    planner = validate_planner_config(config)
    try:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2VLForConditionalGeneration,
        )
    except ImportError as error:
        raise RuntimeError("Qwen2-VL planning requires torch and a Transformers release with Qwen2-VL support") from error

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[planner["dtype"]]
    model_arguments: dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto"}
    if trainable:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA adapter updates require a CUDA device")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise RuntimeError("QLoRA adapter updates require bitsandbytes") from error
        model_arguments["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen2VLForConditionalGeneration.from_pretrained(planner["model_id"], **model_arguments)
    processor = AutoProcessor.from_pretrained(
        planner["model_id"],
        min_pixels=PLANNER_VISUAL_PIXELS,
        max_pixels=PLANNER_VISUAL_PIXELS,
        use_fast=False,
    )

    if trainable or adapter_path is not None:
        try:
            from peft import (
                LoraConfig,
                PeftModel,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError as error:
            raise RuntimeError("LoRA planner adapters require PEFT") from error
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
        else:
            model = prepare_model_for_kbit_training(model)
            lora = planner["lora"]
            targets = _lora_target_modules(model, lora["trainable_fraction"])
            peft_config = LoraConfig(
                r=lora["rank"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            )
            model = get_peft_model(model, peft_config)
    if not trainable:
        model.eval()
    return {"model": model, "processor": processor}


def _qwen_chat_and_vision(processor: Any, prompt: str, visual: Any) -> tuple[str, Any, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as error:
        raise RuntimeError("Qwen2-VL image input requires qwen-vl-utils") from error
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": visual,
                    "resized_height": PLANNER_VISUAL_HEIGHT,
                    "resized_width": PLANNER_VISUAL_WIDTH,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return chat, image_inputs, video_inputs


def generate_qwen_output(
    model_bundle: dict[str, Any],
    prompt: str,
    visual: Any,
    input_token_budget: int,
    max_new_tokens: int,
) -> str:
    model = model_bundle["model"]
    processor = model_bundle["processor"]
    chat, image_inputs, video_inputs = _qwen_chat_and_vision(processor, prompt, visual)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_tokens = int(inputs["input_ids"].shape[-1])
    if input_tokens > input_token_budget:
        raise ValueError(f"tokenized multimodal planner input has {input_tokens} tokens; budget is {input_token_budget}")
    inputs = inputs.to(next(model.parameters()).device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = generated[:, inputs["input_ids"].shape[-1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def run_planning_window(
    config: dict[str, Any],
    query: dict[str, Any],
    model_bundle: dict[str, Any] | None,
    adapter_path: str | None,
    use_toporag: bool,
) -> dict[str, Any]:
    planner = validate_planner_config(config)
    prepared = prepare_planning_window(config, query, use_toporag)
    if model_bundle is None:
        model_bundle = load_qwen_planner(config, False, adapter_path)
    # The paper's approximate 180-token decode budget is only lossless when
    # applied per sampled candidate; the structured response contains B plans.
    output = generate_qwen_output(
        model_bundle,
        prepared["prompt"],
        prepared["visual"],
        planner["prompt_input_tokens"],
        planner["max_new_tokens"] * planner["candidate_count"],
    )
    decoded = parse_plan_output(output, planner["candidate_count"], query)
    plan = build_dispatch_plan(config, query, decoded, prepared["retrieved_traces"], time.time())
    publish_latest_plan(planner["latest_plan_path"], plan)
    return plan


def record_plan_outcome(
    config: dict[str, Any],
    query: dict[str, Any],
    plan: dict[str, Any],
    placement_routing: dict[str, Any],
    rollout_trace: dict[str, Any],
    qos_summary: dict[str, Any],
    observed_utility: float,
    priority: float,
    success: bool,
    created_at: float,
) -> str:
    planner = validate_planner_config(config)
    validate_query(query)
    validate_dispatch_plan(plan)
    for label, value in (
        ("placement_routing", placement_routing),
        ("rollout_trace", rollout_trace),
        ("qos_summary", qos_summary),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"{label} must be an object")
    utility = _finite_number(observed_utility, "observed_utility")
    trace_priority = _finite_number(priority, "trace priority")
    if not 0.0 <= trace_priority <= 1.0:
        raise ValueError("trace priority must be in [0, 1]")
    if not isinstance(success, bool):
        raise TypeError("trace success must be boolean")
    query_context = {key: query[key] for key in QUERY_FIELDS - {"service_dags", "topology"}}
    raw_candidates = [
        {key: candidate[key] for key in RAW_CANDIDATE_FIELDS}
        for candidate in plan["candidates"]
    ]
    planner_output = {
        "reward_weights": plan["reward_weights"],
        "penalty_weights": plan["penalty_weights"],
        "cooperation_bias": plan["cooperation_bias"],
        "candidates": raw_candidates,
    }
    trace = {
        "trace_id": new_trace_id(query["region_id"], query["controller_id"]),
        "region_id": query["region_id"],
        "controller_id": query["controller_id"],
        "created_at": _finite_number(created_at, "trace created_at"),
        "priority": trace_priority,
        "success": success,
        "service_dag": query["service_dags"],
        "topology": query["topology"],
        "query_context": query_context,
        "placement_routing": placement_routing,
        "rollout_trace": rollout_trace,
        "qos_summary": qos_summary,
        "embedding": build_query_embedding(query),
        "metadata": {
            "plan_id": plan["plan_id"],
            "planner_output": planner_output,
            "selected_candidate_id": plan["selected_candidate_id"],
            "predicted_utility": plan["predicted_utility"],
            "observed_utility": utility,
            "semantic_drift": semantic_drift(plan["predicted_utility"], utility),
        },
    }
    ttl_seconds = planner["memory_ttl_windows"] * planner["planning_interval_s"]
    return record_trace(planner["memory_path"], trace, ttl_seconds, planner["max_traces"])


def adapter_update_due(off_peak: bool, topology_changed: bool) -> bool:
    if not isinstance(off_peak, bool) or not isinstance(topology_changed, bool):
        raise TypeError("off_peak and topology_changed must be booleans")
    return off_peak or topology_changed


def adapter_update_requested(
    predicted_utility: float,
    observed_utility: float,
    drift_threshold: float,
) -> bool:
    threshold = _finite_number(drift_threshold, "drift_threshold")
    if threshold <= 0.0:
        raise ValueError("drift_threshold must be positive")
    return semantic_drift(predicted_utility, observed_utility) > threshold


def _query_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    query = dict(trace["query_context"])
    query["service_dags"] = trace["service_dag"]
    query["topology"] = trace["topology"]
    validate_query(query)
    return query


def build_adapter_examples(
    config: dict[str, Any],
    region_id: str | int,
    limit: int,
    now: float,
) -> list[dict[str, Any]]:
    planner = validate_planner_config(config)
    traces = successful_traces(planner["memory_path"], region_id, limit, now)
    template = load_prompt(planner["prompt_path"])
    examples = []
    for trace in traces:
        query = _query_from_trace(trace)
        prompt = render_prompt(
            template,
            query,
            [],
            planner["candidate_count"],
            planner["prompt_input_tokens"] - VISUAL_TOKEN_RESERVE,
        )
        output = trace["metadata"]["planner_output"]
        parse_plan_output(json.dumps(output, separators=(",", ":")), planner["candidate_count"], query)
        examples.append(
            {
                "trace_id": trace["trace_id"],
                "query": query,
                "prompt": prompt,
                "visual": build_planner_visual(query),
                "target": json.dumps(output, separators=(",", ":")),
                "observed_utility": trace["metadata"]["observed_utility"],
            }
        )
    return examples


def _training_batch(
    processor: Any,
    prompt: str,
    visual: Any,
    target: str,
    input_token_budget: int,
    device: Any,
) -> dict[str, Any]:
    prompt_chat, image_inputs, video_inputs = _qwen_chat_and_vision(processor, prompt, visual)
    processor_arguments = {
        "images": image_inputs,
        "videos": video_inputs,
        "return_tensors": "pt",
    }
    prompt_inputs = processor(text=[prompt_chat], **processor_arguments)
    input_tokens = int(prompt_inputs["input_ids"].shape[-1])
    if input_tokens > input_token_budget:
        raise ValueError(f"tokenized multimodal planner input has {input_tokens} tokens; budget is {input_token_budget}")
    batch = processor(
        text=[prompt_chat + target],
        **processor_arguments,
    )
    labels = batch["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[-1]] = -100
    batch["labels"] = labels
    return batch.to(device)


def update_lora_adapter(
    config: dict[str, Any],
    region_id: str | int,
    adapter_root: str,
    example_limit: int,
    holdout_fraction: float,
    epochs: int,
    learning_rate: float,
    minimum_feasibility_rate: float,
    maximum_utility_mae: float,
    off_peak: bool,
    topology_changed: bool,
) -> dict[str, Any]:
    """Fine-tune QLoRA on successful traces and publish only a valid adapter."""

    if not adapter_update_due(off_peak, topology_changed):
        raise RuntimeError("adapter updates are restricted to off-peak or topology-change windows")
    fraction = _finite_number(holdout_fraction, "holdout_fraction")
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    _positive_integer(epochs, "epochs")
    rate = _finite_number(learning_rate, "learning_rate")
    if rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    minimum_rate = _finite_number(minimum_feasibility_rate, "minimum_feasibility_rate")
    maximum_mae = _finite_number(maximum_utility_mae, "maximum_utility_mae")
    if not 0.0 <= minimum_rate <= 1.0 or maximum_mae < 0.0:
        raise ValueError("adapter validation thresholds are invalid")
    examples = build_adapter_examples(config, region_id, example_limit, time.time())
    if len(examples) < 2:
        raise RuntimeError("adapter update requires at least two successful traces")
    holdout_count = max(1, math.ceil(len(examples) * fraction))
    training_examples = examples[holdout_count:]
    holdout_examples = examples[:holdout_count]
    if not training_examples:
        raise RuntimeError("adapter split contains no training examples")

    bundle = load_qwen_planner(config, True, None)
    model = bundle["model"]
    processor = bundle["processor"]
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("adapter updates require torch") from error
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=rate)
    model.train()
    device = next(model.parameters()).device
    planner = validate_planner_config(config)
    for _ in range(epochs):
        for example in training_examples:
            batch = _training_batch(
                processor,
                example["prompt"],
                example["visual"],
                example["target"],
                planner["prompt_input_tokens"],
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("adapter update produced a non-finite loss")
            loss.backward()
            optimizer.step()

    model.eval()
    feasible_count = 0
    utility_errors = []
    with torch.no_grad():
        for example in holdout_examples:
            output = generate_qwen_output(
                bundle,
                example["prompt"],
                example["visual"],
                planner["prompt_input_tokens"],
                planner["max_new_tokens"] * planner["candidate_count"],
            )
            decoded = parse_plan_output(output, planner["candidate_count"], example["query"])
            expanded = [
                {
                    **candidate,
                    "assignments": expand_candidate_assignments(candidate, example["query"]),
                }
                for candidate in decoded["candidates"]
            ]
            feasibility = [
                validate_candidate_feasibility(candidate, example["query"])["feasible"]
                for candidate in expanded
            ]
            if any(feasibility):
                feasible_count += 1
                predicted = max(
                    candidate["predicted_utility"]
                    for candidate, is_feasible in zip(decoded["candidates"], feasibility)
                    if is_feasible
                )
                utility_errors.append(semantic_drift(predicted, example["observed_utility"]))
    feasibility_rate = feasible_count / len(holdout_examples)
    utility_mae = sum(utility_errors) / len(utility_errors) if utility_errors else math.inf
    if feasibility_rate < minimum_rate or utility_mae > maximum_mae:
        raise RuntimeError(
            f"adapter validation failed: feasibility_rate={feasibility_rate:.4f}, utility_mae={utility_mae:.4f}"
        )

    version = f"region-{region_id}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    version_path = Path(adapter_root) / "versions" / version
    version_path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(version_path)
    manifest = {
        "schema_version": 1,
        "region_id": str(region_id),
        "adapter_path": str(version_path),
        "published_at": time.time(),
        "training_examples": len(training_examples),
        "holdout_examples": len(holdout_examples),
        "feasibility_rate": feasibility_rate,
        "utility_mae": utility_mae,
    }
    _atomic_write_json(str(Path(adapter_root) / "current.json"), manifest)
    return manifest


def _snapshot_queries(snapshot: Any) -> list[dict[str, Any]]:
    if isinstance(snapshot, dict) and set(snapshot.keys()) == QUERY_FIELDS:
        return [snapshot]
    if isinstance(snapshot, dict) and set(snapshot.keys()) == {"planner_queries"}:
        queries = snapshot["planner_queries"]
        if not isinstance(queries, list):
            raise TypeError("snapshot planner_queries must be a list")
        return queries
    raise ValueError("planner snapshot must be one query or an object containing only planner_queries")


def main(config_path: str = "configs/paper.json") -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    planner = validate_planner_config(config)
    initialize_memory(planner["memory_path"])
    summary: dict[str, Any] = {"memory": memory_stats(planner["memory_path"], time.time()), "prepared_windows": []}
    snapshot_path = Path(planner["snapshot_path"])
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        for query in _snapshot_queries(snapshot):
            prepared = prepare_planning_window(config, query, True)
            summary["prepared_windows"].append(
                {
                    "window_id": query["window_id"],
                    "controller_id": query["controller_id"],
                    "retrieved_traces": len(prepared["retrieved_traces"]),
                    "prompt_characters": len(prepared["prompt"]),
                    "visual_size": list(prepared["visual"].size),
                }
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
