"""Train the paper's decentralized edge policies on one shared simulator.

The file is intentionally executable without a command-line interface.  Its
functions are also reused by ``evaluate.py`` so training and evaluation cannot
drift into different action or reward semantics.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.mappo import (
    add_rollout_entry,
    blend_semantic_parameters,
    build_models,
    compute_gae,
    configuration_hash,
    critic_value,
    gumbel_temperature,
    initial_hidden,
    load_checkpoint,
    sample_action,
    save_checkpoint,
    semantic_placement_prior,
    trust_coefficient,
    update_mappo,
)
from baselines import (
    ABLATION_METHODS,
    METHODS,
    build_cromadqn,
    deterministic_routing,
    load_cromadqn_checkpoint,
    method_id,
    method_profile,
    push_dqn_transition,
    save_cromadqn_checkpoint,
    select_cromadqn_action,
    update_cromadqn,
)
from env.constants import load_config
from env.envMulti import (
    controller_observation,
    global_state_vector,
    inject_node_failures,
    pending_controller_decisions,
    planner_query,
    prepare_scenario,
    reset_environment,
    sample_environment,
    scenario_json,
    schedule_burst_trace,
    schedule_diurnal_trace,
    step_environment,
)
from env.helper import controller_utilization
from evaluate import (
    append_training_record,
    convergence_episode,
    planner_worker_rows,
    result_row,
)
from llmPlanner import (
    adapter_update_requested,
    build_adapter_examples,
    load_qwen_planner,
    record_plan_outcome,
    run_planning_window,
    semantic_drift,
    update_lora_adapter,
    validate_candidate_feasibility,
)

CORE_METHODS = ("LLM-MAPPO", "RSPPO", "GNN-DRL", "mu-DDRL", "CROMADQN")
LOAD_METHODS = METHODS
ABLATION_FIGURE_METHODS = {
    "full": "LLM-MAPPO",
    "no_toporag": "LLM-MAPPO-no-TopoRAG",
    "no_reward_shaping": "LLM-MAPPO-no-reward-shaping",
    "no_llm_guidance": "MAPPO-no-LLM-guidance",
}
SENSITIVITY_RESPONSES = (
    "end_to_end_latency",
    "sla_violation_rate",
    "convergence_time",
    "resource_efficiency",
    "planning_overhead",
    "adaptation_quality",
)
BURST_TIMELINE_S = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0)
SEMANTIC_DRIFT_WINDOWS = 10
DIURNAL_ACCELERATED_DURATION_S = 14_400.0
SENSITIVITY_TRACE_DURATION_S = 2_400.0


def _sensitivity_conditions(config: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Return explicit low/high one-factor configurations around the paper setup."""

    conditions: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    settings = {
        "lora_rank": (("planner", "lora", "rank"), 8, 32),
        "frozen_layer_ratio": (("planner", "lora", "trainable_fraction"), 0.50, 0.10),
        "planning_frequency": (("planner", "planning_interval_s"), 1200, 300),
        "learning_rate": (("training", "actor_lr"), 0.0001, 0.0005),
        "clip_epsilon": (("training", "clip_ratio"), 0.10, 0.30),
        "entropy_coefficient": (("training", "entropy_start"), 0.005, 0.020),
        "trust_threshold": (("training", "activation_threshold"), 0.005, 0.050),
        "interpolation_decay": (("training", "kl_decay_steps"), 500, 2000),
    }
    for name, (path, low, high) in settings.items():
        variants = []
        for value in (low, high):
            variant = copy.deepcopy(config)
            target: dict[str, Any] = variant
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            if name == "entropy_coefficient":
                variant["training"]["entropy_end"] = float(value) / 10.0
            variants.append(variant)
        conditions[name] = (variants[0], variants[1])
    return conditions


def _artifact_parts(artifact_key: str | None) -> tuple[str, ...]:
    if artifact_key is None:
        return ()
    parts = tuple(part for part in artifact_key.split("/") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid experiment artifact key {artifact_key!r}")
    return ("experiments", *parts)


def _checkpoint_path(
    config: dict[str, Any], method: str, seed: int, artifact_key: str | None
) -> Path:
    return (
        Path(config["paths"]["checkpoints"])
        .joinpath(*_artifact_parts(artifact_key), method_id(method), f"seed-{seed}", "latest.pt")
    )


def _training_log_path(
    config: dict[str, Any], method: str, seed: int, artifact_key: str | None
) -> Path:
    base = Path(config["paths"]["training_logs"])
    if artifact_key is None:
        return base / f"{method_id(method)}-seed-{seed}.jsonl"
    return base.joinpath(*_artifact_parts(artifact_key), f"{method_id(method)}-seed-{seed}.jsonl")


def _paper_diurnal_rates(config: dict[str, Any]) -> list[float]:
    rates = config["workloads"]["arrival_rates"]
    off_peak = float(rates["suburban"][0])
    urban_peak = float(rates["urban"][1])
    profile = np.asarray(
        (
            0.012,
            0.005,
            0.000,
            0.000,
            0.005,
            0.012,
            0.050,
            0.180,
            0.360,
            0.520,
            0.650,
            0.760,
            0.840,
            0.880,
            0.820,
            0.740,
            0.690,
            0.740,
            0.850,
            1.000,
            0.940,
            0.740,
            0.440,
            0.130,
        ),
        dtype=np.float64,
    )
    return (off_peak + profile * (urban_peak - off_peak)).tolist()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _runtime_config(
    config: dict[str, Any],
    method: str,
    seed: int,
    artifact_key: str | None = None,
) -> dict[str, Any]:
    """Isolate planner state for one method/seed without expanding user config."""

    runtime = copy.deepcopy(config)
    directory = (
        Path(config["paths"]["checkpoints"])
        .joinpath(*_artifact_parts(artifact_key), "planner", method_id(method), f"seed-{seed}")
    )
    memory_path = directory / "toporag.sqlite"
    latest_path = directory / "latest_plan.json"
    snapshot_path = directory / "snapshot.json"
    runtime["planner"]["memory_path"] = str(memory_path)
    runtime["planner"]["latest_plan_path"] = str(latest_path)
    runtime["planner"]["snapshot_path"] = str(snapshot_path)
    return runtime


def _adapter_root(config: dict[str, Any]) -> Path:
    return Path(config["planner"]["memory_path"]).with_name("adapter")


def _published_adapter_path(config: dict[str, Any]) -> str | None:
    manifest_path = _adapter_root(config) / "current.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "region_id",
        "adapter_path",
        "published_at",
        "training_examples",
        "holdout_examples",
        "feasibility_rate",
        "utility_mae",
    }
    if set(manifest) != required:
        raise ValueError(f"Published adapter manifest has an invalid schema: {manifest_path}")
    adapter_path = Path(manifest["adapter_path"])
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Published planner adapter is missing: {adapter_path}")
    return str(adapter_path)


def _method_observation(
    state: dict[str, Any], controller_id: int, method: str
) -> dict[str, Any]:
    observation = controller_observation(state, controller_id)
    if not bool(method_profile(method)["neighbour_summary"]):
        return observation
    updated = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }
    controller_count = int(state["config"]["simulation"]["edge_controllers"])
    neighbours = {
        (controller_id - 1) % controller_count,
        (controller_id + 1) % controller_count,
    }
    utilization = controller_utilization(state)
    queues = np.asarray(
        [len(state["controller_queues"][index]) for index in range(controller_count)],
        dtype=np.float32,
    ) / float(len(state["scenario"]["requests"]))
    context = np.asarray(updated["context"], dtype=np.float32)
    semantic_dim = int(state["config"]["training"]["semantic_context_dim"])
    start = len(context) - semantic_dim
    context[start : start + 4] = [
        float(utilization[controller_id]),
        float(np.mean(utilization[list(neighbours)])),
        float(queues[controller_id]),
        float(np.mean(queues[list(neighbours)])),
    ]
    updated["context"] = context
    return updated


def _critic_state(
    config: dict[str, Any],
    state: dict[str, Any],
    observation: dict[str, Any],
    method: str,
) -> np.ndarray:
    if bool(method_profile(method)["central_critic"]):
        return global_state_vector(state)
    infrastructure = np.asarray(observation["infrastructure_nodes"], dtype=np.float32)
    infrastructure_mask = np.asarray(observation["infrastructure_mask"], dtype=bool)
    service = np.asarray(observation["service_nodes"], dtype=np.float32)
    service_mask = np.asarray(observation["service_mask"], dtype=bool)
    path = np.asarray(observation["path_features"], dtype=np.float32)
    path_mask = np.asarray(observation["path_mask"], dtype=bool)
    valid_path_features = path[path_mask]
    local_infrastructure = infrastructure[infrastructure_mask]
    local_service = service[service_mask]
    current_markers = local_service[:, -2]
    if np.count_nonzero(current_markers) != 1:
        raise ValueError("decentralized critic requires one current microservice marker")
    current_service = local_service[int(np.argmax(current_markers))]
    context = np.asarray(observation["context"], dtype=np.float32)
    mean_path = (
        valid_path_features.mean(axis=0)
        if len(valid_path_features)
        else np.zeros(path.shape[-1], dtype=np.float32)
    )
    local = np.concatenate(
        [
            local_infrastructure.mean(axis=0),
            local_infrastructure.max(axis=0)[[0, 1, 2, 3, 19, 20, 21]],
            current_service,
            local_service[:, :6].mean(axis=0),
            context[:9],
            context[12:16],
            mean_path[[0, 1, 2, 4]],
        ]
    )
    required = int(config["training"]["global_state_dim"])
    if local.size > required:
        raise ValueError("training.global_state_dim cannot hold the decentralized critic state")
    return np.pad(local, (0, required - local.size)).astype(np.float32)


def _placement_id(decision: dict[str, Any]) -> str:
    return (
        f"{int(decision['request_id'])}/"
        f"{int(decision['microservice_id'])}/"
        f"{int(decision['replica_id'])}"
    )


def _plan_prior(
    plan: dict[str, Any] | None,
    decision: dict[str, Any],
    local_node_ids: list[int],
    placement_size: int,
) -> np.ndarray:
    if plan is None:
        return np.zeros(placement_size, dtype=np.float32)
    placement_id = _placement_id(decision)
    if placement_id in plan["placement_prior"]:
        node_probabilities = plan["placement_prior"][placement_id]
    else:
        feasible = [
            candidate
            for candidate in plan["candidates"]
            if bool(candidate["feasibility"]["feasible"])
        ]
        maximum_utility = max(
            float(candidate["predicted_utility"]) for candidate in feasible
        )
        node_probabilities: dict[str, float] = {}
        for candidate in feasible:
            weight = math.exp(
                float(candidate["predicted_utility"]) - maximum_utility
            )
            assignments = candidate["assignments"][:-1]
            for _, node_id in assignments:
                key = str(node_id)
                node_probabilities[key] = node_probabilities.get(key, 0.0) + weight
            for rank, node_id in enumerate(candidate["node_preferences"][:-1]):
                key = str(node_id)
                node_probabilities[key] = node_probabilities.get(key, 0.0) + (
                    weight * len(assignments) / (rank + 1)
                )
        total = sum(node_probabilities.values())
        node_probabilities = {
            node_id: probability / total
            for node_id, probability in node_probabilities.items()
        }
    prior = semantic_placement_prior(
        {"placement_prior": node_probabilities}, local_node_ids
    )
    padded = np.zeros(placement_size, dtype=np.float32)
    padded[: len(prior)] = prior
    return padded


def _inject_plan_context(observation: dict[str, Any], plan: dict[str, Any] | None, semantic_dim: int) -> dict[str, Any]:
    updated = {key: np.asarray(value).copy() if isinstance(value, (list, np.ndarray)) else value for key, value in observation.items()}
    context = np.asarray(updated["context"], dtype=np.float32).copy()
    if semantic_dim > len(context):
        raise ValueError("training.semantic_context_dim exceeds the observation context size")
    context[-semantic_dim:] = 0.0
    if plan is not None:
        embedding = np.asarray(plan["plan_embedding"], dtype=np.float32).reshape(-1)
        context[-semantic_dim:] = embedding[:semantic_dim]
    updated["context"] = context
    return updated


def _controller_ids(scenario: dict[str, Any]) -> list[int]:
    return list(range(int(scenario["config"]["simulation"]["edge_controllers"])))


def _controller_model(
    models: dict[str, Any], method: str, controller_id: int
) -> dict[str, Any]:
    if method_profile(method)["algorithm"] == "independent_ppo":
        return models["controller_models"][controller_id]
    return models


def _bootstrap_value(
    config: dict[str, Any],
    state: dict[str, Any],
    method: str,
    models: dict[str, Any],
    controller_id: int,
    done: bool,
) -> float:
    if done:
        return 0.0
    observation = _method_observation(state, controller_id, method)
    if int(observation["request_id"]) < 0:
        return 0.0
    critic_state = _critic_state(config, state, observation, method)
    return critic_value(
        _controller_model(models, method, controller_id), critic_state
    )


def _update_independent_controller(
    models: dict[str, Any],
    rollout: list[dict[str, Any]],
    controller_id: int,
    last_value: float,
    training_config: dict[str, Any],
) -> None:
    controller_rollout = [
        entry
        for entry in rollout
        if int(entry["controller_id"]) == controller_id
    ]
    if not controller_rollout:
        return
    compute_gae(
        controller_rollout,
        {controller_id: last_value},
        float(training_config["gamma"]),
        float(training_config["gae_lambda"]),
    )
    update_mappo(
        models["controller_models"][controller_id],
        controller_rollout,
        training_config,
    )
    rollout[:] = [
        entry
        for entry in rollout
        if int(entry["controller_id"]) != controller_id
    ]


def _planner_model(config: dict[str, Any], methods: list[str]) -> dict[str, Any] | None:
    semantic_methods = [method for method in methods if bool(method_profile(method)["semantic_guidance"])]
    if not semantic_methods:
        return None
    return load_qwen_planner(
        config,
        trainable=False,
        adapter_path=_published_adapter_path(config),
    )


def _plans_for_state(
    config: dict[str, Any],
    state: dict[str, Any],
    method: str,
    planner_model: dict[str, Any] | None,
    previous_plans: dict[int, dict[str, Any]],
    measurements: list[dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    profile = method_profile(method)
    if not bool(profile["semantic_guidance"]):
        return {}
    if planner_model is None:
        raise RuntimeError("Semantic guidance requires an explicitly loaded Qwen2-VL planner")
    sim_time = float(state["time_s"])
    plans = dict(previous_plans)
    for controller_id in sorted(pending_controller_decisions(state)):
        current = plans.get(controller_id)
        if current is not None and sim_time < float(current["valid_until_s"]):
            continue
        query = planner_query(state, controller_id)
        started = time.perf_counter()
        plan = run_planning_window(
            config,
            query,
            planner_model,
            adapter_path=None,
            use_toporag=bool(profile["toporag"]),
        )
        elapsed = time.perf_counter() - started
        plans[controller_id] = plan
        if measurements is not None:
            measurements.append(
                {
                    "controller_id": int(controller_id),
                    "sim_time_s": sim_time,
                    "wall_time_s": elapsed,
                    "query": query,
                    "plan": plan,
                    "outcomes": [],
                }
            )
    return plans


def _semantic_parameters(
    config: dict[str, Any],
    method: str,
    plan: dict[str, Any] | None,
    previous_return: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    reward_config = config["reward"]
    static_weights = [
        float(reward_config["local_weights"][name])
        for name in ("latency", "resource", "load_balance", "reliability")
    ]
    static_penalties = [
        float(reward_config["penalty_weights"][name])
        for name in ("resource", "temporal", "robustness")
    ]
    static_cooperation = float(reward_config["cooperation"])
    if method == "LLM-MAPPO-no-reward-shaping":
        return (
            np.asarray(static_weights, dtype=np.float64),
            np.asarray(static_penalties, dtype=np.float64),
            static_cooperation,
            0.0,
        )
    if not bool(method_profile(method)["reward_shaping"]):
        static_penalties = [0.0, 0.0, 0.0]
        static_cooperation = 1.0 if method != "mu-DDRL" else 0.0
        return (
            np.asarray(static_weights, dtype=np.float64),
            np.asarray(static_penalties, dtype=np.float64),
            static_cooperation,
            0.0,
        )
    if plan is None:
        return (
            np.asarray(static_weights, dtype=np.float64),
            np.asarray(static_penalties, dtype=np.float64),
            static_cooperation,
            0.0,
        )
    trust = trust_coefficient(
        float(plan["predicted_utility"]),
        previous_return,
        float(config["training"]["trust_alpha"]),
        float(config["training"]["trust_cap"]),
    )
    semantic_plan = {
        "reward_weights": [
            float(plan["reward_weights"][name])
            for name in ("latency", "resource_utilization", "load_balance", "reliability")
        ],
        "penalty_weights": [
            float(plan["penalty_weights"][name])
            for name in ("resource", "temporal", "robustness")
        ],
        "cooperation_bias": float(plan["cooperation_bias"]),
    }
    reward_weights, penalties, cooperation = blend_semantic_parameters(
        static_weights, static_penalties, static_cooperation, semantic_plan, trust
    )
    return reward_weights, penalties, cooperation, trust


def _action_record(
    decision: dict[str, Any],
    node_id: int,
    routing_fractions: np.ndarray,
    reward_weights: np.ndarray,
    penalties: np.ndarray,
    cooperation: float,
) -> dict[str, Any]:
    return {
        "controller_id": int(decision["controller_id"]),
        "request_id": int(decision["request_id"]),
        "microservice_id": int(decision["microservice_id"]),
        "replica_id": int(decision["replica_id"]),
        "node_id": int(node_id),
        "routing_fractions": np.asarray(routing_fractions, dtype=np.float64).tolist(),
        "reward_weights": np.asarray(reward_weights, dtype=np.float64).tolist(),
        "penalty_weights": np.asarray(penalties, dtype=np.float64).tolist(),
        "cooperation": float(cooperation),
    }


def _episode_metrics(infos: list[dict[str, Any]], rewards: list[float]) -> dict[str, float]:
    if not infos:
        return {
            "normalized_cumulative_reward": 0.0,
            "end_to_end_latency_ms": 0.0,
            "load_balance_index": 0.0,
            "sla_violation_rate_percent": 0.0,
            "compute_utilization_percent": 0.0,
            "bandwidth_utilization_percent": 0.0,
        }
    events = [event for info in infos for event in info["controller_events"]]
    terminal = [
        event
        for event in events
        if bool(event["request_completed"]) or not bool(event["accepted"])
    ]
    completed = [
        event
        for event in terminal
        if bool(event["request_completed"]) and bool(event["accepted"])
    ]
    normalized = [float(info["normalized_reward"]) for info in infos]
    if completed:
        end_to_end_latency_ms = float(
            np.mean([float(event["latency_ms"]) for event in completed])
        )
    else:
        # No completed workflow has an observable end-to-end latency. Treat the
        # episode horizon as a right-censored lower bound instead of averaging
        # partial-task latency from rejected workflows.
        end_to_end_latency_ms = 1000.0 * max(
            float(info["time_s"]) for info in infos
        )
    load = [float(info["load_balance_index"]) for info in infos]
    violations = [bool(event["sla_violation"]) for event in terminal]
    compute = [float(info["compute_utilization_percent"]) for info in infos]
    bandwidth = [float(info["bandwidth_utilization_percent"]) for info in infos]
    return {
        "normalized_cumulative_reward": float(np.mean(normalized)),
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "load_balance_index": float(np.mean(load)),
        "sla_violation_rate_percent": 100.0 * float(np.mean(violations)) if violations else 0.0,
        "compute_utilization_percent": float(np.mean(compute)),
        "bandwidth_utilization_percent": float(np.mean(bandwidth)),
        "signed_return": float(np.sum(rewards)),
    }


def _measurement_sample(state: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    raw = sample_environment(state)
    return {
        "time_s": float(raw["time_s"]),
        "normalized_reward": float(info["normalized_reward"]),
        "mean_latency_ms": float(info["mean_latency_ms"]),
        "throughput_rps": float(raw["throughput_rps"]),
        "availability_percent": float(raw["availability_percent"]),
        "load_balance_index": float(raw["load_balance_index"]),
        "compute_utilization_percent": float(raw["compute_utilization_percent"]),
        "bandwidth_utilization_percent": float(raw["bandwidth_utilization_percent"]),
        "node_utilization_percent": np.asarray(
            raw["per_node_utilization_percent"], dtype=np.float64
        ).tolist(),
        "completed_requests": len(state["completed_requests"]),
        "rejected_requests": len(state["rejected_requests"]),
        "controller_events": scenario_json(info["controller_events"]),
    }


def _append_planner_outcome(
    planner_windows: list[dict[str, Any]],
    plan: dict[str, Any],
    action: dict[str, Any],
    reward: dict[str, Any],
    event: dict[str, Any],
    info: dict[str, Any],
) -> None:
    plan_id = str(plan["plan_id"])
    for window in reversed(planner_windows):
        if str(window["plan"]["plan_id"]) != plan_id:
            continue
        placement_id = (
            f"{action['request_id']}/{action['microservice_id']}/{action['replica_id']}"
        )
        window["outcomes"].append(
            {
                "time_s": float(info["time_s"]),
                "assignment": [placement_id, str(action["node_id"])],
                "reward": float(reward["total"]),
                "normalized_reward": float(reward["normalized_reward"]),
                "load_balance_index": float(info["load_balance_index"]),
                "event": scenario_json(event),
            }
        )
        return
    raise RuntimeError(f"planner outcome has no matching planning window {plan_id}")


def _close_planner_windows(
    config: dict[str, Any],
    planner_windows: list[dict[str, Any]],
    *,
    record_memory: bool,
) -> None:
    for window in planner_windows:
        plan = window["plan"]
        outcomes = window["outcomes"]
        if not outcomes:
            raise RuntimeError(f"planning window {plan['plan_id']} has no controller outcomes")
        normalized_reward = float(
            np.mean([float(outcome["normalized_reward"]) for outcome in outcomes])
        )
        observed_utility = 2.0 * normalized_reward - 1.0
        events = [outcome["event"] for outcome in outcomes]
        terminal = [
            event
            for event in events
            if bool(event["request_completed"]) or not bool(event["accepted"])
        ]
        completed = [
            event
            for event in terminal
            if bool(event["request_completed"]) and bool(event["accepted"])
        ]
        if completed:
            mean_latency_ms = float(
                np.mean([float(event["latency_ms"]) for event in completed])
            )
        else:
            mean_latency_ms = 1000.0 * (
                float(plan["valid_until_s"]) - float(plan["valid_from_s"])
            )
        violations = [
            bool(event["sla_violation"]) or not bool(event["accepted"])
            for event in terminal
        ]
        sla_violation_rate = (
            100.0 * float(np.mean(violations)) if violations else 0.0
        )
        placement_routing = {
            "assignments": [outcome["assignment"] for outcome in outcomes]
            + ["<eop>"]
        }
        rollout_trace = {
            "steps": len(outcomes),
            "completed_requests": sum(
                bool(event["request_completed"]) and bool(event["accepted"])
                for event in terminal
            ),
            "rejected_requests": sum(not bool(event["accepted"]) for event in terminal),
            "events": terminal,
        }
        qos_summary = {
            "mean_latency_ms": mean_latency_ms,
            "sla_violation_rate_percent": sla_violation_rate,
            "load_balance_index": float(
                np.mean(
                    [float(outcome["load_balance_index"]) for outcome in outcomes]
                )
            ),
        }
        drift = semantic_drift(float(plan["predicted_utility"]), observed_utility)
        window["observed_utility"] = observed_utility
        window["semantic_drift"] = drift
        window["outcome_summary"] = {
            "steps": len(outcomes),
            "completed_requests": rollout_trace["completed_requests"],
            "rejected_requests": rollout_trace["rejected_requests"],
            "mean_latency_ms": mean_latency_ms,
            "sla_violation_rate_percent": sla_violation_rate,
        }
        del window["outcomes"]
        window["adapter_update_requested"] = adapter_update_requested(
            float(plan["predicted_utility"]),
            observed_utility,
            float(config["planner"]["drift_threshold"]),
        )
        if record_memory:
            window["trace_id"] = record_plan_outcome(
                config,
                window["query"],
                plan,
                placement_routing,
                rollout_trace,
                qos_summary,
                observed_utility,
                max(0.0, 1.0 - min(1.0, drift)),
                bool(terminal and sla_violation_rate == 0.0),
                time.time(),
            )
            timing_path = Path(config["planner"]["memory_path"]).with_name(
                "planner_windows.jsonl"
            )
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            with timing_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "trace_id": window["trace_id"],
                            "controller_id": window["controller_id"],
                            "sim_time_s": window["sim_time_s"],
                            "wall_time_s": window["wall_time_s"],
                            "retrieved_traces": len(plan["retrieval_trace_ids"]),
                            "semantic_drift": drift,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            if window["adapter_update_requested"]:
                queue_path = Path(config["planner"]["memory_path"]).with_name(
                    "adapter_queue.jsonl"
                )
                queue_path.parent.mkdir(parents=True, exist_ok=True)
                with queue_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "trace_id": window["trace_id"],
                                "controller_id": window["controller_id"],
                                "predicted_utility": plan["predicted_utility"],
                                "observed_utility": observed_utility,
                                "semantic_drift": drift,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )


def run_mappo_episode(
    config: dict[str, Any],
    scenario: dict[str, Any],
    method: str,
    models: dict[str, Any],
    hidden: dict[int, torch.Tensor],
    planner_model: dict[str, Any] | None,
    plans: dict[int, dict[str, Any]],
    previous_returns: dict[int, float],
    rollout: list[dict[str, Any]],
    episode: int,
    *,
    training: bool,
    capture: dict[str, Any] | None = None,
    state_hook: Any | None = None,
) -> tuple[dict[str, float], dict[int, dict[str, Any]], int]:
    state = reset_environment(scenario, seed=int(scenario["seed"]) + episode)
    plans = {}
    for controller_id, controller_hidden in hidden.items():
        controller_hidden.zero_()
        previous_returns[controller_id] = 0.0
    planner_windows: list[dict[str, Any]] = []
    planning_queries: list[dict[str, Any]] = []
    action_trials: list[dict[str, Any]] = []
    captured_query_windows: set[tuple[int, int]] = set()
    samples: list[dict[str, Any]] = []
    if state_hook is not None:
        state_hook(state)
    infos: list[dict[str, Any]] = []
    episode_rewards: list[float] = []
    steps = 0
    profile = method_profile(method)
    total_episodes = int(config["training"]["max_episodes"])
    temperature = gumbel_temperature(
        episode,
        total_episodes,
        float(config["training"]["gumbel_start"]),
        float(config["training"]["gumbel_end"]),
    )

    while not bool(state["done"]):
        if state_hook is not None:
            state_hook(state)
        decisions = pending_controller_decisions(state)
        if capture is not None:
            interval = int(config["planner"]["planning_interval_s"])
            window_id = int(float(state["time_s"]) // interval)
            for controller_id in sorted(decisions):
                key = (int(controller_id), window_id)
                if key not in captured_query_windows:
                    planning_queries.append(planner_query(state, int(controller_id)))
                    captured_query_windows.add(key)
        if not decisions:
            state, _, done, info = step_environment(state, [], dt_s=float(config["simulation"]["decision_interval_s"]))
            infos.append(info)
            if capture is not None:
                samples.append(_measurement_sample(state, info))
            steps += 1
            if done:
                break
            continue

        plans = _plans_for_state(
            config,
            state,
            method,
            planner_model,
            plans,
            planner_windows,
        )
        pending_entries: dict[int, dict[str, Any]] = {}
        step_trials: dict[int, dict[str, Any]] = {}
        actions = []
        for controller_id, decision in sorted(decisions.items()):
            observation = _method_observation(state, int(controller_id), method)
            plan = plans.get(int(controller_id)) if bool(profile["semantic_guidance"]) else None
            if plan is not None and not (
                float(plan["valid_from_s"])
                <= float(state["time_s"])
                < float(plan["valid_until_s"])
            ):
                raise RuntimeError("edge action received an expired semantic plan")
            if bool(profile["semantic_guidance"]):
                observation = _inject_plan_context(
                    observation, plan, int(config["training"]["semantic_context_dim"])
                )
            local_node_ids = [int(value) for value in decision["local_node_ids"]]
            prior = _plan_prior(plan, decision, local_node_ids, len(observation["placement_mask"]))
            prior *= np.asarray(observation["placement_mask"], dtype=np.float32)
            if float(prior.sum()) > 0.0:
                prior /= float(prior.sum())
            reward_weights, penalties, cooperation, trust = _semantic_parameters(
                config, method, plan, previous_returns[int(controller_id)]
            )
            critic_state = _critic_state(config, state, observation, method)
            policy_model = _controller_model(models, method, int(controller_id))
            value = critic_value(policy_model, critic_state)
            hidden_before = hidden[int(controller_id)].detach()
            selected = sample_action(
                policy_model,
                observation,
                hidden_before,
                temperature,
                prior=prior,
                trust=trust,
                activation_threshold=float(config["training"]["activation_threshold"]),
                deterministic=not training,
            )
            placement_index = int(selected["placement_index"])
            if placement_index >= len(local_node_ids):
                raise ValueError("Actor selected padded placement entry")
            hidden[int(controller_id)] = selected["hidden"]
            action = _action_record(
                decision,
                local_node_ids[placement_index],
                selected["routing_fractions"],
                reward_weights,
                penalties,
                cooperation,
            )
            actions.append(action)
            if capture is not None:
                trial = {
                    "controller_id": int(controller_id),
                    "time_s": float(state["time_s"]),
                    "placement_id": _placement_id(decision),
                    "node_id": int(action["node_id"]),
                    "predicted_utility": float(np.tanh(value)),
                }
                action_trials.append(trial)
                step_trials[int(controller_id)] = trial
            pending_entries[int(controller_id)] = {
                "controller_id": int(controller_id),
                "observation": observation,
                "hidden": hidden_before,
                "global_state": critic_state,
                "placement_index": placement_index,
                "routing_fractions": selected["routing_fractions"],
                "route_count": selected["route_count"],
                "old_log_prob": selected["log_prob"],
                "value": value,
                "prior": prior,
                "trust": trust,
                "temperature": temperature,
            }

        state, rewards, done, info = step_environment(
            state, actions, dt_s=float(config["simulation"]["decision_interval_s"])
        )
        infos.append(info)
        if capture is not None:
            samples.append(_measurement_sample(state, info))
        events_by_controller = {
            int(event["controller_id"]): event
            for event in info["controller_events"]
        }
        for controller_id, entry in pending_entries.items():
            reward = float(rewards[controller_id]["total"])
            episode_rewards.append(reward)
            previous_returns[controller_id] = (
                2.0 * float(rewards[controller_id]["normalized_reward"]) - 1.0
            )
            if controller_id in step_trials:
                step_trials[controller_id]["normalized_reward"] = float(
                    rewards[controller_id]["normalized_reward"]
                )
            plan = plans.get(controller_id)
            if plan is not None and bool(profile["semantic_guidance"]):
                _append_planner_outcome(
                    planner_windows,
                    plan,
                    next(
                        action
                        for action in actions
                        if int(action["controller_id"]) == controller_id
                    ),
                    rewards[controller_id],
                    events_by_controller[controller_id],
                    info,
                )
            if training:
                add_rollout_entry(
                    rollout,
                    **entry,
                    reward=reward,
                    done=bool(done),
                    episode=episode,
                )
        steps += 1
        if training:
            rollout_steps = int(config["training"]["rollout_steps"])
            if profile["algorithm"] == "independent_ppo":
                for controller_id in hidden:
                    controller_entries = sum(
                        int(entry["controller_id"]) == controller_id
                        for entry in rollout
                    )
                    if controller_entries < rollout_steps:
                        continue
                    last_value = _bootstrap_value(
                        config,
                        state,
                        method,
                        models,
                        controller_id,
                        bool(done),
                    )
                    _update_independent_controller(
                        models,
                        rollout,
                        controller_id,
                        last_value,
                        config["training"],
                    )
            elif len(rollout) >= rollout_steps:
                last_values = {
                    controller_id: _bootstrap_value(
                        config,
                        state,
                        method,
                        models,
                        controller_id,
                        bool(done),
                    )
                    for controller_id in hidden
                }
                compute_gae(
                    rollout,
                    last_values,
                    float(config["training"]["gamma"]),
                    float(config["training"]["gae_lambda"]),
                )
                update_mappo(models, rollout, config["training"])
                rollout.clear()
        if done:
            break
    if training:
        unfinished = set(hidden)
        for entry in reversed(rollout):
            controller_id = int(entry["controller_id"])
            if controller_id in unfinished and int(entry["episode"]) == episode:
                entry["done"] = True
                unfinished.remove(controller_id)
            if not unfinished:
                break
    metrics = _episode_metrics(infos, episode_rewards)
    _close_planner_windows(
        config,
        planner_windows,
        record_memory=training and bool(method_profile(method)["toporag"]),
    )
    if capture is not None:
        capture.update(
            {
                "scenario_hash": str(scenario["scenario_hash"]),
                "planning_interval_s": int(config["planner"]["planning_interval_s"]),
                "samples": samples,
                "planner_windows": scenario_json(planner_windows),
                "planning_queries": scenario_json(planning_queries),
                "action_trials": scenario_json(action_trials),
                "terminal_requests": scenario_json(
                    [
                        event
                        for info in infos
                        for event in info["controller_events"]
                        if bool(event["request_completed"]) or not bool(event["accepted"])
                    ]
                ),
                "node_coordinates_km": scenario["topology"]["coordinates_km"].tolist(),
            }
        )
    return metrics, plans, steps


def run_cromadqn_episode(
    config: dict[str, Any],
    scenario: dict[str, Any],
    model: dict[str, Any],
    episode: int,
    *,
    training: bool,
    capture: dict[str, Any] | None = None,
    state_hook: Any | None = None,
) -> tuple[dict[str, float], int]:
    state = reset_environment(scenario, seed=int(scenario["seed"]) + episode)
    rng = np.random.default_rng(int(scenario["seed"]) + episode)
    infos: list[dict[str, Any]] = []
    rewards_seen: list[float] = []
    steps = 0
    samples: list[dict[str, Any]] = []
    epsilon = max(0.05, 1.0 - episode / max(1, int(config["training"]["max_episodes"]) - 1))
    while not bool(state["done"]):
        if state_hook is not None:
            state_hook(state)
        decisions = pending_controller_decisions(state)
        if not decisions:
            state, _, done, info = step_environment(state, [], dt_s=float(config["simulation"]["decision_interval_s"]))
            infos.append(info)
            if capture is not None:
                samples.append(_measurement_sample(state, info))
            steps += 1
            if done:
                break
            continue
        actions = []
        transitions = []
        for controller_id, decision in sorted(decisions.items()):
            observation = _method_observation(state, int(controller_id), "CROMADQN")
            action_index = select_cromadqn_action(model, observation, epsilon if training else 0.0, rng)
            local_node_ids = [int(value) for value in decision["local_node_ids"]]
            routing = deterministic_routing(
                observation["path_features"],
                observation["path_mask"],
                observation["incoming_edge_mask"],
                action_index,
            )
            reward_weights, penalties, cooperation, _ = _semantic_parameters(
                config, "CROMADQN", None, 0.0
            )
            actions.append(
                _action_record(
                    decision,
                    local_node_ids[action_index],
                    routing,
                    reward_weights,
                    penalties,
                    cooperation,
                )
            )
            transitions.append((int(controller_id), observation, action_index))
        next_state, rewards, done, info = step_environment(
            state, actions, dt_s=float(config["simulation"]["decision_interval_s"])
        )
        infos.append(info)
        if capture is not None:
            samples.append(_measurement_sample(next_state, info))
        for controller_id, observation, action_index in transitions:
            next_observation = _method_observation(next_state, controller_id, "CROMADQN")
            reward = float(rewards[controller_id]["total"])
            rewards_seen.append(reward)
            if training:
                terminal = bool(done) or int(next_observation["request_id"]) < 0
                push_dqn_transition(
                    model,
                    (observation, action_index, reward, next_observation, terminal),
                )
                update_cromadqn(model, config["training"])
        state = next_state
        steps += 1
        if done:
            break
    metrics = _episode_metrics(infos, rewards_seen)
    if capture is not None:
        capture.update(
            {
                "scenario_hash": str(scenario["scenario_hash"]),
                "samples": samples,
                "planner_windows": [],
                "planning_queries": [],
                "action_trials": [],
                "terminal_requests": scenario_json(
                    [
                        event
                        for info in infos
                        for event in info["controller_events"]
                        if bool(event["request_completed"]) or not bool(event["accepted"])
                    ]
                ),
                "node_coordinates_km": scenario["topology"]["coordinates_km"].tolist(),
            }
        )
    return metrics, steps


def _build_method_state(
    config: dict[str, Any], scenario: dict[str, Any], method: str, device: torch.device
) -> tuple[dict[str, Any], dict[int, torch.Tensor] | None]:
    state = reset_environment(scenario, seed=int(scenario["seed"]))
    decisions = pending_controller_decisions(state)
    while not decisions and not state["done"]:
        state, _, _, _ = step_environment(state, [], dt_s=float(config["simulation"]["decision_interval_s"]))
        decisions = pending_controller_decisions(state)
    if not decisions:
        raise RuntimeError("Prepared scenario produced no controller decisions")
    controller_id = int(min(decisions))
    observation = _method_observation(state, controller_id, method)
    profile = method_profile(method)
    if profile["algorithm"] == "dqn":
        return build_cromadqn(observation, config["training"], device), None
    if profile["algorithm"] == "independent_ppo":
        models = {
            "controller_models": {
                candidate_controller: build_models(
                    observation,
                    int(config["training"]["global_state_dim"]),
                    config["training"],
                    device,
                )
                for candidate_controller in _controller_ids(scenario)
            },
            "device": device,
        }
    else:
        models = build_models(
            observation,
            int(config["training"]["global_state_dim"]),
            config["training"],
            device,
        )
    hidden = initial_hidden(
        len(_controller_ids(scenario)), int(config["training"]["gru_hidden"]), device
    )
    return models, hidden


def _publish_queued_adapter(config: dict[str, Any]) -> dict[str, Any] | None:
    """Consume a drift queue at the post-training off-peak boundary."""

    queue_path = Path(config["planner"]["memory_path"]).with_name("adapter_queue.jsonl")
    if not queue_path.exists():
        return None
    queued = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queued:
        return None
    controller_ids = sorted({int(item["controller_id"]) for item in queued})
    candidates = [
        (
            len(build_adapter_examples(config, controller_id, 64, time.time())),
            controller_id,
        )
        for controller_id in controller_ids
    ]
    example_count, region_id = max(candidates)
    if example_count < 2:
        return None
    manifest = update_lora_adapter(
        config,
        region_id,
        str(_adapter_root(config)),
        example_limit=64,
        holdout_fraction=0.20,
        epochs=1,
        learning_rate=0.0002,
        minimum_feasibility_rate=0.80,
        maximum_utility_mae=0.25,
        off_peak=True,
        topology_changed=False,
    )
    consumed_path = queue_path.with_name("adapter_queue.consumed.jsonl")
    with consumed_path.open("a", encoding="utf-8") as handle:
        for item in queued:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    queue_path.unlink()
    return manifest


def _reset_planner_artifacts(config: dict[str, Any]) -> None:
    memory_path = Path(config["planner"]["memory_path"])
    files = (
        memory_path,
        Path(str(memory_path) + "-shm"),
        Path(str(memory_path) + "-wal"),
        Path(config["planner"]["latest_plan_path"]),
        Path(config["planner"]["snapshot_path"]),
        memory_path.with_name("planner_windows.jsonl"),
        memory_path.with_name("adapter_queue.jsonl"),
        memory_path.with_name("adapter_queue.consumed.jsonl"),
        memory_path.with_name("runtime_manifest.json"),
    )
    for path in files:
        if path.is_file():
            path.unlink()
    adapter_root = _adapter_root(config)
    if adapter_root.is_dir():
        shutil.rmtree(adapter_root)


def _planner_runtime_manifest(
    config: dict[str, Any],
    method: str,
    seed: int,
    artifact_key: str | None,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    return {
        "configuration_hash": configuration_hash(config),
        "method": method_id(method),
        "seed": int(seed),
        "artifact_key": "base" if artifact_key is None else artifact_key,
        "scenario_hash": str(scenario["scenario_hash"]),
    }


def _runtime_manifest_path(runtime_config: dict[str, Any]) -> Path:
    return Path(runtime_config["planner"]["memory_path"]).with_name(
        "runtime_manifest.json"
    )


def _write_runtime_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def train_method(
    config: dict[str, Any],
    method: str,
    seed: int,
    planner_model: dict[str, Any] | None,
    *,
    scenario_overrides: dict[str, Any] | None = None,
    artifact_key: str | None = None,
    workload_label: str = "diurnal_24h",
    scheduled_duration_s: float | None = None,
) -> Path:
    seed_everything(seed)
    runtime_config = _runtime_config(config, method, seed, artifact_key)
    scenario = prepare_scenario(
        runtime_config,
        seed=seed,
        overrides=scenario_overrides,
    )
    if scenario_overrides is None:
        selected_duration_s = (
            float(runtime_config["simulation"]["duration_s"])
            if scheduled_duration_s is None
            else float(scheduled_duration_s)
        )
        scenario = schedule_diurnal_trace(
            scenario,
            _paper_diurnal_rates(runtime_config),
            selected_duration_s,
        )
    checkpoint_path = _checkpoint_path(config, method, seed, artifact_key)
    training_log = _training_log_path(config, method, seed, artifact_key)
    expected_manifest = _planner_runtime_manifest(
        config, method, seed, artifact_key, scenario
    )
    if checkpoint_path.exists():
        if not training_log.is_file():
            raise FileNotFoundError(
                f"Checkpoint resume requires its matching training log: {training_log}"
            )
        if bool(method_profile(method)["toporag"]):
            memory_path = Path(runtime_config["planner"]["memory_path"])
            timing_path = memory_path.with_name("planner_windows.jsonl")
            manifest_path = _runtime_manifest_path(runtime_config)
            if (
                not memory_path.is_file()
                or not timing_path.is_file()
                or not manifest_path.is_file()
            ):
                raise FileNotFoundError(
                    "TopoRAG checkpoint resume requires its memory, planner-window log, and runtime manifest: "
                    f"{memory_path}, {timing_path}, {manifest_path}"
                )
            observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if observed_manifest != expected_manifest:
                raise ValueError(
                    f"Planner runtime manifest does not match checkpoint condition: {manifest_path}"
                )
    else:
        _reset_planner_artifacts(runtime_config)
        if bool(method_profile(method)["toporag"]):
            _write_runtime_manifest(
                _runtime_manifest_path(runtime_config), expected_manifest
            )
        if training_log.exists():
            training_log.unlink()
    if planner_model is None and bool(method_profile(method)["semantic_guidance"]):
        planner_model = _planner_model(runtime_config, [method])
    device = _device()
    model, hidden = _build_method_state(runtime_config, scenario, method, device)
    method_name = method_id(method)
    training_log.parent.mkdir(parents=True, exist_ok=True)
    episode_start = 0
    global_step = 0
    if checkpoint_path.exists():
        if method_profile(method)["algorithm"] == "dqn":
            restored = load_cromadqn_checkpoint(checkpoint_path, model, config)
        else:
            restored = load_checkpoint(checkpoint_path, model, config)
        episode_start = int(restored["episode"])
        global_step = int(restored["global_step"])

    plans: dict[int, dict[str, Any]] = {}
    previous_returns = {controller_id: 0.0 for controller_id in _controller_ids(scenario)}
    rollout: list[dict[str, Any]] = []
    run_hash = configuration_hash(config)
    condition_id = "base" if artifact_key is None else artifact_key.replace("/", "-")
    run_id = f"train-{run_hash}-{condition_id}-{method_name}-{seed}"
    for episode in range(episode_start, int(config["training"]["max_episodes"])):
        if method_profile(method)["algorithm"] == "dqn":
            metrics, steps = run_cromadqn_episode(
                runtime_config, scenario, model, episode, training=True
            )
        else:
            assert hidden is not None
            metrics, plans, steps = run_mappo_episode(
                runtime_config,
                scenario,
                method,
                model,
                hidden,
                planner_model,
                plans,
                previous_returns,
                rollout,
                episode,
                training=True,
            )
        global_step += steps
        final_episode = episode + 1 == int(config["training"]["max_episodes"])
        if final_episode and rollout:
            if method_profile(method)["algorithm"] == "independent_ppo":
                for controller_id in previous_returns:
                    _update_independent_controller(
                        model,
                        rollout,
                        controller_id,
                        0.0,
                        config["training"],
                    )
            else:
                compute_gae(
                    rollout,
                    {controller_id: 0.0 for controller_id in previous_returns},
                    float(config["training"]["gamma"]),
                    float(config["training"]["gae_lambda"]),
                )
                update_mappo(model, rollout, config["training"])
                rollout.clear()
        append_training_record(
            training_log,
            {
                "run_id": run_id,
                "config_hash": run_hash,
                "seed": seed,
                "method": method_name,
                "condition": condition_id,
                "workload": workload_label,
                "scenario_hash": str(scenario["scenario_hash"]),
                "episode": episode,
                "normalized_cumulative_reward": metrics["normalized_cumulative_reward"],
                "end_to_end_latency_ms": metrics["end_to_end_latency_ms"],
                "load_balance_index": metrics["load_balance_index"],
                "sla_violation_rate_percent": metrics["sla_violation_rate_percent"],
                "compute_utilization_percent": metrics["compute_utilization_percent"],
                "bandwidth_utilization_percent": metrics["bandwidth_utilization_percent"],
                "steps": steps,
            },
        )
        if (
            (episode + 1) % int(config["training"]["checkpoint_every"]) == 0
            or final_episode
        ):
            if method_profile(method)["algorithm"] == "dqn":
                save_cromadqn_checkpoint(
                    checkpoint_path, model, config, episode + 1, global_step
                )
            else:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    config,
                    episode + 1,
                    global_step,
                    planner_version=str(config["planner"]["model_id"]),
                )
            if bool(method_profile(method)["toporag"]):
                queue_path = Path(runtime_config["planner"]["memory_path"]).with_name(
                    "adapter_queue.jsonl"
                )
                if queue_path.is_file():
                    planner_model = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _publish_queued_adapter(runtime_config)
                if planner_model is None and not final_episode:
                    planner_model = _planner_model(runtime_config, [method])
        print(
            f"{method_name} seed={seed} episode={episode + 1}/"
            f"{config['training']['max_episodes']} reward={metrics['normalized_cumulative_reward']:.4f}"
        )
    return checkpoint_path


def _measured_row(
    config: dict[str, Any],
    run_id: str,
    seed: int,
    figure: int,
    panel: str,
    method: str,
    scenario: str,
    workload: str,
    x: float,
    x_label: str,
    metric: str,
    value: float,
    unit: str,
) -> dict[str, Any]:
    return result_row(
        run_id=run_id,
        configuration_hash=configuration_hash(config),
        seed=seed,
        figure=figure,
        panel=panel,
        method=method,
        scenario=scenario,
        workload=workload,
        x=x,
        x_label=x_label,
        metric=metric,
        value=value,
        unit=unit,
    )


def _timeline_values(
    samples: list[dict[str, Any]], field: str, points_s: np.ndarray
) -> np.ndarray:
    if len(samples) < 2:
        raise ValueError(f"A measured {field} timeline needs at least two samples")
    grouped: dict[float, list[float]] = {}
    for sample in samples:
        grouped.setdefault(float(sample["time_s"]), []).append(float(sample[field]))
    times = np.asarray(sorted(grouped), dtype=np.float64)
    values = np.asarray(
        [float(np.mean(grouped[time_s])) for time_s in times], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError(f"Measured {field} timeline contains non-finite values")
    return np.interp(points_s, times, values)


def _terminal_latency_timeline(
    samples: list[dict[str, Any]], points_s: np.ndarray
) -> np.ndarray:
    grouped: dict[float, list[float]] = {}
    for sample in samples:
        terminal_latencies = [
            float(event["latency_ms"])
            for event in sample["controller_events"]
            if bool(event["request_completed"]) and bool(event["accepted"])
        ]
        if terminal_latencies:
            grouped.setdefault(float(sample["time_s"]), []).extend(terminal_latencies)
    if len(grouped) < 2:
        raise ValueError(
            "Burst adaptation needs at least two completed-workflow latency samples"
        )
    times = np.asarray(sorted(grouped), dtype=np.float64)
    latencies = np.asarray(
        [float(np.mean(grouped[time_s])) for time_s in times], dtype=np.float64
    )
    if np.any(latencies <= 0.0) or not np.isfinite(latencies).all():
        raise ValueError("Measured terminal latency timeline must be positive and finite")
    return np.interp(points_s, times, latencies)


def _mean_node_utilization(capture: dict[str, Any], edge_servers: int) -> np.ndarray:
    samples = capture["samples"]
    if len(samples) < 2:
        raise ValueError("Spatial utilization needs at least two raw simulator samples")
    times = np.asarray([float(sample["time_s"]) for sample in samples], dtype=np.float64)
    values = np.asarray(
        [sample["node_utilization_percent"][:edge_servers] for sample in samples],
        dtype=np.float64,
    )
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    unique_times, first = np.unique(times, return_index=True)
    if len(unique_times) == 1:
        return values[first[0]]
    reduced = np.vstack(
        [values[times == time_s].mean(axis=0) for time_s in unique_times]
    )
    duration = float(unique_times[-1] - unique_times[0])
    return np.trapz(reduced, unique_times, axis=0) / duration


def _spatial_grid(
    config: dict[str, Any],
    coordinates: list[np.ndarray],
    utilizations: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_coordinates = np.vstack(coordinates)
    node_values = np.concatenate(utilizations)
    if len(node_coordinates) != len(node_values):
        raise ValueError("Spatial node coordinates and utilization values do not align")
    area = np.asarray(config["simulation"]["area_km"], dtype=np.float64)
    x = np.linspace(0.0, area[0], 60)
    y = np.linspace(0.0, area[1], 60)
    mesh_x, mesh_y = np.meshgrid(x, y)
    query = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    grid = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), 256):
        batch = query[start : start + 256]
        distances = np.linalg.norm(
            batch[:, None, :] - node_coordinates[None, :, :], axis=2
        )
        exact = distances <= 1e-12
        weights = 1.0 / np.maximum(distances, 0.05) ** 2
        estimated = (weights @ node_values) / weights.sum(axis=1)
        for row_index in np.flatnonzero(exact.any(axis=1)):
            estimated[row_index] = float(node_values[exact[row_index]].mean())
        grid[start : start + len(batch)] = estimated
    return (
        x,
        y,
        np.clip(grid.reshape(mesh_x.shape), 0.0, 100.0),
        node_coordinates[:, 0],
        node_coordinates[:, 1],
    )


def _convergence_from_log(
    config: dict[str, Any], method: str, seed: int, artifact_key: str | None
) -> int:
    from evaluate import read_training_records

    records = read_training_records(
        _training_log_path(config, method, seed, artifact_key)
    )
    settings = config["evaluation"]["convergence"]
    rewards = [float(record["normalized_cumulative_reward"]) for record in records]
    return convergence_episode(
        rewards,
        int(settings["window"]),
        int(settings["min_episode"]),
        int(settings["patience"]),
        float(settings["relative_delta"]),
    )


def _figure5_rows(
    config: dict[str, Any], run_id: str
) -> list[dict[str, Any]]:
    rows = []
    metrics = (
        ("a", "end_to_end_latency_ms", "end_to_end_latency", "ms"),
        ("b", "sla_violation_rate_percent", "sla_violation_rate", "percent"),
        ("c", "load_balance_index", "load_balance_index", "index"),
    )
    for seed in config["evaluation"]["seeds"]:
        for rate in config["evaluation"]["arrival_rates"]:
            artifact_key = f"arrival-rate-{int(rate)}"
            for method in LOAD_METHODS:
                runtime = _runtime_config(config, method, int(seed), artifact_key)
                scenario = prepare_scenario(
                    runtime,
                    seed=int(seed),
                    overrides={"arrival_rate": float(rate), "burst": False},
                )
                measured = evaluate_saved_method(
                    config,
                    scenario,
                    method,
                    int(seed),
                    0,
                    artifact_key=artifact_key,
                )
                for panel, source, metric, unit in metrics:
                    rows.append(
                        _measured_row(
                            config,
                            run_id,
                            int(seed),
                            5,
                            panel,
                            method_id(method),
                            "arrival_rate_scaling",
                            "metaverse_mixed",
                            float(rate),
                            str(rate),
                            metric,
                            float(measured[source]),
                            unit,
                        )
                    )
                rows.append(
                    _measured_row(
                        config,
                        run_id,
                        int(seed),
                        5,
                        "d",
                        method_id(method),
                        "arrival_rate_scaling",
                        "metaverse_mixed",
                        float(rate),
                        str(rate),
                        "convergence_time",
                        float(
                            _convergence_from_log(
                                config, method, int(seed), artifact_key
                            )
                        ),
                        "episodes",
                    )
                )
    return rows


def _figure6_rows(
    config: dict[str, Any], run_id: str
) -> list[dict[str, Any]]:
    rows = []
    for seed in config["evaluation"]["seeds"]:
        planner_models = {
            method: _planner_model(
                _runtime_config(config, method, int(seed)), [method]
            )
            for method in CORE_METHODS
        }
        for complexity in config["evaluation"]["complexity"]:
            microservices = int(complexity["microservices"])
            replicas = int(complexity["replicas"])
            label = f"{microservices}MS/{replicas}R"
            for method in CORE_METHODS:
                runtime = _runtime_config(config, method, int(seed))
                scenario = prepare_scenario(
                    runtime,
                    seed=int(seed),
                    overrides={
                        "microservices": microservices,
                        "replicas": replicas,
                        "burst": False,
                    },
                )
                measured = evaluate_saved_method(
                    config,
                    scenario,
                    method,
                    int(seed),
                    0,
                    planner_model=planner_models[method],
                )
                for panel, source, metric, unit in (
                    ("a", "end_to_end_latency_ms", "end_to_end_latency", "ms"),
                    ("b", "sla_violation_rate_percent", "sla_violation_rate", "percent"),
                ):
                    rows.append(
                        _measured_row(
                            config,
                            run_id,
                            int(seed),
                            6,
                            panel,
                            method_id(method),
                            "service_complexity",
                            "metaverse_service_dag",
                            float(microservices),
                            label,
                            metric,
                            float(measured[source]),
                            unit,
                        )
                    )
    return rows


def _failure_hook(
    failed_node_ids: np.ndarray, injection_time_s: float, event: dict[str, Any]
) -> Any:
    def inject(state: dict[str, Any]) -> None:
        if event or float(state["time_s"]) < injection_time_s:
            return
        event.update(inject_node_failures(state, failed_node_ids))

    return inject


def _measured_recovery_time(
    failure_event: dict[str, Any],
    terminal_requests: list[dict[str, Any]],
    observation_horizon_s: float,
) -> float:
    impacted = {int(request_id) for request_id in failure_event["impacted_request_ids"]}
    if not impacted:
        return 0.0
    recovered = {
        int(event["request_id"]): float(event["recovery_time_s"])
        for event in terminal_requests
        if int(event["request_id"]) in impacted
        and float(event["recovery_time_s"]) > 0.0
    }
    if set(recovered) != impacted:
        return observation_horizon_s - float(failure_event["time_s"])
    return max(recovered.values())


def _figure7_rows(
    config: dict[str, Any], run_id: str
) -> list[dict[str, Any]]:
    rows = []
    diurnal_rates = _paper_diurnal_rates(config)
    edge_servers = int(config["simulation"]["edge_servers"])
    failure_duration_s = 420.0
    injection_time_s = 20.0
    burst_multiplier = float(config["workloads"]["pareto"]["burst_multiplier"])
    burst_end_s = 20.0
    for seed in config["evaluation"]["seeds"]:
        planner_models = {
            method: _planner_model(
                _runtime_config(config, method, int(seed)), [method]
            )
            for method in CORE_METHODS
        }
        for failure_rate in config["evaluation"]["failure_rates"]:
            failed_count = round(edge_servers * float(failure_rate) / 100.0)
            failure_rng = np.random.default_rng(int(seed) * 1000 + int(failure_rate))
            failed_ids = (
                np.sort(failure_rng.choice(edge_servers, failed_count, replace=False))
                if failed_count
                else np.empty(0, dtype=np.int64)
            )
            for method in CORE_METHODS:
                runtime = _runtime_config(config, method, int(seed))
                base = prepare_scenario(
                    runtime,
                    seed=int(seed),
                    overrides={"burst": False},
                )
                scenario = schedule_diurnal_trace(
                    base, diurnal_rates, failure_duration_s
                )
                capture: dict[str, Any] = {}
                failure_event: dict[str, Any] = {}
                hook = (
                    _failure_hook(failed_ids, injection_time_s, failure_event)
                    if failed_count
                    else None
                )
                evaluate_saved_method(
                    config,
                    scenario,
                    method,
                    int(seed),
                    0,
                    capture=capture,
                    state_hook=hook,
                    planner_model=planner_models[method],
                )
                if failed_count and not failure_event:
                    raise RuntimeError("Failure injection time was not reached by the measured trace")
                recovery_time = (
                    _measured_recovery_time(
                        failure_event,
                        capture["terminal_requests"],
                        failure_duration_s,
                    )
                    if failed_count
                    else 0.0
                )
                rows.append(
                    _measured_row(
                        config,
                        run_id,
                        int(seed),
                        7,
                        "a",
                        method_id(method),
                        "node_failures",
                        "metaverse_mixed",
                        float(failure_rate),
                        str(failure_rate),
                        "service_recovery_time",
                        recovery_time,
                        "s",
                    )
                )

        for method in CORE_METHODS:
            runtime = _runtime_config(config, method, int(seed))
            unscheduled = prepare_scenario(
                runtime,
                seed=int(seed),
                overrides={"burst": False},
            )
            source_duration = float(unscheduled["config"]["simulation"]["duration_s"])
            baseline_rate = float(unscheduled["trace_schedule"]["mean_rate_rps"])
            baseline_scenario = schedule_diurnal_trace(
                unscheduled,
                [baseline_rate] * 24,
                source_duration,
            )
            burst_scenario = schedule_burst_trace(
                baseline_scenario,
                0.0,
                burst_end_s,
                burst_multiplier,
            )
            baseline_capture: dict[str, Any] = {}
            burst_capture: dict[str, Any] = {}
            evaluate_saved_method(
                config,
                baseline_scenario,
                method,
                int(seed),
                0,
                capture=baseline_capture,
                planner_model=planner_models[method],
            )
            evaluate_saved_method(
                config,
                burst_scenario,
                method,
                int(seed),
                0,
                capture=burst_capture,
                planner_model=planner_models[method],
            )
            baseline_timeline = _terminal_latency_timeline(
                baseline_capture["samples"],
                np.asarray(BURST_TIMELINE_S, dtype=np.float64),
            )
            burst_timeline = _terminal_latency_timeline(
                burst_capture["samples"],
                np.asarray(BURST_TIMELINE_S, dtype=np.float64),
            )
            ratios = burst_timeline / baseline_timeline
            for time_s, ratio in zip(BURST_TIMELINE_S, ratios, strict=True):
                rows.append(
                    _measured_row(
                        config,
                        run_id,
                        int(seed),
                        7,
                        "b",
                        method_id(method),
                        "traffic_burst",
                        "metaverse_mixed",
                        time_s,
                        str(int(time_s)),
                        "latency_to_baseline",
                        float(ratio),
                        "ratio",
                    )
                )
    return rows


def _figure8_measurements(
    config: dict[str, Any], run_id: str
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    before_coordinates = []
    before_utilization = []
    after_coordinates = []
    after_utilization = []
    edge_servers = int(config["simulation"]["edge_servers"])
    accelerated_points = np.linspace(
        0.0, DIURNAL_ACCELERATED_DURATION_S, 24, endpoint=False
    )
    day_points = accelerated_points * (86_400.0 / DIURNAL_ACCELERATED_DURATION_S)
    for seed in config["evaluation"]["seeds"]:
        for method in CORE_METHODS:
            runtime = _runtime_config(config, method, int(seed))
            base = prepare_scenario(runtime, seed=int(seed), overrides={"burst": False})
            scenario = schedule_diurnal_trace(
                base,
                _paper_diurnal_rates(runtime),
                DIURNAL_ACCELERATED_DURATION_S,
            )
            capture: dict[str, Any] = {}
            evaluate_saved_method(
                config,
                scenario,
                method,
                int(seed),
                0,
                capture=capture,
            )
            compute = _timeline_values(
                capture["samples"], "compute_utilization_percent", accelerated_points
            )
            bandwidth = _timeline_values(
                capture["samples"], "bandwidth_utilization_percent", accelerated_points
            )
            for index, time_s in enumerate(day_points):
                for panel, metric, value in (
                    ("a", "compute_utilization", compute[index]),
                    ("b", "bandwidth_utilization", bandwidth[index]),
                ):
                    rows.append(
                        _measured_row(
                            config,
                            run_id,
                            int(seed),
                            8,
                            panel,
                            method_id(method),
                            "resource_utilization",
                            "diurnal_24h",
                            float(time_s),
                            f"hour-{index}",
                            metric,
                            float(value),
                            "percent",
                        )
                    )
            if method == "LLM-MAPPO":
                coordinates = np.asarray(capture["node_coordinates_km"], dtype=np.float64)[
                    :edge_servers
                ]
                after_coordinates.append(coordinates)
                after_utilization.append(_mean_node_utilization(capture, edge_servers))
                before_capture: dict[str, Any] = {}
                before_artifact = "before-convergence"
                before_runtime = _runtime_config(
                    config, method, int(seed), before_artifact
                )
                _reset_planner_artifacts(before_runtime)
                evaluate_untrained_method(
                    config,
                    scenario,
                    method,
                    int(seed),
                    0,
                    before_capture,
                    planner_model=load_qwen_planner(
                        before_runtime, trainable=False, adapter_path=None
                    ),
                    artifact_key=before_artifact,
                )
                before_coordinates.append(coordinates)
                before_utilization.append(
                    _mean_node_utilization(before_capture, edge_servers)
                )
    x, y, before_grid, node_x, node_y = _spatial_grid(
        config, before_coordinates, before_utilization
    )
    _, _, after_grid, _, _ = _spatial_grid(
        config, after_coordinates, after_utilization
    )
    spatial = {
        "x": x,
        "y": y,
        "utilization_before": before_grid,
        "utilization_after": after_grid,
        "node_x": node_x,
        "node_y": node_y,
    }
    return rows, spatial


def _sensitivity_response(
    config: dict[str, Any],
    metrics: dict[str, float],
    capture: dict[str, Any],
    convergence: int,
) -> dict[str, float]:
    windows = capture["planner_windows"]
    if not windows:
        raise ValueError("Sensitivity measurement requires raw planner windows")
    overhead = float(np.mean([float(window["wall_time_s"]) for window in windows]))
    drift = float(np.mean([float(window["semantic_drift"]) for window in windows]))
    return {
        "end_to_end_latency": float(metrics["end_to_end_latency_ms"]),
        "sla_violation_rate": float(metrics["sla_violation_rate_percent"]),
        "convergence_time": float(convergence),
        "resource_efficiency": 0.5
        * (
            float(metrics["compute_utilization_percent"])
            + float(metrics["bandwidth_utilization_percent"])
        ),
        "planning_overhead": overhead,
        "adaptation_quality": 100.0 * max(0.0, 1.0 - min(1.0, drift)),
    }


def _relative_sensitivity(low: float, high: float) -> float:
    scale = max(abs(low), abs(high), 1e-12)
    return 100.0 * abs(high - low) / scale


def _figure9_rows(
    config: dict[str, Any], run_id: str
) -> list[dict[str, Any]]:
    rows = []
    for seed in config["evaluation"]["seeds"]:
        for figure_method, method in ABLATION_FIGURE_METHODS.items():
            runtime = _runtime_config(config, method, int(seed))
            scenario = prepare_scenario(
                runtime, seed=int(seed), overrides={"burst": True}
            )
            metrics = evaluate_saved_method(
                config, scenario, method, int(seed), 0
            )
            convergence = _convergence_from_log(config, method, int(seed), None)
            for panel, metric, value, unit in (
                ("a", "end_to_end_latency", metrics["end_to_end_latency_ms"], "ms"),
                ("b", "sla_violation_rate", metrics["sla_violation_rate_percent"], "percent"),
                ("c", "convergence_time", convergence, "episodes"),
            ):
                rows.append(
                    _measured_row(
                        config,
                        run_id,
                        int(seed),
                        9,
                        panel,
                        figure_method,
                        "component_ablation",
                        "metaverse_mixed",
                        0.0,
                        "measured",
                        metric,
                        float(value),
                        unit,
                    )
                )

        for parameter, variants in _sensitivity_conditions(config).items():
            measured_variants = []
            for level, variant in zip(("low", "high"), variants, strict=True):
                artifact_key = f"sensitivity/{parameter}-{level}"
                runtime = _runtime_config(
                    variant, "LLM-MAPPO", int(seed), artifact_key
                )
                scenario = prepare_scenario(
                    runtime, seed=int(seed), overrides={"burst": False}
                )
                scenario = schedule_diurnal_trace(
                    scenario,
                    _paper_diurnal_rates(runtime),
                    SENSITIVITY_TRACE_DURATION_S,
                )
                capture: dict[str, Any] = {}
                metrics = evaluate_saved_method(
                    variant,
                    scenario,
                    "LLM-MAPPO",
                    int(seed),
                    0,
                    artifact_key=artifact_key,
                    capture=capture,
                )
                convergence = _convergence_from_log(
                    variant,
                    "LLM-MAPPO",
                    int(seed),
                    artifact_key,
                )
                measured_variants.append(
                    _sensitivity_response(variant, metrics, capture, convergence)
                )
            low, high = measured_variants
            for response_index, response in enumerate(SENSITIVITY_RESPONSES):
                rows.append(
                    _measured_row(
                        config,
                        run_id,
                        int(seed),
                        9,
                        "d",
                        parameter,
                        "hyperparameter_sensitivity",
                        "metaverse_mixed",
                        float(response_index),
                        response,
                        "sensitivity",
                        _relative_sensitivity(low[response], high[response]),
                        "percent",
                    )
                )
    return rows


def _plan_trials(
    method: str,
    capture: dict[str, Any],
) -> list[dict[str, Any]]:
    if bool(method_profile(method)["semantic_guidance"]):
        return [
            {
                "candidate_feasibility": [
                    bool(candidate["feasibility"]["feasible"])
                    for candidate in window["plan"]["candidates"]
                ],
                "predicted_utility": float(window["plan"]["predicted_utility"]),
                "observed_utility": float(window["observed_utility"]),
                "semantic_drift": float(window["semantic_drift"]),
            }
            for window in capture["planner_windows"]
        ]
    trials = []
    for query_index, query in enumerate(capture["planning_queries"]):
        placement_ids = {
            f"{service['request_id']}/{microservice['microservice_id']}/{replica_id}"
            for service in query["service_dags"]
            for microservice in service["microservices"]
            for replica_id in microservice["replica_ids"]
        }
        executed = [
            action
            for action in capture["action_trials"]
            if action["placement_id"] in placement_ids
            and int(action["controller_id"]) == int(query["controller_id"])
            and int(float(action["time_s"]) // int(capture["planning_interval_s"]))
            == int(query["window_id"])
        ]
        if not executed:
            continue
        assignments = [
            [str(action["placement_id"]), str(action["node_id"])]
            for action in executed
        ]
        assignments.append("<eop>")
        predicted_utility = float(
            np.mean([float(action["predicted_utility"]) for action in executed])
        )
        observed_utility = 2.0 * float(
            np.mean([float(action["normalized_reward"]) for action in executed])
        ) - 1.0
        node_preferences = list(
            dict.fromkeys(str(action["node_id"]) for action in executed)
        )[:8]
        node_preferences.append("<eop>")
        candidate = {
            "candidate_id": f"actor-execution-{query_index}",
            "node_preferences": node_preferences,
            "placement_strategy": (
                "consolidate" if len(node_preferences) == 2 else "spread"
            ),
            "assignments": assignments,
            "predicted_utility": predicted_utility,
        }
        feasibility = validate_candidate_feasibility(candidate, query)
        drift = semantic_drift(predicted_utility, observed_utility)
        trials.append(
            {
                "candidate_feasibility": [bool(feasibility["feasible"])],
                "predicted_utility": predicted_utility,
                "observed_utility": observed_utility,
                "semantic_drift": drift,
            }
        )
    return trials


def _figure10_rows(
    config: dict[str, Any], run_id: str
) -> list[dict[str, Any]]:
    rows = []
    seeds = [int(seed) for seed in config["evaluation"]["seeds"]]
    candidate_count = int(config["planner"]["candidate_count"])
    candidate_trials_per_seed = int(config["evaluation"]["monte_carlo_runs"])
    densities = ("urban", "medium", "suburban")
    for seed in seeds:
        for figure_method, method in ABLATION_FIGURE_METHODS.items():
            planner_model = _planner_model(
                _runtime_config(config, method, seed), [method]
            )
            urban_drifts: list[float] = []
            for density_index, density in enumerate(densities):
                feasibility: list[bool] = []
                trial_index = 0
                while len(feasibility) < candidate_trials_per_seed or (
                    density == "urban" and len(urban_drifts) < SEMANTIC_DRIFT_WINDOWS
                ):
                    scenario_seed = (
                        seed + (density_index + 1) * 100_000 + trial_index * 1000
                    )
                    runtime = _runtime_config(config, method, seed)
                    scenario = prepare_scenario(
                        runtime,
                        seed=scenario_seed,
                        overrides={"density": density, "burst": True},
                    )
                    capture: dict[str, Any] = {}
                    evaluate_saved_method(
                        config,
                        scenario,
                        method,
                        seed,
                        trial_index,
                        capture=capture,
                        planner_model=planner_model,
                    )
                    measured_trials = _plan_trials(
                        method,
                        capture,
                    )
                    if not measured_trials:
                        raise RuntimeError(
                            f"{method} produced no measured planning trials for {density}"
                        )
                    for trial in measured_trials:
                        feasibility.extend(trial["candidate_feasibility"])
                        if density == "urban":
                            urban_drifts.append(float(trial["semantic_drift"]))
                    trial_index += 1
                    if trial_index > candidate_trials_per_seed * candidate_count:
                        raise RuntimeError("Planner feasibility trial collection did not progress")
                for feasible in feasibility[:candidate_trials_per_seed]:
                    rows.append(
                        _measured_row(
                            config,
                            run_id,
                            seed,
                            10,
                            "a",
                            figure_method,
                            "network_density",
                            "metaverse_mixed",
                            float(density_index),
                            density,
                            "feasible_plan_rate",
                            100.0 if feasible else 0.0,
                            "percent",
                        )
                    )
            for window_index, drift in enumerate(
                urban_drifts[:SEMANTIC_DRIFT_WINDOWS]
            ):
                rows.append(
                    _measured_row(
                        config,
                        run_id,
                        seed,
                        10,
                        "b",
                        figure_method,
                        "semantic_drift",
                        "metaverse_mixed",
                        float(window_index),
                        f"window-{window_index + 1}",
                        "semantic_drift",
                        drift,
                        "score",
                    )
                )
    return rows


def _require_measurement_artifacts(config: dict[str, Any]) -> None:
    required_paths: set[Path] = set()
    base_methods = set(CORE_METHODS) | set(ABLATION_FIGURE_METHODS.values())
    for seed in config["evaluation"]["seeds"]:
        for method in base_methods:
            required_paths.add(_checkpoint_path(config, method, int(seed), None))
            required_paths.add(_training_log_path(config, method, int(seed), None))
            if bool(method_profile(method)["toporag"]):
                runtime = _runtime_config(config, method, int(seed))
                memory_path = Path(runtime["planner"]["memory_path"])
                required_paths.add(memory_path)
                required_paths.add(memory_path.with_name("planner_windows.jsonl"))
                required_paths.add(_runtime_manifest_path(runtime))
        for rate in config["evaluation"]["arrival_rates"]:
            artifact_key = f"arrival-rate-{int(rate)}"
            for method in LOAD_METHODS:
                required_paths.add(
                    _checkpoint_path(config, method, int(seed), artifact_key)
                )
                required_paths.add(
                    _training_log_path(config, method, int(seed), artifact_key)
                )
                if bool(method_profile(method)["toporag"]):
                    runtime = _runtime_config(
                        config, method, int(seed), artifact_key
                    )
                    memory_path = Path(runtime["planner"]["memory_path"])
                    required_paths.add(memory_path)
                    required_paths.add(
                        memory_path.with_name("planner_windows.jsonl")
                    )
                    required_paths.add(_runtime_manifest_path(runtime))
        for parameter, variants in _sensitivity_conditions(config).items():
            for level, variant in zip(("low", "high"), variants, strict=True):
                artifact_key = f"sensitivity/{parameter}-{level}"
                required_paths.add(
                    _checkpoint_path(
                        variant, "LLM-MAPPO", int(seed), artifact_key
                    )
                )
                required_paths.add(
                    _training_log_path(
                        variant, "LLM-MAPPO", int(seed), artifact_key
                    )
                )
                runtime = _runtime_config(
                    variant, "LLM-MAPPO", int(seed), artifact_key
                )
                memory_path = Path(runtime["planner"]["memory_path"])
                required_paths.add(memory_path)
                required_paths.add(memory_path.with_name("planner_windows.jsonl"))
                required_paths.add(_runtime_manifest_path(runtime))
    missing = sorted(str(path) for path in required_paths if not path.is_file())
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(
            "Measured Figure 5--10 artifacts are incomplete. Run train_agentMulti.py; "
            f"missing {len(missing)} files:\n{preview}{suffix}"
        )


def run_measured_figures_5_10(
    *, config: dict[str, Any], measurement_suite: str
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if measurement_suite != "figures_5_10":
        raise ValueError(f"Unknown measured evaluation suite {measurement_suite!r}")
    _require_measurement_artifacts(config)
    run_id = f"evaluate-{configuration_hash(config)}"
    rows = []
    rows.extend(_figure5_rows(config, run_id))
    rows.extend(_figure6_rows(config, run_id))
    rows.extend(_figure7_rows(config, run_id))
    figure8_rows, spatial = _figure8_measurements(config, run_id)
    rows.extend(figure8_rows)
    rows.extend(_figure9_rows(config, run_id))
    rows.extend(_figure10_rows(config, run_id))
    return rows, spatial


run_measured_figures_5_10.measurement_capabilities = {
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


def evaluate_experiments(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Run the configured evaluation matrix using saved measured policies.

    Scenario execution is delegated to ``run_evaluation_matrix`` in the
    simulator, keeping all perturbations on identical seeded request traces.
    """
    from env.envMulti import run_evaluation_matrix
    from rag import benchmark_memory_scaling

    rows, spatial = run_evaluation_matrix(config, run_measured_figures_5_10)
    run_hash = configuration_hash(config)
    run_id = f"evaluate-{run_hash}"
    for seed in config["evaluation"]["seeds"]:
        timing_path = Path(
            _runtime_config(config, "LLM-MAPPO", int(seed))["planner"]["memory_path"]
        ).with_name("planner_windows.jsonl")
        if not timing_path.exists():
            raise FileNotFoundError(
                f"Measured planner timing log is missing: {timing_path}"
            )
        with timing_path.open("r", encoding="utf-8") as handle:
            service_times = [
                float(json.loads(line)["wall_time_s"])
                for line in handle
                if line.strip()
            ]
        if not service_times or any(value <= 0.0 for value in service_times):
            raise ValueError(f"Planner timing log is empty or invalid: {timing_path}")
        rows.extend(
            planner_worker_rows(
                run_id,
                run_hash,
                int(seed),
                controller_counts=[16, 32, 48, 64, 80, 100],
                workers=[1, 2, 3, 4, 5],
                planning_interval_s=float(config["planner"]["planning_interval_s"]),
                measured_service_time_s=float(np.mean(service_times)),
            )
        )
        rows.extend(benchmark_memory_scaling(config, int(seed), run_id, run_hash, result_row))
    return rows, spatial


def evaluate_saved_method(
    config: dict[str, Any],
    scenario: dict[str, Any],
    method: str,
    seed: int,
    episode: int,
    *,
    artifact_key: str | None = None,
    capture: dict[str, Any] | None = None,
    state_hook: Any | None = None,
    planner_model: dict[str, Any] | None = None,
) -> dict[str, float]:
    runtime_config = _runtime_config(config, method, seed, artifact_key)
    device = _device()
    model, hidden = _build_method_state(runtime_config, scenario, method, device)
    if method_profile(method)["algorithm"] == "dqn":
        path = _checkpoint_path(config, method, seed, artifact_key)
        if not path.exists():
            raise FileNotFoundError(f"Measured baseline checkpoint is missing: {path}")
        load_cromadqn_checkpoint(path, model, config)
        metrics, _ = run_cromadqn_episode(
            runtime_config,
            scenario,
            model,
            episode,
            training=False,
            capture=capture,
            state_hook=state_hook,
        )
        if capture is not None:
            capture["metrics"] = dict(metrics)
        return metrics
    checkpoint = _checkpoint_path(config, method, seed, artifact_key)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Measured policy checkpoint is missing: {checkpoint}")
    load_checkpoint(checkpoint, model, config)
    assert hidden is not None
    if planner_model is None:
        planner_model = _planner_model(runtime_config, [method])
    metrics, _, _ = run_mappo_episode(
        runtime_config,
        scenario,
        method,
        model,
        hidden,
        planner_model,
        {},
        {controller_id: 0.0 for controller_id in hidden},
        [],
        episode,
        training=False,
        capture=capture,
        state_hook=state_hook,
    )
    if capture is not None:
        capture["metrics"] = dict(metrics)
    return metrics


def evaluate_untrained_method(
    config: dict[str, Any],
    scenario: dict[str, Any],
    method: str,
    seed: int,
    episode: int,
    capture: dict[str, Any],
    planner_model: dict[str, Any] | None = None,
    artifact_key: str | None = None,
) -> dict[str, float]:
    """Measure the initialized policy on the same trace used after convergence."""

    seed_everything(seed)
    runtime_config = _runtime_config(config, method, seed, artifact_key)
    model, hidden = _build_method_state(runtime_config, scenario, method, _device())
    if method_profile(method)["algorithm"] == "dqn":
        metrics, _ = run_cromadqn_episode(
            runtime_config,
            scenario,
            model,
            episode,
            training=False,
            capture=capture,
        )
    else:
        assert hidden is not None
        if planner_model is None:
            planner_model = _planner_model(runtime_config, [method])
        metrics, _, _ = run_mappo_episode(
            runtime_config,
            scenario,
            method,
            model,
            hidden,
            planner_model,
            {},
            {controller_id: 0.0 for controller_id in hidden},
            [],
            episode,
            training=False,
            capture=capture,
        )
    capture["metrics"] = dict(metrics)
    return metrics


def train_measurement_artifacts(config: dict[str, Any]) -> None:
    configured_methods = list(config["evaluation"]["methods"])
    for method in configured_methods:
        if method not in METHODS:
            raise ValueError(f"Configured unknown method {method!r}")
    base_methods = list(dict.fromkeys(configured_methods + list(ABLATION_METHODS)))
    for method in base_methods:
        for seed in config["evaluation"]["seeds"]:
            train_method(config, method, int(seed), None)

    for rate in config["evaluation"]["arrival_rates"]:
        artifact_key = f"arrival-rate-{int(rate)}"
        for method in configured_methods:
            for seed in config["evaluation"]["seeds"]:
                train_method(
                    config,
                    method,
                    int(seed),
                    None,
                    scenario_overrides={
                        "arrival_rate": float(rate),
                        "burst": False,
                    },
                    artifact_key=artifact_key,
                    workload_label="metaverse_mixed",
                )

    for parameter, variants in _sensitivity_conditions(config).items():
        for level, variant in zip(("low", "high"), variants, strict=True):
            artifact_key = f"sensitivity/{parameter}-{level}"
            for seed in config["evaluation"]["seeds"]:
                train_method(
                    variant,
                    "LLM-MAPPO",
                    int(seed),
                    None,
                    artifact_key=artifact_key,
                    scheduled_duration_s=SENSITIVITY_TRACE_DURATION_S,
                )


def main(config_path: str = "configs/paper.json") -> None:
    train_measurement_artifacts(load_config(config_path))


if __name__ == "__main__":
    main()
