"""Configuration loading and shared names for the paper simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESOURCE_NAMES = ("cpu_vcpu", "memory_gb", "gpu_units", "npu_units")
NODE_TIERS = ("edge_server", "edge_controller", "cloud_core")
RADIO_PROFILES = ("sub6", "mmwave", "thz")
WORKLOAD_CLASSES = ("xr", "digital_twin", "holographic")
SLICE_NAMES = ("immersive", "massive", "digital_twin")
SCENARIO_OVERRIDE_KEYS = {
    "arrival_rate",
    "microservices",
    "replicas",
    "failure_rate",
    "burst",
    "density",
    "duration_s",
}


def _require_keys(value: dict[str, Any], required: set[str], where: str) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"{where}: " + ", ".join(details))


def _number(value: Any, where: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    return result


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where} must be an integer")
    if value < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    return value


def _pair(value: Any, where: str, minimum: float = 0.0) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{where} must be a two-item list")
    low = _number(value[0], f"{where}[0]", minimum)
    high = _number(value[1], f"{where}[1]", minimum)
    if low > high:
        raise ValueError(f"{where} lower bound exceeds upper bound")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the single project configuration without silent defaults."""

    if not isinstance(config, dict):
        raise TypeError("configuration root must be an object")
    _require_keys(
        config,
        {"simulation", "workloads", "reward", "planner", "training", "evaluation", "paths"},
        "config",
    )

    simulation = config["simulation"]
    _require_keys(
        simulation,
        {
            "source",
            "seed",
            "base_stations",
            "edge_servers",
            "edge_controllers",
            "core_nodes",
            "area_km",
            "duration_s",
            "decision_interval_s",
            "candidate_paths",
            "max_local_nodes",
            "nearest_neighbors",
            "mobility_update_s",
            "cross_region_fraction",
            "mips_per_vcpu",
            "edge_server_resources",
            "controller_resources",
            "core_resources",
            "edge_bandwidth_gbps",
            "controller_bandwidth_gbps",
            "core_bandwidth_gbps",
            "slice_shares",
            "external_data",
        },
        "simulation",
    )
    if simulation["source"] not in {"synthetic", "external"}:
        raise ValueError("simulation.source must be 'synthetic' or 'external'")
    _integer(simulation["seed"], "simulation.seed")
    base_stations = _integer(simulation["base_stations"], "simulation.base_stations", 1)
    edge_servers = _integer(simulation["edge_servers"], "simulation.edge_servers", 1)
    controllers = _integer(simulation["edge_controllers"], "simulation.edge_controllers", 1)
    _integer(simulation["core_nodes"], "simulation.core_nodes", 1)
    if not 8 <= controllers <= 16:
        raise ValueError("simulation.edge_controllers must be in the paper's 8-16 range")
    if edge_servers != 200:
        raise ValueError("simulation.edge_servers must be 200 for the paper scenario")
    if base_stations != 4328:
        raise ValueError("simulation.base_stations must be 4328 for the paper scenario")
    _pair(simulation["area_km"], "simulation.area_km", 0.1)
    _number(simulation["duration_s"], "simulation.duration_s", 0.001)
    _number(simulation["decision_interval_s"], "simulation.decision_interval_s", 0.0001)
    _integer(simulation["candidate_paths"], "simulation.candidate_paths", 1)
    _integer(simulation["max_local_nodes"], "simulation.max_local_nodes", 1)
    required_local_nodes = (
        (edge_servers + controllers - 1) // controllers
        + 1
        + int(simulation["core_nodes"])
    )
    if simulation["max_local_nodes"] < required_local_nodes:
        raise ValueError(
            "simulation.max_local_nodes cannot pad the largest controller region and cloud core"
        )
    _integer(simulation["nearest_neighbors"], "simulation.nearest_neighbors", 2)
    _number(simulation["mobility_update_s"], "simulation.mobility_update_s", 0.001)
    _pair(simulation["cross_region_fraction"], "simulation.cross_region_fraction")
    _number(simulation["mips_per_vcpu"], "simulation.mips_per_vcpu", 1.0)
    for name in ("edge_server_resources", "controller_resources", "core_resources"):
        ranges = simulation[name]
        _require_keys(ranges, set(RESOURCE_NAMES), f"simulation.{name}")
        for resource in RESOURCE_NAMES:
            _pair(ranges[resource], f"simulation.{name}.{resource}")
    for name in ("edge_bandwidth_gbps", "controller_bandwidth_gbps", "core_bandwidth_gbps"):
        _pair(simulation[name], f"simulation.{name}", 0.001)
    _require_keys(simulation["slice_shares"], set(SLICE_NAMES), "simulation.slice_shares")
    slice_total = sum(
        _number(simulation["slice_shares"][name], f"simulation.slice_shares.{name}", 0.0)
        for name in SLICE_NAMES
    )
    if abs(slice_total - 1.0) > 1e-9:
        raise ValueError("simulation.slice_shares must sum to 1")
    _require_keys(
        simulation["external_data"],
        {"opencellid_csv", "wsdot_manifest", "alibaba_trace"},
        "simulation.external_data",
    )
    for name, value in simulation["external_data"].items():
        if not isinstance(value, str) or not value:
            raise TypeError(f"simulation.external_data.{name} must be a non-empty path")

    workloads = config["workloads"]
    _require_keys(
        workloads,
        {
            "catalog_size",
            "class_mix",
            "classes",
            "arrival_rates",
            "payload_lognormal_mb",
            "cpu_workload_mi",
            "lifetime_s",
            "dag_extra_edge_probability",
            "pareto",
        },
        "workloads",
    )
    _integer(workloads["catalog_size"], "workloads.catalog_size", 3)
    _require_keys(workloads["class_mix"], set(WORKLOAD_CLASSES), "workloads.class_mix")
    mix_total = sum(_number(workloads["class_mix"][name], f"workloads.class_mix.{name}", 0) for name in WORKLOAD_CLASSES)
    if abs(mix_total - 1.0) > 1e-9:
        raise ValueError("workloads.class_mix must sum to 1")
    _require_keys(workloads["classes"], set(WORKLOAD_CLASSES), "workloads.classes")
    class_keys = {"microservices", "latency_ms", "jitter_ms", "sync_ms", "min_reliability", "replicas", "resources"}
    for class_name in WORKLOAD_CLASSES:
        spec = workloads["classes"][class_name]
        _require_keys(spec, class_keys, f"workloads.classes.{class_name}")
        for key in ("microservices", "latency_ms", "jitter_ms", "sync_ms", "replicas"):
            _pair(spec[key], f"workloads.classes.{class_name}.{key}")
        reliability = _number(spec["min_reliability"], f"workloads.classes.{class_name}.min_reliability", 0)
        if reliability > 1:
            raise ValueError(f"workloads.classes.{class_name}.min_reliability must be <= 1")
        _require_keys(spec["resources"], set(RESOURCE_NAMES), f"workloads.classes.{class_name}.resources")
        for resource in RESOURCE_NAMES:
            _pair(spec["resources"][resource], f"workloads.classes.{class_name}.resources.{resource}")
    _require_keys(workloads["arrival_rates"], {"urban", "medium", "suburban"}, "workloads.arrival_rates")
    for density in ("urban", "medium", "suburban"):
        _pair(workloads["arrival_rates"][density], f"workloads.arrival_rates.{density}")
    _require_keys(workloads["payload_lognormal_mb"], {"mean", "std"}, "workloads.payload_lognormal_mb")
    _number(workloads["payload_lognormal_mb"]["mean"], "workloads.payload_lognormal_mb.mean", 0.001)
    _number(workloads["payload_lognormal_mb"]["std"], "workloads.payload_lognormal_mb.std", 0.001)
    _pair(workloads["cpu_workload_mi"], "workloads.cpu_workload_mi", 1)
    _pair(workloads["lifetime_s"], "workloads.lifetime_s", 0.001)
    probability = _number(workloads["dag_extra_edge_probability"], "workloads.dag_extra_edge_probability", 0)
    if probability > 1:
        raise ValueError("workloads.dag_extra_edge_probability must be <= 1")
    _require_keys(workloads["pareto"], {"on_shape", "off_shape", "on_scale_s", "off_scale_s", "burst_multiplier"}, "workloads.pareto")
    for key in workloads["pareto"]:
        _number(workloads["pareto"][key], f"workloads.pareto.{key}", 0.001)

    reward = config["reward"]
    _require_keys(
        reward,
        {"global_weights", "local_weights", "penalty_weights", "cooperation", "stability_utilization", "slice_isolation_utilization"},
        "reward",
    )
    objective_keys = {"latency", "resource", "load_balance", "reliability"}
    for group in ("global_weights", "local_weights"):
        _require_keys(reward[group], objective_keys, f"reward.{group}")
        total = sum(_number(reward[group][key], f"reward.{group}.{key}", 0) for key in objective_keys)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"reward.{group} must sum to 1")
    _require_keys(reward["penalty_weights"], {"resource", "temporal", "robustness"}, "reward.penalty_weights")
    for key in reward["penalty_weights"]:
        _number(reward["penalty_weights"][key], f"reward.penalty_weights.{key}", 0)
    cooperation = _number(reward["cooperation"], "reward.cooperation", 0)
    if cooperation > 1:
        raise ValueError("reward.cooperation must be <= 1")
    for key in ("stability_utilization", "slice_isolation_utilization"):
        value = _number(reward[key], f"reward.{key}", 0)
        if value > 1:
            raise ValueError(f"reward.{key} must be <= 1")

    planner = config["planner"]
    planner_keys = {
        "backend", "model_id", "dtype", "planning_interval_s", "candidate_count", "prompt_input_tokens",
        "max_new_tokens", "prompt_path", "memory_path", "latest_plan_path", "snapshot_path", "retrieval_pool",
        "retrieval_top_k", "similarity_weights", "soft_prune_threshold", "memory_ttl_windows", "max_traces",
        "drift_threshold", "lora",
    }
    _require_keys(planner, planner_keys, "planner")
    if planner["backend"] != "qwen2_vl":
        raise ValueError("planner.backend must be 'qwen2_vl'")
    if planner["dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("planner.dtype is unsupported")
    for key in ("model_id", "prompt_path", "memory_path", "latest_plan_path", "snapshot_path"):
        if not isinstance(planner[key], str) or not planner[key]:
            raise TypeError(f"planner.{key} must be a non-empty string")
    for key in ("planning_interval_s", "candidate_count", "prompt_input_tokens", "max_new_tokens", "retrieval_pool", "retrieval_top_k", "memory_ttl_windows", "max_traces"):
        _integer(planner[key], f"planner.{key}", 1)
    _require_keys(planner["similarity_weights"], {"latent", "topology", "placement"}, "planner.similarity_weights")
    similarity_total = sum(_number(planner["similarity_weights"][key], f"planner.similarity_weights.{key}", 0) for key in planner["similarity_weights"])
    if abs(similarity_total - 1.0) > 1e-9:
        raise ValueError("planner.similarity_weights must sum to 1")
    for key in ("soft_prune_threshold", "drift_threshold"):
        _number(planner[key], f"planner.{key}", 0)
    _require_keys(planner["lora"], {"rank", "alpha", "dropout", "trainable_fraction"}, "planner.lora")
    _integer(planner["lora"]["rank"], "planner.lora.rank", 1)
    _number(planner["lora"]["alpha"], "planner.lora.alpha", 0)
    _number(planner["lora"]["dropout"], "planner.lora.dropout", 0)
    _number(planner["lora"]["trainable_fraction"], "planner.lora.trainable_fraction", 0)

    training = config["training"]
    training_keys = {
        "max_episodes", "rollout_steps", "minibatch_size", "update_epochs", "actor_lr", "critic_lr",
        "adam_beta1", "adam_beta2", "adam_eps", "gamma", "gae_lambda", "clip_ratio", "value_coef",
        "entropy_start", "entropy_end", "max_grad_norm", "graph_hidden", "gru_hidden", "kl_coef",
        "kl_decay_steps", "gumbel_start", "gumbel_end", "trust_alpha", "trust_cap", "activation_threshold",
        "checkpoint_every", "dqn_replay_size", "dqn_batch_size", "dqn_target_every",
        "coral_population", "coral_mutation_rate", "max_microservices", "max_paths", "context_dim",
        "global_state_dim", "semantic_context_dim",
    }
    _require_keys(training, training_keys, "training")
    integer_training = {
        "max_episodes", "rollout_steps", "minibatch_size", "update_epochs", "graph_hidden", "gru_hidden",
        "kl_decay_steps", "checkpoint_every", "dqn_replay_size", "dqn_batch_size", "dqn_target_every",
        "coral_population", "max_microservices", "max_paths", "context_dim", "global_state_dim", "semantic_context_dim",
    }
    for key in integer_training:
        _integer(training[key], f"training.{key}", 1)
    for key in training_keys - integer_training:
        _number(training[key], f"training.{key}", 0)
    if training["rollout_steps"] % training["minibatch_size"]:
        raise ValueError("training.rollout_steps must be divisible by training.minibatch_size")
    if training["max_paths"] != simulation["candidate_paths"]:
        raise ValueError("training.max_paths must equal simulation.candidate_paths")
    max_paper_microservices = max(int(config["workloads"]["classes"][name]["microservices"][1]) for name in WORKLOAD_CLASSES)
    if training["max_microservices"] < max_paper_microservices:
        raise ValueError("training.max_microservices is smaller than a configured service DAG")

    evaluation = config["evaluation"]
    _require_keys(
        evaluation,
        {"seeds", "monte_carlo_runs", "methods", "arrival_rates", "complexity", "failure_rates", "convergence"},
        "evaluation",
    )
    if not isinstance(evaluation["seeds"], list) or not evaluation["seeds"]:
        raise TypeError("evaluation.seeds must be a non-empty list")
    for index, seed in enumerate(evaluation["seeds"]):
        _integer(seed, f"evaluation.seeds[{index}]")
    _integer(evaluation["monte_carlo_runs"], "evaluation.monte_carlo_runs", 1)
    if not isinstance(evaluation["methods"], list) or not all(isinstance(value, str) and value for value in evaluation["methods"]):
        raise TypeError("evaluation.methods must contain method names")
    if not isinstance(evaluation["arrival_rates"], list) or not evaluation["arrival_rates"]:
        raise TypeError("evaluation.arrival_rates must be a non-empty list")
    for index, rate in enumerate(evaluation["arrival_rates"]):
        _number(rate, f"evaluation.arrival_rates[{index}]", 0)
    if not isinstance(evaluation["complexity"], list) or not evaluation["complexity"]:
        raise TypeError("evaluation.complexity must be a non-empty list")
    for index, item in enumerate(evaluation["complexity"]):
        _require_keys(item, {"name", "microservices", "replicas"}, f"evaluation.complexity[{index}]")
        if not isinstance(item["name"], str) or not item["name"]:
            raise TypeError(f"evaluation.complexity[{index}].name must be text")
        _integer(item["microservices"], f"evaluation.complexity[{index}].microservices", 1)
        _integer(item["replicas"], f"evaluation.complexity[{index}].replicas", 1)
    if not isinstance(evaluation["failure_rates"], list):
        raise TypeError("evaluation.failure_rates must be a list")
    for index, rate in enumerate(evaluation["failure_rates"]):
        value = _number(rate, f"evaluation.failure_rates[{index}]", 0)
        if value > 100:
            raise ValueError("evaluation.failure_rates are percentages in [0,100]")
    _require_keys(evaluation["convergence"], {"window", "min_episode", "patience", "relative_delta", "episode_ceiling"}, "evaluation.convergence")
    for key in ("window", "min_episode", "patience", "episode_ceiling"):
        _integer(evaluation["convergence"][key], f"evaluation.convergence.{key}", 1)
    _number(evaluation["convergence"]["relative_delta"], "evaluation.convergence.relative_delta", 0)

    paths = config["paths"]
    _require_keys(
        paths,
        {"generated_scenario", "checkpoints", "figure_data", "training_logs"},
        "paths",
    )
    for key, value in paths.items():
        if not isinstance(value, str) or not value:
            raise TypeError(f"paths.{key} must be a non-empty string")
    return config


def load_config(path: str | Path = "configs/paper.json") -> dict[str, Any]:
    """Load and validate the project configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return validate_config(config)
