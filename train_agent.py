"""Train the conference RAG-DRL scheduler and its two baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import agent
import baselines
import llmPlanner
import rag
import simulator

METHODS = ("RAG-DRL", "DQN Scheduler", "DRP")
GLOBAL_PLAN_INTERVAL_S = 30.0 * 60.0
LOG_COLUMNS = (
    "method",
    "seed",
    "episode",
    "reward",
    "score",
    "latency_ms",
    "resource_use",
    "sla_violation",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _scale(config: dict[str, Any]) -> dict[str, int]:
    return dict(config["experiment"]["latency_scales"][0])


def _service_signature(service: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("cpu", "memory", "traffic", "adjacency"):
        digest.update(np.asarray(service[key], dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def _example_observation(
    config: dict[str, Any], scenario: dict[str, Any], seed: int
) -> dict[str, Any]:
    for offset in range(1000):
        state = simulator.reset(config, scenario, _scale(config), seed + offset)
        if not state["done"]:
            return simulator.observe(state)
    raise RuntimeError("1,000 seeded Poisson episodes produced no schedulable request")


def _simulator_action(
    observation: dict[str, Any], decision: dict[str, Any]
) -> dict[str, Any]:
    index = int(decision["placement_index"])
    candidate_ids = np.asarray(observation["candidate_node_ids"], dtype=np.int64)
    if index < 0 or index >= len(candidate_ids):
        raise ValueError("policy selected a candidate outside the observation")
    return {
        "placement_index": index,
        "node_id": int(candidate_ids[index]),
        "routing_fractions": np.asarray(
            decision["routing_fractions"], dtype=np.float32
        ),
    }


def _episode_log(
    method: str, seed: int, episode: int, metrics: dict[str, float]
) -> dict[str, Any]:
    return {
        "method": method,
        "seed": seed,
        "episode": episode,
        "reward": float(metrics["reward"]),
        "score": float(metrics["score"]),
        "latency_ms": float(metrics["end_to_end_latency_ms"]),
        "resource_use": float(metrics["resource_use"]),
        "sla_violation": float(metrics["sla_violation_rate_percent"]),
    }


def _learning_reward(config: dict[str, Any], info: dict[str, Any]) -> float:
    rewards = [float(value) for value in info["request_rewards"]]
    if not rewards:
        return 0.0
    slots = int(info["slot_advance"])
    if slots < 1:
        raise ValueError("request rewards require at least one advanced slot")
    gamma = float(config["ppo"]["gamma"])
    final_slot = slots - 1
    return float(
        sum(
            gamma ** min(index, final_slot) * reward
            for index, reward in enumerate(rewards)
        )
    )


def _write_log(path: str, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _load_planner_encoder(
    config: dict[str, Any], scenario: dict[str, Any], device: torch.device
) -> Any:
    encoder = agent.build_graph_encoder(config, scenario).to(device)
    path = Path(config["llm"]["adapter"]) / "graph_encoder.pt"
    if not path.is_file():
        raise FileNotFoundError(f"planner graph encoder not found: {path}")
    encoder.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    encoder.eval()
    return encoder


def _semantic_query(
    policy: dict[str, Any], observation: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    embedding = (
        agent.graph_embedding(policy["planner_encoder"], observation)
        .squeeze(0)
        .cpu()
        .tolist()
    )
    return simulator.planner_query(state, embedding)


def _planner_decision(
    policy: dict[str, Any], observation: dict[str, Any], state: dict[str, Any]
) -> None:
    request_id = int(state["current_request"])
    controller = int(state["current_controller"])
    service = state["service"]
    cache_key = (
        controller,
        int(service["microservices"]),
        int(service["replicas"]),
        _service_signature(service),
    )
    reuse_s = float(policy["config"]["llm"]["edge_push_ms"]) / 1000.0
    expired = float(state["time_s"]) - float(policy["plan_time_s"]) >= reuse_s
    exhausted = policy["plan"] is None or policy["plan_index"] >= len(
        policy["plan"]["placement"] if policy["plan"] is not None else []
    )
    planned_node_is_feasible = False
    if not exhausted:
        planned_node = int(policy["plan"]["placement"][policy["plan_index"]])
        ids = np.asarray(observation["candidate_node_ids"])
        mask = np.asarray(observation["placement_mask"], dtype=bool)
        planned_node_is_feasible = bool(np.any(ids[mask] == planned_node))
    if (
        not exhausted
        and policy["planned_request"] == request_id
        and not expired
        and not state["anomaly"]
        and planned_node_is_feasible
    ):
        return
    if (
        policy["planned_request"] == request_id
        and not exhausted
        and not planned_node_is_feasible
    ):
        policy["plan_cache"].pop(cache_key, None)
    cached = policy["plan_cache"].get(cache_key)
    global_time = float(policy["global_time_s"]) + float(state["time_s"])
    if (
        policy["planned_request"] != request_id
        and cached is not None
        and global_time - float(cached["generated_at_s"]) < GLOBAL_PLAN_INTERVAL_S
        and not state["anomaly"]
    ):
        selected = int(cached["plan"]["placement"][0])
        ids = np.asarray(observation["candidate_node_ids"])
        mask = np.asarray(observation["placement_mask"], dtype=bool)
        if np.any(ids[mask] == selected):
            if policy["record_memory"]:
                policy["request_query"] = _semantic_query(policy, observation, state)
            policy["plan"] = cached["plan"]
            policy["plan_index"] = 0
            if expired:
                policy["plan_time_s"] = float(state["time_s"])
            policy["planned_request"] = request_id
            return
    query = _semantic_query(policy, observation, state)
    if policy["record_memory"] and policy["planned_request"] != request_id:
        policy["request_query"] = query
    candidates = llmPlanner.plan(policy["planner"], policy["config"], query)
    policy["plan"] = candidates[0]
    policy["plan_index"] = 0
    policy["plan_time_s"] = float(state["time_s"])
    policy["planned_request"] = request_id
    if int(state["microservice_index"]) == 0 and int(state["replica_index"]) == 0:
        policy["plan_cache"][cache_key] = {
            "plan": policy["plan"],
            "generated_at_s": global_time,
        }
    state["anomaly"] = False


def _rag_action(
    policy: dict[str, Any],
    observation: dict[str, Any],
    state: dict[str, Any],
    deterministic: bool,
) -> dict[str, Any]:
    if policy["episode_state"] is not state:
        if policy["episode_state"] is not None:
            policy["global_time_s"] += float(policy["config"]["traffic"]["duration_s"])
        policy["episode_state"] = state
        policy["plan"] = None
        policy["plan_index"] = 0
        policy["plan_time_s"] = -float("inf")
        policy["planned_request"] = None
        policy["request_query"] = None
    _planner_decision(policy, observation, state)
    prior = llmPlanner.soft_placement_prior(
        policy["plan"],
        policy["plan_index"],
        [int(item) for item in observation["candidate_node_ids"]],
        float(policy["config"]["llm"]["prior_strength"]),
    )
    guided = dict(observation)
    guided["placement_prior_logits"] = prior
    decision = agent.sample_action(policy["ppo"], guided, deterministic=deterministic)
    decision["policy_observation"] = guided
    policy["plan_index"] += 1
    return decision


def _record_online_trace(
    policy: dict[str, Any],
    info: dict[str, Any],
) -> None:
    completed = info["completed_request"]
    if completed is None:
        return
    query = policy["request_query"]
    objective = simulator.request_objective(policy["config"], completed)
    rag.record_trace(
        policy["config"]["rag"]["database"],
        {
            "timestamp": time.time(),
            "embedding": query["embedding"],
            "telemetry": query["telemetry"],
            "intent": query["intent"],
            "dag": query["dag"],
            "placement": np.asarray(info["completed_placement"]).tolist(),
            "outcome": {
                "latency_ms": float(completed["latency_ms"]),
                "resource_use": float(completed["resource_utilization"]),
                "sla_violations": float(completed["sla_violation"]),
                **objective,
            },
        },
        int(policy["config"]["rag"]["max_traces"]),
    )


def _collect_meta_rollout(
    policy: dict[str, Any],
    config: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rollout: list[dict[str, Any]] = []
    episode = 0
    target_steps = int(config["ppo"]["rollout_steps"])
    while len(rollout) < target_steps:
        state = simulator.reset(config, scenario, _scale(config), seed + episode)
        observation = simulator.observe(state)
        done = bool(state["done"])
        while not done:
            decision = agent.sample_action(policy["ppo"], observation)
            next_observation, _, done, info = simulator.step(
                state, _simulator_action(observation, decision)
            )
            reward = _learning_reward(config, info)
            slot_advance = int(info["slot_advance"])
            rollout.append(
                {
                    "observation": observation,
                    "placement_index": decision["placement_index"],
                    "routing_fractions": decision["routing_fractions"],
                    "old_log_prob": decision["log_prob"],
                    "value": decision["value"],
                    "reward": reward,
                    "slot_advance": slot_advance,
                    "done": done,
                }
            )
            observation = next_observation
            if len(rollout) >= target_steps and (slot_advance > 0 or done):
                break
        episode += 1
    last_value = (
        0.0 if rollout[-1]["done"] else agent.estimate_value(policy["ppo"], observation)
    )
    agent.compute_gae(
        rollout,
        last_value,
        float(config["ppo"]["gamma"]),
        float(config["ppo"]["gae_lambda"]),
    )
    return rollout


def _bootstrap_rag(
    policy: dict[str, Any],
    config: dict[str, Any],
    scenario: dict[str, Any],
) -> None:
    raw_samples = llmPlanner.load_offline_samples(config["llm"]["offline_data"])
    examples = [
        {
            "observation": sample["observation"],
            "placement_index": int(sample["candidate_index"]),
        }
        for sample in raw_samples
    ]
    agent.behavior_clone(
        policy["ppo"],
        examples,
        epochs=int(config["llm"]["behavior_epochs"]),
        minibatch_size=int(config["ppo"]["minibatch_size"]),
    )
    task_rollouts = []
    for offset, kind in enumerate(("node_failure", "bandwidth", "node_churn")):
        perturbed = simulator.perturb_scenario(scenario, kind, config["seed"] + offset)
        task_rollouts.append(
            _collect_meta_rollout(
                policy, config, perturbed, config["seed"] + 1000 * (offset + 1)
            )
        )
    agent.first_order_meta_update(
        policy["ppo"],
        task_rollouts,
        config["ppo"],
        inner_steps=int(config["ppo"]["meta_steps"]),
        meta_step_size=float(config["llm"]["meta_rate"]),
        total_episodes=int(config["experiment"]["episodes"]),
    )


def _train_rag(
    config: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    example = _example_observation(config, scenario, seed)
    policy = {
        "method": "RAG-DRL",
        "ppo": agent.build_ppo(example, config["ppo"], device),
        "planner": llmPlanner.load_qwen_planner(config, trainable=False),
        "planner_encoder": _load_planner_encoder(config, scenario, device),
        "config": config,
        "plan": None,
        "plan_index": 0,
        "plan_time_s": -float("inf"),
        "planned_request": None,
        "request_query": None,
        "episode_state": None,
        "plan_cache": {},
        "global_time_s": 0.0,
        "record_memory": True,
    }
    _bootstrap_rag(policy, config, scenario)
    rollout: list[dict[str, Any]] = []
    rows = []
    global_step = 0
    episodes = int(config["experiment"]["episodes"])
    for episode in range(1, episodes + 1):
        state = simulator.reset(config, scenario, _scale(config), seed + episode)
        observation = simulator.observe(state)
        policy["plan"] = None
        done = bool(state["done"])
        while not done:
            decision = _rag_action(policy, observation, state, deterministic=False)
            guided_observation = decision["policy_observation"]
            next_observation, _, done, info = simulator.step(
                state, _simulator_action(observation, decision)
            )
            reward = _learning_reward(config, info)
            slot_advance = int(info["slot_advance"])
            rollout.append(
                {
                    "observation": guided_observation,
                    "placement_index": decision["placement_index"],
                    "routing_fractions": decision["routing_fractions"],
                    "old_log_prob": decision["log_prob"],
                    "value": decision["value"],
                    "reward": reward,
                    "slot_advance": slot_advance,
                    "done": done,
                }
            )
            observation = next_observation
            global_step += 1
            _record_online_trace(policy, info)
            if len(rollout) >= int(config["ppo"]["rollout_steps"]) and (
                slot_advance > 0 or done
            ):
                agent.compute_gae(
                    rollout,
                    0.0
                    if done
                    else agent.estimate_value(policy["ppo"], next_observation),
                    float(config["ppo"]["gamma"]),
                    float(config["ppo"]["gae_lambda"]),
                )
                agent.ppo_update(
                    policy["ppo"],
                    rollout,
                    config["ppo"],
                    episode=episode,
                    total_episodes=episodes,
                )
                rollout.clear()
        rows.append(
            _episode_log("RAG-DRL", seed, episode, simulator.episode_metrics(state))
        )
    if rollout:
        agent.compute_gae(
            rollout,
            0.0,
            float(config["ppo"]["gamma"]),
            float(config["ppo"]["gae_lambda"]),
        )
        agent.ppo_update(
            policy["ppo"],
            rollout,
            config["ppo"],
            episode=episodes,
            total_episodes=episodes,
        )
    agent.save_checkpoint(
        config["paths"]["rag_drl_checkpoint"],
        policy["ppo"],
        config,
        episode=episodes,
        global_step=global_step,
    )
    return policy, rows


def _train_dqn(
    config: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    policy = baselines.build_dqn(
        _example_observation(config, scenario, seed), config["baseline"], device
    )
    rows = []
    global_step = 0
    episodes = int(config["experiment"]["episodes"])
    for episode in range(1, episodes + 1):
        state = simulator.reset(config, scenario, _scale(config), seed + episode)
        observation = simulator.observe(state)
        done = bool(state["done"])
        epsilon = 1.0 - 0.95 * (episode - 1) / max(1, episodes - 1)
        while not done:
            index = baselines.select_dqn_action(policy, observation, epsilon, rng)
            routing = baselines.shortest_feasible_routing(observation, index)
            next_observation, _, done, info = simulator.step(
                state,
                _simulator_action(
                    observation,
                    {"placement_index": index, "routing_fractions": routing},
                ),
            )
            reward = _learning_reward(config, info)
            slot_advance = int(info["slot_advance"])
            baselines.store_dqn_transition(
                policy,
                observation,
                index,
                reward,
                slot_advance,
                None if done else next_observation,
                done,
            )
            baselines.update_dqn(
                policy,
                config["baseline"],
                gamma=float(config["ppo"]["gamma"]),
            )
            observation = next_observation
            global_step += 1
        rows.append(
            _episode_log(
                "DQN Scheduler", seed, episode, simulator.episode_metrics(state)
            )
        )
    baselines.save_dqn_checkpoint(
        config["paths"]["dqn_checkpoint"],
        policy,
        config,
        episode=episodes,
        global_step=global_step,
    )
    return policy, rows


def _configuration_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _save_drp(path: str, policy: dict[str, Any], config: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "population": np.asarray(policy["population"], dtype=np.int64).tolist(),
        "action_count": int(policy["action_count"]),
        "generation": int(policy["generation"]),
        "configuration_hash": _configuration_hash(config),
    }
    destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _train_drp(
    config: dict[str, Any], scenario: dict[str, Any], seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    policy = baselines.build_drp(_example_observation(config, scenario, seed), rng)
    rows = []
    episodes = int(config["experiment"]["episodes"])
    for episode in range(1, episodes + 1):
        state = simulator.reset(config, scenario, _scale(config), seed + episode)
        observation = simulator.observe(state)
        first_observation = observation
        done = bool(state["done"])
        should_evolve = not done
        while not done:
            index = baselines.select_drp_action(policy, observation)
            routing = baselines.shortest_feasible_routing(observation, index)
            observation, _, done, _ = simulator.step(
                state,
                _simulator_action(
                    observation,
                    {"placement_index": index, "routing_fractions": routing},
                ),
            )
        if should_evolve:
            baselines.evolve_drp(policy, first_observation, rng)
        rows.append(
            _episode_log("DRP", seed, episode, simulator.episode_metrics(state))
        )
    _save_drp(config["paths"]["drp_checkpoint"], policy, config)
    return policy, rows


def load_policy(config: dict[str, Any], method: str) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown conference method: {method}")
    scenario = simulator.load_scenario(config["data"]["scenario"])
    observation = _example_observation(config, scenario, config["seed"])
    device = _device()
    if method == "RAG-DRL":
        policy = {
            "method": method,
            "ppo": agent.build_ppo(observation, config["ppo"], device),
            "planner": llmPlanner.load_qwen_planner(config, trainable=False),
            "planner_encoder": _load_planner_encoder(config, scenario, device),
            "config": config,
            "plan": None,
            "plan_index": 0,
            "plan_time_s": -float("inf"),
            "planned_request": None,
            "request_query": None,
            "episode_state": None,
            "plan_cache": {},
            "global_time_s": 0.0,
            "record_memory": False,
        }
        agent.load_checkpoint(
            config["paths"]["rag_drl_checkpoint"], policy["ppo"], config
        )
        return policy
    if method == "DQN Scheduler":
        policy = baselines.build_dqn(observation, config["baseline"], device)
        baselines.load_dqn_checkpoint(config["paths"]["dqn_checkpoint"], policy, config)
        policy["method"] = method
        return policy
    value = json.loads(
        Path(config["paths"]["drp_checkpoint"]).read_text(encoding="utf-8")
    )
    if value["configuration_hash"] != _configuration_hash(config):
        raise ValueError("DRP checkpoint configuration does not match paper config")
    return {
        "method": method,
        "population": np.asarray(value["population"], dtype=np.int64),
        "action_count": int(value["action_count"]),
        "generation": int(value["generation"]),
    }


def reset_evaluation_state(policy: dict[str, Any]) -> None:
    """Isolate transient RAG planner state between Monte Carlo trials."""

    if policy["method"] != "RAG-DRL":
        return
    policy["plan"] = None
    policy["plan_index"] = 0
    policy["plan_time_s"] = -float("inf")
    policy["planned_request"] = None
    policy["request_query"] = None
    policy["episode_state"] = None
    policy["plan_cache"].clear()
    policy["global_time_s"] = 0.0


def policy_action(
    policy: dict[str, Any],
    observation: dict[str, Any],
    method: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    if method != policy["method"]:
        raise ValueError("policy and requested method differ")
    if method == "RAG-DRL":
        decision = _rag_action(policy, observation, state, deterministic=True)
    elif method == "DQN Scheduler":
        index = baselines.select_dqn_action(
            policy, observation, 0.0, np.random.default_rng(0)
        )
        decision = {
            "placement_index": index,
            "routing_fractions": baselines.shortest_feasible_routing(
                observation, index
            ),
        }
    else:
        index = baselines.select_drp_action(policy, observation)
        decision = {
            "placement_index": index,
            "routing_fractions": baselines.shortest_feasible_routing(
                observation, index
            ),
        }
    return _simulator_action(observation, decision)


def main() -> None:
    config = simulator.load_config("configs/paper.json")
    scenario = simulator.load_scenario(config["data"]["scenario"])
    if tuple(config["experiment"]["methods"]) != METHODS:
        raise ValueError(f"experiment methods must be {METHODS}")
    required = (
        Path(config["llm"]["offline_data"]),
        Path(config["llm"]["adapter"]),
        Path(config["llm"]["adapter"]) / "graph_encoder.pt",
        Path(config["rag"]["database"]),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"planner artifacts are missing: {missing}. Run python llmPlanner.py first."
        )
    device = _device()
    all_rows: list[dict[str, Any]] = []
    for seed in config["experiment"]["seeds"]:
        seed = int(seed)
        seed_everything(seed)
        _, rows = _train_rag(config, scenario, seed, device)
        all_rows.extend(rows)
        seed_everything(seed)
        _, rows = _train_dqn(config, scenario, seed, device)
        all_rows.extend(rows)
        _, rows = _train_drp(config, scenario, seed)
        all_rows.extend(rows)
    _write_log(config["paths"]["training_log"], all_rows)
    print(
        f"trained {', '.join(METHODS)} for {config['experiment']['episodes']} episodes"
    )


if __name__ == "__main__":
    main()
