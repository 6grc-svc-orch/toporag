"""Adapted comparison methods evaluated on the shared simulator.

The paper does not publish its baseline integrations.  These profiles keep
the comparison controlled: every method receives the same workload trace and
differs only in visibility, semantic guidance, coordination, or learning rule.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from agent.gnn import MaskedGraphSAGE

METHODS = (
    "LLM-MAPPO",
    "LLM-MAPPO-no-TopoRAG",
    "RSPPO",
    "GNN-DRL",
    "mu-DDRL",
    "CROMADQN",
)

ABLATION_METHODS = (
    "LLM-MAPPO-no-reward-shaping",
    "MAPPO-no-LLM-guidance",
)

METHOD_IDS = {
    "LLM-MAPPO": "llm_mappo",
    "LLM-MAPPO-no-TopoRAG": "llm_mappo_no_toporag",
    "RSPPO": "rsppo",
    "GNN-DRL": "gnn_drl",
    "mu-DDRL": "mu_ddrl",
    "CROMADQN": "cromadqn",
    "LLM-MAPPO-no-reward-shaping": "llm_mappo_no_reward_shaping",
    "MAPPO-no-LLM-guidance": "mappo_no_llm_guidance",
}


def method_id(method: str) -> str:
    if method not in METHOD_IDS:
        raise ValueError(
            f"Unknown method {method!r}; expected one of {METHODS + ABLATION_METHODS}"
        )
    return METHOD_IDS[method]


def method_profile(method: str) -> dict[str, bool | str]:
    profiles: dict[str, dict[str, bool | str]] = {
        "LLM-MAPPO": {
            "algorithm": "mappo",
            "semantic_guidance": True,
            "toporag": True,
            "central_critic": True,
            "neighbour_summary": False,
            "reward_shaping": True,
        },
        "LLM-MAPPO-no-TopoRAG": {
            "algorithm": "mappo",
            "semantic_guidance": True,
            "toporag": False,
            "central_critic": True,
            "neighbour_summary": False,
            "reward_shaping": True,
        },
        "RSPPO": {
            "algorithm": "ppo",
            "semantic_guidance": False,
            "toporag": False,
            "central_critic": True,
            "neighbour_summary": False,
            "reward_shaping": True,
        },
        "GNN-DRL": {
            "algorithm": "ppo",
            "semantic_guidance": False,
            "toporag": False,
            "central_critic": True,
            "neighbour_summary": False,
            "reward_shaping": False,
        },
        "mu-DDRL": {
            "algorithm": "independent_ppo",
            "semantic_guidance": False,
            "toporag": False,
            "central_critic": False,
            "neighbour_summary": True,
            "reward_shaping": False,
        },
        "CROMADQN": {
            "algorithm": "dqn",
            "semantic_guidance": False,
            "toporag": False,
            "central_critic": False,
            "neighbour_summary": False,
            "reward_shaping": False,
        },
        "LLM-MAPPO-no-reward-shaping": {
            "algorithm": "mappo",
            "semantic_guidance": True,
            "toporag": True,
            "central_critic": True,
            "neighbour_summary": False,
            "reward_shaping": False,
        },
        "MAPPO-no-LLM-guidance": {
            "algorithm": "mappo",
            "semantic_guidance": False,
            "toporag": False,
            "central_critic": True,
            "neighbour_summary": False,
            "reward_shaping": True,
        },
    }
    if method not in profiles:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS + ABLATION_METHODS}")
    return profiles[method]


class PlacementQNetwork(nn.Module):
    """Graph-encoded placement Q-network for the adapted CROMADQN baseline."""

    def __init__(self, infrastructure_dim: int, service_dim: int, context_dim: int, hidden_dim: int, actions: int):
        super().__init__()
        self.infrastructure = MaskedGraphSAGE(infrastructure_dim, hidden_dim)
        self.service = MaskedGraphSAGE(service_dim, hidden_dim)
        self.context = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, actions)
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        infrastructure = self.infrastructure(
            observation["infrastructure_nodes"],
            observation["infrastructure_adjacency"],
            observation["infrastructure_mask"],
        )
        service = self.service(
            observation["service_nodes"],
            observation["service_adjacency"],
            observation["service_mask"],
        )
        context = self.context(observation["context"])
        values = self.head(torch.cat((infrastructure, service, context), dim=-1))
        return values.masked_fill(~observation["placement_mask"].bool(), torch.finfo(values.dtype).min)


def build_cromadqn(example_observation: dict[str, Any], training_config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    infrastructure_dim = int(np.asarray(example_observation["infrastructure_nodes"]).shape[-1])
    service_dim = int(np.asarray(example_observation["service_nodes"]).shape[-1])
    context_dim = int(np.asarray(example_observation["context"]).shape[-1])
    actions = int(np.asarray(example_observation["placement_mask"]).shape[-1])
    hidden = int(training_config["graph_hidden"])
    online = PlacementQNetwork(infrastructure_dim, service_dim, context_dim, hidden, actions).to(device)
    target = PlacementQNetwork(infrastructure_dim, service_dim, context_dim, hidden, actions).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(
        online.parameters(),
        lr=float(training_config["actor_lr"]),
        betas=(float(training_config["adam_beta1"]), float(training_config["adam_beta2"])),
        eps=float(training_config["adam_eps"]),
    )
    return {
        "online": online,
        "target": target,
        "optimizer": optimizer,
        "replay": deque(maxlen=int(training_config["dqn_replay_size"])),
        "device": device,
        "updates": 0,
        "coral_population": int(training_config["coral_population"]),
        "coral_mutation_rate": float(training_config["coral_mutation_rate"]),
    }


def _batch_observations(observations: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "infrastructure_nodes",
        "infrastructure_adjacency",
        "infrastructure_mask",
        "service_nodes",
        "service_adjacency",
        "service_mask",
        "context",
        "placement_mask",
    )
    return {
        key: torch.as_tensor(
            np.stack([np.asarray(observation[key]) for observation in observations]),
            dtype=torch.float32,
            device=device,
        )
        for key in keys
    }


def select_cromadqn_action(
    state: dict[str, Any], observation: dict[str, Any], epsilon: float, rng: np.random.Generator
) -> int:
    feasible = np.flatnonzero(np.asarray(observation["placement_mask"], dtype=bool))
    if len(feasible) == 0:
        raise ValueError("CROMADQN received an observation without feasible placements")
    if rng.random() < epsilon:
        return int(rng.choice(feasible))
    tensors = _batch_observations([observation], state["device"])
    with torch.no_grad():
        q_values = state["online"](tensors)[0].cpu().numpy()
    candidates = coral_reef_candidates(
        q_values,
        observation["placement_mask"],
        state["coral_population"],
        state["coral_mutation_rate"],
        rng,
    )
    return int(candidates[0])


def push_dqn_transition(state: dict[str, Any], transition: tuple[Any, ...]) -> None:
    if len(transition) != 5:
        raise ValueError("DQN transitions must be (observation, action, reward, next_observation, done)")
    state["replay"].append(transition)


def update_cromadqn(state: dict[str, Any], training_config: dict[str, Any]) -> float | None:
    batch_size = int(training_config["dqn_batch_size"])
    if len(state["replay"]) < batch_size:
        return None
    batch = random.sample(state["replay"], batch_size)
    observations, actions, rewards, next_observations, dones = zip(*batch)
    device = state["device"]
    current = _batch_observations(list(observations), device)
    following = _batch_observations(list(next_observations), device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
    q_values = state["online"](current).gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_values = torch.zeros(batch_size, dtype=torch.float32, device=device)
        live = torch.nonzero(done_tensor == 0.0, as_tuple=False).flatten()
        if len(live):
            live_observations = {key: value[live] for key, value in following.items()}
            next_actions = torch.argmax(state["online"](live_observations), dim=-1)
            next_values[live] = state["target"](live_observations).gather(
                1, next_actions.unsqueeze(-1)
            ).squeeze(-1)
        targets = reward_tensor + float(training_config["gamma"]) * (1.0 - done_tensor) * next_values
    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    state["optimizer"].zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(state["online"].parameters(), float(training_config["max_grad_norm"]))
    state["optimizer"].step()
    state["updates"] += 1
    if state["updates"] % int(training_config["dqn_target_every"]) == 0:
        state["target"].load_state_dict(state["online"].state_dict())
    return float(loss.detach().cpu())


def coral_reef_candidates(
    q_values: np.ndarray,
    feasible_mask: np.ndarray,
    population_size: int,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve a compact candidate reef from feasible placement actions."""
    feasible = np.flatnonzero(np.asarray(feasible_mask, dtype=bool))
    if len(feasible) == 0:
        raise ValueError("Coral evolution requires at least one feasible action")
    population = rng.choice(feasible, size=population_size, replace=True)
    for _ in range(3):
        order = population[np.argsort(q_values[population])[::-1]]
        survivors = order[: max(1, population_size // 2)]
        offspring = rng.choice(survivors, size=population_size - len(survivors), replace=True)
        mutations = rng.random(len(offspring)) < mutation_rate
        offspring[mutations] = rng.choice(feasible, size=int(mutations.sum()), replace=True)
        population = np.concatenate((survivors, offspring))
    ranked = population[np.argsort(q_values[population])[::-1]]
    _, first_indices = np.unique(ranked, return_index=True)
    return ranked[np.sort(first_indices)]


def deterministic_routing(
    path_features: np.ndarray,
    path_mask: np.ndarray,
    incoming_edge_mask: np.ndarray,
    placement_index: int,
) -> np.ndarray:
    """Choose the lowest-delay route for each incoming DAG dependency."""

    features = np.asarray(path_features, dtype=np.float64)
    masks = np.asarray(path_mask, dtype=bool)
    incoming = np.asarray(incoming_edge_mask, dtype=bool)
    fractions = np.zeros((features.shape[0], features.shape[2]), dtype=np.float32)
    for edge_index in np.flatnonzero(incoming):
        valid = np.flatnonzero(masks[edge_index, placement_index])
        if len(valid) == 0:
            raise ValueError("A feasible placement requires a route for every incoming edge")
        delays = features[edge_index, placement_index, valid, 0]
        chosen = int(valid[int(np.argmin(delays))])
        fractions[edge_index, chosen] = 1.0
    return fractions


def _configuration_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_cromadqn_checkpoint(
    path: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    episode: int,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "online": state["online"].state_dict(),
        "target": state["target"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "replay": list(state["replay"]),
        "updates": int(state["updates"]),
        "episode": int(episode),
        "global_step": int(global_step),
        "configuration_hash": _configuration_hash(config),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def load_cromadqn_checkpoint(
    path: Path,
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, int]:
    checkpoint = torch.load(path, map_location=state["device"], weights_only=False)
    expected_hash = _configuration_hash(config)
    if checkpoint["configuration_hash"] != expected_hash:
        raise ValueError(
            f"Checkpoint config hash {checkpoint['configuration_hash']} does not match {expected_hash}"
        )
    state["online"].load_state_dict(checkpoint["online"])
    state["target"].load_state_dict(checkpoint["target"])
    state["optimizer"].load_state_dict(checkpoint["optimizer"])
    state["replay"].extend(checkpoint["replay"])
    state["updates"] = int(checkpoint["updates"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
    return {
        "episode": int(checkpoint["episode"]),
        "global_step": int(checkpoint["global_step"]),
    }
