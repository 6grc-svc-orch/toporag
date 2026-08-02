"""Conference baselines on the same masked placement candidates as PPO."""

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

from agent import (
    GAT_OUTPUT_DIM,
    DAGEncoder,
    compact_placement_observation,
    observation_tensors,
)

DRP_POPULATION_SIZE = 10
DRP_CROSSOVER_RATE = 0.8
DRP_MUTATION_RATE = 0.1


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class DQNNetwork(nn.Module):
    """Graph-aware Q-network for the conference DQN Scheduler."""

    def __init__(self, service_dim: int, node_dim: int, context_dim: int):
        super().__init__()
        self.dag = DAGEncoder(service_dim)
        self.node = nn.Linear(node_dim, GAT_OUTPUT_DIM)
        self.state = nn.Linear(context_dim, GAT_OUTPUT_DIM)
        self.context = nn.Linear(GAT_OUTPUT_DIM * 3, GAT_OUTPUT_DIM)
        self.score = nn.Linear(GAT_OUTPUT_DIM, 1)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        dag = self.dag(
            observation["service_features"],
            observation["service_adjacency"],
            observation["service_mask"],
        )
        candidates = torch.tanh(self.node(observation["node_features"]))
        infrastructure = _masked_mean(candidates, observation["node_mask"])
        state = torch.tanh(self.state(observation["context"]))
        context = torch.tanh(
            self.context(torch.cat((dag, infrastructure, state), dim=-1))
        )
        values = self.score(torch.tanh(candidates + context.unsqueeze(1))).squeeze(-1)
        return values.masked_fill(
            ~observation["placement_mask"].bool(), torch.finfo(values.dtype).min
        )


def _batch_observations(
    observations: list[dict[str, Any]], device: torch.device
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("Cannot batch an empty observation list")
    converted = [
        observation_tensors(observation, device, add_batch=False)
        for observation in observations
    ]
    return {key: torch.stack([item[key] for item in converted]) for key in converted[0]}


def build_dqn(
    example_observation: dict[str, Any],
    baseline_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    tensors = observation_tensors(example_observation, device)
    online = DQNNetwork(
        service_dim=int(tensors["service_features"].shape[-1]),
        node_dim=int(tensors["node_features"].shape[-1]),
        context_dim=int(tensors["context"].shape[-1]),
    ).to(device)
    target = _copy_dqn(online, device)
    optimizer = torch.optim.Adam(
        online.parameters(),
        lr=float(baseline_config["dqn_lr"]),
    )
    return {
        "online": online,
        "target": target,
        "optimizer": optimizer,
        "replay": deque(maxlen=int(baseline_config["dqn_replay_size"])),
        "device": device,
        "updates": 0,
    }


def _copy_dqn(network: DQNNetwork, device: torch.device) -> DQNNetwork:
    copied = DQNNetwork(
        service_dim=network.dag.layers[0].projection.shape[1],
        node_dim=network.node.in_features,
        context_dim=network.state.in_features,
    ).to(device)
    copied.load_state_dict(network.state_dict())
    return copied


def select_dqn_action(
    state: dict[str, Any],
    observation: dict[str, Any],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    feasible = np.flatnonzero(np.asarray(observation["placement_mask"], dtype=bool))
    if not len(feasible):
        raise ValueError("DQN received no feasible placement candidate")
    if rng.random() < epsilon:
        return int(rng.choice(feasible))
    tensors = observation_tensors(observation, state["device"])
    with torch.no_grad():
        return int(torch.argmax(state["online"](tensors), dim=-1).item())


def store_dqn_transition(
    state: dict[str, Any],
    observation: dict[str, Any],
    action: int,
    reward: float,
    slot_advance: int,
    next_observation: dict[str, Any] | None,
    done: bool,
) -> None:
    if done != (next_observation is None):
        raise ValueError("Terminal DQN transitions must use next_observation=None")
    if slot_advance < 0:
        raise ValueError("slot_advance cannot be negative")
    state["replay"].append(
        (
            compact_placement_observation(observation),
            int(action),
            float(reward),
            int(slot_advance),
            None
            if next_observation is None
            else compact_placement_observation(next_observation),
            bool(done),
        )
    )


def update_dqn(
    state: dict[str, Any], baseline_config: dict[str, Any], *, gamma: float
) -> float | None:
    batch_size = int(baseline_config["dqn_batch_size"])
    if len(state["replay"]) < batch_size:
        return None
    batch = random.sample(state["replay"], batch_size)
    observations, actions, rewards, slot_advances, next_observations, dones = zip(
        *batch
    )
    device = state["device"]
    current = _batch_observations(list(observations), device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    values = state["online"](current).gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)

    targets = rewards_tensor.clone()
    live_indices = [index for index, done in enumerate(dones) if not done]
    if live_indices:
        live_observations = [next_observations[index] for index in live_indices]
        following = _batch_observations(live_observations, device)
        with torch.no_grad():
            next_values = state["target"](following).max(dim=-1).values
        selected = torch.as_tensor(live_indices, dtype=torch.long, device=device)
        discounts = torch.as_tensor(
            [float(gamma) ** int(slot_advances[index]) for index in live_indices],
            dtype=torch.float32,
            device=device,
        )
        targets[selected] += discounts * next_values

    loss = nn.functional.smooth_l1_loss(values, targets)
    state["optimizer"].zero_grad()
    loss.backward()
    state["optimizer"].step()
    state["updates"] = int(state["updates"]) + 1
    if state["updates"] % int(baseline_config["dqn_target_update"]) == 0:
        state["target"].load_state_dict(state["online"].state_dict())
    return float(loss.detach().cpu())


def build_drp(
    example_observation: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    mask = np.asarray(example_observation["placement_mask"], dtype=bool)
    return {
        "population": np.stack(
            [rng.permutation(len(mask)) for _ in range(DRP_POPULATION_SIZE)]
        ).astype(np.int64),
        "action_count": len(mask),
        "generation": 0,
    }


def _drp_arrays(
    state: dict[str, Any], observation: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(observation["placement_mask"], dtype=bool)
    costs = np.asarray(observation["candidate_costs"], dtype=np.float64)
    if mask.shape != (int(state["action_count"]),):
        raise ValueError("DRP placement_mask changed width")
    if costs.shape != mask.shape or not np.all(np.isfinite(costs[mask])):
        raise ValueError("candidate_costs must be finite and match placement_mask")
    if not np.any(mask):
        raise ValueError("DRP received no feasible placement candidate")
    population = np.asarray(state["population"], dtype=np.int64)
    expected = (DRP_POPULATION_SIZE, len(mask))
    if population.shape != expected:
        raise ValueError(f"DRP population must have shape {expected}")
    genes = np.arange(len(mask))
    if any(not np.array_equal(np.sort(chromosome), genes) for chromosome in population):
        raise ValueError("Each DRP chromosome must be a candidate permutation")
    return mask, costs


def _chromosome_action(chromosome: np.ndarray, mask: np.ndarray) -> int:
    for gene in chromosome:
        if mask[int(gene)]:
            return int(gene)
    raise ValueError("DRP chromosome contains no feasible candidate")


def select_drp_action(state: dict[str, Any], observation: dict[str, Any]) -> int:
    """Select the best feasible action proposed by the placement chromosomes."""

    mask, costs = _drp_arrays(state, observation)
    population = np.asarray(state["population"], dtype=np.int64)
    actions = np.asarray(
        [_chromosome_action(chromosome, mask) for chromosome in population]
    )
    return int(actions[np.argmin(costs[actions])])


def evolve_drp(
    state: dict[str, Any],
    observation: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Advance the persistent DRP population by exactly one generation."""

    mask, costs = _drp_arrays(state, observation)
    population = np.asarray(state["population"], dtype=np.int64)
    actions = np.asarray(
        [_chromosome_action(chromosome, mask) for chromosome in population]
    )
    ranked = population[np.argsort(costs[actions])]
    survivors = ranked[: DRP_POPULATION_SIZE // 2].copy()
    children: list[np.ndarray] = []
    while len(survivors) + len(children) < DRP_POPULATION_SIZE:
        parent_indices = rng.integers(len(survivors), size=2)
        first, second = survivors[parent_indices]
        child = first.copy()
        if len(mask) > 1 and rng.random() < DRP_CROSSOVER_RATE:
            cut = int(rng.integers(1, len(mask)))
            prefix = first[:cut]
            child = np.concatenate((prefix, second[~np.isin(second, prefix)]))
        if len(mask) > 1 and rng.random() < DRP_MUTATION_RATE:
            left, right = rng.choice(len(mask), size=2, replace=False)
            child[left], child[right] = child[right], child[left]
        children.append(child)
    state["population"] = np.vstack((survivors, np.stack(children))).astype(np.int64)
    state["generation"] = int(state["generation"]) + 1


def shortest_feasible_routing(
    observation: dict[str, Any], placement_index: int
) -> np.ndarray:
    """Return one-hot shortest routes for DQN or DRP placement decisions."""

    placement_mask = np.asarray(observation["placement_mask"], dtype=bool)
    if (
        not 0 <= placement_index < len(placement_mask)
        or not placement_mask[placement_index]
    ):
        raise ValueError("placement_index is not feasible")
    path_mask = np.asarray(observation["path_mask"], dtype=bool)
    if "path_delays" in observation:
        delays = np.asarray(observation["path_delays"], dtype=np.float64)
    elif "path_features" in observation:
        delays = np.asarray(observation["path_features"], dtype=np.float64)[..., 0]
    else:
        raise KeyError("Observation requires path_delays or path_features")
    if delays.shape != path_mask.shape:
        raise ValueError("Path delays and path_mask must have the same shape")
    incoming = np.asarray(
        observation.get("incoming_edge_mask", path_mask.any(axis=(1, 2))),
        dtype=bool,
    )
    fractions = np.zeros((path_mask.shape[0], path_mask.shape[2]), dtype=np.float32)
    for edge_index in np.flatnonzero(incoming):
        valid = np.flatnonzero(path_mask[edge_index, placement_index])
        if not len(valid):
            raise ValueError(
                "A feasible placement requires a route for every incoming dependency"
            )
        chosen = int(valid[np.argmin(delays[edge_index, placement_index, valid])])
        fractions[edge_index, chosen] = 1.0
    return fractions


def _configuration_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_dqn_checkpoint(
    path: str | Path,
    state: dict[str, Any],
    config: dict[str, Any],
    *,
    episode: int,
    global_step: int,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "online": state["online"].state_dict(),
        "updates": int(state["updates"]),
        "episode": int(episode),
        "global_step": int(global_step),
        "configuration_hash": _configuration_hash(config),
    }
    torch.save(checkpoint, destination)


def load_dqn_checkpoint(
    path: str | Path, state: dict[str, Any], config: dict[str, Any]
) -> dict[str, int]:
    checkpoint = torch.load(
        Path(path), map_location=state["device"], weights_only=False
    )
    expected_hash = _configuration_hash(config)
    if checkpoint["configuration_hash"] != expected_hash:
        raise ValueError(
            f"Checkpoint configuration hash {checkpoint['configuration_hash']} "
            f"does not match {expected_hash}"
        )
    state["online"].load_state_dict(checkpoint["online"])
    state["target"].load_state_dict(checkpoint["online"])
    state["updates"] = int(checkpoint["updates"])
    return {
        "episode": int(checkpoint["episode"]),
        "global_step": int(checkpoint["global_step"]),
    }
