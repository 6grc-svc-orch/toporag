"""Semantic-guided MAPPO building blocks.

Only the neural networks are classes. Rollout collection, action sampling,
reward shaping, GAE, PPO updates, and checkpointing remain plain functions so
the experiment flow stays explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Dirichlet

from agent.gnn import MaskedGraphSAGE

OBSERVATION_ARRAY_KEYS = (
    "infrastructure_nodes",
    "infrastructure_adjacency",
    "infrastructure_mask",
    "service_nodes",
    "service_adjacency",
    "service_mask",
    "path_features",
    "path_mask",
    "incoming_edge_mask",
    "context",
    "placement_mask",
)


class SemanticActor(nn.Module):
    """Graph-aware decentralized actor with a recurrent belief state."""

    def __init__(
        self,
        infrastructure_dim: int,
        service_dim: int,
        path_dim: int,
        context_dim: int,
        hidden_dim: int,
        placement_dim: int,
    ):
        super().__init__()
        self.infrastructure_encoder = MaskedGraphSAGE(infrastructure_dim, hidden_dim)
        self.service_encoder = MaskedGraphSAGE(service_dim, hidden_dim)
        self.path_encoder = nn.Sequential(
            nn.Linear(path_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.belief = nn.GRUCell(128, hidden_dim)
        self.placement_head = nn.Linear(hidden_dim, placement_dim)
        self.routing_belief = nn.Linear(hidden_dim, hidden_dim)
        self.routing_score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        observation: dict[str, torch.Tensor],
        hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        infrastructure = self.infrastructure_encoder(
            observation["infrastructure_nodes"],
            observation["infrastructure_adjacency"],
            observation["infrastructure_mask"],
        )
        service = self.service_encoder(
            observation["service_nodes"],
            observation["service_adjacency"],
            observation["service_mask"],
        )
        encoded_paths = self.path_encoder(observation["path_features"])
        path_mask = observation["path_mask"].to(encoded_paths.dtype)
        path_mask = path_mask * observation["placement_mask"][:, None, :, None]
        expanded_path_mask = path_mask.unsqueeze(-1)
        path_count = expanded_path_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        paths = (encoded_paths * expanded_path_mask).sum(dim=(1, 2, 3)) / path_count
        context = self.context_encoder(observation["context"])
        fused = self.fusion(torch.cat((infrastructure, service, paths, context), dim=-1))
        belief = self.belief(fused, hidden)
        placement_logits = self.placement_head(belief)
        placement_logits = placement_logits.masked_fill(
            ~observation["placement_mask"].bool(), torch.finfo(placement_logits.dtype).min
        )
        return {
            "belief": belief,
            "placement_logits": placement_logits,
            "encoded_paths": encoded_paths,
        }

    def route_concentration(
        self,
        observation: dict[str, torch.Tensor],
        encoded_paths: torch.Tensor,
        belief: torch.Tensor,
        placement_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Condition each dependency-edge route distribution on placement."""

        batch_indices = torch.arange(encoded_paths.shape[0], device=encoded_paths.device)
        selected_paths = encoded_paths[batch_indices, :, placement_indices]
        conditioned = torch.relu(
            selected_paths + self.routing_belief(belief)[:, None, None, :]
        )
        concentrations = torch.nn.functional.softplus(
            self.routing_score(conditioned).squeeze(-1)
        ) + 1.0
        selected_mask = observation["path_mask"][batch_indices, :, placement_indices].bool()
        return concentrations.masked_fill(~selected_mask, 1.0)


class CentralCritic(nn.Module):
    """Centralized value function used only during training."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.network(global_state).squeeze(-1)


def _tensor(value: Any, device: torch.device, *, add_batch: bool) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0) if add_batch else tensor


def observation_to_tensors(
    observation: dict[str, Any], device: torch.device, *, add_batch: bool = True
) -> dict[str, torch.Tensor]:
    missing = [key for key in OBSERVATION_ARRAY_KEYS if key not in observation]
    if missing:
        raise KeyError(f"Observation is missing required fields: {missing}")
    return {key: _tensor(observation[key], device, add_batch=add_batch) for key in OBSERVATION_ARRAY_KEYS}


def stack_observations(observations: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("Cannot stack an empty observation list")
    return {
        key: torch.as_tensor(
            np.stack([np.asarray(observation[key]) for observation in observations]),
            dtype=torch.float32,
            device=device,
        )
        for key in OBSERVATION_ARRAY_KEYS
    }


def build_models(
    example_observation: dict[str, Any],
    global_state_dim: int,
    training_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    hidden_dim = int(training_config["gru_hidden"])
    actor = SemanticActor(
        infrastructure_dim=int(np.asarray(example_observation["infrastructure_nodes"]).shape[-1]),
        service_dim=int(np.asarray(example_observation["service_nodes"]).shape[-1]),
        path_dim=int(np.asarray(example_observation["path_features"]).shape[-1]),
        context_dim=int(np.asarray(example_observation["context"]).shape[-1]),
        hidden_dim=hidden_dim,
        placement_dim=int(np.asarray(example_observation["placement_mask"]).shape[-1]),
    ).to(device)
    critic = CentralCritic(global_state_dim).to(device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=float(training_config["actor_lr"]),
        betas=(float(training_config["adam_beta1"]), float(training_config["adam_beta2"])),
        eps=float(training_config["adam_eps"]),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=float(training_config["critic_lr"]),
        betas=(float(training_config["adam_beta1"]), float(training_config["adam_beta2"])),
        eps=float(training_config["adam_eps"]),
    )
    return {
        "actor": actor,
        "critic": critic,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "device": device,
        "ppo_updates": 0,
    }


def initial_hidden(num_controllers: int, hidden_dim: int, device: torch.device) -> dict[int, torch.Tensor]:
    return {controller_id: torch.zeros(1, hidden_dim, device=device) for controller_id in range(num_controllers)}


def trust_coefficient(predicted_utility: float, previous_return: float, alpha: float, cap: float) -> float:
    if alpha <= 0:
        raise ValueError("trust_alpha must be positive")
    if not 0 < cap < 1:
        raise ValueError("trust_cap must be between zero and one")
    return min(cap, math.exp(-alpha * abs(predicted_utility - previous_return)))


def blend_semantic_parameters(
    static_reward_weights: list[float],
    static_penalties: list[float],
    static_cooperation: float,
    plan: dict[str, Any],
    trust: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    reward_weights = (1.0 - trust) * np.asarray(static_reward_weights, dtype=np.float64)
    reward_weights += trust * np.asarray(plan["reward_weights"], dtype=np.float64)
    reward_weights /= reward_weights.sum()
    penalties = (1.0 - trust) * np.asarray(static_penalties, dtype=np.float64)
    penalties += trust * np.asarray(plan["penalty_weights"], dtype=np.float64)
    cooperation = (1.0 - trust) * float(static_cooperation)
    bias = float(plan["cooperation_bias"])
    if bias >= 0.0:
        semantic_cooperation = 1.0 / (1.0 + math.exp(-bias))
    else:
        exponential = math.exp(bias)
        semantic_cooperation = exponential / (1.0 + exponential)
    cooperation += trust * semantic_cooperation
    return reward_weights, penalties, cooperation


def semantic_placement_prior(plan: dict[str, Any] | None, local_node_ids: list[int]) -> np.ndarray:
    prior = np.zeros(len(local_node_ids), dtype=np.float32)
    if plan is None:
        return prior
    node_probabilities = plan["placement_prior"]
    for local_index, node_id in enumerate(local_node_ids):
        key = str(node_id)
        if key in node_probabilities:
            prior[local_index] = float(node_probabilities[key])
    total = float(prior.sum())
    if total > 0:
        prior /= total
    return prior


def _guided_logits(
    logits: torch.Tensor,
    prior: torch.Tensor | None,
    trust: torch.Tensor | float,
    activation_threshold: float,
) -> torch.Tensor:
    if prior is None:
        return logits
    if not 0.0 <= activation_threshold < 1.0:
        raise ValueError("activation_threshold must be in [0, 1)")
    valid = prior.sum(dim=-1, keepdim=True) > 0
    normalized = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    guided = logits + torch.as_tensor(trust, device=logits.device).reshape(-1, 1) * torch.log(normalized.clamp_min(1e-8))
    active = normalized >= activation_threshold
    active |= normalized == normalized.max(dim=-1, keepdim=True).values
    guided = guided.masked_fill(valid & ~active, torch.finfo(guided.dtype).min)
    return torch.where(valid, guided, logits)


def sample_action(
    models: dict[str, Any],
    observation: dict[str, Any],
    hidden: torch.Tensor,
    temperature: float,
    *,
    prior: np.ndarray | None,
    trust: float,
    activation_threshold: float,
    deterministic: bool,
) -> dict[str, Any]:
    if temperature <= 0.0:
        raise ValueError("Gumbel temperature must be positive")
    device = models["device"]
    actor: SemanticActor = models["actor"]
    tensors = observation_to_tensors(observation, device)
    with torch.no_grad():
        output = actor(tensors, hidden)
        prior_tensor = None if prior is None else _tensor(prior, device, add_batch=True)
        logits = _guided_logits(
            output["placement_logits"], prior_tensor, trust, activation_threshold
        )
        if deterministic:
            behavior_logits = logits
            placement_index = torch.argmax(logits, dim=-1)
        else:
            # Sampling a categorical distribution from temperature-scaled
            # logits is the hard Gumbel-max draw used for the discrete
            # environment action. Store the same temperature for PPO replay.
            behavior_logits = logits / float(temperature)
            placement_index = Categorical(logits=behavior_logits).sample()
        distribution = Categorical(logits=behavior_logits)

        routing_concentrations = actor.route_concentration(
            tensors,
            output["encoded_paths"],
            output["belief"],
            placement_index,
        )[0]
        incoming_mask = tensors["incoming_edge_mask"][0].bool()
        selected_path_mask = tensors["path_mask"][0, :, placement_index.item()].bool()
        route_counts = selected_path_mask.sum(dim=-1).to(torch.long)
        padded_route = torch.zeros_like(routing_concentrations)
        route_log_prob = torch.zeros((), device=device)
        route_entropy = torch.zeros((), device=device)
        for edge_index in torch.nonzero(incoming_mask, as_tuple=False).flatten().tolist():
            route_count = int(route_counts[edge_index].item())
            if route_count < 1:
                raise ValueError("A feasible placement requires a path for every incoming edge")
            concentrations = routing_concentrations[edge_index, :route_count]
            if route_count == 1:
                padded_route[edge_index, 0] = 1.0
                continue
            route_distribution = Dirichlet(concentrations)
            if deterministic:
                route = concentrations / concentrations.sum()
            else:
                route = route_distribution.sample()
            padded_route[edge_index, :route_count] = route
            route_log_prob += route_distribution.log_prob(route)
            route_entropy += route_distribution.entropy()
        placement_log_prob = distribution.log_prob(placement_index)
        entropy = distribution.entropy() + route_entropy
        probabilities = torch.softmax(logits, dim=-1)
    return {
        "placement_index": int(placement_index.item()),
        "routing_fractions": padded_route.cpu().numpy(),
        "route_count": route_counts.cpu().numpy(),
        "log_prob": float((placement_log_prob + route_log_prob).item()),
        "entropy": float(entropy.item()),
        "hidden": output["belief"].detach(),
        "placement_distribution": probabilities[0].cpu().numpy(),
    }


def critic_value(models: dict[str, Any], global_state: np.ndarray) -> float:
    critic: CentralCritic = models["critic"]
    state = _tensor(global_state, models["device"], add_batch=True)
    with torch.no_grad():
        return float(critic(state).item())


def add_rollout_entry(rollout: list[dict[str, Any]], **entry: Any) -> None:
    required = {
        "controller_id",
        "observation",
        "hidden",
        "global_state",
        "placement_index",
        "routing_fractions",
        "route_count",
        "old_log_prob",
        "value",
        "reward",
        "done",
        "prior",
        "trust",
        "temperature",
    }
    missing = sorted(required.difference(entry))
    if missing:
        raise KeyError(f"Rollout entry is missing fields: {missing}")
    rollout.append(entry)


def compute_gae(
    rollout: list[dict[str, Any]],
    last_values: dict[int, float],
    gamma: float,
    gae_lambda: float,
) -> None:
    by_controller: dict[int, list[int]] = {}
    for index, entry in enumerate(rollout):
        by_controller.setdefault(int(entry["controller_id"]), []).append(index)
    for controller_id, indices in by_controller.items():
        advantage = 0.0
        next_value = float(last_values[controller_id])
        for index in reversed(indices):
            entry = rollout[index]
            continuation = 0.0 if bool(entry["done"]) else 1.0
            delta = float(entry["reward"]) + gamma * next_value * continuation - float(entry["value"])
            advantage = delta + gamma * gae_lambda * continuation * advantage
            entry["advantage"] = advantage
            entry["return"] = advantage + float(entry["value"])
            next_value = float(entry["value"])


def _route_log_prob_and_entropy(
    concentrations: torch.Tensor,
    routes: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = []
    entropies = []
    for batch_index in range(concentrations.shape[0]):
        batch_log_prob = torch.zeros((), device=concentrations.device)
        batch_entropy = torch.zeros((), device=concentrations.device)
        for edge_index in range(concentrations.shape[1]):
            count = int(counts[batch_index, edge_index].item())
            if count == 0:
                continue
            if count == 1:
                continue
            distribution = Dirichlet(
                concentrations[batch_index, edge_index, :count]
            )
            route = routes[batch_index, edge_index, :count].clamp_min(1e-8)
            route = route / route.sum()
            batch_log_prob += distribution.log_prob(route)
            batch_entropy += distribution.entropy()
        log_probs.append(batch_log_prob)
        entropies.append(batch_entropy)
    return torch.stack(log_probs), torch.stack(entropies)


def _semantic_kl(probabilities: torch.Tensor, priors: torch.Tensor, trust: torch.Tensor) -> torch.Tensor:
    prior_total = priors.sum(dim=-1, keepdim=True)
    valid = prior_total.squeeze(-1) > 0
    normalized_priors = priors / prior_total.clamp_min(1e-8)
    kl = (
        probabilities
        * (torch.log(probabilities.clamp_min(1e-8)) - torch.log(normalized_priors.clamp_min(1e-8)))
    ).sum(dim=-1)
    return torch.where(valid, kl * trust, torch.zeros_like(kl))


def entropy_coefficient(
    ppo_updates: int, schedule_horizon_updates: int, start: float, end: float
) -> float:
    progress = min(1.0, ppo_updates / max(1, schedule_horizon_updates))
    return start + progress * (end - start)


def gumbel_temperature(episode: int, total_episodes: int, start: float, end: float) -> float:
    if start <= 0 or end <= 0:
        raise ValueError("Gumbel temperatures must be positive")
    progress = min(1.0, episode / max(1, total_episodes))
    return start * ((end / start) ** progress)


def update_mappo(models: dict[str, Any], rollout: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    if not rollout:
        raise ValueError("Cannot update MAPPO from an empty rollout")
    missing_advantages = [index for index, entry in enumerate(rollout) if "advantage" not in entry]
    if missing_advantages:
        raise ValueError("compute_gae must be called before update_mappo")

    device = models["device"]
    actor: SemanticActor = models["actor"]
    critic: CentralCritic = models["critic"]
    actor_optimizer = models["actor_optimizer"]
    critic_optimizer = models["critic_optimizer"]
    advantages = torch.as_tensor([entry["advantage"] for entry in rollout], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    returns = torch.as_tensor([entry["return"] for entry in rollout], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor([entry["old_log_prob"] for entry in rollout], dtype=torch.float32, device=device)
    placement_actions = torch.as_tensor(
        [entry["placement_index"] for entry in rollout], dtype=torch.long, device=device
    )
    routes = torch.as_tensor(
        np.stack([entry["routing_fractions"] for entry in rollout]), dtype=torch.float32, device=device
    )
    route_counts = torch.as_tensor(
        np.stack([entry["route_count"] for entry in rollout]),
        dtype=torch.long,
        device=device,
    )
    hidden = torch.cat([entry["hidden"].to(device) for entry in rollout], dim=0)
    global_states = torch.as_tensor(
        np.stack([entry["global_state"] for entry in rollout]), dtype=torch.float32, device=device
    )
    priors = torch.as_tensor(np.stack([entry["prior"] for entry in rollout]), dtype=torch.float32, device=device)
    trusts = torch.as_tensor([entry["trust"] for entry in rollout], dtype=torch.float32, device=device)
    temperatures = torch.as_tensor(
        [entry["temperature"] for entry in rollout], dtype=torch.float32, device=device
    )
    if torch.any(temperatures <= 0.0):
        raise ValueError("Rollout contains a non-positive Gumbel temperature")

    batch_size = int(config["minibatch_size"])
    indices = np.arange(len(rollout))
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    kl_values: list[float] = []
    entropy_values: list[float] = []
    schedule_horizon_updates = int(config["max_episodes"])
    entropy_coef = entropy_coefficient(
        int(models["ppo_updates"]),
        schedule_horizon_updates,
        float(config["entropy_start"]),
        float(config["entropy_end"]),
    )
    semantic_decay = math.exp(
        -float(models["ppo_updates"]) / float(config["kl_decay_steps"])
    )

    for _ in range(int(config["update_epochs"])):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected_np = indices[start : start + batch_size]
            selected = torch.as_tensor(selected_np, dtype=torch.long, device=device)
            observations = stack_observations([rollout[i]["observation"] for i in selected_np], device)
            output = actor(observations, hidden[selected])
            guided_logits = _guided_logits(
                output["placement_logits"],
                priors[selected],
                trusts[selected],
                float(config["activation_threshold"]),
            )
            guided_logits = guided_logits / temperatures[selected, None]
            placement_distribution = Categorical(logits=guided_logits)
            placement_log_prob = placement_distribution.log_prob(placement_actions[selected])
            routing_concentrations = actor.route_concentration(
                observations,
                output["encoded_paths"],
                output["belief"],
                placement_actions[selected],
            )
            route_log_prob, route_entropy = _route_log_prob_and_entropy(
                routing_concentrations, routes[selected], route_counts[selected]
            )
            new_log_prob = placement_log_prob + route_log_prob
            ratio = torch.exp(new_log_prob - old_log_probs[selected])
            unclipped = ratio * advantages[selected]
            clipped = torch.clamp(
                ratio, 1.0 - float(config["clip_ratio"]), 1.0 + float(config["clip_ratio"])
            ) * advantages[selected]
            policy_gain = torch.minimum(unclipped, clipped).mean()
            probabilities = torch.softmax(guided_logits, dim=-1)
            semantic_kl = _semantic_kl(probabilities, priors[selected], trusts[selected]).mean()
            entropy = (placement_distribution.entropy() + route_entropy).mean()
            actor_loss = -policy_gain + float(config["kl_coef"]) * semantic_decay * semantic_kl - entropy_coef * entropy

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), float(config["max_grad_norm"]))
            actor_optimizer.step()

            values = critic(global_states[selected])
            critic_loss = float(config["value_coef"]) * torch.nn.functional.mse_loss(values, returns[selected])
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), float(config["max_grad_norm"]))
            critic_optimizer.step()

            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            kl_values.append(float(semantic_kl.detach().cpu()))
            entropy_values.append(float(entropy.detach().cpu()))
    models["ppo_updates"] = int(models["ppo_updates"]) + 1

    return {
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "semantic_kl": float(np.mean(kl_values)),
        "entropy": float(np.mean(entropy_values)),
        "entropy_coefficient": entropy_coef,
    }


def configuration_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def save_checkpoint(
    path: Path,
    models: dict[str, Any],
    config: dict[str, Any],
    episode: int,
    global_step: int,
    planner_version: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    independent = "controller_models" in models
    state: dict[str, Any] = {
        "model_kind": "independent" if independent else "shared",
        "episode": int(episode),
        "global_step": int(global_step),
        "configuration_hash": configuration_hash(config),
        "planner_version": planner_version,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if independent:
        state["controller_models"] = {
            int(controller_id): {
                "actor": controller_model["actor"].state_dict(),
                "critic": controller_model["critic"].state_dict(),
                "actor_optimizer": controller_model["actor_optimizer"].state_dict(),
                "critic_optimizer": controller_model["critic_optimizer"].state_dict(),
                "ppo_updates": int(controller_model["ppo_updates"]),
            }
            for controller_id, controller_model in models["controller_models"].items()
        }
    else:
        state.update(
            {
                "actor": models["actor"].state_dict(),
                "critic": models["critic"].state_dict(),
                "actor_optimizer": models["actor_optimizer"].state_dict(),
                "critic_optimizer": models["critic_optimizer"].state_dict(),
                "ppo_updates": int(models["ppo_updates"]),
            }
        )
    if torch.cuda.is_available():
        state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(path: Path, models: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    independent = "controller_models" in models
    checkpoint = torch.load(path, map_location=models["device"], weights_only=False)
    expected_hash = configuration_hash(config)
    if checkpoint["configuration_hash"] != expected_hash:
        raise ValueError(
            f"Checkpoint config hash {checkpoint['configuration_hash']} does not match {expected_hash}"
        )
    expected_kind = "independent" if independent else "shared"
    if checkpoint["model_kind"] != expected_kind:
        raise ValueError(
            f"Checkpoint model kind {checkpoint['model_kind']!r} does not match {expected_kind!r}"
        )
    if independent:
        saved_models = checkpoint["controller_models"]
        if set(saved_models) != set(models["controller_models"]):
            raise ValueError("Checkpoint controller IDs do not match the current scenario")
        for controller_id, controller_model in models["controller_models"].items():
            saved = saved_models[controller_id]
            controller_model["actor"].load_state_dict(saved["actor"])
            controller_model["critic"].load_state_dict(saved["critic"])
            controller_model["actor_optimizer"].load_state_dict(saved["actor_optimizer"])
            controller_model["critic_optimizer"].load_state_dict(saved["critic_optimizer"])
            controller_model["ppo_updates"] = int(saved["ppo_updates"])
    else:
        models["actor"].load_state_dict(checkpoint["actor"])
        models["critic"].load_state_dict(checkpoint["critic"])
        models["actor_optimizer"].load_state_dict(checkpoint["actor_optimizer"])
        models["critic_optimizer"].load_state_dict(checkpoint["critic_optimizer"])
        models["ppo_updates"] = int(checkpoint["ppo_updates"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
    return {
        "episode": int(checkpoint["episode"]),
        "global_step": int(checkpoint["global_step"]),
        "planner_version": checkpoint["planner_version"],
    }
