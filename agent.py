"""Conference RAG-DRL learning components.

Observations are plain dictionaries.  The policy requires ``node_features``
(``infrastructure_nodes`` is accepted as an alias), ``service_features``
(``service_nodes`` alias), ``service_adjacency``, and ``placement_mask``.
Optional ``node_mask`` and ``service_mask`` arrays mark padding; when omitted,
all supplied rows are treated as real.  Candidate row ``i`` corresponds to
``candidate_node_ids[i]`` in the simulator, while ``placement_mask[i]`` says
whether that candidate is feasible.  ``context``, ``path_features``,
``path_mask``, and ``incoming_edge_mask`` complete the joint action state.
Optional ``placement_prior_logits`` carries the RAG soft plan into sampling
and PPO replay.

Only neural networks are classes.  Sampling, GAE, PPO, behavior cloning,
first-order meta-learning, and checkpointing remain explicit functions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Dirichlet

GAT_HEADS = 8
GAT_HEAD_DIM = 64
GAT_OUTPUT_DIM = GAT_HEADS * GAT_HEAD_DIM


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class GraphAttentionLayer(nn.Module):
    """One eight-head graph-attention layer with 64 features per head."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.projection = nn.Parameter(torch.empty(GAT_HEADS, input_dim, GAT_HEAD_DIM))
        self.source_attention = nn.Parameter(torch.empty(GAT_HEADS, GAT_HEAD_DIM))
        self.target_attention = nn.Parameter(torch.empty(GAT_HEADS, GAT_HEAD_DIM))
        nn.init.xavier_uniform_(self.projection)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)
        self.activation = nn.ELU()
        self.attention_activation = nn.LeakyReLU(0.2)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("GAT features must have shape [batch, nodes, features]")
        if adjacency.shape != features.shape[:2] + (features.shape[1],):
            raise ValueError("GAT adjacency must have shape [batch, nodes, nodes]")
        if node_mask.shape != features.shape[:2]:
            raise ValueError("GAT node_mask must have shape [batch, nodes]")

        transformed = torch.einsum("bni,hio->bhno", features, self.projection)
        source = torch.einsum("bhnd,hd->bhn", transformed, self.source_attention)
        target = torch.einsum("bhnd,hd->bhn", transformed, self.target_attention)
        scores = self.attention_activation(source.unsqueeze(-1) + target.unsqueeze(-2))

        nodes = features.shape[1]
        self_edges = torch.eye(nodes, dtype=torch.bool, device=features.device)
        edges = adjacency.bool() | self_edges.unsqueeze(0)
        valid_nodes = node_mask.bool()
        edges &= valid_nodes.unsqueeze(-1) & valid_nodes.unsqueeze(-2)
        head_edges = edges.unsqueeze(1)
        scores = scores.masked_fill(~head_edges, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.where(head_edges, attention, torch.zeros_like(attention))
        attention /= attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        updated = torch.einsum("bhij,bhjd->bhid", attention, transformed)
        updated = updated.transpose(1, 2).reshape(
            features.shape[0], nodes, GAT_OUTPUT_DIM
        )
        return self.activation(updated) * node_mask.to(updated.dtype).unsqueeze(-1)


class DAGEncoder(nn.Module):
    """Paper-specified three-layer GAT followed by 512-D mean pooling."""

    output_dim = GAT_OUTPUT_DIM

    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                GraphAttentionLayer(input_dim),
                GraphAttentionLayer(GAT_OUTPUT_DIM),
                GraphAttentionLayer(GAT_OUTPUT_DIM),
            )
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_features.ndim != 3:
            raise ValueError(
                "DAGEncoder node_features must have shape [batch, nodes, features]"
            )
        if node_mask is None:
            node_mask = torch.ones(
                node_features.shape[:2],
                dtype=torch.bool,
                device=node_features.device,
            )
        if not torch.all(node_mask.bool().any(dim=1)):
            raise ValueError("Every DAG in a batch must contain at least one node")
        hidden = node_features
        for layer in self.layers:
            hidden = layer(hidden, adjacency, node_mask)
        return _masked_mean(hidden, node_mask)


def build_graph_encoder(config: dict[str, Any], scenario: dict[str, Any]) -> DAGEncoder:
    """Build the shared 512-D DAG encoder from the simulator schema."""

    rag = config["rag"]
    architecture = (
        int(rag["gat_layers"]),
        int(rag["gat_heads"]),
        int(rag["gat_head_dim"]),
        int(rag["embedding_dim"]),
    )
    if architecture != (3, GAT_HEADS, GAT_HEAD_DIM, GAT_OUTPUT_DIM):
        raise ValueError(
            "The conference encoder must be 3 GAT layers, 8 heads, 64 features "
            "per head, and a 512-D pooled embedding"
        )
    return DAGEncoder(int(scenario["service_feature_dim"]))


def graph_embedding(encoder: DAGEncoder, observation: dict[str, Any]) -> torch.Tensor:
    """Encode one service DAG as a batched 512-D tensor."""

    features = _required_array(observation, "service_features", "service_nodes")
    adjacency = _required_array(observation, "service_adjacency")
    mask = np.asarray(
        observation.get("service_mask", np.ones(features.shape[0], dtype=bool)),
        dtype=bool,
    )
    device = next(encoder.parameters()).device
    with torch.no_grad():
        return encoder(
            torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(adjacency, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0),
        )


class PPOActorCritic(nn.Module):
    """One actor/critic shared by all edge controllers."""

    def __init__(
        self,
        service_dim: int,
        node_dim: int,
        context_dim: int,
        path_dim: int,
    ):
        super().__init__()
        self.actor_dag = DAGEncoder(service_dim)
        self.critic_dag = DAGEncoder(service_dim)
        self.actor_node = nn.Linear(node_dim, GAT_OUTPUT_DIM)
        self.critic_node = nn.Linear(node_dim, GAT_OUTPUT_DIM)
        self.actor_state = nn.Linear(context_dim, GAT_OUTPUT_DIM)
        self.critic_state = nn.Linear(context_dim, GAT_OUTPUT_DIM)
        self.actor_context = nn.Linear(GAT_OUTPUT_DIM * 3, GAT_OUTPUT_DIM)
        self.actor_score = nn.Linear(GAT_OUTPUT_DIM, 1)
        self.path_encoder = nn.Linear(path_dim, GAT_OUTPUT_DIM)
        self.route_context = nn.Linear(GAT_OUTPUT_DIM, GAT_OUTPUT_DIM)
        self.route_score = nn.Linear(GAT_OUTPUT_DIM, 1)
        self.critic = nn.Sequential(
            nn.Linear(GAT_OUTPUT_DIM * 3, GAT_OUTPUT_DIM),
            nn.Tanh(),
            nn.Linear(GAT_OUTPUT_DIM, 1),
        )

    def actor_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.actor_dag,
            self.actor_node,
            self.actor_state,
            self.actor_context,
            self.actor_score,
            self.path_encoder,
            self.route_context,
            self.route_score,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def critic_parameters(self) -> list[nn.Parameter]:
        modules = (self.critic_dag, self.critic_node, self.critic_state, self.critic)
        return [parameter for module in modules for parameter in module.parameters()]

    def actor_encoding(
        self, observation: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dag = self.actor_dag(
            observation["service_features"],
            observation["service_adjacency"],
            observation["service_mask"],
        )
        node_states = torch.tanh(self.actor_node(observation["node_features"]))
        infrastructure = _masked_mean(node_states, observation["node_mask"])
        state = torch.tanh(self.actor_state(observation["context"]))
        context = torch.tanh(
            self.actor_context(torch.cat((dag, infrastructure, state), dim=-1))
        )
        return node_states, context

    def policy(
        self,
        observation: dict[str, torch.Tensor],
        placement_prior_logits: torch.Tensor | None = None,
        actor_encoding: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        node_states, context = (
            self.actor_encoding(observation)
            if actor_encoding is None
            else actor_encoding
        )
        logits = self.actor_score(
            torch.tanh(node_states + context.unsqueeze(1))
        ).squeeze(-1)
        prior = (
            observation["placement_prior_logits"]
            if placement_prior_logits is None
            else placement_prior_logits
        )
        if prior.shape != logits.shape:
            raise ValueError("placement_prior_logits shape must match placement logits")
        logits = logits + prior
        return logits.masked_fill(
            ~observation["placement_mask"].bool(), torch.finfo(logits.dtype).min
        )

    def value(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        dag = self.critic_dag(
            observation["service_features"],
            observation["service_adjacency"],
            observation["service_mask"],
        )
        node_states = torch.tanh(self.critic_node(observation["node_features"]))
        infrastructure = _masked_mean(node_states, observation["node_mask"])
        state = torch.tanh(self.critic_state(observation["context"]))
        return self.critic(torch.cat((dag, infrastructure, state), dim=-1)).squeeze(-1)

    def routing_concentrations(
        self,
        observation: dict[str, torch.Tensor],
        placement_indices: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return masked Dirichlet concentrations for the selected nodes."""

        if context is None:
            _, context = self.actor_encoding(observation)
        batch = torch.arange(
            placement_indices.shape[0], device=placement_indices.device
        )
        selected_paths = observation["path_features"][batch, :, placement_indices]
        encoded = torch.tanh(self.path_encoder(selected_paths))
        conditioned = torch.tanh(encoded + self.route_context(context)[:, None, None])
        concentrations = (
            nn.functional.softplus(self.route_score(conditioned).squeeze(-1)) + 1.0
        )
        selected_mask = observation["path_mask"][batch, :, placement_indices].bool()
        return concentrations.masked_fill(~selected_mask, 1.0)

    def forward(
        self, observation: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(observation), self.value(observation)


def _required_array(
    observation: dict[str, Any], canonical: str, alias: str | None = None
) -> np.ndarray:
    if canonical in observation:
        return np.asarray(observation[canonical])
    if alias is not None and alias in observation:
        return np.asarray(observation[alias])
    names = canonical if alias is None else f"{canonical} (or {alias})"
    raise KeyError(f"Observation is missing required field {names}")


def _normalise_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    node_features = _required_array(
        observation, "node_features", "infrastructure_nodes"
    )
    service_features = _required_array(observation, "service_features", "service_nodes")
    service_adjacency = _required_array(observation, "service_adjacency")
    placement_mask = _required_array(observation, "placement_mask").astype(bool)
    context = _required_array(observation, "context")
    if "path_features" in observation and "path_mask" in observation:
        path_features = np.asarray(observation["path_features"])
        path_mask = np.asarray(observation["path_mask"], dtype=bool)
    elif "path_features" not in observation and "path_mask" not in observation:
        path_features = np.zeros(
            (service_features.shape[0], node_features.shape[0], 3, 1),
            dtype=np.float32,
        )
        path_mask = np.zeros(path_features.shape[:3], dtype=bool)
    else:
        raise KeyError("path_features and path_mask must be provided together")
    node_mask = np.asarray(
        observation["node_mask"]
        if "node_mask" in observation
        else observation.get(
            "infrastructure_mask", np.ones(node_features.shape[-2], dtype=bool)
        ),
        dtype=bool,
    )
    service_mask = np.asarray(
        observation.get(
            "service_mask", np.ones(service_features.shape[-2], dtype=bool)
        ),
        dtype=bool,
    )
    incoming_edge_mask = np.asarray(
        observation.get("incoming_edge_mask", path_mask.any(axis=(1, 2))),
        dtype=bool,
    )
    placement_prior_logits = np.asarray(
        observation.get(
            "placement_prior_logits",
            np.zeros(node_features.shape[-2], dtype=np.float32),
        ),
        dtype=np.float32,
    )
    if node_features.ndim != 2 or service_features.ndim != 2:
        raise ValueError(
            "An unbatched observation must contain two-dimensional feature arrays"
        )
    if service_adjacency.shape != (
        service_features.shape[0],
        service_features.shape[0],
    ):
        raise ValueError("service_adjacency shape does not match service_features")
    if node_mask.shape != (node_features.shape[0],):
        raise ValueError("node_mask shape does not match node_features")
    if service_mask.shape != (service_features.shape[0],):
        raise ValueError("service_mask shape does not match service_features")
    if placement_mask.shape != (node_features.shape[0],):
        raise ValueError("placement_mask shape does not match candidate node rows")
    if placement_prior_logits.shape != placement_mask.shape:
        raise ValueError("placement_prior_logits shape does not match placement_mask")
    if not np.all(np.isfinite(placement_prior_logits[placement_mask])):
        raise ValueError(
            "placement_prior_logits must be finite for feasible candidates"
        )
    if context.ndim != 1:
        raise ValueError("context must be a one-dimensional state vector")
    expected_path_prefix = (service_features.shape[0], node_features.shape[0])
    if path_features.ndim != 4 or path_features.shape[:2] != expected_path_prefix:
        raise ValueError(
            "path_features must have shape [service slots, candidates, paths, features]"
        )
    if path_mask.shape != path_features.shape[:3]:
        raise ValueError("path_mask shape does not match path_features")
    if incoming_edge_mask.shape != (service_features.shape[0],):
        raise ValueError("incoming_edge_mask shape does not match service slots")
    active_edges = np.flatnonzero(incoming_edge_mask)
    feasible_actions = np.flatnonzero(placement_mask)
    if active_edges.size and np.any(
        ~path_mask[np.ix_(active_edges, feasible_actions)].any(axis=-1)
    ):
        raise ValueError(
            "Each feasible placement must offer a path for every incoming dependency"
        )
    return {
        "node_features": node_features,
        "node_mask": node_mask,
        "service_features": service_features,
        "service_adjacency": service_adjacency,
        "service_mask": service_mask,
        "context": context,
        "path_features": path_features,
        "path_mask": path_mask,
        "incoming_edge_mask": incoming_edge_mask,
        "placement_prior_logits": placement_prior_logits,
        "placement_mask": placement_mask,
    }


def compact_placement_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Copy only arrays consumed by placement-only BC and DQN."""

    arrays = _normalise_observation(observation)
    keys = (
        "node_features",
        "node_mask",
        "service_features",
        "service_adjacency",
        "service_mask",
        "context",
        "placement_prior_logits",
        "placement_mask",
    )
    return {key: arrays[key].copy() for key in keys}


def observation_tensors(
    observation: dict[str, Any],
    device: torch.device,
    *,
    add_batch: bool = True,
) -> dict[str, torch.Tensor]:
    """Validate one observation and convert its policy arrays to tensors."""

    arrays = _normalise_observation(observation)
    tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in arrays.items()
    }
    if add_batch:
        tensors = {key: value.unsqueeze(0) for key, value in tensors.items()}
    return tensors


def _stack_observations(
    observations: list[dict[str, Any]], device: torch.device
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("Cannot stack an empty observation list")
    arrays = [_normalise_observation(observation) for observation in observations]
    return {
        key: torch.as_tensor(
            np.stack([observation[key] for observation in arrays]),
            dtype=torch.float32,
            device=device,
        )
        for key in arrays[0]
    }


def _optimizers(
    model: PPOActorCritic,
    training_config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    actor = torch.optim.Adam(
        model.actor_parameters(), lr=float(training_config["actor_lr"])
    )
    critic = torch.optim.Adam(
        model.critic_parameters(), lr=float(training_config["critic_lr"])
    )
    return actor, critic


def build_ppo(
    example_observation: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    arrays = _normalise_observation(example_observation)
    model = PPOActorCritic(
        service_dim=int(arrays["service_features"].shape[-1]),
        node_dim=int(arrays["node_features"].shape[-1]),
        context_dim=int(arrays["context"].shape[-1]),
        path_dim=int(arrays["path_features"].shape[-1]),
    ).to(device)
    actor_optimizer, critic_optimizer = _optimizers(model, training_config)
    return {
        "model": model,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "device": device,
        "updates": 0,
    }


def sample_action(
    state: dict[str, Any],
    observation: dict[str, Any],
    *,
    placement_prior_logits: np.ndarray | None = None,
    deterministic: bool = False,
) -> dict[str, Any]:
    if not np.any(np.asarray(observation["placement_mask"], dtype=bool)):
        raise ValueError("Observation has no feasible placement candidate")
    if placement_prior_logits is not None:
        observation["placement_prior_logits"] = np.asarray(
            placement_prior_logits, dtype=np.float32
        ).copy()
    tensors = observation_tensors(observation, state["device"])
    model: PPOActorCritic = state["model"]
    with torch.no_grad():
        encoding = model.actor_encoding(tensors)
        logits = model.policy(tensors, actor_encoding=encoding)
        value = model.value(tensors)
        distribution = Categorical(logits=logits)
        action = (
            torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        )
        concentrations = model.routing_concentrations(
            tensors, action, context=encoding[1]
        )
        selected_mask = tensors["path_mask"][0, :, action.item()].bool()
        incoming = tensors["incoming_edge_mask"][0].bool()
        routing_fractions = torch.zeros_like(concentrations[0])
        route_log_prob = torch.zeros((), device=state["device"])
        for edge_index in torch.nonzero(incoming, as_tuple=False).flatten().tolist():
            valid = torch.nonzero(selected_mask[edge_index], as_tuple=False).flatten()
            if len(valid) == 0:
                raise ValueError(
                    "A feasible placement requires a route for every incoming dependency"
                )
            if len(valid) == 1:
                routing_fractions[edge_index, valid[0]] = 1.0
                continue
            route_distribution = Dirichlet(concentrations[0, edge_index, valid])
            route = (
                concentrations[0, edge_index, valid]
                / concentrations[0, edge_index, valid].sum()
                if deterministic
                else route_distribution.sample()
            )
            routing_fractions[edge_index, valid] = route
            route_log_prob += route_distribution.log_prob(route)
    return {
        "placement_index": int(action.item()),
        "routing_fractions": routing_fractions.cpu().numpy(),
        "log_prob": float((distribution.log_prob(action) + route_log_prob).item()),
        "value": float(value.item()),
    }


def estimate_value(state: dict[str, Any], observation: dict[str, Any]) -> float:
    """Return the critic bootstrap value for one nonterminal observation."""

    tensors = observation_tensors(observation, state["device"])
    with torch.no_grad():
        return float(state["model"].value(tensors).item())


def _routing_log_prob_and_entropy(
    concentrations: torch.Tensor,
    path_mask: torch.Tensor,
    incoming_edge_mask: torch.Tensor,
    routes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for batch_index in range(concentrations.shape[0]):
        log_probability = torch.zeros((), device=concentrations.device)
        entropy = torch.zeros((), device=concentrations.device)
        active_edges = torch.nonzero(
            incoming_edge_mask[batch_index].bool(), as_tuple=False
        ).flatten()
        for edge_index in active_edges.tolist():
            valid = torch.nonzero(
                path_mask[batch_index, edge_index].bool(), as_tuple=False
            ).flatten()
            if len(valid) == 0:
                raise ValueError(
                    "A rollout placement has no route for an incoming dependency"
                )
            if len(valid) == 1:
                continue
            distribution = Dirichlet(concentrations[batch_index, edge_index, valid])
            route = routes[batch_index, edge_index, valid].clamp_min(1e-8)
            route = route / route.sum()
            log_probability += distribution.log_prob(route)
            entropy += distribution.entropy()
        log_probabilities.append(log_probability)
        entropies.append(entropy)
    return torch.stack(log_probabilities), torch.stack(entropies)


def compute_gae(
    rollout: list[dict[str, Any]],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Add ``advantage`` and ``return`` to rollout entries in place."""

    if not rollout:
        raise ValueError("Cannot compute GAE for an empty rollout")
    advantage = 0.0
    next_value = float(last_value)
    for transition in reversed(rollout):
        slot_advance = int(transition["slot_advance"])
        if slot_advance < 0:
            raise ValueError("rollout slot_advance cannot be negative")
        step_gamma = gamma**slot_advance
        step_lambda = gae_lambda**slot_advance
        continuation = 0.0 if bool(transition["done"]) else 1.0
        delta = (
            float(transition["reward"])
            + step_gamma * next_value * continuation
            - float(transition["value"])
        )
        advantage = delta + step_gamma * step_lambda * continuation * advantage
        transition["advantage"] = advantage
        transition["return"] = advantage + float(transition["value"])
        next_value = float(transition["value"])


def _entropy_coefficient(
    config: dict[str, Any], episode: int, total_episodes: int
) -> float:
    if total_episodes < 1:
        raise ValueError("total_episodes must be positive")
    if not 0 <= episode <= total_episodes:
        raise ValueError("episode must be in [0, total_episodes]")
    progress = episode / total_episodes
    start = float(config["entropy_start"])
    return start + progress * (float(config["entropy_end"]) - start)


def ppo_update(
    state: dict[str, Any],
    rollout: list[dict[str, Any]],
    training_config: dict[str, Any],
    *,
    episode: int,
    total_episodes: int,
) -> dict[str, float]:
    if not rollout:
        raise ValueError("Cannot update PPO from an empty rollout")
    if any("advantage" not in transition for transition in rollout):
        raise ValueError("compute_gae must run before ppo_update")

    device = state["device"]
    model: PPOActorCritic = state["model"]
    actions = torch.as_tensor(
        [
            int(
                item["placement_index"] if "placement_index" in item else item["action"]
            )
            for item in rollout
        ],
        dtype=torch.long,
        device=device,
    )
    routes = torch.as_tensor(
        np.stack([item["routing_fractions"] for item in rollout]),
        dtype=torch.float32,
        device=device,
    )
    old_log_probs = torch.as_tensor(
        [float(item["old_log_prob"]) for item in rollout],
        dtype=torch.float32,
        device=device,
    )
    advantages = torch.as_tensor(
        [float(item["advantage"]) for item in rollout],
        dtype=torch.float32,
        device=device,
    )
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-8)
    returns = torch.as_tensor(
        [float(item["return"]) for item in rollout],
        dtype=torch.float32,
        device=device,
    )
    entropy_coefficient = _entropy_coefficient(training_config, episode, total_episodes)
    order = torch.arange(len(rollout), device=device)
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    entropies: list[float] = []

    for _ in range(int(training_config["update_epochs"])):
        order = order[torch.randperm(len(order), device=device)]
        for start in range(0, len(order), int(training_config["minibatch_size"])):
            selected = order[start : start + int(training_config["minibatch_size"])]
            indices = selected.detach().cpu().tolist()
            observations = _stack_observations(
                [rollout[index]["observation"] for index in indices], device
            )

            encoding = model.actor_encoding(observations)
            logits = model.policy(observations, actor_encoding=encoding)
            distribution = Categorical(logits=logits)
            selected_actions = actions[selected]
            placement_log_prob = distribution.log_prob(selected_actions)
            concentrations = model.routing_concentrations(
                observations, selected_actions, context=encoding[1]
            )
            batch = torch.arange(len(selected), device=device)
            selected_path_mask = observations["path_mask"][batch, :, selected_actions]
            route_log_prob, route_entropy = _routing_log_prob_and_entropy(
                concentrations,
                selected_path_mask,
                observations["incoming_edge_mask"],
                routes[selected],
            )
            log_prob = placement_log_prob + route_log_prob
            ratio = torch.exp(log_prob - old_log_probs[selected])
            unclipped = ratio * advantages[selected]
            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - float(training_config["clip_ratio"]),
                    1.0 + float(training_config["clip_ratio"]),
                )
                * advantages[selected]
            )
            entropy = (distribution.entropy() + route_entropy).mean()
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            actor_loss -= entropy_coefficient * entropy
            state["actor_optimizer"].zero_grad()
            actor_loss.backward()
            state["actor_optimizer"].step()

            values = model.value(observations)
            critic_loss = float(training_config["value_coef"]) * nn.functional.mse_loss(
                values, returns[selected]
            )
            state["critic_optimizer"].zero_grad()
            critic_loss.backward()
            state["critic_optimizer"].step()

            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))

    state["updates"] = int(state["updates"]) + 1
    return {
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "entropy": float(np.mean(entropies)),
        "entropy_coefficient": entropy_coefficient,
    }


def behavior_clone(
    state: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    epochs: int,
    minibatch_size: int,
) -> float:
    """Warm-start the policy from LLM placement examples."""

    if not examples:
        raise ValueError("Behavior cloning requires at least one example")
    if epochs < 1 or minibatch_size < 1:
        raise ValueError("Behavior-cloning epochs and minibatch_size must be positive")
    device = state["device"]
    model: PPOActorCritic = state["model"]
    losses: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        for start in range(0, len(order), minibatch_size):
            chosen = order[start : start + minibatch_size]
            batch = [examples[index] for index in chosen]
            observations = _stack_observations(
                [example["observation"] for example in batch], device
            )
            target_indices = []
            for example in batch:
                if "placement_index" in example:
                    target_indices.append(int(example["placement_index"]))
                elif "candidate_index" in example:
                    target_indices.append(int(example["candidate_index"]))
                else:
                    target_indices.append(int(example["action"]["placement_index"]))
            targets = torch.as_tensor(target_indices, dtype=torch.long, device=device)
            batch_indices = torch.arange(len(targets), device=device)
            if not torch.all(
                observations["placement_mask"][batch_indices, targets].bool()
            ):
                raise ValueError("Behavior-cloning target is infeasible")
            logits = model.policy(observations)
            loss = nn.functional.cross_entropy(logits, targets)
            state["actor_optimizer"].zero_grad()
            loss.backward()
            state["actor_optimizer"].step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def first_order_meta_update(
    state: dict[str, Any],
    perturbed_rollouts: list[list[dict[str, Any]]],
    training_config: dict[str, Any],
    *,
    inner_steps: int,
    meta_step_size: float,
    total_episodes: int,
) -> dict[str, float]:
    """Apply one first-order Reptile update over perturbed environments."""

    if not perturbed_rollouts:
        raise ValueError("Meta-learning requires at least one perturbed rollout")
    if inner_steps < 1:
        raise ValueError("inner_steps must be positive")
    if not 0.0 < meta_step_size <= 1.0:
        raise ValueError("meta_step_size must be in (0, 1]")

    base_model: PPOActorCritic = state["model"]
    adapted_parameters: list[list[torch.Tensor]] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    for rollout in perturbed_rollouts:
        task_model = copy.deepcopy(base_model)
        actor_optimizer, critic_optimizer = _optimizers(task_model, training_config)
        task_state = {
            "model": task_model,
            "actor_optimizer": actor_optimizer,
            "critic_optimizer": critic_optimizer,
            "device": state["device"],
            "updates": int(state["updates"]),
        }
        for _ in range(inner_steps):
            metrics = ppo_update(
                task_state,
                rollout,
                training_config,
                episode=0,
                total_episodes=total_episodes,
            )
        actor_losses.append(metrics["actor_loss"])
        critic_losses.append(metrics["critic_loss"])
        adapted_parameters.append(
            [parameter.detach().clone() for parameter in task_model.parameters()]
        )

    with torch.no_grad():
        for index, parameter in enumerate(base_model.parameters()):
            mean_adapted = torch.stack(
                [task[index] for task in adapted_parameters]
            ).mean(dim=0)
            parameter.add_(meta_step_size * (mean_adapted - parameter))
    return {
        "tasks": float(len(perturbed_rollouts)),
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
    }


def _configuration_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_checkpoint(
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
        "model": state["model"].state_dict(),
        "updates": int(state["updates"]),
        "episode": int(episode),
        "global_step": int(global_step),
        "configuration_hash": _configuration_hash(config),
    }
    torch.save(checkpoint, destination)


def load_checkpoint(
    path: str | Path,
    state: dict[str, Any],
    config: dict[str, Any],
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
    state["model"].load_state_dict(checkpoint["model"])
    state["updates"] = int(checkpoint["updates"])
    return {
        "episode": int(checkpoint["episode"]),
        "global_step": int(checkpoint["global_step"]),
    }
