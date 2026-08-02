"""Small graph encoders used by the paper-aligned policies.

The simulator emits padded node-feature matrices, adjacency matrices, and
masks.  Keeping the aggregation here avoids a dependency on a particular
graph batching framework and makes the same encoder usable by every method.
"""

from __future__ import annotations

import torch
from torch import nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool ``values`` over nodes selected by ``mask``."""
    weights = mask.to(values.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / count


def mean_neighbours(features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Return mean neighbouring features for a padded graph batch."""
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.bmm(adjacency, features) / degree


class MaskedGraphSAGE(nn.Module):
    """Two-layer mean-aggregation GraphSAGE with masked graph pooling."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.self_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)]
        )
        self.neighbour_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)])
        self.activation = nn.ReLU()

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = features
        node_mask = mask.to(features.dtype).unsqueeze(-1)
        for self_layer, neighbour_layer, norm in zip(
            self.self_layers, self.neighbour_layers, self.norms
        ):
            neighbours = mean_neighbours(hidden, adjacency)
            hidden = self.activation(norm(self_layer(hidden) + neighbour_layer(neighbours)))
            hidden = hidden * node_mask
        return masked_mean(hidden, mask)
