import numpy as np
from typing import Dict

import torch
import torch.nn as nn

from env.constants import N_MS


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        h: torch.Tensor = self.encoder(x)
        # a feature is represented at the end with [1, hidden_dim]

        return h.mean(dim=1)


class DictEncoder(nn.Module):
    def __init__(self, obs_space: Dict, hidden_dim: int):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for key, space in obs_space.spaces.items():
            if key in ["next_service_idx", "next_ms_idx", "svc_id"]:
                input_dim = int(space.high.max() + 1)
                self.encoders[key] = nn.Embedding(input_dim, hidden_dim)
            elif key not in ["placement_status"]:
                input_dim = int(np.prod(space.shape[1:]))  # e.g., (N, F) -> F
                self.encoders[key] = FeatureEncoder(input_dim, hidden_dim)

    def forward(self, obs_dict: Dict):
        features = []
        for key, encoder in self.encoders.items():
            val: torch.Tensor = obs_dict[key]
            B = val.shape[0]
            if key in ["next_service_idx", "next_ms_idx", "svc_id"]:
                val = val.squeeze(-1).long()  # (B,)
                out = encoder(val)  # (B, hidden_dim)
                out = out.view(B, -1)  # (B, hidden_dim)
            elif key not in ["placement_status"]:
                val = val.view(B, -1, val.shape[-1]).float()  # (B, N, F)
                out = encoder(val)
            features.append(out)
        out = torch.cat(features, dim=-1)  # (B, total_hidden)
        return out


class Actor(nn.Module):
    def __init__(self, obs_space: Dict, action_dim: int, hidden_dim=256):
        super().__init__()
        self.encoder = DictEncoder(obs_space, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * (len(obs_space.spaces) - 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def diff(self, obs_dict: Dict):
        """Calculate the difference between node availability and microservice demand."""
        node_avail: torch.Tensor = obs_dict["node_feats"][:, :, :2]  # (B, N, 2)
        current_svc: torch.Tensor = obs_dict["svc_id"]  # (B, 1)
        ms_feats: torch.Tensor = obs_dict["ms_feats"]  # (B, N_MS * N_SRVS, F_MS)

        ms_idx: torch.Tensor = current_svc * N_MS + obs_dict["next_ms_idx"]  # (B, 1)
        ms_idx = ms_idx.squeeze(-1).long()  # (B,)
        batch_size = ms_feats.shape[0]
        ms_demand = ms_feats[torch.arange(batch_size), ms_idx, :2]  # (B, 2)

        diff = node_avail - ms_demand.unsqueeze(1)  # (B, N, 2)
        diff = diff.sum(dim=-1)  # (B, N)
        diff = (diff - diff.min(dim=1, keepdim=True).values) / (
            diff.max(dim=1, keepdim=True).values - diff.min(dim=1, keepdim=True).values + 1e-6
        )  # minmax normalize diff to [0, 1]
        return diff

    def forward(self, obs_dict: Dict, deterministic=False):
        state = self.encoder(obs_dict)
        x = self.net(state)

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        z = mean if deterministic else dist.rsample()
        action = torch.tanh(z)  # Keep within [-1, 1], SB3-style

        diff = self.diff(obs_dict)  # (B, N)
        action = action * (diff + 0.1)  # scale action based on node availability
        action = torch.clamp(action, -0.999, 0.999)

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_space: Dict, action_dim: int, hidden_dim=256):
        super().__init__()
        self.encoder = DictEncoder(obs_space, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim *(len(obs_space.spaces) - 1) + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_dict: Dict, action):
        state = self.encoder(obs_dict)
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        return out


if __name__ == "__main__":
    from env.env import obs_space, action_space

    obs_dict = obs_space.sample()
    print(obs_dict.keys())
    print(action_space.shape)

    print(action_space.sample())

    print([f.shape for f in obs_dict.values()])

    # Create instances of the networks
    hidden_dim = 256
    action_dim = action_space.shape[0]  # Example action_dim
    # Example input_dim

    actor = Actor(obs_space, action_dim, hidden_dim)
    critic = Critic(obs_space, action_dim, hidden_dim)

    # Example SINGLE OBS
    print("--" * 20)
    print("Single observation")
    print("--" * 20)
    obs_dict = {k: torch.tensor(v).unsqueeze(0) for k, v in obs_dict.items()}
    action, log_prob = actor(obs_dict)
    print(f"Action: {torch.argmax(action, dim=-1)}")
    print(f"Log prob: {log_prob}")

    # critc
    q1 = critic(obs_dict, action)
    print(f"Q1: {q1}")

    # here for batch testing: sample 3 obs
    print("--" * 20)
    print("Batch observation")
    print("--" * 20)
    obs_dict = {k: torch.tensor(v).unsqueeze(0) for k, v in obs_dict.items()}
    obs_dict = {k: torch.cat([v, v, v], dim=0) for k, v in obs_dict.items()}
    action, log_prob = actor(obs_dict)
    print(f"Action: {torch.argmax(action, dim=-1)}")
    print(f"Log prob: {log_prob}")
    # critc
    q1 = critic(obs_dict, action)
    print(f"Q1: {q1}")
