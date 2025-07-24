import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import torch.nn as nn
from typing import Dict

from env.constants import N_LINKS_MS, N_MS
from agent.gnn import EncoderFusion


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
            if key in ["node_feats", "link_feats", "service_feats", "ms_feats", "link_ms_feats", "placement_status"]:
                input_dim = int(np.prod(space.shape[1:]))  # e.g., (N, F) -> F
                self.encoders[key] = FeatureEncoder(input_dim, hidden_dim)

    def forward(self, obs_dict: Dict[str, torch.Tensor]):
        obs_dict = self.per_svc({k: v.clone() for k, v in obs_dict.items()})  # send deepcopied dict

        features = []
        for key, encoder in self.encoders.items():
            val: torch.Tensor = obs_dict[key]
            if key in ["node_feats", "link_feats", "service_feats", "ms_feats", "link_ms_feats", "placement_status"]:
                val = val.view(val.shape[0], -1, val.shape[-1]).float()  # (B, N, F)
                out = encoder(val)
            features.append(out)
        out = torch.cat(features, dim=-1)  # (B, total_hidden)
        return out

    def per_svc(self, obs_dict: Dict[str, torch.Tensor]):
        filter_svc = lambda feats, idx: feats.gather(
            dim=1, index=idx.unsqueeze(-1).expand(-1, -1, feats.size(-1)).long()
        )

        svc = obs_dict["svc_id"]  # (B, 1)
        ms_idx = svc * N_MS + torch.arange(N_MS, device=svc.device)
        link_ms_idx = svc * N_LINKS_MS + torch.arange(N_LINKS_MS, device=svc.device)

        obs_dict["service_feats"] = filter_svc(obs_dict["service_feats"], svc)
        obs_dict["ms_feats"] = filter_svc(obs_dict["ms_feats"], ms_idx)
        obs_dict["link_ms_feats"] = filter_svc(obs_dict["link_ms_feats"], link_ms_idx)
        obs_dict["placement_status"] = filter_svc(obs_dict["placement_status"], ms_idx)
        return obs_dict


class Actor(nn.Module):
    def __init__(self, obs_space: Dict, action_dim: int, hidden_dim=256, encoder="default"):
        super().__init__()
        if encoder == "gnn":
            self.encoder = EncoderFusion(obs_space, hidden_dim=hidden_dim)
            net_in_dim = hidden_dim * 2
        else:
            self.encoder = DictEncoder(obs_space, hidden_dim)
            net_in_dim = hidden_dim * 6  # (len(obs_space.spaces) - 1)

        self.net = nn.Sequential(
            nn.Linear(net_in_dim, hidden_dim),
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
        ms_feats: torch.Tensor = obs_dict["ms_feats"]  # (B, N_MS, F_MS)

        ms_idx: torch.Tensor = obs_dict["next_ms_idx"]  # (B, 1)
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
    def __init__(self, obs_space: Dict, action_dim: int, hidden_dim=256, encoder="default"):
        super().__init__()
        if encoder == "gnn":
            self.encoder = EncoderFusion(obs_space, hidden_dim=hidden_dim)
            net_in_dim = hidden_dim * 2+ action_dim
        else:
            self.encoder = DictEncoder(obs_space, hidden_dim)
            net_in_dim = hidden_dim * 6 + action_dim  # (len(obs_space.spaces) - 1)

        self.net = nn.Sequential(
            nn.Linear(net_in_dim, hidden_dim),
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
    from env.envMulti import obs_space, action_space

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
