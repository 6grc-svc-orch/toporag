import numpy as np
from gymnasium.spaces import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
            if key in ["next_service_idx", "next_ms_idx"]:
                input_dim = int(space.high.max() + 1)
                self.encoders[key] = nn.Embedding(input_dim, hidden_dim)
            else:
                input_dim = int(np.prod(space.shape[1:]))  # e.g., (N, F) -> F
                self.encoders[key] = FeatureEncoder(input_dim, hidden_dim)

    def forward(self, obs_dict: Dict):
        features = []
        for key, encoder in self.encoders.items():
            val: torch.Tensor = obs_dict[key]
            B = val.shape[0]
            if key in ["next_service_idx", "next_ms_idx"]:
                val = val.squeeze(-1).long()  # (B,)
                out = encoder(val)  # (B, hidden_dim)
                out = out.view(B, -1)  # (B, hidden_dim)
            else:
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
            nn.Linear(hidden_dim * len(obs_space.spaces), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_dict: Dict):
        state = self.encoder(obs_dict)
        x = self.net(state)

        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)  # Keep within [-1, 1], SB3-style

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_space: Dict, action_dim: int, hidden_dim=256):
        super().__init__()
        self.encoder = DictEncoder(obs_space, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * len(obs_space.spaces) + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_dict: Dict, action):
        state = self.encoder(obs_dict)
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        return out


if __name__ == "__main__":
    from env import obs_space, action_space

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
