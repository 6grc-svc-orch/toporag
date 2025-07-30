from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.data import Data, Batch

from env.constants import N_LINKS_MS, N_MS


def per_svc(obs_dict: Dict[str, torch.Tensor]):
    filter_svc = lambda feats, idx: feats.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, feats.size(-1)).long())

    svc = obs_dict["svc_id"]  # (B, 1)
    ms_idx = svc * N_MS + torch.arange(N_MS, device=svc.device)
    link_ms_idx = svc * N_LINKS_MS + torch.arange(N_LINKS_MS, device=svc.device)

    obs_dict["service_feats"] = filter_svc(obs_dict["service_feats"], svc)
    obs_dict["ms_feats"] = filter_svc(obs_dict["ms_feats"], ms_idx)
    obs_dict["link_ms_feats"] = filter_svc(obs_dict["link_ms_feats"], link_ms_idx)
    obs_dict["placement_status"] = filter_svc(obs_dict["placement_status"], ms_idx)
    return obs_dict


class DictExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        h: torch.Tensor = self.encoder(x)
        return h.mean(dim=-2)


class DictEncoder(nn.Module):
    def __init__(self, obs_space: Dict, hidden_dim: int):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for key, space in obs_space.spaces.items():
            if key in ["node_feats", "link_feats", "service_feats", "ms_feats", "link_ms_feats", "placement_status"]:
                input_dim = int(np.prod(space.shape[1:]))  # e.g., (N, F) -> F
                self.encoders[key] = DictExtractor(input_dim, hidden_dim)

    def forward(self, obs_batch: Dict[str, torch.Tensor]):
        obs_batch = per_svc({k: v.clone() for k, v in obs_batch.items()})  # send deepcopied dict

        features = []
        for key, encoder in self.encoders.items():
            val: torch.Tensor = obs_batch[key]
            if key in ["node_feats", "link_feats", "service_feats", "ms_feats", "link_ms_feats", "placement_status"]:
                val = val.view(val.shape[0], -1, val.shape[-1]).float()  # (B, N, F)
                out = encoder(val)
            features.append(out)
        out = torch.cat(features, dim=-1)  # (B, total_hidden)
        return out


class GNNExtractor(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, node_dim * hidden_dim),
        )

        # NNConv will do:  x_i' = ∑_j  W_e(i,j) x_j: W_e(i,j) is produced by self.edge_mlp(edge_attr_ij).
        self.conv = NNConv(in_channels=node_dim, out_channels=hidden_dim, nn=self.edge_mlp, aggr="mean")

        self.act = nn.ReLU()

    def forward(self, node_feats, edge_index, edge_attr, batch_size=1, pp=None):
        node_feats = self.conv.forward(node_feats, edge_index, edge_attr)
        node_feats = self.act(node_feats)
        node_feats = node_feats.view(batch_size, -1, node_feats.shape[-1])  # [B, N, hidden_dim]
        return node_feats.mean(dim=-2)  # [B, hidden_dim]


class GNNEncoder(nn.Module):
    def __init__(self, obs_space: dict, hidden_dim: int):
        super().__init__()
        node_cfg = {
            "node_dim": obs_space["node_feats"].shape[1],
            "edge_dim": obs_space["link_feats"].shape[1],
        }
        ms_cfg = {
            "node_dim": obs_space["ms_feats"].shape[1],
            "edge_dim": obs_space["link_ms_feats"].shape[1],
        }
        self.node_encoder = GNNExtractor(**node_cfg, hidden_dim=hidden_dim)
        self.ms_encoder = GNNExtractor(**ms_cfg, hidden_dim=hidden_dim)

        self.edge_index = torch.combinations(torch.arange(obs_space["node_feats"].shape[0]), r=2).t()
        self.edge_ms_index = torch.tensor([[i, i + 1] for i in range(N_MS - 1)], dtype=torch.long).t()

    def forward_single(self, obs_batch: dict[str, torch.Tensor]):
        node_feats = obs_batch["node_feats"].squeeze(0)  # [4, 4]
        edge_feats = obs_batch["link_feats"].squeeze(0)  # [45, 3]
        ms_feats = obs_batch["ms_feats"].squeeze(0)  # [N_MS, 2]
        link_ms_feats = obs_batch["link_ms_feats"].squeeze(0)  # [N_MS-1, 1]

        out_node = self.node_encoder(
            node_feats=node_feats,
            edge_index=self.edge_index.to(node_feats.device),
            edge_attr=edge_feats,
        )
        out_ms = self.ms_encoder(
            node_feats=ms_feats, edge_index=self.edge_ms_index.to(ms_feats.device), edge_attr=link_ms_feats
        )

        return torch.cat([out_node, out_ms], dim=-1)  # [1, hidden_dim * 2]

    def forward(self, obs_batch: dict[str, torch.Tensor]):
        obs_batch = per_svc({k: v.clone() for k, v in obs_batch.items()})  # send deepcopied dict

        batch_size = len(obs_batch["node_feats"])
        if batch_size == 1:
            return self.forward_single(obs_batch)

        batch_node, batch_ms = self.create_data_objects(obs_batch)

        # batch_node.x.shape # [B*N, node_feat_dim]
        out_node = self.node_encoder(
            node_feats=batch_node.x,
            edge_index=batch_node.edge_index,
            edge_attr=batch_node.edge_attr,
            batch_size=batch_size,
        )
        out_ms = self.ms_encoder(
            node_feats=batch_ms.x, edge_index=batch_ms.edge_index, edge_attr=batch_ms.edge_attr, batch_size=batch_size
        )

        return torch.cat([out_node, out_ms], dim=-1)  # [B, hidden_dim * 2]

    def create_data_objects(self, obs_batch):
        data_list_nodes = []
        data_list_ms = []

        for i in range(len(obs_batch["node_feats"])):
            node_feats = obs_batch["node_feats"][i]  # [4, node_feat_dim]
            edge_feats = obs_batch["link_feats"][i]  # [num_edges, edge_feat_dim]
            ms_feats = obs_batch["ms_feats"][i]  # [num_ms_nodes, ms_feat_dim]
            link_ms_feats = obs_batch["link_ms_feats"][i]  # [num_ms_edges, ms_edge_feat_dim]

            data_node = Data(
                x=node_feats,
                edge_index=self.edge_index.to(node_feats.device),  # edge_index should be the same for all node graphs
                edge_attr=edge_feats,
            )

            data_ms = Data(
                x=ms_feats,
                edge_index=self.edge_ms_index.to(ms_feats.device),  # fixed layout
                edge_attr=link_ms_feats,
            )

            data_list_nodes.append(data_node)
            data_list_ms.append(data_ms)

        return Batch.from_data_list(data_list_nodes), Batch.from_data_list(data_list_ms)


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from env.envMulti import obs_space

    from env.envMulti import FogEnvMulti as FogEnv
    from template import loadFogNetwork
    from pathlib import Path
    from train_agent import build_run_name, parse_args

    encoder = GNNEncoder(obs_space, hidden_dim=32).to("cuda")
    args = parse_args()

    print(f"Arguments: {args}")
    print("==" * 20)

    run_name = build_run_name(args.name)
    CONTEXT_LOG_PATH = Path(f"logs/context-{run_name}.json")

    print(f"Training SAC agent...  Run name = {run_name}")
    print(f"Context log path: {CONTEXT_LOG_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = loadFogNetwork("pkl/network-connected-svcs.pkl")
    env = FogEnv(run_name=run_name, network=network, vis_placement=False)
    _ = env.reset()
    obs = env.get_state(0)
    obs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in obs.items()}
    for k, v in obs.items():
        print(v.dtype)
    out = encoder.forward(obs)
    print(out.shape)  # torch.Size([32])
