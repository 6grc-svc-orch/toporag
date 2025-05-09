import time
from typing import List, TypedDict
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from numpy.typing import NDArray
from requests import init_plt, update_plt
from template import FogNetwork
from utils import read_json, write_json
import matplotlib.pyplot as plt


# num
N_NODES = 10
N_LINKS = (N_NODES * (N_NODES - 1)) // 2
MAX_SRVS = 50  # max number of req services
N_MS = 5
N_LINKS_MS = N_MS - 1

# features
F_NODE = 8  # (avail resources, total)*3 (CPU, MEM, STO), idle, max power
F_LINK = 4  # total, avail bandwidth, energy_per_bit, delay
F_SRV = 3  # deadline, lifetime, arrival_rate
F_MS = 7  # (avail resources, total)*3 (CPU, MEM, STO), computational_intensity
F_LINK_MS = 1  # data_size
REQ_FILENAME = "requests.json"


class FogEnvState(TypedDict):
    node_feats: NDArray[np.float64]
    link_feats: NDArray[np.float64]
    service_feats: NDArray[np.float64]
    ms_feats: NDArray[np.float64]
    link_ms_feats: NDArray[np.float64]
    placement_status: NDArray[np.float64]
    next_ms_idx: int
    next_service_idx: int


class FogEnv(gym.Env):
    def __init__(self, network: FogNetwork = None, vis_reqs=False, vis_placement=False):
        # TODO: have a list of already explored nodes
        # TODO: observation space is 0,1 should everything be normalized?
        # TODO: Edge case 0 requests is not handled
        # TODO: keep node placement as suggestion until all ms in a s are placed
        super().__init__()
        assert network is not None, "Network cannot be None"
        self.observation_space = Dict(
            {
                # network
                "node_feats": Box(0.0, 1.0, shape=(N_NODES, F_NODE)),
                "link_feats": Box(0.0, 1.0, shape=(N_LINKS, F_LINK)),
                # requests (all services)
                "service_feats": Box(0.0, 1.0, shape=(MAX_SRVS, F_SRV)),
                "ms_feats": Box(0.0, 1.0, shape=(N_MS * MAX_SRVS, F_MS)),
                "link_ms_feats": Box(
                    0.0, 1.0, shape=(N_LINKS_MS * MAX_SRVS, F_LINK_MS)
                ),
                # current
                "placement_status": Box(0, 1, shape=(N_MS * MAX_SRVS, N_NODES)),
                # idx
                "next_ms_idx": Discrete(N_MS * MAX_SRVS),
                "next_service_idx": Discrete(MAX_SRVS),
            }
        )
        self.action_space = Discrete(N_NODES, seed=646)
        self.network = network

        self.state: FogEnvState = {}
        self.services_requested = []

        self.vis_placement = vis_placement
        self.services_requested_len = []

        if self.vis_placement:
            plt.ion()  # Interactive mode on
            self.fig, self.ax = plt.subplots()
            # TODO: not the best way to init it
            self.im = self.ax.imshow(
                np.zeros((N_MS * MAX_SRVS, N_NODES)),
                cmap="gray",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            (self.break_line,) = self.ax.plot([-0.5, 9.5], [0, 0], "r-", linewidth=2)
            self.ax.set_xlabel("Nodes")
            self.ax.set_ylabel("All MS in the requested services")
            self.ax.set_xticks(range(N_NODES))

        self.reset()

    def reset(self, seed=646):
        #  NOTE: next_service_idx is not the service_id but just a counter over self.services_requested
        node_feats = np.array(
            [
                [
                    *n.available_resources,
                    *n.total_resources,
                    n.idle_power,
                    n.max_power,
                ]
                for n in self.network.fog_nodes
            ]
        )

        link_feats = np.array(
            [
                [
                    l.total_bandwidth,
                    l.available_bandwidth,
                    l.energy_per_bit,
                    l.delay,
                ]
                for l in self.network.links
            ]
        )
        self.state.update(
            {
                "node_feats": node_feats,
                "link_feats": link_feats,
                "service_feats": np.zeros((MAX_SRVS, F_SRV)),
                "ms_feats": np.zeros((N_MS * MAX_SRVS, F_MS)),
                "link_ms_feats": np.zeros((N_LINKS_MS * MAX_SRVS, F_LINK_MS)),
                "placement_status": np.zeros(
                    (N_MS * MAX_SRVS, N_NODES)
                ),  # see when it has to be reset
                "next_ms_idx": 0,
                "next_service_idx": 0,
            }
        )
        return self.state, {}

    def get_new_requests(self, verbose=False):
        """Get the current state of the environment."""

        all_reqs = read_json(REQ_FILENAME)

        if len(all_reqs) == 0:
            retry_after = 2
            print(f"No requests found retrying in {retry_after} seconds.")
            time.sleep(retry_after)
            return self.get_new_requests(verbose=verbose)

        all_reqs_flattened = [req for reqs in all_reqs for req in reqs]
        num_requests = len(all_reqs_flattened)

        # fill up the self.services_requested from self.network.services
        self.services_requested = []
        for i, req in enumerate(all_reqs_flattened):
            if i == MAX_SRVS:
                break
            self.services_requested.append(self.network.services[req])

        # empty the reqs file (so we dont fetch it again)
        write_json(REQ_FILENAME, [])

        assert (
            self.state["next_service_idx"] == 0
        ), f"`next_service_idx` != 0 instead is {self.state['next_service_idx']}"
        assert (
            self.state["next_ms_idx"] == 0
        ), f"`next_ms_idx` != 0 instead is {self.state['next_ms_idx']}"

        self.state["placement_status"] = np.zeros((N_MS * MAX_SRVS, N_NODES))
        self._update_state()

        self.services_requested_len.append(num_requests)
        if self.vis_placement:
            self.break_line.set_ydata(
                [
                    N_MS * len(self.services_requested),
                    N_MS * len(self.services_requested),
                ]
            )
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        return self.state, {
            "num_reqs": num_requests,
            "len_reqs": [len(reqs) for reqs in all_reqs],
            "dropped_reqs": num_requests - len(self.services_requested),
        }

    def _update_state(self):
        """Fill up the state based on the `self.services_requested`."""
        # service
        # s_tbd = self.services_requested[next_service_idx]
        self.state["service_feats"] = np.array(
            [
                s_tbd.deadline,
                s_tbd.lifetime,
                s_tbd.arrival_rate,
            ]
            for s_tbd in self.services_requested
        )

        # microservices
        self.state["ms_feats"] = np.array(
            [
                [*ms.resource_demand, ms.computational_intensity]
                for s_tbd in self.services_requested
                for ms in s_tbd.microservices
            ]
        )

        # links
        self.state["link_ms_feats"] = np.array(
            [
                [l.data_size]
                for s_tbd in self.services_requested
                for l in s_tbd.invocation_edges
            ]
        )

    def update_state(self, last_step=False):
        """Update the state of the environment and check if done."""

        done = False
        self.state["next_ms_idx"] += 1

        assert self.state["next_ms_idx"] <= N_MS * len(
            self.services_requested
        ), f"next_ms_idx > N_ms*len(self.services_requested), i.e {self.state['next_ms_idx']} > {N_MS*len(self.services_requested)}"

        if (
            self.state["next_ms_idx"] == N_MS * len(self.services_requested)
            or last_step
        ):
            # we move to the next service
            self.state["next_ms_idx"] = 0
            self.state["next_service_idx"] += 1

        if self.state["next_service_idx"] == len(self.services_requested) or last_step:
            # we have placed all the ms of all the services
            self.state["next_service_idx"] = 0
            done = True
            assert (
                self.state["next_ms_idx"] == 0
            ), f"All services (and ms) have been placed but `next_ms_idx` == {self.state['next_ms_idx']} != 0"

        # self._update_state()
        # NOTE: This is the beauty of the multiservice env POV

        return done

    def step(self, action, last_step=False):
        """action SO FAR is the selected node index to host the ms at `self.state["next_ms_idx"]`."""
        selected_node_idx = action
        done, truncated = False, False
        info = {}

        # Constraint 1: check if the ms_feats are bigger than the node_feats
        if np.any(
            self.state["node_feats"][selected_node_idx, :3]
            < self.state["ms_feats"][self.state["next_ms_idx"], :3]
        ):
            reward = -1
            if last_step:
                # we couldnt place the current ms and reached last step so we gotta move to the next service.
                _ = self.update_state(last_step=True)
            return self.state, reward, done, truncated, info

        # NOTE: All constraints need to be satisfied before updating the state
        # 1. update the placement status
        self.state["placement_status"][self.state["next_ms_idx"], selected_node_idx] = 1

        # 2. decrement the available resources of the node by the ms resources
        self.state["node_feats"][selected_node_idx, :3] -= self.state["ms_feats"][
            self.state["next_ms_idx"], :3
        ]

        reward = self._compute_reward()

        # we have placed the ms at the selected node so we now want to update the state
        done = self.update_state()

        if self.vis_placement:
            self.im.set_data(self.state["placement_status"])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return self.state, reward, done, truncated, info

    # TODO: dont care for now

    def get_state_info(self, return_=False):
        for key, value in self.state.items():
            print(f"{key}: {value.shape}")
        if return_:
            return self.state

    def render(self, mode="human"):
        """Render the environment."""
        pass

    def _compute_reward(self):
        """Reward: TODO: Random for now"""
        return np.random.rand()


if __name__ == "__main__":
    env = FogEnv()
    env.get_state_info()
