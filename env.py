import time
from typing import List, TypedDict
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from numpy.typing import NDArray
from template import FogNetwork, Service
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
F_MS = 4  # (avail resources)*3 (CPU, MEM, STO), computational_intensity
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
        # TODO: observation space is 0,1 should everything be normalized? [not yet used]
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
                "next_ms_idx": Discrete(N_MS),
                "next_service_idx": Discrete(MAX_SRVS),
            }
        )
        self.action_space = Discrete(N_NODES, seed=646)
        self.network = network

        self.state: FogEnvState = {}
        # overwritten on every load_new_requests
        self.services_requested: List[Service] = []
        self._all_services_requested: List[Service] = []  # not overwritten

        # only increments not set back to 0 # i wish i can remove this
        self._all_services_requested_idx = 0

        self.vis_placement = vis_placement
        self.services_requested_len = []
        # for service lifetime its idx is self._all_services_requested_idx
        self._service_start_times: Dict[int, float] = {}

        if self.vis_placement:
            self._render_placement()

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

    def fetch_requests(self, verbose=False):
        """Get the current state of the environment."""

        all_reqs = read_json(REQ_FILENAME)

        if len(all_reqs) == 0:
            return self.state, {
                "num_reqs": 0,
                "len_reqs": [],
                "dropped_reqs": 0,
            }

        all_reqs_flattened = [req for reqs in all_reqs for req in reqs]
        num_requests = len(all_reqs_flattened)

        # fill up the self.services_requested from self.network.services
        self.services_requested = []
        for i, req in enumerate(all_reqs_flattened):
            if i == MAX_SRVS:
                break
            self.services_requested.append(self.network.services[req])

        # extend or append? (append would have kept track but needs to be flattened to be accessed)
        self._all_services_requested.extend(self.services_requested)

        # empty the reqs file (so we dont fetch it again)
        write_json(REQ_FILENAME, [])

        assert (
            self.state["next_service_idx"] == 0
        ), f"`next_service_idx` != 0 instead is {self.state['next_service_idx']}"
        assert (
            self.state["next_ms_idx"] == 0
        ), f"`next_ms_idx` != 0 instead is {self.state['next_ms_idx']}"

        # AM I GOING CRAZY? WHAT IS THIS? I SHOULD NEVER ZERO OUT PLACEMENT STATUS
        # self.state["placement_status"] = np.zeros((N_MS * MAX_SRVS, N_NODES))
        self._populate_observation()

        self.services_requested_len.append(num_requests)

        return self.state, {
            "num_reqs": num_requests,
            "len_reqs": [len(reqs) for reqs in all_reqs],
            "dropped_reqs": num_requests - len(self.services_requested),
        }

    def step(self, action, last_step=False):
        """Perform one environment step by placing a microservice on a selected node (and expire a service if lifetime is over)."""
        selected_node_idx = action
        info = {}
        done = False
        truncated = False  # always False so not used

        # # 1) Expiration
        _ = self._expire_services()

        # 2) DRL placement constraints
        if not self._check_resource_constraints(selected_node_idx):
            # invalid placement
            reward = -1
            if last_step:
                self._advance_state(last_step=True)
            return self.state, reward, done, False, info

        # 3) valid: place it, maybe record start time if last ms
        self._place_microservice(selected_node_idx)

        # 4) Reward & advance
        reward = self._compute_reward()
        done = self._advance_state(last_step)

        # 5) Visualization
        if self.vis_placement:
            self._update_placement_plot()

        return self.state, reward, done, False, info

    def _populate_observation(self):
        """Fill up the state based on the `self.services_requested`."""
        for s_tbd in self.services_requested:
            # service
            self.state["service_feats"][s_tbd.id] = np.array(
                [
                    s_tbd.deadline,
                    s_tbd.lifetime,
                    s_tbd.arrival_rate,
                ]
            )

            # microservice
            for i, ms in enumerate(s_tbd.microservices):
                self.state["ms_feats"][s_tbd.id * N_MS + i] = np.array(
                    [
                        *ms.resource_demand,
                        ms.computational_intensity,
                    ]
                )

            # link
            for i, link in enumerate(s_tbd.invocation_edges):
                self.state["link_ms_feats"][s_tbd.id * N_LINKS_MS + i] = np.array(
                    [link.data_size]
                )

    def _check_resource_constraints(self, node_idx: int) -> bool:
        ms_demand = self.state["ms_feats"][self.state["next_ms_idx"], :3]
        node_avail = self.state["node_feats"][node_idx, :3]
        return np.all(node_avail >= ms_demand)

    def _place_microservice(self, node_idx: int):
        service_idx = self.state["next_service_idx"]
        current_service = self.services_requested[service_idx]
        ms_idx = self.state["next_ms_idx"]
        global_ms_idx = current_service.id * N_MS + ms_idx

        # Update placement status
        self.state["placement_status"][global_ms_idx, node_idx] += 1

        # Deduct used resources
        self.state["node_feats"][node_idx, :3] -= self.state["ms_feats"][ms_idx, :3]

        # If this was the final MS of a service, record its start time
        if ms_idx == N_MS - 1:
            self._service_start_times[self._all_services_requested_idx] = time.time()

    def _advance_state(self, last_step=False):
        """Update the state of the environment and check if done."""

        self.state["next_ms_idx"] += 1

        assert (
            self.state["next_ms_idx"] <= N_MS
        ), f"next_ms_idx > N_ms, i.e {self.state['next_ms_idx']} > {N_MS}"

        if self.state["next_ms_idx"] == N_MS or last_step:
            # we move to the next service
            self.state["next_ms_idx"] = 0
            self.state["next_service_idx"] += 1
            self._all_services_requested_idx += 1

        if self.state["next_service_idx"] == len(self.services_requested) or last_step:
            # we have placed all the ms of all the services
            self.state["next_service_idx"] = 0
            assert (
                self.state["next_ms_idx"] == 0
            ), f"All services (and ms) have been placed but `next_ms_idx` == {self.state['next_ms_idx']} != 0"
            return True

        return False

    def _update_placement_plot(self):
        self.im.set_data(self.state["placement_status"])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _expire_services(self):
        """Expire any services whose lifetime has passed."""
        now = time.time()
        expired_services_ids = set()
        reclaimed_nodes = set()

        for idx, start_time in list(self._service_start_times.items()):
            if start_time is None:
                continue

            svc: Service = self._all_services_requested[idx]
            if (now - start_time) > svc.lifetime:
                # 1. find all microservice rows for this service
                base = svc.id * N_MS
                rows = range(base, base + N_MS)

                # 2. for each placed MS, reclaim resources
                for row in rows:
                    node_idxs = np.where(self.state["placement_status"][row] >= 1)[0]

                    # Find all 1's in the placement_status matrix
                    if len(node_idxs) == 0 and row == base:
                        print(
                            f"\033[91mWarning: Service {svc.id} not placed on any node!\033[0m"
                        )

                    for node in node_idxs:
                        # MS demand vector
                        demand = self.state["ms_feats"][row, :3]
                        self.state["node_feats"][node, :3] += demand
                        self.state["placement_status"][row, node] -= 1

                        reclaimed_nodes.add(int(node))
                        expired_services_ids.add(svc.id)

                assert np.all(
                    self.state["placement_status"] >= 0
                ), f"Placement status has negative values"

                # 4. cleanup
                self._service_start_times[idx] = None

        if expired_services_ids:
            print(f"Expired services: {expired_services_ids}")
            print(f"Reclaimed nodes: {list(reclaimed_nodes)}")

        return expired_services_ids, reclaimed_nodes

    def _render_placement(self):
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots(figsize=(5, 10))
        self.im = self.ax.imshow(
            np.zeros((N_MS * MAX_SRVS, N_NODES)),
            cmap="gray",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        self.ax.set_xlabel("Nodes")
        self.ax.set_ylabel("All MS in the requested services")
        self.ax.set_xticks(range(N_NODES))
        self.ax.set_yticks(
            [i * N_MS + N_MS // 2 for i in range(MAX_SRVS)],
            labels=[f"s_{i}" for i in range(MAX_SRVS)],
        )
        self.ax.set_yticklabels(
            [f"s_{i}" for i in range(MAX_SRVS)],
            rotation=0,
        )

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

    def close(self):
        """Clean up resources when environment is closed."""
        pass


if __name__ == "__main__":
    env = FogEnv()
    env.get_state_info()
