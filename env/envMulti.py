import sys
import os

from env.multiProducer import generate_requests_nr as generate_requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List, Tuple, TypedDict, Dict
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as GymDict, Discrete
from numpy.typing import NDArray
from template import FogNetwork, Service, loadFogNetwork
from utils import read_json, write_json
import matplotlib.pyplot as plt
from env.constants import *
from env.helper import FogEnvAssertionsMixin, FogEnvState


# take observation space out to use it before the env is created
obs_space = GymDict(
    {
        # network
        "node_feats": Box(0.0, np.inf, shape=(N_NODES, F_NODE)),
        "link_feats": Box(0.0, np.inf, shape=(N_LINKS, F_LINK)),
        # requests (all services)
        "service_feats": Box(0.0, np.inf, shape=(MAX_SRVS, F_SRV)),
        "ms_feats": Box(0.0, np.inf, shape=(N_MS * MAX_SRVS, F_MS)),
        "link_ms_feats": Box(0.0, np.inf, shape=(N_LINKS_MS * MAX_SRVS, F_LINK_MS)),
        # current
        "placement_status": Box(0, MAX_SRVS, shape=(N_MS * MAX_SRVS, N_NODES)),
        # idx
        "next_ms_idx": Box(0, N_MS - 1, shape=(1,), dtype=int),
        "next_service_idx": Box(0, REQ_BUFFER_SIZE - 1, shape=(1,), dtype=int),
        "svc_id": Box(0, MAX_SRVS - 1, shape=(1,), dtype=int),  # current service id
    },
    seed=646,
)
action_space = Box(low=0, high=1, shape=(10,), dtype=int, seed=646)
# action_space = Discrete(N_NODES, seed=646)


class FogEnvMulti(FogEnvAssertionsMixin, gym.Env):
    def __init__(self, network: FogNetwork = None, vis_reqs=False, vis_placement=False):
        # TODO: have a list of already explored nodes
        # TODO: keep node placement as suggestion until all ms in a s are placed
        super().__init__()
        assert network is not None, "Network cannot be None"
        self.observation_space = obs_space  # use get_obs_space
        self.action_space = action_space  # use get_act_space
        self.network = network
        self.num_agents = N_AGENTS

        self.state: FogEnvState = {}
        self.svcs_reqd: List[List[Service, List]] = []  # List[svc, [loc, closest_agent_idx]]
        self._all_svcs_reqd: List[List[int, int, float, List, List]] = (
            []
        )  # List[idx, svc.id, time, more, node_ids] # -1 node_idx nt placed

        # only increments not set back to 0
        self.c_svc_reqd_idx = 0  # current service requested index

        self.vis_placement = vis_placement
        self.services_requested_len = []

        self.weights = [0.333, 0.333, 0.333]  # alpha, beta, gamma

        if self.vis_placement:
            self._render_placement()

    @staticmethod
    def _nodes2link(i: int, j: int) -> int:
        """Convert node indices to link index."""
        if i > j:
            i, j = j, i
        return i * (2 * N_NODES - i - 1) // 2 + j - i - 1

    def _get_ms_idx(self) -> int:
        """Get the GLOBAL `ms_idx` for the `current service` and `microservice`."""
        svc = self.svcs_reqd[self.state["next_service_idx"]]
        if svc[0].id != self.state["svc_id"]:
            print(f"Service ID mismatch: {svc[0].id} != {self.state['svc_id']}")
        ms_idx = svc[0].id * N_MS + self.state["next_ms_idx"]
        return ms_idx

    def _get_l_idx(self, node_idx: int, svc_reqd_idx: int = None, ms_idx: int = None) -> int:
        """Calculate the link index (`l_idx`) between the previous node and the current node.

        If `ms_idx` is `None`, the method uses the current microservice index (`ms_idx`).
        If `svc_reqd_idx` is `None`, it uses the current service index.
        """

        if svc_reqd_idx is None:
            svc_reqd_idx = self.c_svc_reqd_idx

        if ms_idx is None:
            ms_idx = self._get_ms_idx()

        svc_nodes = self._all_svcs_reqd[svc_reqd_idx][4]  # List of nodes for the req. svc
        p_node = svc_nodes[ms_idx % N_MS - 1]

        l_idx = self._nodes2link(p_node, node_idx)
        return l_idx

    def reset(self, seed=646):
        #  NOTE: next_service_idx is not the service_id but just a counter over self.services_requested
        node_feats = np.array(
            [
                [
                    *n.available_resources,
                    *n.total_resources,
                    # n.idle_power,
                    # n.max_power,
                ]
                for n in self.network.fog_nodes
            ]
        )

        link_feats = np.zeros((N_LINKS, F_LINK))
        for l in self.network.links:
            link_feats[self._nodes2link(l.source, l.destination)] = np.array(
                [
                    l.available_bandwidth,
                    l.total_bandwidth,
                    # l.energy_per_bit,
                    l.delay,
                ]
            )

        self.state.update(
            {
                "node_feats": node_feats,
                "link_feats": link_feats,
                "service_feats": np.zeros((MAX_SRVS, F_SRV)),
                "ms_feats": np.zeros((N_MS * MAX_SRVS, F_MS)),
                "link_ms_feats": np.zeros((N_LINKS_MS * MAX_SRVS, F_LINK_MS)),
                "placement_status": np.zeros((N_MS * MAX_SRVS, N_NODES)),  # see when it has to be reset
                "next_ms_idx": 0,
                "next_service_idx": 0,
                "svc_id": -1,  # current service id
            }
        )
        return self.state, {}

    def get_agent_order(self, agent_idx: int) -> List[int]:
        """Get the order of agents based on the closest agent index."""
        return [(agent_idx + i) % self.num_agents for i in range(self.num_agents)]

    def get_state(self, agent_idx: int, verbose=False) -> FogEnvState:
        """Get the state for a specific agent."""
        self._assert_agent_idx(agent_idx)

        # nodes corresponding to this agent
        _nodes = [n for n, a in N2A.items() if a == agent_idx]

        # Filtered state for the agent
        state_per_agent: FogEnvState = {
            **{k: v for k, v in self.state.items() if k not in {"node_feats", "placement_status"}},
            "node_feats": self.state["node_feats"][_nodes],
            "placement_status": self.state["placement_status"][:, _nodes],
        }
        if verbose:
            self._print_state_info(state_per_agent)
        return state_per_agent

    def get_obs_space(self, agent_idx: int):
        """Get the observation space for a specific agent."""
        self._assert_agent_idx(agent_idx)
        _nodes = [n for n, a in N2A.items() if a == agent_idx]
        return GymDict(
            {
                **{
                    k: v
                    for k, v in self.observation_space.spaces.items()
                    if k not in {"node_feats", "placement_status"}
                },
                "node_feats": Box(0.0, np.inf, shape=(len(_nodes), F_NODE)),
                "placement_status": Box(0, MAX_SRVS, shape=(N_MS * MAX_SRVS, len(_nodes))),
            }
        )

    def get_act_space(self, agent_idx: int):
        """Get the action space for a specific agent."""
        self._assert_agent_idx(agent_idx)
        _nodes = [n for n, a in N2A.items() if a == agent_idx]

        return Box(low=0, high=1, shape=(len(_nodes),), dtype=int, seed=646)

    def get_node(self, selected_idx: int, agent_idx: int) -> int:
        self._assert_agent_idx(agent_idx)
        _nodes = [n for n, a in N2A.items() if a == agent_idx]
        return _nodes[selected_idx]

    def _find_closest_agent(self, loc: Tuple[float, float]) -> int:
        """Find the closest agent to the given location."""
        distances = [np.linalg.norm(np.array(N_LOC[n.id]) - np.array(loc)) for n in self.network.fog_nodes]
        closest_node = int(np.argmin(distances))
        return N2A[closest_node]

    def fetch_requests(self, verbose=False):
        """Get the current state of the environment."""
        #  0-12 (except 2)
        all_reqs = generate_requests()
        # print(f"Fetched {len(all_reqs)} requests")
        # print([req[0] for req in all_reqs])  # print service ids
        # print([self._find_closest_agent(req[1]) for req in all_reqs])  # print locations

        if len(all_reqs) == 0:
            return {
                "num_reqs": 0,
                # "len_reqs": [],
                "dropped_reqs": 0,
            }

        num_requests = len(all_reqs)

        # fill up the self.services_requested from self.network.services
        self.svcs_reqd = []
        for i, req in enumerate(all_reqs):
            if i == REQ_BUFFER_SIZE:
                break
            # find closest agent for req
            closest_agent_idx = self._find_closest_agent(req[1])
            self.svcs_reqd.append([self.network.services[req[0]], [req[1], closest_agent_idx]])

        # extend or append? (append would have kept track but needs to be flattened to be accessed)
        start_idx = len(self._all_svcs_reqd)
        self._all_svcs_reqd.extend(
            [[start_idx + i, s.id, np.inf, more, [-1] * N_MS] for i, [s, more] in enumerate(self.svcs_reqd)]
        )

        # empty the reqs file (so we dont fetch it again)
        write_json(REQ_FILENAME, [])

        self._assert_next_service_idx()
        self._assert_next_ms_idx()

        self.state["svc_id"] = self.svcs_reqd[self.state["next_service_idx"]][0].id
        self._populate_observation()

        self.services_requested_len.append(num_requests)

        return {
            "num_reqs": num_requests,
            # "len_reqs": [len(reqs) for reqs in all_reqs],
            "dropped_reqs": num_requests - len(self.svcs_reqd),
        }

    def step(self, agent_idx, action, last_step=False):
        """Perform one environment step by placing a microservice on a selected node (and expire a service if lifetime is over)."""
        selected_node_idx = self.get_node(action, agent_idx)
        info = {"sp": False}
        done = False
        truncated = False  # always False so not used

        # # 1) Expiration
        exp_svcs, re_nodes = self._expire_services()  # global
        if len(exp_svcs) > 0:
            info["exp_svcs"] = exp_svcs
            info["re_nodes"] = re_nodes
            # print(f"Expired services: {exp_svcs}, reclaimed nodes: {re_nodes}")

        # 2) DRL placement constraints C1
        c1_passed, demand, n_avail = self._check_node_constraints(selected_node_idx)
        if not c1_passed:
            info["node/demand"] = demand
            info["node/avail"] = n_avail
            info["how_many"] = np.sum(self.state["placement_status"][:, selected_node_idx] > 0)
            # invalid placement
            reward = 0
            if last_step:
                done, _ = self._advance_state(last_step=True)
                info["sp"] = True
            return self.get_state(agent_idx), reward, done, False, info

        # check if link has enough bandwidth C2
        c2_passed, l_demand, l_avail = self._check_link_constraints(selected_node_idx)
        if not c2_passed:
            info["link/demand"] = l_demand
            info["link/avail"] = l_avail
            # invalid placement
            reward = 0
            if last_step:
                done, _ = self._advance_state(last_step=True)
                info["sp"] = True
            return self.get_state(agent_idx), reward, done, False, info

        # 3) valid: place it, maybe record start time if last ms
        self._place_microservice(selected_node_idx, demand=demand, l_demand=l_demand)

        base_reward, reward_comp = self._compute_reward()  # 0-1
        info.update(
            {
                "latency": reward_comp[0],
                "lb": reward_comp[1],
                "ru": reward_comp[2],
                "alpha": reward_comp[3],
                "beta": reward_comp[4],
                "gamma": reward_comp[5],
            }
        )
        progress_reward = (self.state["next_service_idx"] * N_MS + self.state["next_ms_idx"]) / (
            len(self.svcs_reqd) * N_MS
        )  # 0-1
        reward = base_reward * 10 + progress_reward * 10
        if base_reward == 0:
            reward = 0

        done, sp = self._advance_state(last_step)
        info["sp"] = sp

        # 5) Visualization
        if self.vis_placement:
            self._update_placement_plot()

        return self.get_state(agent_idx), reward, done, False, info

    def _populate_observation(self):
        """Fill up the state based on the `self.services_requested`."""
        for s_tbd, more in self.svcs_reqd:
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
                        # ms.computational_intensity,
                    ]
                )

            # link
            for i, link in enumerate(s_tbd.invocation_edges):
                self.state["link_ms_feats"][s_tbd.id * N_LINKS_MS + i] = np.array([link.data_size])

    def _check_node_constraints(self, node_idx: int) -> tuple[bool, NDArray[np.float64]]:
        ms_idx = self._get_ms_idx()
        ms_demand = self.state["ms_feats"][ms_idx, :2]  # BUG: was just next_ms_idx
        node_avail = self.state["node_feats"][node_idx, :2]
        return np.all(node_avail >= ms_demand), ms_demand, node_avail

    def _check_link_constraints(self, node_idx: int) -> tuple[bool, float]:
        """Check if the link between `prev_node_idx` and `node_idx` has enough `bandwidth`"""
        if self.state["next_ms_idx"] == 0:
            return True, 0.0, 0.0  # no link constraints for the first ms

        l_idx = self._get_l_idx(node_idx)
        base_l_idx = self.svcs_reqd[self.state["next_service_idx"]][0].id * N_LINKS_MS
        l_ms_idx = base_l_idx + self.state["next_ms_idx"] - 1

        l_avail = self.state["link_feats"][l_idx, 0]
        l_demand = self.state["link_ms_feats"][l_ms_idx, 0]
        return l_avail >= l_demand, l_demand, l_avail

    def _place_microservice(self, node_idx: int, demand: NDArray[np.float64], l_demand: float):
        ms_idx = self._get_ms_idx()

        # Deduct used resources
        self.state["node_feats"][node_idx, :2] -= demand

        if self.state["next_ms_idx"] > 0:
            l_idx = self._get_l_idx(node_idx)
            self.state["link_feats"][l_idx, 0] -= l_demand

        # Update placement status
        self.state["placement_status"][ms_idx, node_idx] += 1
        self._all_svcs_reqd[self.c_svc_reqd_idx][4][self.state["next_ms_idx"]] = node_idx

        # If this was the final MS of a service, record its start time
        if self.state["next_ms_idx"] == N_MS - 1:
            self._all_svcs_reqd[self.c_svc_reqd_idx][2] = time.time()

        # TODO: update self.network with node_feats and placement_status

    def _advance_state(self, last_step=False):
        """Advances the internal state of the environment by updating indices for microservices and services.
        Args:
            last_step (bool, optional)

        Returns:
            svcs_placed (bool), svc_placed (bool)
        """
        svcs_placed, svc_placed = False, False

        self.state["next_ms_idx"] += 1
        self._assert_next_ms_idx_nms()

        if self.state["next_ms_idx"] == N_MS or last_step:
            # we move to the next service
            self.state["next_ms_idx"] = 0
            self.state["next_service_idx"] += 1
            self.c_svc_reqd_idx += 1
            svc_placed = True

        if self.state["next_service_idx"] == len(self.svcs_reqd) or last_step:
            # we have placed all the ms of all the services
            self.state["next_service_idx"] = 0
            self._assert_next_ms_idx_all()
            svcs_placed = True

        self.state["svc_id"] = self.svcs_reqd[self.state["next_service_idx"]][0].id

        return svcs_placed, svc_placed

    def _update_placement_plot(self):
        self.im.set_data(self.state["placement_status"])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _expire_services(self):
        """Expire any services whose lifetime has passed."""
        now = time.time()
        expired_services_ids = set()
        reclaimed_nodes = set()

        for idx, svc_idx, start_time, more, nodes in self._all_svcs_reqd:
            if start_time is None:
                continue

            svc: Service = self.network.services[svc_idx]
            if (now - start_time) > svc.lifetime:
                # 1. find all microservice rows for this service
                base = svc_idx * N_MS
                rows = range(base, base + N_MS)

                # 2. for each placed MS, reclaim resources
                row_node: List[tuple[int, int]] = []
                for i, ms_idx in enumerate(rows):
                    node = nodes[i]
                    row_node.append((ms_idx, node))

                    # MS demand vector
                    demand = self.state["ms_feats"][ms_idx, :2]
                    self.state["node_feats"][node, :2] += demand

                    self._assert_node_claim(node)

                    # Link demand
                    if ms_idx > base:
                        l_idx = self._get_l_idx(node, ms_idx=ms_idx, svc_reqd_idx=idx)

                        base_l_idx = svc_idx * N_LINKS_MS
                        l_ms_idx = base_l_idx + ms_idx - base - 1

                        l_demand = self.state["link_ms_feats"][l_ms_idx, 0]
                        self.state["link_feats"][l_idx, 0] += l_demand

                        self._assert_link_claim(l_idx)

                # Updating placement status at the end (to avoid err in link update)
                for ms_idx, node in row_node:
                    self.state["placement_status"][ms_idx, node] -= 1

                    reclaimed_nodes.add(int(node))
                    expired_services_ids.add(svc_idx)

                self._assert_placement_all()

                # 4. cleanup
                self._all_svcs_reqd[idx][2] = None  # remove start time
                self._all_svcs_reqd[idx][4] = [-1] * N_MS  # reset nodes

        # TODO: update self.network with expired_services_ids and reclaimed_nodes

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

    def _compute_reward_dummy(self):
        """Reward: Random"""
        return np.random.rand()

    def _compute_reward(self):
        """Compute the reward based POV: current svc."""

        if self.state["next_service_idx"] == self.state["next_ms_idx"] == 0:  # read once every episode
            try:
                hp: Dict = read_json("planner/weights.json")
                alpha = hp.get("latency", 0.333)
                beta = hp.get("load_balance", 0.333)
                gamma = hp.get("utilization", 0.333)
                self.weights = [alpha, beta, gamma]
            except Exception:
                alpha = beta = gamma = 0.333
        else:
            alpha, beta, gamma = self.weights

        temp = 1e-1  # cost is ~1

        # 1. E2E latency
        latency, l_comp = self._calc_latency()

        # 2. Resource utilization
        ru, ru_comp = self._calc_resource_utilization()

        # 3. Load balance
        lb, lb_comp = self._calc_lb()

        cost = alpha * latency + beta * lb + gamma * ru
        objective_value = 1 / (1 + temp * cost)
        if cost < 0:
            print(f"cost: {cost} reward: {objective_value:.4f} (latency: {latency:.2f}, lb: {lb:.2f})")

        # TODO: dont even advance if C3 is not met
        # latency C3
        svc = self.svcs_reqd[self.state["next_service_idx"]][0]
        if latency > (svc.deadline / 1000):  # in sec
            # print(f"Latency {latency:.2f} exceeds deadline {svc.deadline / 1000:.2f} sec")
            objective_value = 0

        return objective_value, (latency, lb, ru, alpha, beta, gamma)

    # calcs
    def _calc_latency(self):
        """
        Calculate the end-to-end latency of the network **for the current service**.
        """
        current_svc_info = self._all_svcs_reqd[self.c_svc_reqd_idx]
        svc_nodes = current_svc_info[4]  # List of nodes for the current service
        svc_id = current_svc_info[1]

        base = svc_id * N_MS
        ms_indices = np.arange(base, base + N_MS)

        # 1: Access delay (POV: svc_id)
        loc, _ = current_svc_info[3]
        node_loc = N_LOC[svc_nodes[0]]
        distance = np.linalg.norm(np.array(loc) - np.array(node_loc))
        access_delay = distance * 2 / 1000  # 2msec per unit distance # in sec

        # 2: Processing delay (POV: svc_id)
        cpu_capacity = self.state["node_feats"][svc_nodes, 2]  # shape (N_MS,)
        cpu_demand = self.state["ms_feats"][ms_indices, 0]  # shape (N_MS,)
        processing_delay = np.sum(cpu_demand / cpu_capacity)  # in sec

        # 3: Queuing delay (simple M/M/1) (POV: svc_id)
        # `cpu_throughputs` & `computational_intensities` already computed
        service_rates = cpu_capacity / cpu_demand  # shape (N_MS,)
        arrival_rates = self.state["service_feats"][
            ms_indices // N_MS, 2
        ]  # shape (N_MS,) # arrival rate of ms is arrival rate of svc

        utilizations = np.minimum(0.99, arrival_rates / service_rates)  # shape (N_MS,)
        queueing_delay = np.sum(utilizations / (service_rates * (1 - utilizations)))  # in sec

        # 4: NW delay (POV: svc_id)
        network_delay = 0  # in msec
        placed_ms = range(base + 1, base + self.state["next_ms_idx"] + 1)  # start from 1 to skip the first ms
        for i, ms_idx in enumerate(placed_ms):
            l_idx = self._get_l_idx(svc_nodes[i], ms_idx=ms_idx, svc_reqd_idx=current_svc_info[0])
            network_delay += self.state["link_feats"][l_idx, 2]
        network_delay /= 1000  # in sec

        total_delay = access_delay + processing_delay + queueing_delay + network_delay
        return total_delay, (
            access_delay,
            processing_delay,
            queueing_delay,
            network_delay,
        )

    def _calc_resource_utilization(self):
        """
        Calculate the resource utilization of the network **for the current service**.
        """
        current_svc_info = self._all_svcs_reqd[self.c_svc_reqd_idx]
        svc_nodes: List = current_svc_info[4]  # List of nodes for the current service
        svc_id = current_svc_info[1]

        base = svc_id * N_MS
        ms_indices = np.arange(base, base + N_MS)

        demand = self.state["ms_feats"][ms_indices, :2]  # shape (N_MS, 2)
        capacity = self.state["node_feats"][svc_nodes, 2:4]  # shape (N_MS, 2)

        # Compute utilization for CPU and MEM
        cpu_util = np.mean(demand[:, 0] / capacity[:, 0])
        mem_util = np.mean(demand[:, 1] / capacity[:, 1])
        util = (cpu_util + mem_util) / 2.0

        return util, (cpu_util, mem_util)

    def _calc_lb(self):
        """
        Take variance across nodes for CPU & MEM & STO and sum.  **POV: nodes**
        """
        node_feats = self.state["node_feats"]

        # extract avail & total
        avail = node_feats[:, :2]  # CPU, MEM
        total = node_feats[:, 2:4]  # CPU, MEM
        util = avail / total  # shape (N_NODES, 2)

        # per‐dimension variance
        var_cpu = np.var(util[:, 0])
        var_mem = np.var(util[:, 1])
        var = var_cpu + var_mem
        return var, (var_cpu, var_mem)

    def get_state_info(self, return_=False):
        for key, value in self.state.items():
            if isinstance(value, int):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value.shape}")
        if return_:
            return self.state

    def render(self, mode="human"):
        """Render the environment."""
        pass

    def close(self):
        """Clean up resources when environment is closed."""
        pass

    def get_llm_action(self, svc_id: int, ms_idx: int, closest_agent_idx: int) -> int:
        "Get `global` node_idx from the planner/placement.json"
        suggestions = read_json("planner/placement.json")
        if not suggestions:
            return -1
        for suggestion in suggestions:
            if (
                suggestion["svc_id"] == svc_id
                and suggestion["ms_idx"] == ms_idx
                and suggestion["closest_agent_idx"] == closest_agent_idx
            ):
                return suggestion["node_idx"]
        return -1

if __name__ == "__main__":

    network = loadFogNetwork("pkl/network-connected-svcs.pkl")
    env = FogEnvMulti(network=network, vis_placement=False)
    _ = env.reset()
    rs = env.fetch_requests()
    print(env.svcs_reqd)

    env.get_state(1)
    print(env.get_act_space(1).shape[0])
