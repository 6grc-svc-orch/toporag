import random
import numpy as np
import networkx as nx

from typing import List
from utils import load_gexf, save_gexf, save_services
from dataclasses import dataclass, field, asdict


# seed numpy
np.random.seed(646)
random.seed(646)


@dataclass
class Node:
    id: str
    cpu: float
    mem: float
    location: str

    def to_dict(self):
        return asdict(self)


# TODO: make it with some dependencies
@dataclass
class Microservice:
    id: int
    cpu: float
    mem: float
    data_out: float
    dependency: object = field(default=None)  # Default value for dependency

    def to_dict(self):
        return asdict(self)


@dataclass
class Service:
    id: str
    host: str
    chain: List[Microservice]
    max_delay: float
    arrival_rates: float

    def to_dict(self):
        return asdict(self)


def build_network(
    nodes: List[Node],
    link: np.ndarray,
    bw_matrix: np.ndarray,
    delay_matrix: np.ndarray,
):
    # the physical edge network
    G = nx.Graph()

    for node in nodes:
        G.add_node(
            node.id,
            cpu=node.cpu,
            mem=node.mem,
            # location=node.location,
        )

    # network links with bandwidth and delay
    indices = np.argwhere(link == 1)
    for i, j in indices:
        bw = bw_matrix[i][j]
        delay = delay_matrix[i][j]
        G.add_edge(
            nodes[i].id,
            nodes[j].id,
            bandwidth=bw,
            delay=delay,
        )
    return G


def build_service(s_id, num_ms=3):
    return Service(
        id=f"s{s_id}",
        host=f"n{random.randint(0, 4)}",  # assign a host node
        chain=[
            Microservice(
                f"s{s_id}_ms{i}",
                cpu=random.randint(1, 4),
                mem=random.randint(2, 8),
                data_out=random.randint(10, 100),
            )
            for i in range(num_ms)
        ],
        max_delay=random.uniform(100, 500),  # ms
        arrival_rates=np.random.poisson(lam=5),  # Poisson arrivals per node
    )

# TODO: just thinking out loud
def compute_routing_probabilities(services: List[Service], nodes: List[Node], deployment_status: np.ndarray):
    num_nodes = len(nodes)
    num_services = len(services)
    num_ms = len(services[0].chain)  # assuming all services have same length chains
    num_transitions = num_services * (num_ms - 1)

    routing_probs = np.zeros((num_transitions, num_nodes, num_nodes))  # [transition, from_node, to_node]

    for s_idx, service in enumerate(services):
        for i in range(num_ms - 1):  # skip last MS (no next MS)
            transition_idx = s_idx * (num_ms - 1) + i
            for ni in range(num_nodes):
                if deployment_status[ni][s_idx * num_ms + i] == 1:  # mi deployed at ni
                    weights = []
                    valid_nj = []
                    for nj in range(num_nodes):
                        if deployment_status[nj][s_idx * num_ms + i + 1] == 1:  # mj deployed at nj
                            weight = 1.0 / (1.0 + abs(ni - nj))  # simple proximity weight
                            weights.append(weight)
                            valid_nj.append(nj)
                    if weights:
                        weights = np.array(weights)
                        probs = weights / weights.sum()
                        for idx, nj in enumerate(valid_nj):
                            routing_probs[transition_idx, ni, nj] = probs[idx]
    return routing_probs


def main():
    """
    Netowork build
    """
    num_nodes = 5
    bw_low, bw_high = 10, 100
    delay_low, delay_high = 1, 10
    cpu_options = [1, 64, 128]
    mem_options = [128, 256, 512]

    # create 5 nodes
    nodes = [
        Node(
            id=f"n{i}",
            cpu=random.choice(cpu_options),
            mem=random.choice(mem_options),
            location="",
        )
        for i in range(num_nodes)
    ]

    link = np.tril(np.ones((num_nodes, num_nodes)), -1)
    bw_matrix = np.tril(
        np.random.randint(bw_low, bw_high, size=(num_nodes, num_nodes)), -1
    )
    delay_matrix = np.tril(
        np.random.uniform(delay_low, delay_high, size=(num_nodes, num_nodes)), -1
    )

    # Create graph
    G = build_network(nodes, link, bw_matrix, delay_matrix)
    save_gexf(G, "results/network.gexf")

    # load the same graph
    G = load_gexf("results/network.gexf")
    save_gexf(G, "results/network_resaved.gexf")

    # """
    # Service(s) build
    # """
    # num_services = 1
    # num_ms = 3  # number of microservices in a service
    # services = [build_service(s_id=i, num_ms=num_ms) for i in range(num_services)]
    # save_services(services, "results/services.json")

    # """
    # Deployment
    # """
    # # TODO: function to accept a microservice and a node and updates the deployment status
    # deployment_status = np.random.randint(1, 2, size=(num_nodes, num_services * num_ms))

    
    # # CPU and memory status
    # ms_cpus = np.array([ms.cpu for service in services for ms in service.chain])
    # ms_mems = np.array([ms.mem for service in services for ms in service.chain])
    # cpu_status = deployment_status * ms_cpus
    # mem_status = deployment_status * ms_mems

    # """
    # Constraints
    # """

    # # the check
    # cpu_mask = np.sum(cpu_status, axis=1) > np.array([node.cpu for node in nodes])
    # mem_mask = np.sum(mem_status, axis=1) > np.array([node.mem for node in nodes])

    # routing_table = compute_routing_probabilities(services, nodes, deployment_status)
    # print("Routing Table:", routing_table)
    # routing_probs = compute_routing_probabilities_np(services, nodes, deployment_status)
    # print("Routing Probabilities (NumPy):", routing_probs)
    # print("Routing Probs shape:", routing_probs.shape)



if __name__ == "__main__":
    main()
