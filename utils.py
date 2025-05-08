import json
import networkx as nx


def sanitize_graph_for_gexf(G: nx.Graph) -> nx.Graph:
    G_sanitized = G.copy()

    # Convert complex node attributes to strings
    for n, attr in G_sanitized.nodes(data=True):
        for k, v in attr.items():
            if not isinstance(v, (int, float, str, bool)):
                G_sanitized.nodes[n][k] = str(v)

    # Convert complex edge attributes to strings
    for u, v, attr in G_sanitized.edges(data=True):
        for k, val in attr.items():
            if not isinstance(val, (int, float, str, bool)):
                G_sanitized.edges[u, v][k] = str(val)
    return G_sanitized


def read_json(file_path):
    """
    Return a JSON file's content.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def write_json(file_path, data):
    """
    Write to a JSON file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def save_gexf(G, path):
    G_clean = sanitize_graph_for_gexf(G)
    nx.write_gexf(G_clean, path)

def load_gexf(path):
    """
    Load a GEXF file and return the graph.
    """
    G = nx.read_gexf(path)
    # Convert node attributes to appropriate types
    for n, attr in G.nodes(data=True):
        for k, v in attr.items():
            if isinstance(v, str) and v.isdigit():
                G.nodes[n][k] = int(v)
            elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
                G.nodes[n][k] = float(v)
    return G

# save services to a json
def save_services(services, path):
    
    services_dict = [service.to_dict() for service in services]
    with open(path, "w") as f:
        json.dump(services_dict, f, indent=4)


# load services from a json
def load_services(path):
    from models import Service, Microservice

    services_dict = read_json(path)
    services = []
    for service_dict in services_dict:
        service = Service(
            id=service_dict["id"],
            host=service_dict["host"],
            chain=[
                Microservice(
                    id=ms["id"],
                    cpu=ms["cpu"],
                    mem=ms["mem"],
                    data_out=ms["data_out"],
                )
                for ms in service_dict["chain"]
            ],
            max_delay=service_dict["max_delay"],
            arrival_rates=service_dict["arrival_rates"],
        )
        services.append(service)
    return services

def seed_it(seed):
    """
    Set the seed for reproducibility for `random`, `np`, `torch`, `torch.cuda`.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False