import json
import networkx as nx
import matplotlib.pyplot as plt


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

def read_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


def write_json(file_path, data, indent=4):
    """
    Write to a JSON file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=indent)


def append_json(file_path, data, indent=4):
    """
    Append to a JSON file or create it if it doesn't exist.
    """
    try:
        existing_data = read_json(file_path)
    except FileNotFoundError:
        existing_data = []

    existing_data.append(data)
    write_json(file_path, existing_data, indent=indent)


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
            elif isinstance(v, str) and v.replace(".", "", 1).isdigit():
                G.nodes[n][k] = float(v)
    return G


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


def init_plt(x_lim=10, y_lim=50):

    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots()

    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("# of Requests")
    ax.set_title("# Real-time Requests every 2 seconds")
    ax.grid()
    return fig, ax


def update_plt(line, x_data, y_data):
    # Update line data
    line.set_data(x_data, y_data)

    plt.draw()
    plt.pause(0.01)

    # if (i + 1) % x_lim == 0:
    #     ax.set_xlim(0, (i + x_lim + 1))
