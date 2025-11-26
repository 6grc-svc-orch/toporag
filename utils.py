import json
import networkx as nx
import matplotlib.pyplot as plt

import os
import tempfile


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


def read_json(file_path, create=False):
    """
    Return a JSON file's content. If create is True, it will create an empty list if the file does not exist.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        if create:
            write_json(file_path, [])
            return []
        else:
            raise FileNotFoundError(f"File {file_path} does not exist.")


def read_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


def write_json(file_path, data, indent=4, atomic=False):
    """
    Write to a JSON file. If atomic is True, it will write to a temp file and rename it.
    """
    if atomic:
        dir_name = os.path.dirname(file_path)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tmp_file:
            json.dump(data, tmp_file, indent=indent)
            temp_name = tmp_file.name
        os.replace(temp_name, file_path)  # atomic on most OS
    else:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=indent)


def append_json(file_path, data, indent=4, max_length=None, atomic=False):
    """
    Append to a JSON file or create it if it doesn't exist.
    If max_length is specified, it will limit the number of entries in the file.
    """
    try:
        existing_data = read_json(file_path)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.append(data)
    if max_length is not None and len(existing_data) > max_length:
        existing_data = existing_data[-max_length:]

    write_json(file_path, existing_data, indent=indent, atomic=atomic)


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

def remove_empty_files(directory, dry_run=True, only_json=True):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if only_json and not filename.endswith('.json'):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r') as file:
                    content = file.read().strip()
                if content == '' or content == '[]':
                    if not dry_run:
                        os.remove(filepath)
                    print(f"Removed empty JSON file: {filepath}")
            except Exception as e:
                continue
                # print(f"Error processing file {filepath}: {e}")

# if __name__ == "__main__":
#     import sys
#     directory = sys.argv[1] if len(sys.argv) > 1 else '.'
#     dry_run = '--dry-run' in sys.argv
#     remove_empty_json_files(directory=directory, dry_run=dry_run)
