# num
N_AGENTS = 3
N_NODES = 10
assert N_NODES == 10, "N_NODES needs to be 10"
N_LINKS = (N_NODES * (N_NODES - 1)) // 2  # upper limit

# MAX_SRVS = 50  # max number of req services
# For Services @data/services_.json
MAX_SRVS = 13
N_MS = 5
N_LINKS_MS = N_MS - 1
REQ_BUFFER_SIZE = 15

# features
F_NODE = 4  # avail cpu, avail mem, total cpu, total mem, # idle, max power
F_LINK = 3  # avail, total bandwidth, delay # energy_per_bit
F_SRV = 3  # deadline, lifetime, arrival_rate
F_MS = 2  # demand cpu, demand mem #, computational_intensity
F_LINK_MS = 1  # data_size
REQ_FILENAME = "requests.json"

# Node to Agent mapping
N2A = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
}

# Nodes locations
N_LOC = {
    0: (0.0, 0.0),
    1: (3.5, 0.0),
    2: (2.0, 3.0),
    3: (2.0, 5.0),
    4: (0.0, 8.0),
    5: (3.5, 8.0),
    6: (6.0, 2.0),
    7: (8.0, 2.0),
    8: (6.0, 7.0),
    9: (8.0, 7.0),
}


# plot N_LOC
def plot_node_locations(return_plt=False):
    import matplotlib.pyplot as plt

    x = [loc[0] for loc in N_LOC.values()]
    y = [loc[1] for loc in N_LOC.values()]

    # color based on agent from N2A
    colors = [N2A[node] for node in N_LOC.keys()]
    plt.scatter(x, y, c=colors, cmap="viridis", s=100)
    for node, loc in N_LOC.items():
        plt.annotate(node, (loc[0], loc[1]), textcoords="offset points", xytext=(0, 5), ha="center")

    if return_plt:
        return plt
    plt.title("Node Locations")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_node_locations()
