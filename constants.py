# num
N_NODES = 10
assert N_NODES == 10, "N_NODES needs to be 10"
N_LINKS = (N_NODES * (N_NODES - 1)) // 2  # upper limit
MAX_SRVS = 50  # max number of req services
N_MS = 5
N_LINKS_MS = N_MS - 1
REQ_BUFFER_SIZE = 10

# features
F_NODE = 6  # (avail resources, total)*2 (CPU, MEM), idle, max power
F_LINK = 4  # avail, total bandwidth, energy_per_bit, delay
F_SRV = 3  # deadline, lifetime, arrival_rate
F_MS = 3  # (avail resources)*2 (CPU, MEM), computational_intensity
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