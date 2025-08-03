# multiProducer.py

# A request is [id, (x,y)] where id is a service ID (between 0 and 13) and (x,y) is the location of the request bounded by (0, 0) and (8,8).
# here is the composition of the requests:
# area A0: (0, 0) to (4, 4) 5 requests
# area A1: (0, 4) to (4, 8) 50 requests
# area A2: (4, 0) to (8, 8) 100 requests

import numpy as np
from utils import seed_it

seed_it(646)
SM = 12


def _find_closest_agent(loc) -> int:
    """Find the closest agent to the given location."""
    distances = [np.linalg.norm(np.array(N_LOC[n]) - np.array(loc)) for n in N_LOC.keys()]
    closest_node = int(np.argmin(distances))
    return N2A[closest_node]


# regional
def generate_requests():

    A0_reqs = []
    len_A0_reqs = 1
    for _ in range(len_A0_reqs):
        x = np.random.rand() * 3 + 0.5
        y = np.random.rand() * 3 + 0.5
        s_id = np.random.randint(0, SM)
        req = [s_id, (x, y)]
        A0_reqs.append(req)

    A1_reqs = []
    len_A1_reqs = 4
    for _ in range(len_A1_reqs):
        x = np.random.rand() * 3 + 0.5
        y = np.random.rand() * 3 + 4.5  # 4-8
        s_id = np.random.randint(0, SM)
        req = [s_id, (x, y)]
        A1_reqs.append(req)

    A2_reqs = []
    len_A2_reqs = 10
    for _ in range(len_A2_reqs):
        x = np.random.rand() * 3 + 5  # 4-8
        y = np.random.rand() * 7 + 0.5
        s_id = np.random.randint(0, SM)
        req = [s_id, (x, y)]
        A2_reqs.append(req)

    all_reqs = A0_reqs + A1_reqs + A2_reqs
    # shuffle
    np.random.shuffle(all_reqs)

    return all_reqs


# non regional
def generate_requests_nr(len_reqs=15):
    seed_it(646)
    """Generate requests with random locations and service IDs. (non-regional)"""
    all_reqs = []
    count = 0
    while count < len_reqs:
        x = np.random.rand() * 8
        y = np.random.rand() * 8
        s_id = np.random.randint(0, SM)
        if s_id == 10:
            continue  # skip service ID 10
        req = [s_id, (x, y)]
        all_reqs.append(req)
        count += 1
    return all_reqs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from env.constants import N2A, N_LOC, plot_node_locations

    all_reqs = generate_requests()
    A0_reqs = [req for req in all_reqs if 0 <= req[1][0] <= 4 and 0 <= req[1][1] <= 4]  # (0,0) to (4,4)
    A1_reqs = [req for req in all_reqs if 0 <= req[1][0] <= 4 and 4 < req[1][1] <= 8]  # (0,4) to (4,8)
    A2_reqs = [req for req in all_reqs if 4 < req[1][0] <= 8 and 0 <= req[1][1] <= 8]  # (4,0) to (8,8)

    # plot_node_locations(return_plt=True)

    def plot_requests(requests, area_name, color):
        x = [req[1][0] for req in requests]
        y = [req[1][1] for req in requests]
        colors = [_find_closest_agent(req[1]) for req in requests]
        plt.scatter(x, y, label=area_name, color=color, s=100)

    plot_requests(A0_reqs, "Area A0", color="blue")
    plot_requests(A1_reqs, "Area A1", color="green")
    plot_requests(A2_reqs, "Area A2", color="red")
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Requests in Different Areas")
    plt.legend()
    plt.show()
