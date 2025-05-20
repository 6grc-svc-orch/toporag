import time
import numpy as np
from utils import read_json, seed_it, write_json

FILENAME = "requests.json"
seed_it(646)

vis_reqs = False

if vis_reqs:
    from requests import init_plt, update_plt

    _, ax = init_plt(x_lim=200, y_lim=20)
    (line,) = ax.plot([], [], c="red")

current_reqs = []
i = 0
while True:
    interval = 1  # sec

    request_count = np.random.randint(1, 10)  # number of requests coming every interval
    service_ids_requested: list = np.random.choice(range(50), request_count).tolist()
    print(
        f"[reqs.: {i}] # of reqs.: {request_count}, Service IDs: {service_ids_requested}"
    )

    # Update plt
    if vis_reqs:
        current_reqs.append(request_count)
        update_plt(line, range(len(current_reqs)), current_reqs)

    existing_requests: list = read_json(FILENAME)
    existing_requests.append(service_ids_requested)
    write_json(FILENAME, existing_requests)
    time.sleep(interval)

    i += 1
