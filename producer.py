import time
import numpy as np
from requests import init_plt, update_plt
from utils import read_json, seed_it, write_json

FILENAME = "requests.json"
seed_it(646)

_, ax = init_plt(x_lim=200, y_lim=20)
line = ax.plot([], [], c="red")

current_reqs = []
while True:
    interval = 2  # sec

    request_count = np.random.randint(1, 20)  # number of requests coming every interval
    service_ids_requested: list = np.random.choice(range(50), request_count).tolist()

    # Update plt
    current_reqs.append(request_count)
    update_plt(line, range(len(current_reqs)), current_reqs)

    existing_requests: list = read_json(FILENAME)
    existing_requests.append(service_ids_requested)
    write_json(FILENAME, existing_requests)
    time.sleep(interval)
