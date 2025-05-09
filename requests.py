import time
import random
import numpy as np
import matplotlib.pyplot as plt
from template import returnFogNetwork
from utils import seed_it


seed_it(646)

def generate_requests(size=1000, low=10, high=40, filename="requests_.npz"):
    """
    Generate random integers, save them to a .npz file, and return the array.
    """
    requests = np.random.randint(low, high, size=size)
    np.savez(filename, requests=requests)
    return requests


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

def llm(num=1):
    """
    Dummy LLM processing requests.
    """
    start = time.time()
    outputs = []
    for i in range(num):
        # read ith request from 
        outputs.append(f"Output {i + 1}")
        time.sleep(0.05)
    end = time.time()
    print(f"\t[LLM] Processing {num} requests took {end - start:.2f} seconds")
    return outputs


def main():
    network = returnFogNetwork()


    interval = 1  # seconds
    i_s, current_reqs = [], []

    # visualize plot
    _, ax = init_plt(x_lim=200)
    line, = ax.plot(i_s, current_reqs, c="red")

    i = 0
    while i < 200:
        num_requests = np.random.randint(1, 50)
        print(f"Req {i}: # of Requests: {num_requests}")

        service_requests = np.random.choice(network.services, num_requests)
        # print(f"\tRequsted Service IDs: {[service.id for service in service_requests]}")

        # Update data
        i_s.append(i)
        current_reqs.append(num_requests)
        update_plt(line, i_s, current_reqs)

        time.sleep(interval)
        i += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # generate_requests()

    main()
