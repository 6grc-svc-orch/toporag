import matplotlib.pyplot as plt
import numpy as np
import time

state = {"placement_status": np.random.randint(0, 2, (250, 5))}

plt.ion()
fig, ax = plt.subplots(figsize=(5, 15))  # Wider figure to balance the narrow matrix
im = ax.imshow(state["placement_status"], cmap="gray", vmin=0, vmax=1, aspect="auto")


break_line = ax.plot([-0.5, 4.5], [0, 0], "r-", linewidth=2)

while True:
    state["placement_status"] = np.random.randint(0, 2, (250, 5))
    breakpoint_row = np.random.randint(10, 249)

    im.set_data(state["placement_status"])
    break_line.set_ydata([breakpoint_row, breakpoint_row])

    fig.canvas.draw()
    time.sleep(10.5)
