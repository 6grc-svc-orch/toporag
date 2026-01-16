import numpy as np
import matplotlib.pyplot as plt

# Load data
main_line = np.load("tuned.npy")
drift1_line = np.load("drift1.npy")
drift2_line = np.load("drift2.npy")
fine_tune_indices = np.load("fine_tune_indices.npy")
threshold = 6.0

# Unpack (episode, reward)
t_main, r_main = main_line[:, 0], main_line[:, 1]
t_d1, r_d1 = drift1_line[:, 0], drift1_line[:, 1]
t_d2, r_d2 = drift2_line[:, 0], drift2_line[:, 1]

plt.figure(figsize=(11, 5))

plt.plot(t_main, r_main, lw=2, label="Reward (with fine-tuning)")
plt.plot(t_d1, r_d1, linestyle="--", linewidth=2, alpha=0.8, c="orange", label="drift")
plt.plot(t_d2, r_d2, linestyle="--", linewidth=2, alpha=0.8, c="orange", label="drift")

plt.axhline(threshold, linestyle="--", linewidth=1.5)

# vertical lines + points at fine-tuning moments
for ft in fine_tune_indices[:-1]:
    plt.axvline(ft, linestyle=":", linewidth=1.0, alpha=0.7)
    # find reward at that episode in main_line
    idx = np.where(t_main == ft)[0][0]
    plt.scatter(ft, r_main[idx], s=40)

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend(loc="lower left")
plt.tight_layout()
plt.xlim(0, 700)
plt.ylim(3, 11)
plt.savefig("drift_plot.png")
plt.show()
