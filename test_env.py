# test_env.py
import sys
import random
import numpy as np

from env.constants import A2N, N2A
from env.envMulti import FogEnvMulti as FogEnv

from rich import print
from utils import seed_it, read_json
from template import loadFogNetwork

seed_it(646)
key = sys.argv[1]
keys = ["7b", "q2-1.5b", "0.5b", "0.36b", "1.5-32b"]
assert key in keys, f"Key {key} not in {keys}"

network = loadFogNetwork("pkl/network-connected-svcs.pkl")
run_name = "wz-0728_084550"
env = FogEnv(network=network, vis_placement=False, run_name=run_name)
env.reset()
env.fetch_requests()

weights = read_json(f"lmpf/weights-{run_name}-{key}.json")
env.weights = list(weights.values())

results = np.zeros((len(env.svcs_reqd), 4))  # reward, latency, lb, ru

total_steps = 400
for req in env.svcs_reqd:
    req = env.svcs_reqd[1]
    svc, (loc, closest_agent_idx) = req
    for step in range(total_steps):
        print(f"[Step {step}]", end="\r")
        llm_action = env.get_llm_action(
            svc_id=svc.id,
            ms_idx=env.state["next_ms_idx"],
            closest_agent_idx=closest_agent_idx,
            placement_path=f"lmpf/placement-{run_name}-{key}.json",
        )

        llm_action = random.randint(0, 9) if llm_action == -1 else llm_action

        agent_idx = N2A[llm_action]
        action = A2N[agent_idx].index(llm_action)

        _, reward, _, _, info = env.step(agent_idx=agent_idx, action=action, action_index_b4=action)

        if info["sp"]:
            break
        if step == total_steps - 1:
            reward = 0
            info["latency"] = np.max(results[:, 1])
            info["lb"] = np.max(results[:, 2])
            info["ru"] = np.max(results[:, 3])
            break

    result = (reward, info["latency"], info["lb"], info["ru"])
    if reward != 0:
        results = np.vstack([results, result])
        print(result)


print(f"{run_name}-{key}")

# results where reward != 0
results = results[results[:, 0] != 0]
print("Total Completed Services:", results.shape[0])

print("Avg Reward:", np.mean(results[:, 0]))
print("Avg Latency:", np.mean(results[:, 1]) * 1000, "ms")
print("Avg Load Balance:", np.mean(results[:, 2]) * 100, "%")
print("Avg Resource Utilization:", np.mean(results[:, 3]) * 100, "%")
