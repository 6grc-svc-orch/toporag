# test_drift.py
# test_env but for drift testing
#  let me see if wih time the reward drops down

import sys
import random
import time
import numpy as np

from env.constants import A2N, N2A
from env.envMulti import FogEnvMulti as FogEnv

from rich import print
from utils import seed_it, read_json, write_json
from template import loadFogNetwork

import wandb

seed_it(646)

DIR = "drift"
key = sys.argv[1]
run_name = "wz-0728_084550"
use_wandb = False
if use_wandb:
    wandb.init(
        project="fog-llm-drift",
        name=f"{run_name}-{key}",
        config={
            "run_name": run_name,
            "model_key": key,
        },
    )

keys = ["7b", "q2-1.5b", "0.5b", "0.36b", "1.5-32b"]
assert key in keys, f"Key {key} not in {keys}"

network = loadFogNetwork("pkl/network-connected-svcs.pkl")
env = FogEnv(network=network, vis_placement=False, run_name=run_name)
env.reset()

REWARD_THRESHOLD = 5.0
reward_below_threshold_count = 0

for global_step in range(int(1e6)):
    start = time.time()
    env.reset()
    env.fetch_requests()
    try:
        weights = read_json(f"{DIR}/weights-{run_name}-{key}.json")
    except Exception as e:
        print(f"Error reading weights file: {e}")
        weights = {"latency": 0.333, "load_balance": 0.333, "utilization": 0.333}
    env.weights = list(weights.values())
    results = np.zeros((len(env.svcs_reqd), 4))  # reward, latency, lb, ru
    for req in env.svcs_reqd:
        req = env.svcs_reqd[1]
        svc, (loc, closest_agent_idx) = req
        total_steps = 400
        for step in range(total_steps):
            # print(f"[Step {step}]", end="\r")
            try:
                llm_action = env.get_llm_action(
                    svc_id=svc.id,
                    ms_idx=env.state["next_ms_idx"],
                    closest_agent_idx=closest_agent_idx,
                    placement_path=f"{DIR}/placement-{run_name}-{key}.json",
                )
            except Exception as e:
                llm_action = -1
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
            # print(result)
    end = time.time()
    if end - start < 5:
        time.sleep(5 - (end - start))

    # results where reward != 0
    results = results[results[:, 0] != 0]
    avg_reward = np.mean(results[:, 0])
    avg_latency = np.mean(results[:, 1])  # sec
    avg_lb = np.mean(results[:, 2])
    avg_ru = np.mean(results[:, 3])
    summary = {
        "cmpltd svc": f"{results.shape[0]}/{len(env.svcs_reqd)}",
        "r": avg_reward,
        "l": avg_latency,
        "lb": avg_lb,
        "ru": avg_ru,
    }

    if use_wandb:
        wandb.log(summary, step=global_step)

    print(f"step {global_step}: {run_name}-{key}:", summary)

    # report back to the llm planner

    if avg_reward < REWARD_THRESHOLD:
        reward_below_threshold_count += 1
        if reward_below_threshold_count >= 2:
            print("Average reward below threshold twice in a row")
            write_json(f"{DIR}/{run_name}-{key}-reward_threshold.json", {"below_threshold": True})
            reward_below_threshold_count = 0


if use_wandb:
    wandb.finish()
