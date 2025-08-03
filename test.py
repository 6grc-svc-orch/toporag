# this is to evaluate the trained multi-agent model
import argparse
from itertools import cycle
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
import wandb
import torch
import numpy as np
from agent.SAC import SACAgent
from env.envMulti import FogEnvMulti as FogEnv
from template import Service, loadFogNetwork, returnFogNetwork
from utils import append_json, seed_it
from env.constants import A2N, N2A, N_AGENTS, N_MS
from train_agentMulti import AgentConfig, agent_step, maybe_load_checkpoint
from collections import deque
from pprint import pprint


class TestConfig:
    max_steps = 30000
    use_llm = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test the multi-agent model")
    parser.add_argument("-r", "--run_name", type=str, default="0729_085632", help="Name of the run to test")
    parser.add_argument("-n", "--num_reqs", type=int, default=15, help="Number of requests to test with")
    parser.add_argument("-e", "--encoder", type=str, default="default", help="Encoder type for the agent")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name
    num_reqs = args.num_reqs
    encoder = args.encoder

    from rich import print

    print(f"[green]{args}[/green]")

    eval = True
    global_step = -1
    episodes = 20
    live = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = loadFogNetwork("pkl/network-connected-svcs-nodes.pkl")
    eenv = FogEnv(run_name=run_name, network=network, eval=eval)
    _ = eenv.reset()

    # build agents
    agents = {
        i: SACAgent(
            eenv.get_obs_space(i),
            eenv.get_act_space(i).shape[0],
            device=device,
            encoder=encoder,
            **AgentConfig.get_config(),
            # TODO:should use from the wandb config of the run,
        )
        for i in range(N_AGENTS)
    }

    # load agents
    _, _ = maybe_load_checkpoint(agents, run_name)
    print(f"Arrival Rate: ~{num_reqs} reqs/episode")
    _ = eenv.reset()

    # Start evaluation
    summaries = []
    for episode in tqdm(range(episodes), total=episodes):
        inf = eenv.fetch_requests(verbose=False, num_reqs=num_reqs)

        e_start = time.time()
        eep_reward, eep_max_reward, eep_step = 0.0, 0.0, 0
        emetrics = (np.inf, np.inf, np.inf, 0.333, 0.333, 0.333)

        done = False
        emetricss = []
        ers = []  # evaluation rewards

        for x, req in enumerate(eenv.svcs_reqd):
            _, (_, closest_agent_idx) = req
            agent_order = eenv.get_agent_order(closest_agent_idx)
            for agent_idx in agent_order:
                svc_placed = False
                if done:
                    break
                agent = agents[agent_idx]
                max_agent_steps = TestConfig.max_steps // N_AGENTS
                for a_step in range(max_agent_steps):
                    aa = (eenv.state["next_service_idx"], eenv.state["next_ms_idx"])

                    svc_placed, done, (er, eep_reward, eep_max_reward), emetrics, _ = agent_step(
                        live,
                        eenv,
                        agent,
                        agent_idx,
                        req,
                        a_step,
                        max_agent_steps,
                        emetrics,
                        global_step,
                        episode,
                        ep_step=eep_step,
                        ep_reward=eep_reward,
                        ep_max_reward=eep_max_reward,
                        to_wandb=False,
                        context_path=None,
                        eval=eval,
                    )
                    emetricss.append(aa + emetrics)
                    ers.append(er)

                    eep_step += 1

                    if svc_placed or done:
                        break
        e_end = time.time()
        eelapsed = e_end - e_start

        # collect

        # placement_status = eenv.state["placement_status"]
        # node_sums = np.sum(placement_status, axis=0)
        # print(f"Sum across nodes (axis=0): {node_sums}")
        # print(f"Total placement status: {np.sum(placement_status)}")

        # extract the last record of each service
        last_record = {}
        for em in emetricss:
            svc_id = em[0]
            if svc_id not in last_record:
                last_record[svc_id] = em
            else:
                if em[1] >= last_record[svc_id][1]:
                    last_record[svc_id] = em

        ls = [em[2] for em in last_record.values()]
        lb_s = [em[3] for em in last_record.values()]
        ru_s = [em[4] for em in last_record.values()]

        m_l = round(np.mean(ls), 4)
        m_lb = round(np.mean(lb_s), 4)
        m_ru = round(np.mean(ru_s), 4)
        s_l = round(np.std(ls), 4)
        s_lb = round(np.std(lb_s), 4)
        s_ru = round(np.std(ru_s), 4)
        sla_violation = sum(1 for r in ers if r == 0)

        summary = {
            "run_name": run_name,
            "eval/time": round(eelapsed, 2),
            "eval/ER": float(eep_reward) / (eep_step**1),
            "eval/ER_max": float(eep_max_reward.item()) / (eep_step**1),
            "eval/ep_step": eep_step,
            "max_step": TestConfig.max_steps,
            "ep_step": eep_step,
            "latency": [m_l, s_l],
            "lb": [m_lb, s_lb],
            "ru": [m_ru, s_ru],
            "sla_violation": sla_violation,
            "last_deployed_svc": list(last_record.values())[-1][0] + 1,
            "num_reqs": inf["num_reqs"],
        }
        summaries.append(summary)
    eenv.close()

    summary_mean = {}
    for key in summaries[0].keys():
        if isinstance(summaries[0][key], list):
            summary_mean[key] = [np.mean([s[key][0] for s in summaries]), np.mean([s[key][1] for s in summaries])]
        elif isinstance(summaries[0][key], (int, float, np.number)):
            summary_mean[key] = np.mean([s[key] for s in summaries])
        else:
            summary_mean[key] = summaries[0][key]

    # Print final metrics

    pprint(summary_mean)
    # save
    with open(f"out/{run_name}-summary-{num_reqs}.json", "w") as f:
        import json

        json.dump(summary_mean, f, indent=4)


if __name__ == "__main__":
    main()
