import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import numpy as np
from agent.SAC import SACAgent
from env.envMulti import FogEnvMulti as FogEnv
from template import loadFogNetwork, returnFogNetwork
from utils import append_json, seed_it
from env.constants import N_AGENTS, N_MS

from collections import deque
from rich.live import Live
from rich.table import Table
from rich.columns import Columns

seed_it(646)

step_logs = deque(maxlen=10)
episode_logs = deque(maxlen=10)


class TrainerConfig:
    max_episodes = 2000
    max_steps = 2000
    batch_size = 64
    ckpt_dir = Path("models")
    ckpt_freq = 100  # save every n episodes
    exploration_rate = 0.1


class AgentConfig:
    @staticmethod
    def get_config():
        return {
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "buffer_size": 1e5,
            "batch_size": 64,
            "lr": 3e-4,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--name", type=str, default=None, help="Run name (defaults to timestamp)")
    parser.add_argument("--id", type=str, default=None, help="Run ID for Weights & Biases")
    return parser.parse_args()


def build_run_name(custom_name: str | None) -> str:
    if custom_name:
        return f"{custom_name}"
    now = datetime.now().strftime("%m%d_%H%M%S")
    return now


def get_ckpt_paths(run_name: str):
    model_path = TrainerConfig.ckpt_dir / run_name / "agent"
    trainer_path = TrainerConfig.ckpt_dir / run_name / "trainer_state"
    return model_path, trainer_path


def maybe_load_checkpoint(agent: SACAgent, run_name: str):
    """Load checkpoint if it exists and return starting episode and global_step."""
    # TODO:
    # model_ckpt, trainer_ckpt = get_ckpt_paths(run_name)
    # if model_ckpt.exists() and trainer_ckpt.exists():
    #     print(f"[Checkpoint] Found checkpoint for run '{run_name}', resuming training…")
    #     agent.load_model(model_ckpt)
    #     state_dict = torch.load(trainer_ckpt, map_location="cpu")
    #     return state_dict["episode"], state_dict["global_step"]
    return 0, 0


def save_checkpoint(agents: Dict[int, SACAgent], run_name: str, episode: int, global_step: int):
    model_ckpt, trainer_ckpt = get_ckpt_paths(run_name)
    model_ckpt.mkdir(parents=True, exist_ok=True)
    trainer_ckpt.mkdir(parents=True, exist_ok=True)
    for agent_idx, agent in agents.items():
        agent.save_model(model_ckpt / f"{agent_idx}.pt")
        torch.save({"episode": episode, "global_step": global_step // N_AGENTS}, trainer_ckpt / f"{agent_idx}.pt")


def table(live: Live):
    step_table = Table(title="Step Logs")
    for key in step_logs[0]:
        step_table.add_column(str(key))
    for log in step_logs:
        step_table.add_row(*[str(log.get(k, "")) for k in step_logs[0]])

    ep_table = Table(title="Episode Logs")
    for key in episode_logs[0] if episode_logs else ["episode"]:
        ep_table.add_column(str(key))
    for log in episode_logs:
        ep_table.add_row(*[str(log.get(k, "")) for k in episode_logs[0]])

    live.update(Columns([ep_table, step_table]))


def main():
    args = parse_args()

    print(f"Arguments: {args}")
    print(f"Trainer Config: {TrainerConfig.__dict__}")
    print(f"Agent Config: {AgentConfig.get_config()}")
    print("==" * 20)

    run_name = build_run_name(args.name)
    CONTEXT_LOG_PATH = Path(f"logs/context-{run_name}.json")

    print(f"Training SAC agent...  Run name = {run_name}")
    print(f"Context log path: {CONTEXT_LOG_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = loadFogNetwork("pkl/network-connected-svcs.pkl")
    env = FogEnv(network=network, vis_placement=False)
    _ = env.reset()
    env.get_state_info()

    agents = {
        i: SACAgent(env.get_obs_space(i), env.get_act_space(i).shape[0], device=device, **AgentConfig.get_config())
        for i in range(N_AGENTS)
    }

    # --- Checkpoint resume (if any) ---
    # always returns 0,0 agents chkpt not saved yet
    episode_start, global_step = maybe_load_checkpoint(agents, run_name)

    # --- WandB init ---
    if args.wandb:
        import wandb

        wandb.init(
            project="placemeent",
            entity="my-ku-org",
            name=f"sac--{run_name}",
            # id=args.id,
            # resume="allow",  # will resume if run id exists, else create new
        )
        wandb.config.update({"run_name": run_name, **AgentConfig.get_config()})

    # --- Training loop ---
    episode = episode_start
    p_latency, p_lb, p_ru = np.inf, np.inf, np.inf
    alpha, beta, gamma = 0.333, 0.333, 0.333  # latency, load balancing, ru weigths
    with Live(refresh_per_second=4) as live:
        while episode < TrainerConfig.max_episodes:
            ep_reward = 0
            ep_step = 0
            info = env.fetch_requests(verbose=False)
            start = time.time()
            if info["num_reqs"] == 0:
                time.sleep(0.5)
                continue

            # svcs_reqd gets reset every fetch_requests so we can iterate over it
            for svc, (loc, closest_agent_idx) in env.svcs_reqd:
                done = False
                svc_placed = False
                agent_order = env.get_agent_order(closest_agent_idx)  # [0, 1, 2]
                for agent_idx in agent_order:
                    if svc_placed or done:
                        # if done there should be no more svc to place (that's why the assert)
                        break

                    state = env.get_state(agent_idx)
                    agent = agents[agent_idx]

                    max_agent_steps = TrainerConfig.max_steps // len(agent_order)
                    for a_step in range(max_agent_steps):
                        if np.random.rand() < TrainerConfig.exploration_rate:
                            action = torch.tensor(env.get_act_space(agent_idx).sample())
                        else:
                            action = agent.select_action(state)
                        action_index = torch.argmax(action, dim=-1).item()

                        next_state, reward, done, _, info_ = env.step(
                            agent_idx, action_index, last_step=a_step == max_agent_steps - 1
                        )

                        agent.replay_buffer.push(state, action, reward, next_state, done)
                        state = next_state
                        ep_reward += reward

                        loss = None
                        if agent.replay_buffer.ready(TrainerConfig.batch_size):
                            loss = agent.update()

                        # if "link/demand" in info_ or "exp_svcs" in info_:
                        #     print(f"Info: {info_}")

                        if "latency" in info_:
                            p_latency = info_["latency"]
                            p_lb = info_["lb"]
                            p_ru = info_["ru"]
                            alpha = info_["alpha"]
                            beta = info_["beta"]
                            gamma = info_["gamma"]
                        metrics = {
                            f"latency/{svc.id}": p_latency,
                            f"lb/{svc.id}": p_lb,
                            f"ru/{svc.id}": p_ru,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                        }

                        step_log = {
                            **metrics,
                            "a_step": a_step,
                            "reward": round(reward, 2),
                            f"reward/{svc.id}": round(reward, 2),
                            "ER_": round(ep_reward, 2),
                            "loss/a": loss["al"] if loss else 0,
                            "loss/cl_1": loss["cl_1"] if loss else 0,
                            "loss/cl_2": loss["cl_2"] if loss else 0,
                            "ms_idx": env.state["next_service_idx"] * N_MS + env.state["next_ms_idx"],
                            "action": action_index,
                            "ep_step": ep_step,
                            "global_step": global_step,
                            "agent_idx": agent_idx,
                        }
                        step_logs.append(step_log)
                        if args.wandb:
                            wandb.log(step_log, step=global_step)

                        global_step += 1
                        ep_step += 1
                        table(live)

                        # only log if info_ has latency otherwise the reward is just -1
                        if False:  # "latency" in info_ and info_["latency"] != np.inf:
                            # TODO: Explore more context
                            logger = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                # req
                                "svc_id": svc.id,
                                "loc": (round(loc[0], 2), round(loc[1], 2)),
                                "closest_agent_idx": closest_agent_idx,
                                # agent
                                "agent_idx": agent_idx,
                                "node_idx": env.get_node(action_index, agent_idx),
                                "a_step": a_step,
                                # result
                                "reward": round(reward, 2),
                                **metrics,
                            }

                            append_json(CONTEXT_LOG_PATH, logger, indent=None)

                        svc_placed = info_["sp"]
                        if svc_placed:
                            break

                        if done:
                            by = 1
                            ep_reward *= 1 / ep_step**by
                            break

            # --- episode end logging ---
            end = time.time()
            episode_log = {
                "episode": episode,
                "ER": round(ep_reward, 2),
                "ep_step": ep_step,
                **info,
                # "global_step": global_step,
                "time": round(end - start, 2),
            }
            episode_logs.append(episode_log)
            if args.wandb:
                wandb.log(episode_log, step=global_step)

            table(live)
            if episode % TrainerConfig.ckpt_freq == 0 or episode == TrainerConfig.max_episodes - 1:
                save_checkpoint(agents, run_name, episode + 1, global_step)

            episode += 1

    env.close()
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
