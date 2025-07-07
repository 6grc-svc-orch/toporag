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
from env.constants import A2N, N2A, N_AGENTS, N_MS

from collections import deque
from rich.live import Live
from rich.table import Table
from rich.columns import Columns

seed_it(646)

step_logs = deque(maxlen=10)
episode_logs = deque(maxlen=10)


class TrainerConfig:
    max_episodes = 200000
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


def maybe_load_checkpoint(agents: Dict[int, SACAgent], run_name: str):
    """Load checkpoint if it exists and return starting episode and global_step."""
    # TODO:
    episode = 0
    global_step = 0
    for agent_idx, agent in agents.items():
        model_ckpt, trainer_ckpt = get_ckpt_paths(run_name)
        model_ckpt = model_ckpt / f"{agent_idx}.pt"
        trainer_ckpt = trainer_ckpt / f"{agent_idx}.pt"

        if model_ckpt.exists():
            print(f"[Checkpoint] Found checkpoint for agent {agent_idx} in run '{run_name}', resuming training…")
            agent.load_model(model_ckpt)
            if agent_idx == 0 and trainer_ckpt.exists():
                state_dict = torch.load(trainer_ckpt)
                episode = state_dict["episode"]
                global_step = state_dict["global_step"]
                print(f"Resuming from episode {episode}, global step {global_step}")
    return episode, global_step


def save_checkpoint(agents: Dict[int, SACAgent], run_name: str, episode: int, global_step: int):
    model_ckpt, trainer_ckpt = get_ckpt_paths(run_name)
    model_ckpt.mkdir(parents=True, exist_ok=True)
    trainer_ckpt.mkdir(parents=True, exist_ok=True)
    for agent_idx, agent in agents.items():
        agent.save_model(model_ckpt / f"{agent_idx}.pt")
        torch.save({"episode": episode, "global_step": global_step}, trainer_ckpt / f"{agent_idx}.pt")


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


def guide_action(action: torch.Tensor, llm_action: int, agent_idx: int, method: str = "default"):
    if llm_action == -1 or N2A[llm_action] != agent_idx:
        action_index = torch.argmax(action, dim=-1).item()
        return action_index, action_index

    if method == "default":
        agent_nodes = A2N[agent_idx]
        llm_node_idx = agent_nodes.index(llm_action)

        norm_bias = 0.1
        llm_action = [norm_bias] * len(agent_nodes)
        llm_action[llm_node_idx] = 1  # one-hot
        llm_action = torch.tensor(llm_action)
        llm_action = torch.softmax(llm_action, dim=-1)  # convert to probabilities

        action_index_b4 = torch.argmax(action, dim=-1).item()
        # TODO: b4 does not consider if action_b4 is offloaded i.e caught on prev check.
        action = action * llm_action
        action_index = torch.argmax(action, dim=-1).item()
        return action_index, action_index_b4
    else:
        raise NotImplementedError(f"Action guiding method '{method}' is not implemented.")


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
    env = FogEnv(run_name=run_name, network=network, vis_placement=False)
    _ = env.reset()
    env.get_state_info()

    agents = {
        i: SACAgent(env.get_obs_space(i), env.get_act_space(i).shape[0], device=device, **AgentConfig.get_config())
        for i in range(N_AGENTS)
    }

    # --- Checkpoint resume (if any) ---
    episode_start, global_step = maybe_load_checkpoint(agents, run_name)

    # --- WandB init ---
    if args.wandb:
        import wandb

        WANDB_DIR = "/dpc/kunf0097/.cache/wandb"
        wandb.init(
            dir=WANDB_DIR,
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
                            action_index = torch.argmax(action, dim=-1).item()
                        else:
                            action = agent.select_action(state)  # size (N_NODES_AGENT,)
                            llm_action = env.get_llm_action(
                                svc_id=svc.id, ms_idx=state["next_ms_idx"], closest_agent_idx=closest_agent_idx
                            )
                            action_index, action_index_b4 = guide_action(action, llm_action, agent_idx)

                        next_state, reward, done, _, info_ = env.step(
                            agent_idx,
                            action_index,
                            action_index_b4=action_index_b4,
                            last_step=a_step == max_agent_steps - 1,
                            a_step=a_step,
                        )

                        agent.replay_buffer.push(state, action, reward, next_state, done)
                        state = next_state
                        ep_reward += reward

                        loss = None
                        if agent.replay_buffer.ready(TrainerConfig.batch_size):
                            loss = agent.update()

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
                            "latency": p_latency,
                            "lb": p_lb,
                            "ru": p_ru,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                        }

                        step_log = {
                            **metrics,
                            "a_step": a_step,
                            "reward": round(reward, 2),
                            f"reward/{svc.id}": round(reward, 2),
                            f"max_reward": round(info_["max_reward"], 2),
                            f"best_node/{svc.id}": info_["best_node"],
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
                        if "latency" in info_ and info_["latency"] != np.inf:
                            # TODO: Explore more context
                            logger = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                # req
                                "service": {
                                    "svc_id": svc.id,
                                    "ms_idx": state["next_ms_idx"],
                                    "requested_loc": (round(loc[0], 2), round(loc[1], 2)),
                                    "closest_agent_idx": closest_agent_idx,
                                },
                                # agent
                                "deployment": {
                                    "agent_idx": agent_idx,
                                    "node_idx": env.get_node(action_index, agent_idx),
                                    "steps_taken": a_step,
                                },
                                # result
                                "result": {
                                    "reward": round(reward, 2),
                                    f"latency/{svc.id}": p_latency,
                                    f"lb/{svc.id}": p_lb,
                                    f"ru/{svc.id}": p_ru,
                                    "plan_feedback": info_["plan_feedback"],
                                    "weights_feedback": info_["weights_feedback"],
                                    "weights_old": info_["weights_old"],
                                },
                            }

                            append_json(CONTEXT_LOG_PATH, logger, indent=1, max_length=1000, atomic=True)
                        else:
                            # TODO: collect the 0 rewarded decisions systematically!

                            pass

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
