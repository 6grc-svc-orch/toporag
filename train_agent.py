import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from agent.SAC import SACAgent
from env.env import FogEnv, FogEnvState
from template import loadFogNetwork, returnFogNetwork
from utils import seed_it

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
    model_path = TrainerConfig.ckpt_dir / run_name / "agent.pt"
    trainer_path = TrainerConfig.ckpt_dir / run_name / "trainer_state.pt"
    return model_path, trainer_path


def maybe_load_checkpoint(agent: SACAgent, run_name: str):
    """Load checkpoint if it exists and return starting episode and global_step."""
    model_ckpt, trainer_ckpt = get_ckpt_paths(run_name)
    if model_ckpt.exists() and trainer_ckpt.exists():
        print(f"[Checkpoint] Found checkpoint for run '{run_name}', resuming training…")
        agent.load_model(model_ckpt)
        state_dict = torch.load(trainer_ckpt, map_location="cpu")
        return state_dict["episode"], state_dict["global_step"]
    return 0, 0


def save_checkpoint(agent: SACAgent, run_name: str, episode: int, global_step: int):
    model_ckpt, trainer_ckpt = get_ckpt_paths(run_name)
    model_ckpt.parent.mkdir(parents=True, exist_ok=True)
    agent.save_model(model_ckpt)
    torch.save({"episode": episode, "global_step": global_step}, trainer_ckpt)


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

    print(f"Training SAC agent...  Run name = {run_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = loadFogNetwork("pkl/network-connected.pkl")
    env = FogEnv(network=network, vis_placement=False)
    _ = env.reset()
    env.get_state_info()

    agent = SACAgent(env.observation_space, 10, device=device, **AgentConfig.get_config())

    # --- Checkpoint resume (if any) ---
    episode_start, global_step = maybe_load_checkpoint(agent, run_name)

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
    with Live(refresh_per_second=4) as live:
        while episode < TrainerConfig.max_episodes:
            episode_reward, done = 0, False
            state, info = env.fetch_requests(verbose=False)

            if info["num_reqs"] == 0:
                time.sleep(0.5)
                continue

            for step in range(TrainerConfig.max_steps):
                if np.random.rand() < TrainerConfig.exploration_rate:
                    action = torch.tensor(env.action_space.sample())
                else:
                    action = agent.select_action(state)
                action_index = torch.argmax(action, dim=-1).item()
                next_state, reward, done, _, info_ = env.step(
                    action_index, last_step=step == TrainerConfig.max_steps - 1
                )

                agent.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                loss = None
                if agent.replay_buffer.ready(TrainerConfig.batch_size):
                    loss = agent.update()

                step_log = {
                    "step": step,
                    "reward": round(reward, 2),
                    "ER_": round(episode_reward, 2),
                    "loss/a": loss["al"] if loss else 0,
                    "loss/cl_1": loss["cl_1"] if loss else 0,
                    "loss/cl_2": loss["cl_2"] if loss else 0,
                    "global_step": global_step,
                    "action": action_index,
                    "ms_idx": env.state["next_service_idx"] * 5 + env.state["next_ms_idx"],
                }
                step_logs.append(step_log)
                if args.wandb:
                    wandb.log(step_log)

                # Update side-by-side display during steps
                # if step % 10 == 0:
                table(live)
                if "link/demand" in info_ or "exp_svcs" in info_:
                    print(f"Info: {info_}")

                if "latency" in info_:
                    wandb.log({"global_step": global_step, "latency": info_["latency"], "lb": info_["lb"]})

                global_step += 1
                if done:
                    by = 1
                    episode_reward *= 1 / step**by
                    break

            # --- episode end logging ---
            episode_log = {
                "episode": episode,
                "ER": round(episode_reward, 2),
                "last_step": step,
                **info,
                "global_step": global_step,
            }
            episode_logs.append(episode_log)
            if args.wandb:
                wandb.log(episode_log)

            table(live)
            if episode % TrainerConfig.ckpt_freq == 0 or episode == TrainerConfig.max_episodes - 1:
                save_checkpoint(agent, run_name, episode + 1, global_step)

            episode += 1

    env.close()
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
