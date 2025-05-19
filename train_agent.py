import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from agent.SAC import SACAgent
from env import FogEnv, FogEnvState
from template import returnFogNetwork
from utils import seed_it

seed_it(646)


class TrainerConfig:
    max_episodes = 2000
    max_steps = 1000
    batch_size = 64
    ckpt_dir = Path("models")
    ckpt_freq = 100  # save every n episodes


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
    parser = argparse.ArgumentParser(
        description="Train an SAC agent with robust checkpointing and optional Weights & Biases logging."
    )
    parser.add_argument("-w", "--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--name", type=str, default=None, help="Run name (defaults to timestamp)")
    parser.add_argument("--id", type=str, default=None, help="Run ID for Weights & Biases")
    return parser.parse_args()


def build_run_name(custom_name: str | None) -> str:
    now = datetime.now().strftime("%m%d_%H%M%S")
    if custom_name:
        return f"{now}-{custom_name}"
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


def main():
    args = parse_args()
    run_name = build_run_name(args.name)

    print(f"Training SAC agent...  Run name = {run_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = returnFogNetwork()
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
            id=args.id,
            resume="allow",  # will resume if run id exists, else create new
        )
        wandb.config.update({"run_name": run_name, **AgentConfig.get_config()})

    # --- Training loop ---
    for episode in range(episode_start, TrainerConfig.max_episodes):
        episode_reward, done = 0, False
        state, info = env.fetch_requests(verbose=False)

        if info["num_reqs"] == 0:
            time.sleep(0.5)
            continue

        for step in range(TrainerConfig.max_steps):
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

            if args.wandb:
                wandb.log(
                    {
                        "reward": reward,
                        "ER_": episode_reward,
                        "loss/a": loss["al"] if loss else 0,
                        "loss/cl_1": loss["cl_1"] if loss else 0,
                        "loss/cl_2": loss["cl_2"] if loss else 0,
                        "global_step": global_step,
                        "action": action_index,
                    }
                )
            global_step += 1

            print(
                f"\t[{done}][Step {step}/{TrainerConfig.max_steps}]: a_idx: {action_index} reward: {reward:.2f}: n_ms: {env.state['next_ms_idx']}  n_s: {env.state['next_service_idx']} loss: {loss} {info_}",
                end="\r",
            )
            if done:
                by = 0.1
                episode_reward *= 1 / step**by
                break

        # --- episode end logging ---
        if args.wandb:
            wandb.log(
                {
                    "ER": episode_reward,
                    "episode": episode,
                    "last_step": step,
                    **info,
                    "global_step": global_step,
                }
            )

        print(
            f"[Episode {episode}/{TrainerConfig.max_episodes}]: Info: {info}, Steps: {step+1}, EpisodeR: {episode_reward:.2f}"
        )

        # --- checkpoint save ---
        if episode % TrainerConfig.ckpt_freq == 0 or episode == TrainerConfig.max_episodes - 1:
            save_checkpoint(agent, run_name, episode + 1, global_step)  # +1 so next start is correct

    env.close()
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
