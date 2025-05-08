import sys
import random
import time
import numpy as np
import torch

# import wandb
from tqdm import tqdm

from env import FogEnv
from template import returnFogNetwork

seed = 646
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
from datetime import datetime

now = datetime.now().strftime("%m%d_%H%M%S")
# assert len(sys.argv) > 2, "Agent name? Run name?"
AGENT = "run"  # sys.argv[1]
run_name = f"{now}-{sys.argv[2]}" if len(sys.argv) > 2 else now
use_wandb = False


def donothing(state):
    """
    Dummy function to simulate agent action.
    """
    pass


def main():
    print(f"Training {AGENT} agent...")

    network = returnFogNetwork()
    env = FogEnv(network=network, vis_reqs=True, vis_placement=True)
    env.reset()

    # Hyperparameters
    max_episodes = 6_000
    max_steps = 1000
    learning_start = 0  # change it to fill up the replay buffer
    print(f"Max episodes: {max_episodes}, Learning start: {learning_start}")

    if use_wandb:
        wandb.init(
            project="placemeent", entity="my-ku-org", name=f"{AGENT}--{run_name}"
        )

    interval = 1
    # time.sleep(100)
    for episode in range(max_episodes):
        episode_reward, done = 0, False
        state, info = env.get_new_requests(
            verbose=False
        )  # request new services to be deployed
        # time.sleep(interval)

        for step in range(max_steps):
            donothing(state)  # get the action from the agent
            action = env.action_space.sample()
            next_state, reward, done, _, info_ = env.step(
                action, last_step=step == max_steps - 1
            )
            state = next_state

            episode_reward += reward
            step += 1

            if done:
                by = 0.1
                episode_reward *= 1 / step**by
                break

            # print(f"\t[Step {step}/{max_steps}]: Step reward: {reward}: {env.state['next_ms_idx']}  {env.state['next_service_idx']}")
        print(
            f"[Episode {episode}/{max_episodes}]: Episode reward: {episode_reward}, Steps: {step}, Info: {info}"
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
