import sys
import time
from datetime import datetime

import torch
from agent.SAC import SACAgent
from env import FogEnv, FogEnvState
from template import returnFogNetwork
from utils import seed_it

seed_it(646)

now = datetime.now().strftime("%m%d_%H%M%S")
AGENT = "sac"
run_name = f"{now}-{sys.argv[2]}" if len(sys.argv) > 2 else now
use_wandb = False


def main():
    print(f"Training {AGENT} agent...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = returnFogNetwork()
    env = FogEnv(network=network, vis_placement=False)
    _ = env.reset()
    env.get_state_info()

    agent = SACAgent(env.observation_space, 10, device=device)

    max_episodes = 200
    max_steps = 1000
    learning_start = 1000
    batch_size = 64
    print(f"Max episodes: {max_episodes}, Learning start: {learning_start}")

    if use_wandb:
        import wandb

        wandb.init(
            project="placemeent", entity="my-ku-org", name=f"{AGENT}--{run_name}"
        )

    episode = 0
    while episode < max_episodes:
        episode_reward, done = 0, False
        state, info = env.fetch_requests(verbose=False)

        if info["num_reqs"] == 0:
            time.sleep(2)
            print("Let me sleep for 2 secs...")
            continue

        for step in range(max_steps):
            action: torch.Tensor = agent.select_action(state)
            # NOTE: output is a 10d tensor arg max the action
            action_index = torch.argmax(action, dim=-1).item()
            next_state, reward, done, _, info_ = env.step(
                action_index, last_step=step == max_steps - 1
            )

            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if agent.replay_buffer.ready(batch_size):
                agent.update()

            print(
                f"\t[{done}][Step {step}/{max_steps}]: action: {action_index} reward: {reward:.2f}: n_ms: {env.state['next_ms_idx']}  n_s: {env.state['next_service_idx']}",
                end="\r",
            )
            if done:
                by = 0.1
                episode_reward *= 1 / step**by
                break

        print(f"[Episode {episode}/{max_episodes}]: Info: {info}, Steps: {step+1}")
        episode += 1

    env.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
