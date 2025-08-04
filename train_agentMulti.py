import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import wandb
import torch
import numpy as np
from agent.SAC import SACAgent
from env.envMulti import FogEnvMulti as FogEnv
from template import Service, loadFogNetwork, returnFogNetwork
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
    max_episodes = 30000
    max_steps = 30000  # avg=77.83
    batch_size = 64
    ckpt_dir = Path("models")
    ckpt_freq = 100  # save every n episodes
    # exploration_rate = 0.5
    use_llm = True  # False  # Use LLM for action guidance

    def get_config():
        return {
            "max_episodes": TrainerConfig.max_episodes,
            "max_steps": TrainerConfig.max_steps,
            "batch_size": TrainerConfig.batch_size,
            "ckpt_dir": str(TrainerConfig.ckpt_dir),
            "ckpt_freq": TrainerConfig.ckpt_freq,
            "use_llm": TrainerConfig.use_llm,
        }


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
    parser.add_argument("-w", "--wandb", action="store_true", default=False, help="Log to Weights & Biases")
    parser.add_argument("-n", "--name", type=str, default=None, help="Run name (defaults to timestamp)")
    parser.add_argument("-e", "--encoder", default="default", type=str, help="Encoder type: default, gnn")
    parser.add_argument("-l", "--log-context", action="store_true", default=False, help="Log context to JSON file")
    return parser.parse_args()


def build_run_name(custom_name: str | None) -> str:
    now = datetime.now().strftime("%m%d_%H%M%S")
    if custom_name:
        return f"{custom_name}-{now}"
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


def table(live: Live, step_logs, episode_logs):
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


def get_action(
    env: FogEnv,
    agent: SACAgent,
    agent_idx: int,
    global_step: int,
    state: Dict,
    svc_id: int,
    closest_agent_idx: int,
    method: str = "default",
    eval: bool = False,
):
    """
    Decide the agent's action, optionally guiding it based on LLM recommendations.

    Returns:
        action_index (int): Selected action index after guidance.
        action_index_b4 (int): Action index before applying any guidance (same as action_index if no guidance applied).
        action (torch.Tensor): Action tensor after applying guidance.
    """
    # epsilon = TrainerConfig.exploration_rate(1 - global_step / (TrainerConfig.max_episodes * 78))
    # epsilon =  TrainerConfig.exploration_rate / (1 + 1e-3 * global_step) # e0 / 2 @ 1000
    if method not in ["default"]:
        raise NotImplementedError(f"Action guiding method '{method}' is not implemented.")

    # Exploitation step
    action: torch.Tensor = agent.select_action(state, deterministic=eval)  # Tensor of shape (N_NODES_AGENT,)
    action = torch.softmax(action, dim=-1) if not torch.allclose(action.sum(), torch.tensor(1.0)) else action

    try:
        action_index_b4 = torch.multinomial(action, num_samples=1).item()
    except RuntimeError as e:
        print(f"Error in multinomial sampling: {e}")
        action_index_b4 = torch.argmax(action, dim=-1).item()
        action_index = action_index_b4
        return action_index, action_index_b4, action

    # Get LLM suggestion
    llm_action = (
        env.get_llm_action(svc_id=svc_id, ms_idx=state["next_ms_idx"], closest_agent_idx=closest_agent_idx)
        if TrainerConfig.use_llm and True  # not eval
        else -1
    )

    if llm_action == -1 or N2A[llm_action] != agent_idx:
        # No guidance needed or LLM points to a different agent
        action_index = action_index_b4
    else:
        # Guidance logic
        agent_nodes = A2N[agent_idx]
        llm_node_idx = agent_nodes.index(llm_action)

        norm_bias = 0.1
        llm_action_probs = [norm_bias] * len(agent_nodes)
        llm_action_probs[llm_node_idx] = 1.0
        llm_action_probs = torch.tensor(llm_action_probs)
        llm_action_probs = torch.softmax(llm_action_probs, dim=-1)

        # Apply guidance
        action = torch.softmax(action * llm_action_probs, dim=-1)
        action_index = torch.multinomial(action, num_samples=1).item()

    return action_index, action_index_b4, action


def log_context(path, info_, svc_info, agent_info, reward_info):
    svc_id, ms_idx, loc, closest_agent_idx = svc_info
    agent_idx, node_idx, a_step = agent_info
    reward, p_latency, p_lb, p_ru = reward_info

    if path and "latency" in info_ and info_["latency"] != np.inf:
        # TODO: Explore more context
        logger = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            # req
            "service": {
                "svc_id": svc_id,
                "ms_idx": ms_idx,
                "requested_loc": (round(loc[0], 2), round(loc[1], 2)),
                "closest_agent_idx": closest_agent_idx,
            },
            # agent
            "deployment": {
                "agent_idx": agent_idx,
                "node_idx": node_idx,
                "steps_taken": a_step,
            },
            # result
            "result": {
                "reward": round(reward, 2),
                f"latency/{svc_id}": p_latency,
                f"lb/{svc_id}": p_lb,
                f"ru/{svc_id}": p_ru,
                "plan_feedback": info_["plan_feedback"],
                "weights_feedback": info_["weights_feedback"],
                "weights_old": info_["weights_old"],
            },
        }

        append_json(path, logger, indent=1, max_length=1000, atomic=True)
    else:
        # TODO: collect the 0 rewarded decisions systematically!
        pass


def build_step_log(info_, idx, metrics: tuple[float], steps, value):
    """Build a step log dictionary with the provided information.
    Args:
        info_ (dict): info_
        idx (tuple): svc_id, ms_idx, agent_idx, action_index
        metrics (tuple): p_latency, p_lb, p_ru, alpha, beta, gamma
        steps (tuple): a_step, ep_step, global_step
        value (tuple): reward, ep_reward, loss
    """
    svc_id, ms_idx, agent_idx, action_index = idx
    p_latency, p_lb, p_ru, alpha, beta, gamma = metrics
    reward, ep_reward, loss = value
    a_step, ep_step, global_step, episode = steps

    metrics = {
        f"latency/{svc_id}": p_latency,
        f"lb/{svc_id}": p_lb,
        f"ru/{svc_id}": p_ru,
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
        f"reward/{svc_id}": round(reward, 2),
        f"max_reward": round(info_["max_reward"], 2),
        f"best_node/{svc_id}": info_["best_node"],
        "ER_": round(ep_reward, 2),
        "loss/a": loss["al"] if loss else 0,
        "loss/cl_1": loss["cl_1"] if loss else 0,
        "loss/cl_2": loss["cl_2"] if loss else 0,
        "ms_idx": ms_idx,
        "action": action_index,
        "ep_step": ep_step,
        "global_step": global_step,
        "episode": episode,
        "agent_idx": agent_idx,
        "plan_fb": info_["plan_feedback"],
        "weights_fb": info_["weights_feedback"],
    }

    return step_log


def build_episode_log(
    info, episode, ep_reward, ep_step, elapse, req_idx, eep: tuple, eep_step, eelapsed, explored_nodes: set
):
    eep_reward, eep_max_reward = eep
    return {
        "episode": episode,
        "ER_adj": ep_reward / (ep_step**1),
        "ep_step": ep_step,
        "explored_nodes": explored_nodes,
        **info,
        "time": round(elapse, 2),
        "completed_req": req_idx + 1,
        "eval/time": round(eelapsed, 2),
        "eval/ER": eep_reward / (eep_step**1),
        "eval/ER_max": eep_max_reward / (eep_step**1),
        "eval/ep_step": eep_step,
    }


def agent_step(
    live: Live,
    env: FogEnv,
    agent: SACAgent,
    agent_idx: int,
    req,
    a_step: int,
    max_agent_steps: int,
    metrics: tuple,
    global_step: int,
    episode: int,
    ep_step: int,
    ep_reward: float,
    ep_max_reward: float,
    to_wandb=True,
    context_path: Path = None,
    eval: bool = False,
) -> tuple[bool, bool, tuple[float, float], tuple[float]]:
    svc, (loc, closest_agent_idx) = req
    svc_id = svc.id
    state = env.get_state(agent_idx)
    action_index, action_index_b4, action = get_action(
        env, agent, agent_idx, global_step, state, svc_id, closest_agent_idx, eval=eval
    )

    next_state, reward, done, _, info_ = env.step(
        agent_idx,
        action_index,
        action_index_b4=action_index_b4,
        last_step=ep_step >= TrainerConfig.max_steps,  # a_step == max_agent_steps - 1,
        a_step=a_step,
    )
    ep_reward += reward
    ep_max_reward += info_["max_reward"]
    if "latency" in info_:
        metrics = (info_["latency"], info_["lb"], info_["ru"], info_["alpha"], info_["beta"], info_["gamma"])

    # NOTE: context will be logged for eval also (not just test)
    p_latency, p_lb, p_ru, _, _, _ = metrics
    log_context(
        context_path,
        info_,
        (svc_id, state["next_ms_idx"], loc, closest_agent_idx),
        (agent_idx, env.get_node(action_index, agent_idx), a_step),
        (reward, p_latency, p_lb, p_ru),
    )
    if eval:
        return info_["sp"], done, (reward, ep_reward, ep_max_reward), metrics, action_index

    agent.replay_buffer.push(state, action, reward, next_state, done)

    loss = None
    if agent.replay_buffer.ready(TrainerConfig.batch_size):
        loss = agent.update()

    ms_idx = env.state["next_service_idx"] * N_MS + env.state["next_ms_idx"]
    step_log = build_step_log(
        info_,
        (svc_id, ms_idx, agent_idx, action_index),
        metrics,
        (a_step, ep_step, global_step, episode),
        (reward, ep_reward, loss),
    )
    step_logs.append(step_log)
    if to_wandb:
        wandb.log(step_log, step=global_step)

    table(live, step_logs, episode_logs)

    return info_["sp"], done, (reward, ep_reward, ep_max_reward), metrics, action_index


def main():
    args = parse_args()
    TrainerConfig.use_llm = args.log_context

    print(f"Arguments: {args}")
    print(f"Trainer Config: {TrainerConfig.__dict__}")
    print(f"Agent Config: {AgentConfig.get_config()}")
    print("==" * 20)

    run_name = build_run_name(args.name)
    CONTEXT_LOG_PATH = Path(f"logs/context-{run_name}.json") if args.log_context else None

    print(f"Training SAC agent...  Run name = {run_name}")
    print(f"Context log path: {CONTEXT_LOG_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = loadFogNetwork("pkl/network-connected-svcs-nodes.pkl")
    env = FogEnv(run_name=run_name, network=network, vis_placement=False)
    _ = env.reset()
    env.get_state_info()
    print(f"Arrival Rate: ~{env.arrival_rate} reqs/sec")

    eenv = FogEnv(run_name=run_name, network=network, vis_placement=False, eval=True)
    _ = eenv.reset()

    agents = {
        i: SACAgent(
            env.get_obs_space(i),
            env.get_act_space(i).shape[0],
            device=device,
            encoder=args.encoder,
            **AgentConfig.get_config(),
        )
        for i in range(N_AGENTS)
    }

    # --- Checkpoint resume (if any) ---
    episode_start, global_step = maybe_load_checkpoint(agents, run_name)

    # --- WandB init ---
    if args.wandb:

        WANDB_DIR = "/dpc/kunf0097/.cache/wandb"
        wandb.init(
            dir=WANDB_DIR,
            project="placemeent",
            entity="my-ku-org",
            name=f"{run_name}",
        )
        wandb.config.update({"run_name": run_name, **AgentConfig.get_config(), **TrainerConfig.get_config()})

    # --- Training loop ---
    episode = episode_start
    # p_latency, p_lb, p_ru = np.inf, np.inf, np.inf
    # alpha, beta, gamma = 0.333, 0.333, 0.333  # latency, load balancing, ru weigths
    metrics = (np.inf, np.inf, np.inf, 0.333, 0.333, 0.333)
    with Live(refresh_per_second=4) as live:
        while episode < TrainerConfig.max_episodes:
            ep_reward = 0
            ep_step = 0
            info = env.fetch_requests(verbose=False)
            explored_nodes = set()
            start = time.time()
            if info["num_reqs"] == 0:
                time.sleep(0.5)
                continue

            done = False
            # svcs_reqd gets reset every fetch_requests so we can iterate over it
            for req_idx, req in enumerate(env.svcs_reqd):
                svc_placed = False
                if done:
                    break

                _, (_, closest_agent_idx) = req
                agent_order = env.get_agent_order(closest_agent_idx)  # [0, 1, 2]
                for agent_idx in agent_order:
                    if svc_placed or done:
                        # if done there should be no more svc to place (that's why the assert)
                        break

                    agent = agents[agent_idx]
                    max_agent_steps = TrainerConfig.max_steps // len(agent_order)
                    for a_step in range(max_agent_steps):
                        svc_placed, done, (_, ep_reward, _), metrics, action_index = agent_step(
                            live,
                            env,
                            agent,
                            agent_idx,
                            req,
                            a_step,
                            max_agent_steps,
                            metrics,
                            global_step,
                            episode,
                            ep_step,
                            ep_reward,
                            ep_max_reward=0,  # not used
                            to_wandb=args.wandb,
                            context_path=CONTEXT_LOG_PATH,
                        )
                        explored_nodes.add(A2N[agent_idx][action_index])

                        global_step += 1
                        ep_step += 1

                        if svc_placed or done:
                            break

            end = time.time()
            elapsed = end - start

            # === Eval ===
            e_start = time.time()
            eep_reward = 0.0
            eep_max_reward = 0.0
            eep_step = 0
            _ = eenv.reset()
            _ = eenv.fetch_requests(verbose=False)
            emetrics = (np.inf, np.inf, np.inf, 0.333, 0.333, 0.333)
            for req in eenv.svcs_reqd:
                _, (_, closest_agent_idx) = req
                done = False
                svc_placed = False
                agent_order = eenv.get_agent_order(closest_agent_idx)
                for agent_idx in agent_order:
                    if svc_placed or done:
                        break
                    agent = agents[agent_idx]
                    max_agent_steps = TrainerConfig.max_steps // len(agent_order)
                    for a_step in range(max_agent_steps):
                        svc_placed, done, (_, eep_reward, eep_max_reward), emetrics, _ = agent_step(
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
                            to_wandb=args.wandb,
                            context_path=CONTEXT_LOG_PATH,
                            eval=True,
                        )

                        eep_step += 1

                        if svc_placed or done:
                            break
            e_end = time.time()
            eelapsed = e_end - e_start

            # --- episode end logging ---
            episode_log = build_episode_log(
                info,
                episode,
                ep_reward,
                ep_step,
                elapsed,
                req_idx,
                (eep_reward, eep_max_reward),
                eep_step,
                eelapsed,
                explored_nodes,
            )
            episode_logs.append(episode_log)
            if args.wandb:
                wandb.log(episode_log, step=global_step)

            table(live, step_logs, episode_logs)
            if episode % TrainerConfig.ckpt_freq == 0 or episode == TrainerConfig.max_episodes - 1:
                save_checkpoint(agents, run_name, episode + 1, global_step)

            episode += 1

    env.close()
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
