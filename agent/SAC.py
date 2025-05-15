import torch
import torch.nn.functional as F
from agent.modules import Actor, Critic
from agent.buffer import DictReplayBuffer
from gymnasium.spaces import Dict


class SACAgent:
    def __init__(
        self,
        obs_space: Dict,
        action_dim,
        device="cpu",
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_size=100000,
        batch_size=64,
        lr=3e-4,
    ):
        self.actor = Actor(obs_space, action_dim)
        self.critic1 = Critic(obs_space, action_dim)
        self.critic2 = Critic(obs_space, action_dim)
        self.critic1_target = Critic(obs_space, action_dim)
        self.critic2_target = Critic(obs_space, action_dim)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.replay_buffer = DictReplayBuffer(buffer_size)

        self.device = device

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

    def select_action(self, state):
        state = {
            k: (
                torch.tensor([v], dtype=torch.float32, device=self.device).unsqueeze(0)
                if isinstance(v, (int, float))
                else torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(
                    0
                )
            )
            for k, v in state.items()
        }
        action, _ = self.actor(state)
        return action[0].detach().cpu()

    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )

        state = {k: torch.FloatTensor(v).to(self.device) for k, v in state.items()}
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = {
            k: torch.FloatTensor(v).to(self.device) for k, v in next_state.items()
        }
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_action, log_prob = self.actor(state)
        q1 = self.critic1(state, new_action)
        q2 = self.critic2(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
