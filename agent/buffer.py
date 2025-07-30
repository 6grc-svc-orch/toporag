import numpy as np
import random
from collections import deque
import copy


class DictReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        # Deepcopy to ensure no external mutation
        transition = (
            copy.deepcopy(state),
            np.array(action, dtype=np.float32),
            np.float32(reward),
            copy.deepcopy(next_state),
            np.float32(done),
        )
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Merge dicts into batched dicts
        batched_state = self._stack_dicts(states)
        batched_next_state = self._stack_dicts(next_states)

        actions = np.stack(actions)
        rewards = np.stack(rewards)
        dones = np.stack(dones)

        return batched_state, actions, rewards, batched_next_state, dones

    def _stack_dicts(self, list_of_dicts):
        keys = list_of_dicts[0].keys()
        stacked = {key: np.stack([d[key] for d in list_of_dicts]) for key in keys if key != "svc_id"}
        if "svc_id" in keys:
            svc_ids = np.array([d["svc_id"] for d in list_of_dicts]).reshape(-1, 1)
            stacked["svc_id"] = svc_ids

        return stacked

    def ready(self, batch_size: int):
        return len(self.buffer) >= batch_size

    def __len__(self):
        return len(self.buffer)
