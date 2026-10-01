"""Fixed-size, NumPy-backed vectorized experience replay."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple[int, ...], seed: int = 0):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = float(done)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        if self.size < batch_size:
            raise ValueError("Not enough transitions in replay buffer")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size
