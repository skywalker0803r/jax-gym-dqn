"""Configuration and hardware-safe defaults for the CartPole experiment."""

import os
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")
os.environ.setdefault("JAX_ENABLE_X64", "false")


@dataclass(frozen=True)
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 42
    episodes: int = 200
    max_steps: int = 500
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    gamma: float = 0.99
    learning_rate: float = 1e-3
    target_update_frequency: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    hidden_dim: int = 128
    checkpoint_dir: str = "checkpoints"
    require_cuda: bool = False
