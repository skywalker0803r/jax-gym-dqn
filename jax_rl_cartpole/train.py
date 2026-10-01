"""Train a DQN agent on Gymnasium CartPole-v1."""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.60")

import argparse
import pickle
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from .agent import make_optimizer, train_step
from .config import Config
from .models import QNetwork
from .replay_buffer import ReplayBuffer


def epsilon_by_step(step: int, config: Config) -> float:
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def train(config: Config) -> list[float]:
    if config.require_cuda and jax.default_backend() != "gpu":
        raise RuntimeError(
            f"CUDA backend required, but JAX selected {jax.default_backend()}. "
            "Install the CUDA JAX wheel in WSL2/Linux and retry."
        )
    env = gym.make(config.env_id)
    observation, _ = env.reset(seed=config.seed)
    network = QNetwork(action_dim=env.action_space.n, hidden_dim=config.hidden_dim)
    params = network.init(jax.random.PRNGKey(config.seed), jnp.asarray(observation[None]))
    target_params = params
    optimizer = make_optimizer(config.learning_rate)
    opt_state = optimizer.init(params)
    buffer = ReplayBuffer(config.buffer_size, env.observation_space.shape, config.seed)
    rng = np.random.default_rng(config.seed)
    rewards = []
    global_step = 0

    for episode in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        episode_reward = 0.0
        for _ in range(config.max_steps):
            epsilon = epsilon_by_step(global_step, config)
            if rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                q_values = network.apply(params, jnp.asarray(observation[None]))
                action = int(jax.device_get(jnp.argmax(q_values)))
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(observation, action, reward, next_observation, terminated)
            observation = next_observation
            episode_reward += reward
            global_step += 1

            if len(buffer) >= max(config.batch_size, config.learning_starts) and global_step % config.train_frequency == 0:
                batch = tuple(jnp.asarray(value) for value in buffer.sample(config.batch_size))
                params, opt_state, _ = train_step(
                    params, target_params, opt_state, batch, config.gamma, network.apply, optimizer.update
                )
            if global_step % config.target_update_frequency == 0:
                target_params = params
            if done:
                break
        rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            print(f"episode={episode + 1:03d} reward={episode_reward:.0f} average10={np.mean(rewards[-10:]):.1f}")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "dqn_params.pkl").open("wb") as file:
        pickle.dump(jax.device_get(params), file)
    env.close()
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    train(Config(episodes=args.episodes, require_cuda=args.require_cuda))
