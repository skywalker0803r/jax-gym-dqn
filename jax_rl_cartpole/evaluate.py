"""Evaluate a saved DQN checkpoint without epsilon exploration."""

import argparse
import pickle

import gymnasium as gym
import jax
import jax.numpy as jnp

from .config import Config
from .models import QNetwork


def evaluate(config: Config, episodes: int = 10, render: bool = False) -> list[float]:
    env = gym.make(config.env_id, render_mode="human" if render else None)
    network = QNetwork(action_dim=env.action_space.n, hidden_dim=config.hidden_dim)
    with open(f"{config.checkpoint_dir}/dqn_params.pkl", "rb") as file:
        params = pickle.load(file)
    scores = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        score = 0.0
        for _ in range(config.max_steps):
            q_values = network.apply(params, jnp.asarray(observation[None]))
            action = int(jax.device_get(jnp.argmax(q_values)))
            observation, reward, terminated, truncated, _ = env.step(action)
            score += reward
            if terminated or truncated:
                break
        scores.append(score)
    env.close()
    print(f"scores={scores} average={sum(scores) / len(scores):.1f}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(Config(), episodes=args.episodes, render=args.render)
