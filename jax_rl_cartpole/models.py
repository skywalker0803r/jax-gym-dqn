"""Flax neural network definitions."""

import flax.linen as nn


class QNetwork(nn.Module):
    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(features=self.action_dim)(x)
