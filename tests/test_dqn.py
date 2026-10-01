import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jax_rl_cartpole.agent import make_optimizer, train_step
from jax_rl_cartpole.models import QNetwork
from jax_rl_cartpole.replay_buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_buffer_wraps_and_samples_vectorized_batch(self):
        buffer = ReplayBuffer(capacity=3, observation_shape=(4,), seed=1)
        for index in range(4):
            state = np.full(4, index, dtype=np.float32)
            buffer.add(state, index % 2, 1.0, state + 1, False)
        self.assertEqual(len(buffer), 3)
        batch = buffer.sample(2)
        self.assertEqual(batch[0].shape, (2, 4))
        self.assertEqual(batch[1].dtype, np.int32)


class AgentTests(unittest.TestCase):
    def test_jitted_train_step_updates_parameters(self):
        network = QNetwork(action_dim=2)
        states = jnp.ones((4, 4), dtype=jnp.float32)
        params = network.init(jax.random.PRNGKey(0), states)
        optimizer = make_optimizer(1e-3)
        opt_state = optimizer.init(params)
        batch = (states, jnp.zeros(4, dtype=jnp.int32), jnp.ones(4), states, jnp.zeros(4))
        new_params, _, loss = train_step(
            params, params, opt_state, batch, 0.99, network.apply, optimizer.update
        )
        self.assertTrue(np.isfinite(float(loss)))
        self.assertFalse(np.array_equal(jax.tree_util.tree_leaves(params)[0], jax.tree_util.tree_leaves(new_params)[0]))


if __name__ == "__main__":
    unittest.main()
