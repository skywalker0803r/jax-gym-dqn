"""Pure JAX DQN update and action-selection functions."""

from functools import partial

import jax
import jax.numpy as jnp
import optax


@partial(jax.jit, static_argnums=2)
def select_greedy_action(params, observations, apply_fn):
    return jnp.argmax(apply_fn(params, observations), axis=-1)


@partial(jax.jit, static_argnums=(5, 6))
def train_step(params, target_params, opt_state, batch, gamma, model_apply_fn, opt_update_fn):
    states, actions, rewards, next_states, dones = batch
    next_q = model_apply_fn(target_params, next_states)
    max_next_q = jnp.max(next_q, axis=-1)
    target_q = rewards + (1.0 - dones) * gamma * max_next_q

    def loss_fn(current_params):
        q_values = model_apply_fn(current_params, states)
        one_hot = jax.nn.one_hot(actions, q_values.shape[-1])
        chosen_q = jnp.sum(q_values * one_hot, axis=-1)
        return jnp.mean((chosen_q - jax.lax.stop_gradient(target_q)) ** 2)

    loss, gradients = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = opt_update_fn(gradients, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


def make_optimizer(learning_rate: float):
    return optax.adam(learning_rate)
