from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp


class DiscreteActionCritic:
    """
    """
    def __init__(
        self,
        net_fn,
        act_dim,
    ):
        # add final layer with a value for each action
        net = lambda obs: jax.nn.relu(hk.Linear(output_size=act_dim)(net_fn(obs)))

        # transform net_fn
        self._net_init, self._net_apply = hk.transform(net)

    @partial(jax.jit, static_argnums=0)
    def init_params(self, rng, obs):
        obs = jax.tree_map(lambda t: t[None, ...], obs).astype(jnp.float32)
        return self._net_init(rng, obs)

    @partial(jax.jit, static_argnums=0)
    def action_values(self, params, obs):
        return self._net_apply(params, obs)


class StateCritic:
    """
    """
    def __init__(
        self,
        net_fn,
    ):
        self._net_init, self._net_apply = hk.transform(net_fn)

    @partial(jax.jit, static_argnums=0)
    def init_params(self, rng, obs):
        obs = jax.tree_map(lambda t: t[None, ...], obs).astype(jnp.float32)
        return self._net_init(rng, obs)

    @partial(jax.jit, static_argnums=0)
    def state_value(self, params, obs):
        return self._net_apply(params, obs)
