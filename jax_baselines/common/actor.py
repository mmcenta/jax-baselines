from functools import partial

import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from gym.spaces import Box, Discrete


def get_actor(
        action_space,
        net_fn,
        **kwargs,
    ):
    if isinstance(action_space, Box):
        return GaussianActor(action_space, net_fn, *kwargs)
    elif isinstance(action_space, Discrete):
        return CategoricalActor(action_space, net_fn, *kwargs)
    raise ValueError("action_space type not supported.")


class CategoricalActor:
    """
    """
    def __init__(
        self,
        action_space,
        net_fn,
    ):
        self.action_space = action_space
        self.act_dim = action_space.n
        self._net_init, self._net_apply = hk.transform(net_fn)

    @partial(jax.jit, static_argnums=0)
    def init_params(self, rng, obs):
        obs = jax.tree_map(lambda t: t[None, ...], obs).astype(jnp.float32)
        return self._net_init(rng, obs)

    @partial(jax.jit, static_argnums=0)
    def step(self, params, rng, obs):
        logits = self._net_apply(params, obs)
        return random.categorical(rng, logits)

    @partial(jax.jit, static_argnums=0)
    def logp(self, params, obs, act):
        logits = self._net_apply(params, obs)
        all_logps = nn.log_softmax(logits)
        return (hk.one_hot(act, self.act_dim) * all_logps).sum(-1)


class GaussianActor:
    """
    """
    def __init__(
        self,
        action_space,
        net_fn,
        logstd=-0.5,
        eps=1e-8,
    ):
        assert isinstance(logstd, (float, int)), 'logstd must be an scalar'

        self.action_space = action_space
        self.act_dim = action_space.sample().shape
        self.logstd = (logstd * jnp.ones(self.act_dim))
        self.std = jnp.exp(self.logstd)
        self.eps = eps
        self._net_init, self._net_apply = hk.transform(net_fn)

    @partial(jax.jit, static_argnums=0)
    def init_params(self, rng, obs):
        return self._net_init(rng, obs)

    @partial(jax.jit, static_argnums=0)
    def step(self, params, rng, obs):
        mean = self._net_apply(params, obs)
        return mean + self.std * random.normal(rng, shape=mean.shape)

    @partial(jax.jit, static_argnums=0)
    def logp(self, params, obs, act):
        print(obs.shape, act.shape)
        mean = self._net_apply(params, obs)
        logps = -0.5 * (((act - mean) / (self.std + self.eps)) ** 2 + 2 * self.logstd + jnp.log(2 * jnp.pi))
        print(logps.shape)
        return jnp.sum(logps, axis=1)
