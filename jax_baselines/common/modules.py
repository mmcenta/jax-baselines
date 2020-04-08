import abc
import flax
import jax
import flax.nn as nn
import jax.numpy as jnp
import jax.random as random
from gym.spaces import Box, Discrete


EPS = 1e-8


def one_hot(indices, dim):
    indices = jnp.array(indices, dtype=jnp.int32)
    print(indices.shape)
    batch_dim = indices.shape[0]
    enc = jnp.zeros(shape=(batch_dim, dim))
    return jax.ops.index_update(enc, indices, 1.)


def logstd_init(rng, shape):
    return -0.5 * jnp.ones(shape=shape)


def gaussian_likelihood(x, mean, logstd):
    pre_sum = -0.5 * (((x - mean) / (jnp.exp(logstd) + EPS)) ** 2 + 2 * logstd + jnp.log(2 * jnp.pi))
    return jnp.sum(pre_sum, axis=1)


class _MLP(nn.Module):
    """
    """
    def apply(self, x, sizes, activation_fn=nn.tanh, output_fn=None):
        for size in sizes:
            x = nn.Dense(x, features=size)
            x = activation_fn(x)
        if output_fn is not None:
            x = output_fn(x)
        return x

class MLPCategoricalActor(nn.Module):
    """
    """
    def apply(self, obs, act, act_rng, action_space=None, hidden_sizes=(64, 64),
              activation_fn=nn.tanh, output_fn=None):
        assert action_space is not None, "Action space must be specified."

        # get parameters for the MLP
        act_dim = action_space.n
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        # compute logits and take action
        logits = _MLP(obs, sizes=list(hidden_sizes) + [act_dim], activation_fn=activation_fn)
        pi = random.categorical(act_rng, logits)

        # compute log probabilities
        logp_all = nn.log_softmax(logits)
        logp = logp_all[act]
        logp_pi = logp_all[pi]

        return pi, logp, logp_pi


class MLPGaussianActor(nn.Module):
    """
    """
    def apply(self, obs, act, act_rng, action_space=None, logstd_init=logstd_init,
              hidden_sizes=(64, 64), activation_fn=nn.tanh, output_fn=None):
        assert action_space is not None, "Action space must be specified."

        # get parameters for the MLP
        act_dim = act.shape[-1]
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        # compute mean and take action
        mean = _MLP(obs, sizes=list(hidden_sizes) + [act_dim],
                    activation_fn=activation_fn, output_fn=output_fn)
        logstd = self.param('logstd', (act_dim,), logstd_init)
        std = jnp.exp(logstd)
        pi = mean + std * random.normal(act_rng, shape=mean.shape)

        # compute log probabilites
        logp = gaussian_likelihood(act, mean, logstd)
        logp_pi = gaussian_likelihood(pi, mean, logstd)

        return pi, logp, logp_pi



class MLPCritic(nn.Module):
    """
    """
    def apply(self, obs, hidden_sizes=(64, 64), activation_fn=nn.tanh):
        val = _MLP(obs, sizes=list(hidden_sizes) + [1], activation_fn=activation_fn)
        return val
