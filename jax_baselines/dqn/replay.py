import jax.numpy as jnp
import numpy as np
from jax.ops import index_update

from jax_baselines.common.util import add_batch_dim


class ReplayMemory:
    def __init__(self, capacity, obs, act):
        # convert obs and act to arrays
        obs = np.array(obs)
        act = np.array(act)

        # use numpy for faster operations
        self.obs = np.zeros(add_batch_dim(capacity, obs.shape))
        self.act = np.zeros(add_batch_dim(capacity, act.shape))
        self.rew = np.zeros((capacity,))
        self.is_terminal = np.zeros((capacity,), dtype=bool)
        self.next_obs = np.zeros(add_batch_dim(capacity, obs.shape))

        self.ptr, self.size, self.capacity = 0, 0, capacity

    def store(self, obs, act, rew,  next_obs, is_terminal=False):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.is_terminal[self.ptr] = is_terminal
        self.next_obs[self.ptr] = next_obs

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_batch(self, batch_size):
        batch_indexes = np.random.choice(self.size, size=batch_size)
        obs = self.obs[batch_indexes]
        act = self.act[batch_indexes]
        rew = self.rew[batch_indexes]
        is_terminal = self.is_terminal[batch_indexes]
        next_obs = self.next_obs[batch_indexes]
        return {
            "observations": jnp.device_put(obs),
            "actions": jnp.device_put(act),
            "rewards": jnp.device_put(rew),
            "is_terminal": jnp.device_put(is_terminal),
            "next_observations": jnp.device_put(next_obs),
        }
