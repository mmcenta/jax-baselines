import jax.numpy as jnp
import numpy as np
from jax.ops import index_update

from jax_baselines.common.util import add_batch_dim, TransitionBatch


class ReplayMemory:
    def __init__(self, capacity, obs, act):
        # convert obs and act to arrays
        obs = np.array(obs)
        act = np.array(act)

        # use numpy for faster operations
        self.obs_buf = np.zeros(add_batch_dim(capacity, obs.shape))
        self.act_buf = np.zeros(add_batch_dim(capacity, act.shape))
        self.rew_buf = np.zeros((capacity,))
        self.next_obs_buf = np.zeros(add_batch_dim(capacity, obs.shape))

        self.ptr, self.size, self.capacity = 0, 0, capacity

    def store(self, obs, act, rew,  next_obs):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_batch(self, batch_size):
        batch_indexes = np.random.choice(self.size, size=batch_size)
        obs = self.obs_buf[batch_indexes]
        act = self.act_buf[batch_indexes]
        rew = self.rew_buf[batch_indexes]
        next_obs = self.next_obs_buf[batch_indexes]
        return TransitionBatch(
            observations=jnp.device_put(obs),
            actions=jnp.device_put(act),
            rewards=jnp.device_put(rew),
            next_observations=jnp.device_put(next_obs)
        )
