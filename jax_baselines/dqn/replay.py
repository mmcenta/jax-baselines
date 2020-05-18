# Adapted from OpenAI's implementation at https://github.com/openai/baselines
import random

import jax.numpy as jnp
import numpy as np
from jax.ops import index_update

from jax_baselines.common.util import add_batch_dim
from jax_baselines.common.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    def __init__(self, capacity, obs, act):
        # convert obs and act to arrays
        obs = np.array(obs)
        act = np.array(act)

        # use numpy for faster operations
        self.obs = np.zeros(add_batch_dim(capacity, obs.shape))
        self.act = np.zeros(add_batch_dim(capacity, act.shape))
        self.rew = np.zeros((capacity,))
        self.next_obs = np.zeros(add_batch_dim(capacity, obs.shape))
        self.is_terminal = np.zeros((capacity,), dtype=bool)

        self._ptr, self._size, self.capacity = 0, 0, capacity

    def __len__(self):
        return self._size

    def store(self, obs, act, rew, next_obs, is_terminal=False):
        self.obs[self._ptr] = obs
        self.act[self._ptr] = act
        self.rew[self._ptr] = rew
        self.next_obs[self._ptr] = next_obs
        self.is_terminal[self._ptr] = is_terminal

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _get_batch_from_indexes(self, idxes):
        obs = self.obs[idxes]
        act = self.act[idxes]
        rew = self.rew[idxes]
        next_obs = self.next_obs[idxes]
        is_terminal = self.is_terminal[idxes]
        return {
            "observations": jnp.device_put(obs),
            "actions": jnp.device_put(act),
            "rewards": jnp.device_put(rew),
            "next_observations": jnp.device_put(next_obs),
            "is_terminal": jnp.device_put(is_terminal),
        }

    def sample(self, batch_size):
        batch_indexes = np.random.choice(self._size, size=batch_size)
        return self._get_batch_from_indexes(batch_indexes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, obs, act, alpha=0.6):
        """
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity, obs, act)

        assert alpha >= 0
        self.alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def store(self, *args, **kwargs):
        """
        """
        idx = self._ptr
        super().store(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        """
        """
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        batch = self._get_batch_from_indexes(idxes)
        batch["is_weights"] = jnp.device_put(weights)
        batch["indexes"] = idxes
        return batch

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
