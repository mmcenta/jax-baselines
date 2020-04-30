import jax
import jax.numpy as jnp
import numpy as np

from jax_baselines.common.functions import reverse_discount_cumsum
from jax_baselines.common.util import AdvantageBatch, add_batch_dim


class GAEBuffer:
    """
    A buffer that stores (observation, action, reward, state value) tuples
    collected while interacting with environment and produces a (observations,
    actions, returns, advantages) batch when full. It uses the General
    Advantage Estimation (GAE) method to compute advantages.
    """
    def __init__(self, batch_size, obs, act,
                 gamma=0.99, lam=0.95, eps=1e-8):
        """
        Constructs the buffer.

        Args:
            batch_size: Number of samples output batch.
            obs: Dummy observation that is used to infer shape.
            act: Dummy action that is used to infor shape.
            gamma: GAE's gamma parameter.
            lam: GAE's lamba parameter.
            eps: Scalar added to denominators to prevent instabilities.
        """

        # use numpy for faster CPU operations
        obs = np.array(obs)
        act = np.array(act)

        self.obs_buf = np.zeros(add_batch_dim(batch_size, obs.shape))
        self.act_buf = np.zeros(add_batch_dim(batch_size, act.shape))
        self.rew_buf = np.zeros((batch_size,))
        self.ret_buf = np.zeros((batch_size,))
        self.val_buf = np.zeros((batch_size,))
        self.adv_buf = np.zeros((batch_size,))
        self.gamma, self.lam, self.eps = gamma, lam, eps
        self.ptr, self.trajectory_start, self.batch_size = 0, 0, batch_size

    def store_timestep(self, obs, act, rew, val):
        """
        Stores a timestep in the buffer.

        Args:
            obs: Observation at the timestep.
            act: Action taken at the timestep.
            rew: Reward received at the timestep.
            val: Estimated value of the current state.

        Returns:
            True if the buffer is full, False otherwise.
        """
        assert self.ptr < self.batch_size, "VPGBuffer is full"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1
        return self.ptr == self.batch_size

    def end_trajectory(self, last_val=0):
        """
        Ends the ongoing trajectory and calculates its return and advantage.

        Args:
            last_val: Value of the last state of the trajectory, useful to
                bootstrap estimates when the trajectory was cut short.
        """
        trajectory = slice(self.trajectory_start, self.ptr)

        # get the rewards and values of the trajectory
        rews = np.append(self.rew_buf[trajectory], last_val)
        vals = np.append(self.val_buf[trajectory], last_val)

        # estimate the advantege using the GAE method
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = reverse_discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf[trajectory] = adv

        # compute the rewards-to-go
        ret = reverse_discount_cumsum(rews, self.gamma)[:-1]
        self.ret_buf[trajectory] = ret

        self.trajectory_start = self.ptr

    def get_batch(self):
        """
        Returns the batch collected on the buffer. Assumes that the buffer is
        full and that the end_trajectory method was called immediately before.

        Returns:
            A batch containing observations, actions, returns and advantages.

        Raises:
            AssertionError: when called on buffer that is not full.

        """
        assert self.ptr == self.batch_size, "VPGBuffer batch not complete"
        self.ptr, self.trajectory_start = 0, 0

        # normalize the advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + self.eps)
        return AdvantageBatch(
            observations=jax.device_put(self.obs_buf),
            actions=jax.device_put(self.act_buf),
            returns=jax.device_put(self.ret_buf),
            advantages=jax.device_put(self.ret_buf),
        )
