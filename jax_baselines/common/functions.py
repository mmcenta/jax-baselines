import jax
import jax.numpy as jnp
import jax.random as random
from jax.ops import index_update
import numpy as np


def reward_clipping(rew):
    """
    Sets positive rewards to +1 and negative rewards to -1.

    Args:
        rew: A scalar reward signal.

    Returns:
        +1. if the scalar is positive, -1. if is negative and 0 otherwise.
    """
    if rew > 0:
        return 1.
    if rew < 0:
        return -1.
    return 0.


def discount_cumsum(x, discount):
    """
    Discounted cumulative sum of an 1D array with JAX backend.

    Args:
        x: input array.
        discount: the discount factor.

    Returns:
        A new array containing the result.
    """
    dim = x.shape[0]
    w = np.full(shape=(dim,), fill_value=discount)
    w[0] = 1.
    w = np.cumprod(w)
    return np.cumsum(w * x)


def reverse_discount_cumsum(x, discount):
    """
    Reverse discounted cumulative sum of an 1D array with JAX backend.

    Args:
        x: input array.
        discount: the discount factor.

    Returns:
        A new array containing the result.
    """
    return discount_cumsum(x[::-1], discount)[::-1]
