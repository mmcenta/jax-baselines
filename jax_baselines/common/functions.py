import jax
import jax.numpy as jnp
from jax.ops import index_update


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
    w = jnp.full(shape=(dim,), fill_value=discount)
    w = index_update(w, 0, 1.)
    w = jnp.cumprod(w)
    return jnp.cumsum(w * x)


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
