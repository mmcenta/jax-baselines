from collections import namedtuple

import jax
import jax.numpy as jnp


AdvantageBatch = namedtuple('AdvantageBatch',
    ['observations', 'actions', 'returns', 'advantages'])

TransitionBatch = namedtuple('TransitionBatch',
    ['observations', 'actions', 'rewards', 'new_observations'])


def add_batch_dim(batch_size, shape):
    if shape is None:
        return (batch_size,)
    return (batch_size, shape) if jnp.isscalar(shape) else (batch_size, *shape)


def get_shape(arr):
    if hasattr(arr, 'shape'):
        return arr.shape
    if isinstance(arr, (int, float)):
        return (1,)
    raise ValueError("Inputs should be scalars or have a 'shape' attribute.")


def make_preprocessor(transforms=None, device_put=False):
    """
    """
    # verify input
    if transforms is not None:
        if not isinstance(transforms, (list, tuple)):
            transforms = (transforms)

        for fn in transforms:
            if not callable(fn):
                raise ValueError("Each element of custom_fns must be callabe")

    def preprocess(obs):
        # apply custom transforms first
        if transforms:
            for fn in transforms:
                obs = fn(obs)

        # convert obs to array
        if isinstance(obs, (int, float)):
            return jnp.array(obs).reshape((1,))
        if not obs.shape:
            return obs.reshape((1,))

        # put array to device if flag is set
        if device_put:
            obs = jax.device_put(obs)
        return obs

    return preprocess
