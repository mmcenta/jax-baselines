import jax.numpy as jnp


def add_batch_dim(batch_size, shape):
    if shape is None or shape == (1,):
        return (batch_size,)
    return (batch_size, shape) if jnp.isscalar(shape) else (batch_size, *shape)


def get_shape(arr):
    if hasattr(arr, 'shape'):
        return arr.shape
    if isinstance(arr, (int, float)):
        return (1,)
    raise ValueError("Inputs should be scalars or have a 'shape' attribute.")


def get_input_shapes(*inputs):
    return [get_shape(inpt) for inpt in inputs]
