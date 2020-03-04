import jax.numpy as np
from jax import grad, jit

@jit
def sphere_jax(x):
    square = x ** 2.0
    j = np.sum(square, axis=1)

    return j