import jax
import jax.numpy as jnp
from jax import lax

import flax
from flax import linen as nn

import einops

from typing import Sequence


def reparameterization(
        rng: jax.random.PRNGKey, mu: jnp.ndarray, sigma: jnp.ndarray
) -> jnp.ndarray:
    eps = jax.random.normal(rng, mu.shape)
    return mu + sigma * eps
