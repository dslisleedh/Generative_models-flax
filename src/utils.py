import gin
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


@gin.configurable
def linear_beta_schedule(
        b_start: float, b_end: float, t: int
) -> jnp.ndarray:
    return jnp.linspace(b_start, b_end, t)


@gin.configurable
def cosine_beta_schedule(
        b_start: float, b_end: float, t: int
) -> jnp.ndarray:
    return b_start + 0.5 * (b_end - b_start) * (
            1 + jnp.cos(jnp.pi * jnp.linspace(0, 1, t))
    )
