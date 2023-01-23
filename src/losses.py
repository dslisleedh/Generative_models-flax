import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn


def binary_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


def kl_divergence(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar))
