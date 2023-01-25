import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn


def binary_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


def kl_divergence_analytic(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar))


def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    log_div = jnp.log(q / p)
    return jnp.sum(p * log_div, axis=-1)


def consistency_loss(*args):
    loss = []
    for arg in args:
        loss.append(kl_divergence(arg[0], arg[1]))
    loss = jnp.mean(jnp.sum(jnp.stack(loss, axis=0), axis=0), axis=0)
    return loss
