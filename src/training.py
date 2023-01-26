import gin
import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn

from src.losses import *


@gin.configurable
def vae_training_step(state, batch):
    x, y = batch
    x = jnp.array(x, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.float32)

    def loss_fn(params):
        logits, z_T, z_from_q, z_from_g, mu, sigma = state.apply_fn(params, x)
        recon_loss = binary_cross_entropy(logits, y)
        kl_loss = kl_divergence(mu, sigma)
        consist_loss = consistency_loss(z_from_q, z_from_g)
        loss = recon_loss + kl_loss + consist_loss  # Maybe use lambda to weight the consistency loss?
        return loss, recon_loss, kl_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, recon_loss, kl_loss), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, (loss, recon_loss, kl_loss)


@gin.configurable
def mhvae_training_step(state, batch):
    x, y = batch
    x = jnp.array(x, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.float32)

    def loss_fn(params):
        logits, z_T, z_from_q, mu, sigma  = state.apply_fn(params, x)
        recon_loss = binary_cross_entropy(logits, y)
        kl_loss = kl_divergence(mu, sigma)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, recon_loss, kl_loss), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, (loss, recon_loss, kl_loss)
