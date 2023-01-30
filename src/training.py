import flax.training.train_state
import gin
import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn

from src.losses import *
from src.utils import *

from tqdm import tqdm

import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')


def train_epoch(
        train_func: callable, train_state: flax.training.train_state.TrainState,
        train_ds: tf.data.Dataset, rng: jax.random.PRNGKey, epoch: int,
) -> flax.training.train_state.TrainState:

    with tqdm(total=train_ds.cardinality(), desc=f"Epoch {epoch}") as pbar:
        for i, batch in enumerate(train_ds):
            rng = jax.random.fold_in(rng, i)
            train_state, loss = train_func(train_state, batch, rng)
            pbar.update(1)
            pbar.set_postfix(loss=loss)

    return train_state


@gin.configurable
def vae_training_step(state, x, **kwargs):
    x = jnp.array(x, dtype=jnp.float32)

    def loss_fn(params):
        logits, z, mu, sigma = state.apply_fn(params, x)
        recon_loss = binary_cross_entropy(logits, x)
        kl_loss = kl_divergence(mu, sigma)
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, recon_loss, kl_loss), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, (loss, recon_loss, kl_loss)


@gin.configurable
def mhvae_training_step(state, x, **kwargs):
    x = jnp.array(x, dtype=jnp.float32)

    def loss_fn(params):
        logits, z_T, z_from_q, z_from_g, mu, sigma = state.apply_fn(params, x)
        recon_loss = binary_cross_entropy(logits, x)
        kl_loss = kl_divergence(mu, sigma)
        consist_loss = consistency_loss(z_from_q, z_from_g)
        loss = recon_loss + kl_loss + consist_loss  # Maybe use lambda to weight the consistency loss?
        return loss, recon_loss, kl_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, recon_loss, kl_loss), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, (loss, recon_loss, kl_loss)


@gin.configurable
def gan_training_step(state, x, rng, **kwargs):
    x = jnp.array(x, dtype=jnp.float32)
    gen_rng, disc_rng = jax.random.split(rng, 2)

    def discriminator_loss_fn(params):
        _, d_real, d_fake = state.apply_fn(params, x, gen_rng)
        loss = discrimination_loss(d_real, d_fake)
        return loss

    disc_grad_fn = jax.value_and_grad(discriminator_loss_fn)
    disc_loss, disc_grads = disc_grad_fn(state.params)
    state = state.apply_gradients(grads=disc_grads)

    def generator_loss_fn(params):
        _, d_real, d_fake = state.apply_fn(params, x, gen_rng)
        loss = generation_loss(d_fake)
        return loss

    gen_grad_fn = jax.value_and_grad(generator_loss_fn)
    gen_loss, gen_grads = gen_grad_fn(state.params)
    state = state.apply_gradients(grads=gen_grads)

    return state, (disc_loss, gen_loss)


@gin.configurable
def diffusion_training_step(state, x, rng, **kwargs):
    x = jnp.array(x, dtype=jnp.float32)

    def loss_fn(params):
        predicted_noise, eps = state.apply_fn(params, x, rng)
        loss = diffusion_loss(eps, predicted_noise)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss
