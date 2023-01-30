import gin
import jax
import jax.numpy as jnp
from jax import lax

import flax
from flax import linen as nn

import einops

from typing import Sequence

import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], 'GPU')

from functools import partial


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


def preprocess(sample: tuple, normalization_range: float = 1., augment: bool = False) -> tf.Tensor:
    x, y = sample

    x = tf.cast(x, tf.float32) / 255. * normalization_range - (normalization_range - 1)

    if augment:
        x = tf.image.random_flip_left_right(x)

        if tf.random.normal() < .5:
            x = tf.image.random_crop(x, size=(26, 26, 1))
            x = tf.image.resize(x, size=(28, 28))

    return x


@gin.configurable
def get_dataset(normalization_range: int = 1, batch_size: int = 512) -> Sequence[tf.data.Dataset]:
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'])

    augment_func = partial(preprocess, normalization_range=normalization_range, augment=True)
    no_augment_func = partial(preprocess, normalization_range=normalization_range, augment=False)

    train_ds = train_ds.map(augment_func, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True)\
        .prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    test_ds = test_ds.map(no_augment_func, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True)\
        .prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

    return train_ds, test_ds


def losses_to_string(losses: dict) -> str:
    return ', '.join([f'{k}: {v:.4f}' for k, v in losses.items()])
