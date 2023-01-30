import flax.training.train_state
import gin
import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn

from tqdm.contrib.concurrent import process_map

from src.losses import *
from src.utils import *

from tqdm import tqdm

import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')


def test_step(
        model: nn.Module, state: flax.training.train_state.TrainState, n_epochs: int,
        rng: jax.random.PRNGKey, z: jnp.ndarray, n_tests: int = 50
):
    x = model.apply(state.parmas, z, method=model.sample, rngs=rng)
    names = [f'{n_epochs}_{i}' for i in range(n_tests)]
    process_map(visualize, x, names)
    return x


def visualize(img, name):
    img = img.reshape(28, 28)
    img = np.array(img * 255, dtype=np.uint8)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.encode_png(img)
    tf.io.write_file(f'./visualize/{name}.png', img)
