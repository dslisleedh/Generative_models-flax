import jax
import jax.numpy as jnp
from jax import lax

import flax
from flax import linen as nn

import einops

from typing import Sequence
from functools import reduce


class MLPEncoder(nn.Module):
    n_layers: int
    n_filters: int | Sequence[int]
    n_latent_dim: int

    @nn.compact
    def __call__(self, x):
        b = x.shape[0]
        x = jnp.reshape(x, (b, -1))

        for _ in range(self.n_layers):
            x = nn.Dense(self.n_filters)(x)
            x = nn.relu(x)

        mu = nn.Dense(self.n_latent_dim)(x)
        logvar = nn.Dense(self.n_latent_dim)(x)

        return mu, logvar


class MLPDecoder(nn.Module):
    n_layers: int
    n_filters: int | Sequence[int]
    output_shape: Sequence[int] = (28, 28, 1)

    @nn.compact
    def __call__(self, z):
        output_dim = reduce((lambda x, y: x * y), self.output_shape)

        for _ in range(self.n_layers):
            z = nn.Dense(self.n_filters)(z)
            z = nn.relu(z)

        y_hat = nn.Dense(output_dim)(z)
        y_hat = jnp.reshape(y_hat, (-1, *self.output_shape))

        return y_hat

