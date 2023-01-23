import jax
import jax.numpy as jnp
from jax import lax

import flax
from flax import linen as nn

import einops

from typing import Sequence

from src.layers import *


class VAE(nn.Module):
    n_layers: int
    n_filters: int
    n_latent_dims: int
    output_shape: Sequence[int] = (28, 28, 1)

    @nn.compact
    def __call__(self, x):
        mu, logvar = MLPEncoder(self.n_layers, self.n_filters, self.n_latent_dims)(x)
        z = Reparameterization()(mu, logvar)
        y_hat = MLPDecoder(self.n_layers, self.n_filters, self.output_shape)(z)
        return y_hat, mu, logvar


