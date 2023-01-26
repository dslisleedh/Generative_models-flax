import sys
import inspect

import gin
import jax
import jax.numpy as jnp
from jax import lax

import flax
from flax import linen as nn

import einops

from typing import Sequence

from src.layers import *
from src.utils import *


@gin.configurable
class VAE(nn.Module):
    n_layers: int
    n_filters: int
    n_latent_dims: int
    output_shape: Sequence[int] = (28, 28, 1)

    def setup(self):
        self.q_phi = MLPEncoder(self.n_layers, self.n_filters, self.n_latent_dims)
        self.g_theta = MLPDecoder(self.n_layers, self.n_filters, self.output_shape)

    def sample(self, z: jnp.ndarray) -> jnp.ndarray:
        n, n_latent_dims = z.shape
        assert n_latent_dims == self.n_latent_dims
        y_hat = self.g_theta(z)
        return y_hat

    def __call__(
            self, x: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> Sequence[jnp.ndarray]:
        mu, logvar = self.q_phi(x)
        sigma = jnp.exp(.5 * logvar)
        z = reparameterization(rng, mu, sigma)
        y_hat = self.g_theta(z)
        return y_hat, z, mu, sigma


@gin.configurable
class MarkovianHVAE(nn.Module):
    n_layers: int
    n_filters: int
    n_latent_dims: int
    output_shape: Sequence[int] = (28, 28, 1)
    n_steps: int = 3

    def setup(self) -> None:
        assert self.n_steps > 1, "For step 1, use Vanilla VAE"

        self.q_phis = [
            MLPEncoder(self.n_layers, self.n_filters, self.n_latent_dims) for _ in range(self.n_steps)
        ]
        self.g_thetas = [
            MLPEncoder(self.n_layers, self.n_filters, self.n_latent_dims) for _ in range(self.n_steps - 1)
        ] + [
            MLPDecoder(self.n_layers, self.n_filters, self.output_shape)
        ]

    def sample(self, z: jnp.ndarray) -> jnp.ndarray:
        n, n_latent_dims = z.shape
        assert n_latent_dims == self.n_latent_dims
        for g_theta in self.g_thetas[:-1]:
            z = g_theta(z)
        y_hat = self.g_thetas[-1](z)
        return y_hat

    def __call__(self, x, rng):
        rngs = jax.random.split(rng, self.n_steps)
        z_from_q = []
        z_from_g = []

        z = x
        for i, r in enumerate(rngs[:-1]):
            mu, logvar = self.q_phis[i](z)
            sigma = jnp.exp(.5 * logvar)
            z = reparameterization(r, mu, sigma)
            z_from_q.append(z)

        mu_T, logvar_T = self.q_phis[-1](z)
        sigma_T = jnp.exp(.5 * logvar_T)
        z = reparameterization(rngs[-1], mu_T, sigma_T)
        z_T = z.copy()

        for g_theta in self.g_thetas[:-1]:
            mu, logvar = g_theta(z)
            sigma = jnp.exp(.5 * logvar)
            z = reparameterization(rngs[-1], mu, sigma)
            z_from_g.append(z)

        y_hat = self.g_thetas[-1](z)

        return y_hat, z_T, z_from_q, z_from_g, mu_T, sigma_T


@gin.configurable
class GAN(nn.Module):
    n_gen_filters: int
    n_gen_layers: int
    n_disc_filters: int
    n_disc_layers: int
    n_latent_dims: int
    output_shape: Sequence[int] = (28, 28, 1)

    def setup(self):
        self.generator = MLPDecoder(self.n_gen_layers, self.n_gen_filters, self.output_shape)
        self.discriminator = MLPDiscriminator(self.n_disc_layers, self.n_disc_filters)

    def sample(self, z: jnp.ndarray) -> jnp.ndarray:
        n, n_latent_dims = z.shape
        assert n_latent_dims == self.n_latent_dims

        y_hat = self.generator(z)
        return y_hat

    def __call__(
            self, y: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> Sequence[jnp.ndarray]:
        z = jax.random.normal(rng, (y.shape[0], self.n_latent_dims))
        y_hat = self.generator(z)

        d_real = self.discriminator(y)
        d_fake = self.discriminator(y_hat)

        return y_hat, d_real, d_fake
