import jax
import jax.numpy as jnp
from jax import lax

import flax
from flax import linen as nn

import einops

from typing import Sequence, Optional
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


class MLPDiscriminator(nn.Module):
    n_layers: int
    n_filters: int | Sequence[int]

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(self.n_filters)(x)
            x = nn.relu(x)

        y_hat = nn.Dense(1)(x)

        return y_hat


class DoubleConv(nn.Module):
    n_filters: int
    residual: bool = False
    middle_filters: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.Conv(self.middle_filters or self.n_filters, (3, 3), padding='SAME', use_bias=False)(x)
        y = nn.GroupNorm(1)(y)
        y = nn.gelu(y)
        y = nn.Conv(self.n_filters, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.GroupNorm(1)(y)
        if self.residual:
            y = nn.gelu(x + y)
        return y


class Down(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray, pos_enc: jnp.ndarray) -> jnp.ndarray:
        n_filters = x.shape[-1]
        y = nn.max_pool(x, (2, 2), strides=(2, 2))
        y = DoubleConv(n_filters, residual=True)(y)
        y = DoubleConv(n_filters * 2)(y)

        emb = nn.silu(pos_enc)
        emb = nn.Dense(n_filters * 2)(emb)
        emb = jnp.expand_dims(emb, axis=(1, 2))

        return y + emb


class Up(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray, x_skip: jnp.ndarray, pos_enc: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        _, _, _, c_skip = x_skip.shape
        y = jax.image.resize(x, (b, h * 2, w * 2, c), method='bilinear')
        y = jnp.concatenate([x_skip, y], axis=-1)
        y = DoubleConv(c + c_skip, residual=True)(y)
        y = DoubleConv(c // 2, middle_filters=c // 2)(y)

        emb = nn.silu(pos_enc)
        emb = nn.Dense(c // 2)(emb)
        emb = jnp.expand_dims(emb, axis=(1, 2))

        return y + emb


class UNet(nn.Module):
    init_filters: int
    n_time_filters: int

    @staticmethod
    def positional_encoding(t, n_filters):
        inv_freq = 1. / (10000. ** (jnp.arange(0, n_filters, 2, dtype=jnp.float32) / n_filters))
        t = jnp.repeat(t, n_filters // 2, axis=-1) * inv_freq
        pos_enc = jnp.concatenate([jnp.sin(t), jnp.cos(t)], axis=-1)
        return pos_enc

    @nn.compact
    def __call__(self, x, t):
        out_filters = x.shape[-1]
        t = t[:, jnp.newaxis]
        t = self.positional_encoding(t, self.n_time_filters)

        x1 = DoubleConv(self.init_filters)(x)
        x2 = Down()(x1, t)
        x2 = DoubleConv(self.init_filters * 2)(x2)
        x3 = Down()(x2, t)
        x3 = DoubleConv(self.init_filters * 4)(x3)

        x3 = DoubleConv(self.init_filters * 8)(x3)
        x3 = DoubleConv(self.init_filters * 4)(x3)

        x = Up()(x3, x2, t)
        x = DoubleConv(self.init_filters * 2)(x)
        x = Up()(x, x1, t)
        x = DoubleConv(self.init_filters)(x)

        x = nn.Conv(out_filters, (1, 1))(x)
        return x
