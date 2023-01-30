import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import inspect
import importlib
import gin

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_swiss_roll

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.models import Diffusion
from test.utils import train


if __name__ == '__main__':
    os.makedirs('./test/figures', exist_ok=True)
    # Load Diffusion Model from gin-config
    rng = jax.random.PRNGKey(0)
    gin.parse_config_file('./config/models/Diffusion.gin')
    model = train()
    params = model.init(rng, jnp.ones((1, 28, 28, 1)), rng=rng)

    # Load swiss-roll data from sklearn
    X, _ = make_swiss_roll(n_samples=500, noise=0.1, random_state=0)
    X = jnp.array(X[:, [0, 2]], dtype=jnp.float32)

    # Visualize Diffusion Process
    steps = jnp.arange(0, 1001, 100)
    noised = []
    for step in steps:
        s = step.reshape(-1)
        noised_img, _ = model.apply(params, x=X, method=model.noise_image, t=s, rng=rng)
        noised.append(np.array(noised_img))

    fig, ax = plt.subplots(1, 11, figsize=(44, 4))
    for i in range(11):
        ax[i].scatter(noised[i][:, 0], noised[i][:, 1], c=noised[i][:, 0], cmap='coolwarm')
        ax[i].set_xlim([-15, 15])
        ax[i].set_ylim([-15, 15])
        ax[i].set_title(f'Step: {steps[i]}')
    plt.tight_layout()
    plt.savefig('./test/figures/swiss_roll.png')
