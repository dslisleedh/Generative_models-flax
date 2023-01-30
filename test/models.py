import os
import sys
import inspect
import importlib
import gin

import jax
import jax.numpy as jnp

from test.utils import train


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    classes = [
        c for c in inspect.getmembers(importlib.import_module('src.models'), inspect.isclass)
        if 'src.models' in c[1].__module__
    ]

    inputs = jnp.ones((1, 28, 28, 1))
    rng = jax.random.PRNGKey(0)

    for name, cls in classes:
        print(f'\nModel: {name}')
        gin.parse_config_file(f'./config/models/{name}.gin')
        model = train()
        print(model.tabulate(rng, inputs, rng=rng))
        gin.clear_config()
