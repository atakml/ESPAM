import jax
from jax import numpy as jnp
import flax


class SequenceEncoder(flax.linen.Module):
    def setup(self):
        self.OR_dense_1 = flax.linen.Dense(256)
        self.OR_dense_2 = flax.linen.Dense(73)
        self.OR_LayerNorm = flax.linen.LayerNorm()

    def __call__(self, s):
        s = self.OR_dense_1(s)
        s = flax.linen.relu(s)
        s = self.OR_dense_2(s)
        s = self.OR_LayerNorm(s)
        return s

