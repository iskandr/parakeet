"""Elementwise rescaling in Parakeet's four standard styles:
``_numpy``, ``_comp``, ``_loop``, and ``_par``.
"""

import numpy as np

import parakeet as par


# --8<-- [start:tanh_rescale_four_ways]
@par.jit
def tanh_rescale_numpy(x, alpha=0.5, beta=0.3):
  return np.tanh(x * alpha) + beta


@par.jit
def tanh_rescale_comp(x, alpha=0.5, beta=0.3):
  return np.array([np.tanh(xi * alpha) + beta for xi in x])


@par.jit
def tanh_rescale_loop(x, alpha=0.5, beta=0.3):
  y = np.empty_like(x, dtype=float)
  for i in range(len(x)):
    y[i] = np.tanh(x[i] * alpha) + beta
  return y


@par.jit
def tanh_rescale_par(x, alpha=0.5, beta=0.3):
  return par.map(lambda xi: np.tanh(xi * alpha) + beta, x)


# --8<-- [end:tanh_rescale_four_ways]


def main(n: int = 1_000_000):
  x = np.random.default_rng(0).random(n)
  expected = np.tanh(x * 0.5) + 0.3
  for fn in (tanh_rescale_numpy, tanh_rescale_comp, tanh_rescale_loop, tanh_rescale_par):
    assert np.allclose(fn(x), expected), fn.__name__
  return expected


if __name__ == "__main__":
  main()
  print(tanh_rescale_numpy.specialize(np.zeros(4)).ir)
