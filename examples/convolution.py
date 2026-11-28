"""Valid 3x3 convolution in four styles: ``_numpy``, ``_comp``, ``_loop``, and ``_par``."""

import numpy as np

import parakeet as par


@par.jit
def conv_3x3_numpy(x, weights):
  h, w = x.shape
  out = np.zeros((h - 2, w - 2), dtype=x.dtype)
  for a in range(3):
    for b in range(3):
      out = out + weights[a, b] * x[2 - a : h - a, 2 - b : w - b]
  return out


@par.jit
def conv_3x3_comp(x, weights):
  flipped = weights[::-1, ::-1]
  return np.array(
    [
      [np.sum(flipped * x[i - 1 : i + 2, j - 1 : j + 2]) for j in range(1, x.shape[1] - 1)]
      for i in range(1, x.shape[0] - 1)
    ]
  )


@par.jit
def conv_3x3_loop(x, weights):
  h, w = x.shape
  out = np.empty((h - 2, w - 2), dtype=x.dtype)
  for i in range(1, h - 1):
    for j in range(1, w - 1):
      total = 0.0
      for a in range(3):
        for b in range(3):
          total += weights[a, b] * x[i - a + 1, j - b + 1]
      out[i - 1, j - 1] = total
  return out


@par.jit
def conv_3x3_par(x, weights):
  flipped = weights[::-1, ::-1]
  return par.imap(
    lambda i, j: np.sum(flipped * x[i : i + 3, j : j + 3]),
    (x.shape[0] - 2, x.shape[1] - 2),
  )


def main(n: int = 200):
  rng = np.random.default_rng(2)
  x, w = rng.normal(size=(n, n)), rng.normal(size=(3, 3))
  ref = sum(w[a, b] * x[2 - a : n - a, 2 - b : n - b] for a in range(3) for b in range(3))
  for fn in (conv_3x3_numpy, conv_3x3_comp, conv_3x3_loop, conv_3x3_par):
    assert np.allclose(fn(x, w), ref), fn.__name__
  return ref


if __name__ == "__main__":
  print(main().shape)
