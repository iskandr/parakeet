"""3x3 convolution written as a nested comprehension over a helper function."""

import numpy as np

from parakeet import jit


@jit
def conv_3x3_trim(x, weights):
  def compute_pixel(i, j):
    total = 0.0
    for ii in range(3):
      for jj in range(3):
        total += weights[ii, jj] * x[i - ii + 1, j - jj + 1]
    return total

  return np.array([[compute_pixel(i, j) for j in range(1, x.shape[1] - 1)] for i in range(1, x.shape[0] - 1)])


def main(n: int = 200):
  rng = np.random.default_rng(2)
  x, w = rng.normal(size=(n, n)), rng.normal(size=(3, 3))
  out = conv_3x3_trim(x, w)
  ref = sum(w[a, b] * x[2 - a : n - a, 2 - b : n - b] for a in range(3) for b in range(3))
  assert np.allclose(out, ref)
  return out


if __name__ == "__main__":
  print(main().shape)
