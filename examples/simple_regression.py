"""Least-squares line fit from scalar reductions."""

import numpy as np

from parakeet import jit


def covariance(x, y):
  return ((x - x.mean()) * (y - y.mean())).mean()


@jit
def fit_simple_regression(x, y):
  slope = covariance(x, y) / covariance(x, x)
  offset = y.mean() - slope * x.mean()
  return slope, offset


def main(n: int = 100_000):
  rng = np.random.default_rng(6)
  x = rng.normal(size=n)
  y = 3 * x + 2 + rng.normal(size=n) * 0.01
  slope, offset = fit_simple_regression(x, y)
  assert abs(slope - 3) < 0.01 and abs(offset - 2) < 0.01
  return slope, offset


if __name__ == "__main__":
  print(main())
