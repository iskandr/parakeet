"""Derivative of the Rosenbrock function, in array-slice style.

From https://github.com/numfocus/python-benchmarks (Travis Oliphant, Serge Guelton).
"""

import numpy as np

from parakeet import jit


def rosenbrock_derivative(x):
  der = np.empty_like(x)
  der[1:-1] = 200 * (x[1:-1] - x[:-2] ** 2) - 400 * (x[2:] - x[1:-1] ** 2) * x[1:-1] - 2 * (1 - x[1:-1])
  der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
  der[-1] = 200 * (x[-1] - x[-2] ** 2)
  return der


fast_rosenbrock_derivative = jit(rosenbrock_derivative)


def main(n: int = 100_000):
  x = np.arange(n) / float(n)
  out = fast_rosenbrock_derivative(x)
  assert np.allclose(out, rosenbrock_derivative(x))
  return out


if __name__ == "__main__":
  print(main()[:5])
