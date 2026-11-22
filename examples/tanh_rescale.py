"""Three ways to write the same computation - all compile to the same fused loop."""

import numpy as np

from parakeet import jit


@jit
def vectorized(x, alpha=0.5, beta=0.3):
    return np.tanh(x * alpha) + beta


@jit
def loopy(x, alpha=0.5, beta=0.3):
    y = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        y[i] = np.tanh(x[i] * alpha) + beta
    return y


@jit
def comprehension(x, alpha=0.5, beta=0.3):
    return np.array([np.tanh(xi * alpha) + beta for xi in x])


def main(n: int = 1_000_000):
    x = np.random.default_rng(0).random(n)
    expected = np.tanh(x * 0.5) + 0.3
    for f in (vectorized, loopy, comprehension):
        assert np.allclose(f(x), expected), f.__name__
    return expected


if __name__ == "__main__":
    main()
    print(vectorized.specialize(np.zeros(4)).ir)
