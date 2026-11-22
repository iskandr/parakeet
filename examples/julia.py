"""Julia set: a data-parallel map over pixels with a data-dependent inner loop."""

import parakeet
from parakeet import jit


@jit
def julia(cr, ci, n, bound=1.5, lim=1000.0, cutoff=100):
  step = 2.0 * bound / n

  def pixel(idx):
    i, j = idx
    x = -bound + i * step
    y = -bound + j * step
    count = 0
    while count < cutoff and x * x + y * y < lim:
      x, y = x * x - y * y + cr, 2.0 * x * y + ci
      count += 1
    return count

  return parakeet.imap(pixel, (n, n))


def main(n: int = 200):
  img = julia(-0.8, 0.156, n)
  assert img.shape == (n, n) and img.min() >= 0 and img.max() == 100
  return img


if __name__ == "__main__":
  print(main()[::20, ::20])
