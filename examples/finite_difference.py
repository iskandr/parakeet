"""2-d wave equation with explicit loops (the kind of code that is slow in Python)."""

import numpy as np

from parakeet import jit


def fdtd(input_grid, steps):
  grid = input_grid.copy()
  old_grid = np.zeros_like(input_grid)
  previous_grid = np.zeros_like(input_grid)
  l_x, l_y = grid.shape
  for _ in range(steps):
    previous_grid[:, :] = old_grid
    old_grid[:, :] = grid
    for x in range(l_x):
      for y in range(l_y):
        total = 0.0
        if x + 1 < l_x:
          total += old_grid[x + 1, y]
        if x - 1 > 0:
          total += old_grid[x - 1, y]
        if y + 1 < l_y:
          total += old_grid[x, y + 1]
        if y - 1 > 0:
          total += old_grid[x, y - 1]
        grid[x, y] = total / 2.0 - previous_grid[x, y]
  return grid


fast_fdtd = jit(fdtd)


def main(n: int = 60, steps: int = 10):
  grid = np.random.default_rng(3).normal(size=(n, n)).astype(np.float32)
  out = fast_fdtd(grid, steps)
  assert np.allclose(out, fdtd(grid, steps), rtol=1e-4, atol=1e-4)
  return out


if __name__ == "__main__":
  print(main().shape)
