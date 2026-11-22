"""Benchmark programs: each entry is (function, argument factory, description).

The functions are plain NumPy-style Python, so the same code is timed as
(a) ordinary NumPy and (b) compiled by Parakeet on each backend.
"""

from __future__ import annotations

import numpy as np

import parakeet


def tanh_rescale(x, alpha, beta):
  return np.tanh(x * alpha) + beta


def euclidean_norm(x, y):
  return np.sqrt(np.sum((x - y) ** 2))


def arc_distance(theta_1, phi_1, theta_2, phi_2):
  temp = np.sin((theta_2 - theta_1) / 2) ** 2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
  return 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))


def rosenbrock_derivative(x):
  der = np.empty_like(x)
  der[1:-1] = 200 * (x[1:-1] - x[:-2] ** 2) - 400 * (x[2:] - x[1:-1] ** 2) * x[1:-1] - 2 * (1 - x[1:-1])
  der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
  der[-1] = 200 * (x[-1] - x[-2] ** 2)
  return der


def diffuse(u, iterations):
  for _ in range(iterations):
    u[1:-1, 1:-1] = u[1:-1, 1:-1] + 0.1 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1])
  return u


def allpairs_distances(X, C):
  return parakeet.allpairs(lambda a, b: np.sum((a - b) ** 2), X, C)


def allpairs_distances_numpy(X, C):
  return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def harris(I):  # noqa: E741
  m, n = I.shape
  dx = (I[1:, :] - I[: m - 1, :])[:, 1:]
  dy = (I[:, 1:] - I[:, : n - 1])[1:, :]
  A = dx * dx
  B = dy * dy
  C = dx * dy
  tr = A + B
  det = A * B - C * C
  return det - 0.05 * tr * tr


def julia(cr, ci, n):
  bound, lim, cutoff = 1.5, 1000.0, 100
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


def julia_numpy(cr, ci, n):
  bound, lim, cutoff = 1.5, 1000.0, 100
  grid = -bound + np.arange(n) * (2.0 * bound / n)
  x = np.repeat(grid[:, None], n, axis=1)
  y = np.repeat(grid[None, :], n, axis=0)
  count = np.zeros((n, n), dtype=np.int64)
  active = np.ones((n, n), dtype=bool)
  for _ in range(cutoff):
    active &= x * x + y * y < lim
    if not active.any():
      break
    xa, ya = x[active], y[active]
    x[active], y[active] = xa * xa - ya * ya + cr, 2.0 * xa * ya + ci
    count[active] += 1
  return count


def convolution_3x3(x, w):
  def pixel(i, j):
    total = 0.0
    for a in range(3):
      for b in range(3):
        total += w[a, b] * x[i - a + 1, j - b + 1]
    return total

  return np.array([[pixel(i, j) for j in range(1, x.shape[1] - 1)] for i in range(1, x.shape[0] - 1)])


def convolution_3x3_numpy(x, w):
  n, m = x.shape
  return sum(w[a, b] * x[2 - a : n - a, 2 - b : m - b] for a in range(3) for b in range(3))


def matmult(X, Y):
  return X @ Y


def sum_loop(x):
  total = 0.0
  for i in range(len(x)):
    total += x[i]
  return total


def growcut(image, state, window_radius):
  height, width = image.shape[0], image.shape[1]

  def attack_cell(i, j):
    winning_colony = state[i, j, 0]
    defense_strength = state[i, j, 1]
    for jj in range(max(0, j - window_radius), min(width, j + window_radius + 1)):
      for ii in range(max(0, i - window_radius), min(height, i + window_radius + 1)):
        if ii != i or jj != j:
          ssd = np.sum((image[i, j, :] - image[ii, jj, :]) ** 2)
          gval = 1.0 - np.sqrt(ssd) / np.sqrt(3.0)
          attack_strength = gval * state[ii, jj, 1]
          if attack_strength > defense_strength:
            defense_strength = attack_strength
            winning_colony = state[ii, jj, 0]
    return np.array([winning_colony, defense_strength])

  return np.array([[attack_cell(i, j) for j in range(width)] for i in range(height)])


def _rng():
  return np.random.default_rng(0)


def make(dtype=np.float64, scale: float = 1.0):
  """Benchmarks as ``name -> (parakeet_fn, numpy_fn or None, args)``.

  ``numpy_fn`` is None when the function itself is already idiomatic NumPy
  (it is then timed uncompiled as the baseline) or "python" when the only
  uncompiled form is a pure-Python loop that would take minutes."""
  r = _rng()
  n = int(4_000_000 * scale)
  side = int(2000 * scale**0.5)
  f = lambda a: a.astype(dtype)  # noqa: E731
  x, y = f(r.random(n)), f(r.random(n))
  t1, p1, t2, p2 = (f(r.random(n) * 3) for _ in range(4))
  grid = f(r.random((side // 2, side // 2)))
  img = f(r.random((side, side)))
  X, C = f(r.random((int(40_000 * scale), 32))), f(r.random((64, 32)))
  A, B = f(r.random((300, 300))), f(r.random((300, 300)))
  gc_image = f(r.random((64, 64, 3)))
  gc_state = f(r.random((64, 64, 2)))
  one = dtype(1)
  return {
    "tanh_rescale": (tanh_rescale, None, (x, dtype(0.5), dtype(0.3))),
    "euclidean_norm": (euclidean_norm, None, (x, y)),
    "arc_distance": (arc_distance, None, (t1, p1, t2, p2)),
    "rosenbrock_derivative": (rosenbrock_derivative, None, (x,)),
    "harris": (harris, None, (img,)),
    "diffuse (20 steps)": (diffuse, None, (grid, 20)),
    "allpairs_distances": (allpairs_distances, allpairs_distances_numpy, (X, C)),
    "convolution_3x3": (convolution_3x3, convolution_3x3_numpy, (img, f(r.random((3, 3))))),
    "julia set": (julia, julia_numpy, (-0.8 * float(one), 0.156, side // 2)),
    "matmult 300x300": (matmult, None, (A, B)),
    "sum_loop": (sum_loop, "python", (x,)),
    "growcut 64x64": (growcut, "python", (gc_image, gc_state, 5)),
  }
