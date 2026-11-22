"""Workloads a GPU is for: a lot of arithmetic per element, on a lot of elements.

    python benchmarks/gpu_showcase.py [--scale 1.0] [--json out.json]

Each kernel is timed in NumPy (where it can be written that way without
running out of memory), on one core, on all cores and on the GPU backend of
this machine, in float32.  The same Python source runs everywhere.
"""

import argparse
import json
import pathlib
import platform
import sys
import time
import warnings

import numpy as np

import parakeet
from parakeet import jit, parfor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmarks import suite  # noqa: E402

F = np.float32


def black_scholes(price, strike, years, rate, vol):
  """European call prices: an element-wise formula with a dozen transcendental functions."""
  root = vol * np.sqrt(years)
  d1 = (np.log(price / strike) + (rate + 0.5 * vol * vol) * years) / root
  d2 = d1 - root
  cdf1 = 0.5 * (1.0 + np.tanh(0.8 * d1))  # a smooth stand-in for the normal CDF
  cdf2 = 0.5 * (1.0 + np.tanh(0.8 * d2))
  return price * cdf1 - strike * np.exp(-rate * years) * cdf2


def pull(i, pos, mass, out):
  ax = ay = az = F(0.0)
  for j in range(len(mass)):
    dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
    r2 = dx * dx + dy * dy + dz * dz + F(1e-3)
    w = mass[j] / (r2 * np.sqrt(r2))
    ax, ay, az = ax + dx * w, ay + dy * w, az + dz * w
  out[i, 0], out[i, 1], out[i, 2] = ax, ay, az


def nbody(pos, mass):
  """Gravitational acceleration of every body from every other: n squared interactions."""
  out = np.zeros_like(pos)
  parfor(lambda i: pull(i, pos, mass, out), len(mass))
  return out


def nbody_numpy(pos, mass):
  d = pos[None, :, :] - pos[:, None, :]
  r2 = (d * d).sum(-1) + F(1e-3)
  return (d * (mass[None, :] / (r2 * np.sqrt(r2)))[:, :, None]).sum(1)


def nearest(i, X, C, out):
  best, where = F(np.inf), 0
  for c in range(C.shape[0]):
    d = F(0.0)
    for k in range(C.shape[1]):
      t = X[i, k] - C[c, k]
      d += t * t
    if d < best:
      best, where = d, c
  out[i] = where


def assign(X, C):
  """The assignment step of k-means: the nearest of the centers for every point."""
  out = np.zeros(X.shape[0], dtype=np.int32)
  parfor(lambda i: nearest(i, X, C, out), X.shape[0])
  return out


def assign_numpy(X, C):
  return np.argmin((X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :], axis=1).astype(np.int32)


def heat(u, steps):
  """Many small steps of a stencil: the grid never has to leave the device."""
  for _ in range(steps):
    v = u.copy()
    v[1:-1, 1:-1] = u[1:-1, 1:-1] + F(0.2) * (
      u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - F(4.0) * u[1:-1, 1:-1]
    )
    u = v
  return u


def cases(rng, scale):
  f = lambda a: a.astype(F)  # noqa: E731
  n = int(20_000_000 * scale)
  opts = (f(rng.uniform(10, 200, n)), f(rng.uniform(10, 200, n)), f(rng.uniform(0.1, 3, n)), F(0.03), F(0.25))
  yield "Black-Scholes", f"{n:,} options", black_scholes, black_scholes, opts
  side = int(4096 * scale**0.5)
  yield "Julia set", f"{side} x {side}", suite.julia, None, (-0.8, 0.156, side)
  bodies = int(16_384 * scale**0.5)
  pos, mass = f(rng.normal(size=(bodies, 3))), f(rng.uniform(0.5, 2, bodies))
  yield "n-body forces", f"{bodies:,} bodies", nbody, nbody_numpy if bodies <= 8192 else None, (pos, mass)
  points = int(2_000_000 * scale)
  X, C = f(rng.random((points, 16))), f(rng.random((256, 16)))
  yield "k-means assignment", f"{points:,} x 256 x 16", assign, assign_numpy, (X, C)
  grid = int(2048 * scale**0.5)
  yield "heat equation", f"{grid}², 200 steps", heat, heat, (f(rng.random((grid, grid))), 200)


def best(fn, args, repeat):
  out = float("inf")
  for _ in range(repeat):
    start = time.perf_counter()
    fn(*args)
    out = min(out, time.perf_counter() - start)
  return out * 1e3


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--scale", type=float, default=1.0)
  p.add_argument("--repeat", type=int, default=2)
  p.add_argument("--json", default="")
  p.add_argument("-k", "--filter", default="")
  opts = p.parse_args(argv)
  available = parakeet.available_backends()
  gpu = next((b for b in ("cuda", "metal") if available.get(b) == "ok"), None)
  backends = ["c", "multicore"] + ([gpu] if gpu else [])
  print(f"# {platform.platform()} / {parakeet.config.num_threads} threads / gpu backend: {gpu} / float32")
  print(f"{'kernel':20s} {'size':>22s} {'numpy':>10s} " + " ".join(f"{b:>11s}" for b in backends) + "   gpu vs cores")
  rows = []
  for name, size, fn, reference, args in cases(np.random.default_rng(0), opts.scale):
    if opts.filter not in name:
      continue
    row = {"kernel": name, "size": size, "numpy": best(reference, args, 1) if reference else None}
    cells = [f"{row['numpy']:8.1f}ms" if reference else "       n/a"]
    want = None
    for backend in backends:
      compiled = jit(fn, backend=backend)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", parakeet.FallbackWarning)
        got = compiled(*args)
        if want is None:
          want = got
        elif not np.allclose(got, want, rtol=5e-3, atol=5e-3):
          raise AssertionError(f"{name}: backend {backend} disagrees with {backends[0]}")
        row[backend] = best(compiled, args, 1 if backend == "c" else opts.repeat)
      cells.append(f"{row[backend]:9.1f}ms")
    ratio = row["multicore"] / row[gpu] if gpu else float("nan")
    row["gpu_speedup"] = ratio
    rows.append(row)
    print(f"{name:20s} {size:>22s} " + " ".join(cells) + f"   {ratio:9.1f}x", flush=True)
  if opts.json:
    pathlib.Path(opts.json).write_text(
      json.dumps({"platform": platform.platform(), "gpu": gpu, "rows": rows}, indent=1)
    )
  return rows


if __name__ == "__main__":
  main()
