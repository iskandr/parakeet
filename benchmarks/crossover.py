"""Where does the GPU start to win?  CPU and GPU times across problem sizes.

    python benchmarks/crossover.py                 # c, multicore and the GPU of this machine
    python benchmarks/crossover.py --json out.json

Every kernel is timed on NumPy (where it is idiomatic NumPy), one core, all
cores and the GPU backend (Metal on macOS, CUDA elsewhere), in float32.  The
last column counts the loops that actually ran on the device: a loop smaller
than ``config.device_threshold`` stays on the CPU, which is why the small
sizes show the same time for both.
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
from parakeet import jit, value_and_grad

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import decision_tree  # noqa: E402

from benchmarks import suite  # noqa: E402

F = np.float32


def mlp_loss(params, x, y):
  (w1, b1), (w2, b2) = params
  h = np.tanh(x @ w1 + b1)
  logits = h @ w2 + b2
  z = logits - np.max(logits, axis=1, keepdims=True)
  logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
  return -np.mean(np.sum(logp * y, axis=1))


def mlp_step(params, x, y):
  return value_and_grad(mlp_loss)(params, x, y)


def tree_fit(X, y):
  return decision_tree.fit(X, y, 8, 64, 2)


def cases(rng):
  """(kernel, size label, function, numpy reference or None, arguments)."""
  f = lambda a: a.astype(F)  # noqa: E731
  for n in (10_000, 100_000, 1_000_000, 10_000_000):
    yield "tanh_rescale", f"{n:,}", suite.tanh_rescale, suite.tanh_rescale, (f(rng.random(n)), F(0.5), F(0.3))
  for n in (1_000, 10_000, 100_000):
    X, C = f(rng.random((n, 32))), f(rng.random((64, 32)))
    yield "allpairs_distances", f"{n:,} x 64", suite.allpairs_distances, suite.allpairs_distances_numpy, (X, C)
  for n in (128, 512, 1024, 2048):
    yield "matmult", f"{n} x {n}", suite.matmult, suite.matmult, (f(rng.random((n, n))), f(rng.random((n, n))))
  for side in (256, 1024, 2048):
    yield "julia set", f"{side} x {side}", suite.julia, None, (-0.8, 0.156, side)
  for side in (256, 1024, 4096):
    yield "harris", f"{side} x {side}", suite.harris, suite.harris, (f(rng.random((side, side))),)
  for batch in (64, 512, 4096):
    params = (
      (f(rng.normal(0, 0.05, (784, 256))), np.zeros(256, F)),
      (f(rng.normal(0, 0.05, (256, 10))), np.zeros(10, F)),
    )
    x, y = f(rng.random((batch, 784))), np.eye(10, dtype=F)[rng.integers(0, 10, batch)]
    yield "mlp gradient", f"batch {batch}", mlp_step, None, (params, x, y)
  for n in (10_000, 100_000):
    centers = rng.normal(size=(8, 32)) * 0.7
    X, y = decision_tree.blobs(n, centers, rng)
    yield "decision tree fit", f"{n:,} x 32", tree_fit, None, (f(X), y)


def best(fn, args, repeat):
  out = float("inf")
  for _ in range(repeat):
    start = time.perf_counter()
    fn(*args)
    out = min(out, time.perf_counter() - start)
  return out * 1e3


def launches(backend):
  dev = parakeet.api.get_backend(backend)._device
  return 0 if dev is None else dev.launches


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("-k", "--filter", default="")
  p.add_argument("--repeat", type=int, default=3)
  p.add_argument("--json", default="")
  opts = p.parse_args(argv)
  available = parakeet.available_backends()
  gpu = next((b for b in ("cuda", "metal") if available.get(b) == "ok"), None)
  backends = ["c", "multicore"] + ([gpu] if gpu else [])
  print(f"# {platform.platform()} / {parakeet.config.num_threads} threads / gpu backend: {gpu} / float32")
  print(f"{'kernel':20s} {'size':>14s} {'numpy':>10s} " + " ".join(f"{b:>10s}" for b in backends) + "  gpu loops")
  rows = []
  for name, size, fn, reference, args in cases(np.random.default_rng(0)):
    if opts.filter not in name:
      continue
    row = {"kernel": name, "size": size, "numpy": best(reference, args, 2) if reference else None}
    cells = [f"{row['numpy']:8.2f}ms" if reference else "       n/a"]
    used = 0
    for backend in backends:
      compiled = jit(fn, backend=backend)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", parakeet.FallbackWarning)
        compiled(*args)
        before = launches(backend)
        row[backend] = best(compiled, args, opts.repeat)
        if backend == gpu:
          used = (launches(backend) - before) // opts.repeat
      cells.append(f"{row[backend]:8.2f}ms")
    row["gpu_loops"] = used
    rows.append(row)
    print(f"{name:20s} {size:>14s} " + " ".join(cells) + f"  {used:9d}", flush=True)
  if opts.json:
    pathlib.Path(opts.json).write_text(
      json.dumps({"platform": platform.platform(), "gpu": gpu, "rows": rows}, indent=1)
    )
  return rows


if __name__ == "__main__":
  main()
