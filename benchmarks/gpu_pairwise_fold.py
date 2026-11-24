"""Check exact GPU indices and compare the pairwise fold with its materialized baseline.

Run locally on CUDA with `python -m benchmarks.gpu_pairwise_fold`, or on a Modal
A10 with `modal run tools/modal_pairwise_fold.py --output pairwise.json`.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import parakeet as pk
from benchmarks.keops_pairwise import exact_1nn, numpy_1nn
from parakeet.backends.cuda import CudaDevice
from parakeet.errors import BackendUnavailable
from parakeet.syntax import ir, ops
from parakeet.transforms.analysis import IndexFacts

PASS = "pairwise fold tiling"


def allocations(fn, arrays):
  """Resolve explicit non-output IR allocations at this input shape (bytes per allocation)."""
  lowered = fn.trace_passes(*arrays, _backend="cuda")
  facts = IndexFacts.from_function(lowered)
  roots = {r.buf.uid for r in lowered.results if isinstance(r, ir.ArrayResult)}
  shared = {
    fold.scores.uid
    for s in ir.walk_stmts(lowered.body)
    if isinstance(s, ir.ParLoop) and (fold := getattr(s.shared_tile, "row_fold", None))
  }
  for param, array in zip(lowered.params, arrays, strict=True):
    facts.defs.update({dim.uid: ir.Const(n, dim.type) for dim, n in zip(param.shape, array.shape, strict=True)})

  def constant(e):
    if isinstance(e, ir.Prim):
      return ops.prim(e.op, tuple(map(constant, e.args)), e.type)
    if isinstance(e, ir.Cast):
      return ops.cast(constant(e.arg), e.type)
    return e

  sizes = []
  for s in ir.walk_stmts(lowered.body):
    if isinstance(s, ir.Alloc) and s.var.uid not in roots:
      count = constant(facts.resolve(s.count))
      if not isinstance(count, ir.Const):
        raise AssertionError(f"Unresolved size for {s.var}: {ir.fmt_expr(count)}")
      sizes.append(
        dict(
          buffer=s.var.name,
          dtype=s.var.type.elt.name,
          bytes=count.value * s.var.type.elt.itemsize,
          shared_score_tile=s.var.uid in shared,
        )
      )
  return sizes


def inputs(n, m, d, seed):
  rng = np.random.default_rng(seed)
  x = rng.integers(-15, 16, size=(n, d)).astype(np.float32)
  y = rng.integers(-15, 16, size=(m, d)).astype(np.float32)
  # Two exact copies of a query at distant positions; the first index wins.
  x[0] = y[1] = y[-2] = 64
  # Integer-valued coordinates keep every float32 feature sum exact. Compute
  # the oracle in float64 over query strips to bound its host memory usage.
  xd, yd = x.astype(np.float64), y.astype(np.float64)
  ynorm = np.sum(yd * yd, axis=1)
  expected = np.concatenate([(ynorm - 2 * rows @ yd.T).argmin(axis=1) for rows in np.array_split(xd, min(n, 32))])
  assert expected[0] == 1
  return (x, y), expected


@pk.jit
def farthest_l1(x, y):
  return np.argmax(np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=-1), axis=1)


def edge_checks():
  """Check partial query/reference/feature tiles and first-NaN argmin/argmax."""
  host, _ = inputs(71, 259, 17, seed=13)
  checks = []
  for has_nan in (False, True):
    x, y = host
    if has_nan:
      y = y.copy()
      y[[3, -11], 0] = np.nan
    diff = x[:, None, :] - y[None, :, :]
    args = pk.to_device((x, y))
    for name, fn, expected in (
      ("numpy-l2-argmin", numpy_1nn, np.argmin(np.sum(diff**2, axis=-1), axis=1)),
      ("dsl-l2-argmin", exact_1nn, np.argmin(np.sum(diff**2, axis=-1), axis=1)),
      ("numpy-l1-argmax", farthest_l1, np.argmax(np.sum(np.abs(diff), axis=-1), axis=1)),
    ):
      np.testing.assert_array_equal(pk.to_host(fn(*args)), expected)
      checks.append(dict(workload=name, has_nan=has_nan, shape=[len(x), len(y), x.shape[1]]))
  return checks


def run(references=(1024, 4096, 16384), queries=1024, features=32, repeats=21, baseline=False):
  import cupy as cp

  sync = cp.cuda.runtime.deviceSynchronize
  device = CudaDevice.get()
  if device.simulated:
    raise RuntimeError("Hardware timing requires a physical CUDA GPU.")
  props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
  results = {
    "gpu": props["name"].decode(),
    "total_memory_bytes": props["totalGlobalMem"],
    "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
    "cupy_version": cp.__version__,
    "numpy_version": np.__version__,
    "method": "Warmed synchronized calls, device-resident float32 inputs, exact indices, no fallback.",
    "warmups": 3,
    "repeats": repeats,
    "rows": [],
  }
  rng = np.random.default_rng(918)
  with pk.options(backend="cuda", blas=False, device_fallback="error", device_threshold=1, tensor_cores=False):
    if not baseline:
      results["edge_checks"] = edge_checks()
    for m in references:
      host, expected = inputs(queries, m, features, seed=71 + m)
      args = pk.to_device(host)
      sync()
      cases = []
      for source, fn in [("numpy", numpy_1nn), ("dsl", exact_1nn)]:
        for enabled in (True,) if baseline else (False, True):
          skip = () if enabled else (PASS,)
          row = dict(queries=queries, references=m, features=features, source=source, enabled=enabled)
          results["rows"].append(row)
          with pk.options(skip_transforms=skip):
            row["ir_scratch"] = allocations(fn, args)
            try:
              out = fn(*args)
              sync()
            except BackendUnavailable as exc:
              if (baseline or not enabled) and source == "dsl" and "temporary arrays inside the loop" in str(exc):
                row["unsupported"] = str(exc)
                continue
              raise
            np.testing.assert_array_equal(pk.to_host(out), expected)
            for _ in range(results["warmups"]):
              out = fn(*args)
              sync()
          if enabled and not baseline:
            scratch = row["ir_scratch"]
            assert any(b["shared_score_tile"] for b in scratch), f"{source}: missing ordered tile fold"
            if any(b["bytes"] >= queries * m * 4 for b in scratch if not b["shared_score_tile"]):
              raise AssertionError(f"{source} allocated a full score matrix: {scratch}")
          row.update(samples_ms=[], launches=0, uploads=0, downloads=0)
          cases.append((fn, skip, row))
      for _ in range(repeats):
        for i in rng.permutation(len(cases)):
          fn, skip, row = cases[i]
          with pk.options(skip_transforms=skip):
            before = device.launches, device.uploads, device.downloads
            start = time.perf_counter_ns()
            out = fn(*args)
            sync()
            row["samples_ms"].append((time.perf_counter_ns() - start) / 1e6)
            for key, value, previous in zip(
              ("launches", "uploads", "downloads"), (device.launches, device.uploads, device.downloads), before
            ):
              row[key] += value - previous
      for _, _, row in cases:
        row["median_ms"] = statistics.median(row["samples_ms"])
        row["giga_pairs_per_s"] = queries * m / row["median_ms"] / 1e6
        assert row["launches"] >= repeats
        print(queries, m, features, row["source"], row["enabled"], row["median_ms"], flush=True)
  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--baseline", action="store_true", help="Run the original pipeline with strict CUDA dispatch")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  report = json.dumps(run(baseline=args.baseline), indent=2) + "\n"
  if args.output:
    args.output.write_text(report)
  else:
    print(report, end="")
