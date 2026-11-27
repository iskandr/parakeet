"""Reference-checked cross-framework benchmarks on CPU or GPU.

Evaluates five workloads (exact 1-NN, RBF kernel regression, causal attention,
cut cross-entropy, and GBDT histogram split) across distinct Parakeet surface
implementations and installed baselines (NumPy, Numba, SciPy, scikit-learn,
Faiss, PyKeOps, CuPy, PyTorch eager/SDPA/compile, JAX jit).

Usage::

    python -m benchmarks.release_comparison cpu out_cpu.json --threads 32
    python -m benchmarks.release_comparison array out_gpu.json --scale release
    python -m benchmarks.release_comparison array out_gpu_tf32.json --scale large --tensor-cores
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


def cases(scale: str = "release"):
  """Yield ``(task, host_args, reference)`` benchmark cases.

  ``scale="release"`` matches the standard comparison shapes.
  ``scale="large"`` targets larger GPU shapes where materializing pairwise or
  score matrices allocates 1-4 GiB intermediates.
  """
  rng = np.random.default_rng(123)
  if scale == "large":
    n, m, d = 16384, 65536, 32
    x = rng.standard_normal((n, d), dtype=np.float32)
    y = rng.standard_normal((m, d), dtype=np.float32)
    v = rng.standard_normal((m, 4), dtype=np.float32)
    sample_idx = np.linspace(0, n - 1, 64, dtype=np.int64)
    y64, v64 = y.astype("d"), v.astype("d")
    nn_ref, rbf_ref = [], []
    for q_row in x[sample_idx]:
      dist = np.square(y64 - q_row).sum(1)
      nn_ref.append(dist.argmin())
      p = np.exp(dist.min() - dist)
      rbf_ref.append((p @ v64) / p.sum())
    yield "nn", (x, y), (sample_idx, np.asarray(nn_ref))
    yield "rbf", (x, y, v), (sample_idx, np.asarray(rbf_ref))

    q, k, va = rng.standard_normal((3, 4, 16, 4096, 64), dtype=np.float32) * np.float32(0.2)
    t_idx = np.concatenate([np.arange(32), np.arange(4064, 4096)])
    q0, k0, v0 = q[0, 0, t_idx].astype("d"), k[0, 0].astype("d"), va[0, 0].astype("d")
    scores = (q0 @ k0.T) / 8.0
    scores = np.where(t_idx[:, None] >= np.arange(4096)[None, :], scores, -np.inf)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    yield "causal", (q, k, va), (t_idx, (probs / probs.sum(-1, keepdims=True)) @ v0)

    h = rng.standard_normal((4096, 512), dtype=np.float32) * np.float32(0.05)
    w = rng.standard_normal((512, 65536), dtype=np.float32) * np.float32(0.05)
    targets = rng.integers(65536, size=4096, dtype=np.int64)
    losses = []
    w64 = w.astype("d")
    for start in range(0, 4096, 256):
      hb = h[start : start + 256].astype("d")
      tb = targets[start : start + 256]
      lg = hb @ w64
      z = lg - lg.max(1, keepdims=True)
      losses.append(np.log(np.exp(z).sum(1)) - z[np.arange(len(hb)), tb])
    yield "cce", (h, w, targets), np.mean(np.concatenate(losses))

    ids = rng.integers(256, size=10_000_000, dtype=np.int64)
    g = rng.standard_normal(len(ids), dtype=np.float32) + np.where(ids < 71, -0.25, 0.25).astype("f")
    hessian = rng.uniform(0.1, 1.0, len(ids)).astype("f")
    gl, hl = [np.bincount(ids, weights=a, minlength=256).cumsum() for a in (g, hessian)]
    gain = gl**2 / (hl + 1) + (gl[-1] - gl) ** 2 / (hl[-1] - hl + 1)
    yield "split", (ids, g, hessian), np.array([gain.argmax(), gain.max()])
    return

  for n, m, d in [(256, 1024, 3), (1024, 4096, 32)]:
    x = rng.standard_normal((n, d), dtype=np.float32)
    y = rng.standard_normal((m, d), dtype=np.float32)
    v = rng.standard_normal((m, 4), dtype=np.float32)
    y64, v64 = y.astype("d"), v.astype("d")
    nn_ref, rbf_ref = [], []
    for q_row in x:
      dist = np.square(y64 - q_row).sum(1)
      nn_ref.append(dist.argmin())
      p = np.exp(dist.min() - dist)
      rbf_ref.append((p @ v64) / p.sum())
    yield "nn", (x, y), np.asarray(nn_ref)
    yield "rbf", (x, y, v), np.asarray(rbf_ref)

  q, k, v = rng.standard_normal((3, 1, 4, 512, 64), dtype=np.float32)
  scores = (q.astype("d") @ k.astype("d").swapaxes(-2, -1)) / 8
  scores = np.where(np.tri(512, dtype=bool), scores, -np.inf)
  probs = np.exp(scores - scores.max(-1, keepdims=True))
  yield "causal", (q, k, v), (probs / probs.sum(-1, keepdims=True)) @ v

  h = rng.standard_normal((512, 128), dtype=np.float32)
  w = rng.standard_normal((128, 2048), dtype=np.float32)
  targets = rng.integers(2048, size=512, dtype=np.int64)
  logits = h.astype("d") @ w.astype("d")
  z = logits - logits.max(1, keepdims=True)
  yield "cce", (h, w, targets), np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(512), targets])

  ids = rng.integers(256, size=262144, dtype=np.int64)
  g = rng.standard_normal(len(ids), dtype=np.float32) + np.where(ids < 71, -0.25, 0.25).astype("f")
  hessian = rng.uniform(0.1, 1.0, len(ids)).astype("f")
  gl, hl = [np.bincount(ids, weights=a, minlength=256).cumsum() for a in (g, hessian)]
  gain = gl**2 / (hl + 1) + (gl[-1] - gl) ** 2 / (hl[-1] - hl + 1)
  yield "split", (ids, g, hessian), np.array([gain.argmax(), gain.max()])


def broadcast(xp, task: str, args):
  """Reference array-API expression shared by NumPy, CuPy, and JAX."""
  x, y, *rest = args
  if task == "cce":
    z = x @ y
    z = z - z.max(1, keepdims=True)
    return (xp.log(xp.exp(z).sum(1)) - z[xp.arange(x.shape[0]), rest[0]]).mean()
  if task == "split":

    def histogram(weights):
      h = xp.zeros(256, dtype=weights.dtype)
      if hasattr(h, "at"):
        return h.at[x].add(weights)
      xp.add.at(h, x, weights)
      return h

    gl, hl = histogram(y).cumsum(), histogram(rest[0]).cumsum()
    gain = gl**2 / (hl + 1) + (gl[-1] - gl) ** 2 / (hl[-1] - hl + 1)
    idx = gain.argmax()
    return xp.stack((idx.astype(gl.dtype), gain[idx]))
  if task == "nn":
    return xp.argmin(xp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=1)
  if task == "rbf":
    scores = -xp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
  else:
    scores = xp.where(xp.tri(x.shape[-2], dtype=bool), (x @ y.swapaxes(-1, -2)) / 8, -float("inf"))
  probs = xp.exp(scores - scores.max(-1, keepdims=True))
  return (probs / probs.sum(-1, keepdims=True)) @ rest[0]


def _numba_cpu_adapter(task: str, args, num_threads: int):
  """Optional Numba ``@njit(parallel=True)`` baseline for CPU comparisons."""
  import numba

  numba.set_num_threads(num_threads)

  if task == "nn":

    @numba.njit(parallel=True, fastmath=False)
    def nb_1nn(x, y):
      n, d = x.shape
      m = y.shape[0]
      out = np.empty(n, dtype=np.int64)
      for i in numba.prange(n):
        best_d = np.float32(np.inf)
        best_j = 0
        for j in range(m):
          acc = np.float32(0.0)
          for k in range(d):
            diff = x[i, k] - y[j, k]
            acc += diff * diff
          if acc < best_d:
            best_d = acc
            best_j = j
        out[i] = best_j
      return out

    x, y = args
    return "numba-parallel", lambda: nb_1nn(x, y)

  if task == "rbf":

    @numba.njit(parallel=True, fastmath=False)
    def nb_rbf(x, y, v):
      n, d = x.shape
      m = y.shape[0]
      c = v.shape[1]
      out = np.empty((n, c), dtype=np.float32)
      for i in numba.prange(n):
        max_s = np.float32(-np.inf)
        denom = np.float32(0.0)
        acc = np.zeros(c, dtype=np.float32)
        for j in range(m):
          dist = np.float32(0.0)
          for k in range(d):
            diff = x[i, k] - y[j, k]
            dist += diff * diff
          s = -dist
          if s > max_s:
            scale = np.float32(np.exp(max_s - s)) if max_s > -np.inf else np.float32(0.0)
            denom = denom * scale + np.float32(1.0)
            for col in range(c):
              acc[col] = acc[col] * scale + v[j, col]
            max_s = s
          else:
            w = np.float32(np.exp(s - max_s))
            denom += w
            for col in range(c):
              acc[col] += w * v[j, col]
        for col in range(c):
          out[i, col] = acc[col] / denom
      return out

    x, y, v = args
    return "numba-parallel", lambda: nb_rbf(x, y, v)

  if task == "split":

    @numba.njit(parallel=False)
    def nb_split(ids, grads, hess):
      g_hist = np.zeros(256, dtype=np.float32)
      h_hist = np.zeros(256, dtype=np.float32)
      for i in range(len(ids)):
        b = ids[i]
        g_hist[b] += grads[i]
        h_hist[b] += hess[i]
      gl = np.cumsum(g_hist)
      hl = np.cumsum(h_hist)
      g_tot = gl[255]
      h_tot = hl[255]
      best_gain = np.float32(-np.inf)
      best_bin = 0
      for b in range(256):
        gr = g_tot - gl[b]
        hr = h_tot - hl[b]
        gain = (gl[b] * gl[b]) / (hl[b] + np.float32(1.0)) + (gr * gr) / (hr + np.float32(1.0))
        if gain > best_gain:
          best_gain = gain
          best_bin = b
      out = np.empty(2, dtype=np.float32)
      out[0] = np.float32(best_bin)
      out[1] = best_gain
      return out

    ids, grads, hess = args
    return "numba", lambda: nb_split(ids, grads, hess)

  return None


def _parakeet_implementations(task: str):
  import parakeet as pk
  from examples.algebraic_gallery import (
    causal_flash_attention,
    cut_cross_entropy,
    exact_1nn,
    gbdt_best_split,
    gbdt_best_split_comprehension,
    gbdt_best_split_loop,
    gbdt_best_split_numpy,
    nearest_fold,
    nearest_numpy,
    rbf_kernel_regression,
  )

  return {
    "nn": [("pk-numpy", nearest_numpy), ("pk-map", exact_1nn), ("pk-fold", nearest_fold)],
    "rbf": [("pk-numpy", rbf_kernel_regression)],
    "causal": [("pk-numpy", causal_flash_attention)],
    "cce": [("pk-numpy", pk.jit(cut_cross_entropy))],
    "split": [
      ("pk-scatter", gbdt_best_split),
      ("pk-comprehension", gbdt_best_split_comprehension),
      ("pk-numpy-at", gbdt_best_split_numpy),
      ("pk-loop", gbdt_best_split_loop),
    ],
  }[task]


def _extra_args_for_parakeet(task: str) -> tuple:
  if task == "rbf":
    return (np.float32(-1.0),)
  if task == "split":
    return (256, np.float32(1.0))
  return ()


def cpu_adapters(task: str, args, num_threads: int):
  import parakeet as pk

  x, y, *rest = args
  extra = _extra_args_for_parakeet(task)

  with pk.options(backend="threads", num_threads=num_threads, blas=False):
    for name, fn in _parakeet_implementations(task):
      yield name, (lambda f=fn, a=args, ex=extra: f(*a, *ex))

  try:
    nb = _numba_cpu_adapter(task, args, num_threads)
    if nb is not None:
      yield nb
  except ImportError:
    pass

  if task == "nn":
    try:
      import faiss

      faiss.omp_set_num_threads(num_threads)
      flat = faiss.IndexFlatL2(x.shape[1])
      flat.add(y)
      yield "faiss-flat", lambda: flat.search(x, 1)[1].ravel()
    except ImportError:
      pass

    try:
      from scipy.spatial import cKDTree

      tree = cKDTree(y)
      yield "scipy-tree", lambda: tree.query(x, k=1, eps=0, workers=num_threads)[1]
    except ImportError:
      pass

    try:
      from sklearn.neighbors import NearestNeighbors

      brute = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=num_threads).fit(y)
      yield "sklearn-brute", lambda: brute.kneighbors(x, return_distance=False).ravel()
    except ImportError:
      pass

  if task in ("nn", "rbf"):
    try:
      from pykeops.numpy import Genred

      aliases = [f"X=Vi({x.shape[1]})", f"Y=Vj({y.shape[1]})"]
      if task == "nn":
        reduce = Genred("SqDist(X,Y)", aliases, reduction_op="ArgMin", axis=1)
        yield "keops", lambda: reduce(x, y, backend="CPU").ravel()
      else:
        reduce = Genred("-SqDist(X,Y)", aliases + ["V=Vj(4)"], reduction_op="SumSoftMaxWeight", formula2="V", axis=1)
        yield "keops", lambda: reduce(*args, backend="CPU")
    except ImportError:
      pass

  if task in ("causal", "cce"):
    try:
      import torch

      torch.set_num_threads(num_threads)
      targs = tuple(torch.from_numpy(a) for a in args)
      if task == "causal":
        yield (
          "torch-sdpa",
          lambda: torch.nn.functional.scaled_dot_product_attention(*targs, is_causal=True).numpy(),
        )
      else:
        yield "torch-cross-entropy", lambda: torch.nn.functional.cross_entropy(targs[0] @ targs[1], targs[2]).numpy()
    except ImportError:
      pass

  yield "numpy", lambda: broadcast(np, task, args)


def _noop_sync(_out):
  return None


def _torch_sync(_out):
  import torch

  torch.cuda.synchronize()


def _torch_to_cpu(out):
  return out.detach().cpu().numpy()


def adapters(provider: str, task: str, host, *, tensor_cores: bool = False, num_threads: int = 4):
  if provider == "cpu":
    for name, fn in cpu_adapters(task, host, num_threads):
      yield name, fn, _noop_sync, np.asarray
    return

  if provider == "array":
    import cupy as cp

    import parakeet as pk

    def sync(_out):
      cp.cuda.runtime.deviceSynchronize()

    extra = _extra_args_for_parakeet(task)
    with pk.options(backend="cuda", blas=False, device_fallback="error", device_threshold=1, tensor_cores=tensor_cores):
      dargs = pk.to_device(host, backend="cuda")
      for name, fn in _parakeet_implementations(task):
        yield name, (lambda f=fn, a=dargs, ex=extra: f(*a, *ex)), sync, pk.to_host

    cp_args = tuple(cp.asarray(a) for a in host)
    yield "cupy", lambda: broadcast(cp, task, cp_args), sync, cp.asnumpy
    return

  if provider == "jax":
    import jax
    import jax.numpy as jnp

    assert jax.devices()[0].platform == "gpu"
    jax.config.update("jax_default_matmul_precision", "tensorfloat32" if tensor_cores else "highest")
    jargs = tuple(jax.device_put(a) for a in host)
    jax.block_until_ready(jargs)
    fn = jax.jit(lambda *a: broadcast(jnp, task, a))
    yield "jax-jit", lambda: fn(*jargs), jax.block_until_ready, np.asarray
    return

  import torch

  torch.set_float32_matmul_precision("high" if tensor_cores else "highest")
  torch.backends.cuda.matmul.allow_tf32 = bool(tensor_cores)
  targs = tuple(torch.from_numpy(a).cuda() for a in host)
  x, y, *_rest = targs

  if provider == "faiss" and task == "nn":
    import faiss
    import faiss.contrib.torch_utils  # noqa: F401 - registers PyTorch GPU tensor interop

    assert faiss.get_num_gpus() > 0
    resources = faiss.StandardGpuResources()
    resources.setDefaultNullStreamAllDevices()
    index = faiss.GpuIndexFlatL2(resources, x.shape[1])
    index.add(y)
    yield "faiss-flat", lambda: index.search(x, 1)[1].ravel(), _torch_sync, _torch_to_cpu
    return

  if provider == "keops" and task in ("nn", "rbf"):
    from pykeops.torch import Genred

    variables = [f"X=Vi({x.shape[1]})", f"Y=Vj({y.shape[1]})"]
    if task == "nn":
      nn = Genred("SqDist(X,Y)", variables, reduction_op="ArgMin", axis=1)
      yield "keops", lambda: nn(x, y, backend="GPU").ravel(), _torch_sync, _torch_to_cpu
    else:
      rbf = Genred("-SqDist(X,Y)", variables + ["V=Vj(4)"], reduction_op="SumSoftMaxWeight", formula2="V", axis=1)
      yield "keops", lambda: rbf(*targs, backend="GPU"), _torch_sync, _torch_to_cpu
    return

  if provider == "torch":
    if task == "cce":

      def fn(a, b, tgt):
        return torch.nn.functional.cross_entropy(a @ b, tgt)

      eager_name = "torch-cross-entropy"
    elif task == "split":

      def fn(ids, grads, hess):
        gl = torch.zeros(256, device=ids.device).scatter_add_(0, ids, grads).cumsum(0)
        hl = torch.zeros_like(gl).scatter_add_(0, ids, hess).cumsum(0)
        gain = gl.square() / (hl + 1) + (gl[-1] - gl).square() / (hl[-1] - hl + 1)
        idx = gain.argmax()
        return torch.stack((idx, gain.gather(0, idx.reshape(1))[0]))

      eager_name = "torch-scatter"
    elif task == "causal":

      def fn(q, k, val):
        return torch.nn.functional.scaled_dot_product_attention(q, k, val, is_causal=True)

      eager_name = "torch-sdpa"
    elif task == "nn":

      def fn(a, b):
        return torch.cdist(a, b).argmin(1)

      eager_name = "torch-cdist"
    else:

      def fn(a, b, val):
        return torch.softmax(-torch.cdist(a, b).square(), -1) @ val

      eager_name = "torch-cdist"

    yield eager_name, lambda: fn(*targs), _torch_sync, _torch_to_cpu
    compiled = torch.compile(fn, fullgraph=True)
    yield "torch-compile", lambda: compiled(*targs), _torch_sync, _torch_to_cpu

    if task in ("nn", "rbf"):

      def formula(a, b, *val):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return d.argmin(1) if task == "nn" else torch.softmax(-d, -1) @ val[0]

      direct = torch.compile(formula, fullgraph=True)
      yield "torch-compile-broadcast", lambda: direct(*targs), _torch_sync, _torch_to_cpu


def _installed_packages() -> dict[str, str]:
  packages = {}
  for pkg in (
    "numpy",
    "numba",
    "cupy-cuda12x",
    "torch",
    "faiss-gpu-cu12",
    "pykeops",
    "jax",
    "jaxlib",
    "faiss-cpu",
    "scipy",
    "scikit-learn",
  ):
    try:
      packages[pkg] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
      pass
  return packages


def _cpu_model_name() -> str:
  try:
    with open("/proc/cpuinfo", encoding="utf-8") as fh:
      for line in fh:
        if line.startswith("model name"):
          return line.split(":", 1)[1].strip()
  except OSError:
    pass
  return platform.processor() or platform.machine()


def run(
  provider: str,
  *,
  scale: str = "release",
  tensor_cores: bool = False,
  num_threads: int | None = None,
  cpu_all_tasks: bool = False,
) -> dict:
  threads = num_threads or (os.cpu_count() or 1)
  unsupported: tuple[type[BaseException], ...] = ()
  if provider == "array":
    from parakeet.errors import BackendUnavailable

    unsupported = (BackendUnavailable,)

  result: dict = {
    "packages": _installed_packages(),
    "scale": scale,
    "tensor_cores": tensor_cores,
    "rows": [],
  }
  if provider == "cpu":
    result["cpu"] = _cpu_model_name()
    result["threads"] = threads
  else:
    result["gpu"] = subprocess.check_output(
      ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
    ).strip()

  atol, rtol = (3e-3, 5e-3) if tensor_cores else (3e-5, 5e-4)
  for task, host, ref in cases(scale=scale):
    if provider == "faiss" and task != "nn":
      continue
    if provider == "keops" and task not in ("nn", "rbf"):
      continue
    if provider == "cpu" and not cpu_all_tasks and task not in ("nn", "rbf"):
      continue

    row = {"task": task, "shapes": [list(a.shape) for a in host], "provider": provider}
    try:
      for name, call, sync, to_host in adapters(provider, task, host, tensor_cores=tensor_cores, num_threads=threads):
        item = dict(row, method=name)
        try:
          if provider != "jax":
            sync(None)
          start = time.perf_counter()
          out = call()
          sync(out)
          item["first_call_s"] = time.perf_counter() - start
          actual = np.asarray(to_host(out))
          if isinstance(ref, tuple):
            sample_idx, ref_vals = ref
            actual_cmp = actual[0, 0, sample_idx] if task == "causal" else actual[sample_idx]
          else:
            actual_cmp, ref_vals = actual, ref

          if task == "split":
            np.testing.assert_equal(actual_cmp[0], ref_vals[0])
          if task == "nn":
            np.testing.assert_array_equal(actual_cmp, ref_vals)
          else:
            np.testing.assert_allclose(actual_cmp, ref_vals, atol=atol, rtol=rtol)
          item["max_abs_error"] = float(np.abs(actual_cmp - ref_vals).max())

          for _ in range(3):
            sync(call())

          device = None
          counters = (0, 0, 0)
          if provider == "array" and name.startswith("pk-"):
            from parakeet.backends.cuda import CudaDevice

            device = CudaDevice.get()
            counters = (device.launches, device.uploads, device.downloads)

          samples = []
          for _ in range(11 if provider == "cpu" else 21):
            t0 = time.perf_counter_ns()
            sync(call())
            samples.append((time.perf_counter_ns() - t0) / 1e6)

          if device is not None:
            after = (device.launches, device.uploads, device.downloads)
            for key, old, new in zip(("launches", "uploads", "downloads"), counters, after):
              item[f"{key}_per_call"] = (new - old) / len(samples)
            if not item["launches_per_call"]:
              raise AssertionError("no CUDA kernel launched")

          item.update(median_ms=statistics.median(samples), samples_ms=samples)
        except unsupported as exc:
          if name in ("pk-map", "pk-dsl") and task == "nn" and "temporary arrays inside the loop" in str(exc):
            item["unsupported"] = str(exc)
          else:
            item["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
          item["error"] = f"{type(exc).__name__}: {exc}"[:1600]

        print(task, name, item.get("median_ms", item.get("unsupported", item.get("error"))), flush=True)
        result["rows"].append(item)
    except Exception as exc:
      result["rows"].append(dict(row, error=f"{type(exc).__name__}: {exc}"[:1600]))
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("provider", choices=("cpu", "array", "faiss", "keops", "torch", "jax"))
  parser.add_argument("output", type=Path)
  parser.add_argument("--scale", choices=("release", "large"), default="release")
  parser.add_argument("--tensor-cores", action="store_true")
  parser.add_argument("--threads", type=int, default=None, help="CPU thread count (default: all cores)")
  parser.add_argument("--all-tasks", action="store_true", help="Include causal, cce, and split on CPU")
  cli = parser.parse_args(argv)
  cli.output.parent.mkdir(parents=True, exist_ok=True)
  cli.output.write_text(
    json.dumps(
      run(
        cli.provider,
        scale=cli.scale,
        tensor_cores=cli.tensor_cores,
        num_threads=cli.threads,
        cpu_all_tasks=cli.all_tasks,
      ),
      indent=2,
    )
    + "\n"
  )


if __name__ == "__main__":
  main()
