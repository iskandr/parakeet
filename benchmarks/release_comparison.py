"""Reference-checked benchmarks on a single CPU host or synchronized GPU."""

import importlib.metadata
import statistics
import subprocess
import time

import numpy as np


def cases():
  rng = np.random.default_rng(123)
  for n, m, d in [(256, 1024, 3), (1024, 4096, 32)]:
    x = rng.standard_normal((n, d), dtype=np.float32)
    y = rng.standard_normal((m, d), dtype=np.float32)
    v = rng.standard_normal((m, 4), dtype=np.float32)
    y64, v64 = y.astype("d"), v.astype("d")
    nn, gaussian = [], []
    for q in x:
      dist = np.square(y64 - q).sum(1)
      nn.append(dist.argmin())
      p = np.exp(dist.min() - dist)
      gaussian.append((p @ v64) / p.sum())
    yield "nn", (x, y), np.asarray(nn)
    yield "rbf", (x, y, v), np.asarray(gaussian)
  q, k, v = rng.standard_normal((3, 1, 4, 512, 64), dtype=np.float32)
  scores = (q.astype("d") @ k.astype("d").swapaxes(-2, -1)) / 8
  scores = np.where(np.tri(512, dtype=bool), scores, -np.inf)
  p = np.exp(scores - scores.max(-1, keepdims=True))
  yield "causal", (q, k, v), (p / p.sum(-1, keepdims=True)) @ v

  h = rng.standard_normal((512, 128), dtype=np.float32)
  w = rng.standard_normal((128, 2048), dtype=np.float32)
  targets = rng.integers(2048, size=512, dtype=np.int64)
  logits = h.astype("d") @ w.astype("d")
  z = logits - logits.max(1, keepdims=True)
  yield "cce", (h, w, targets), np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(512), targets])
  ids = rng.integers(256, size=262144, dtype=np.int64)
  # Positive curvature and a known gradient signal produce a well-separated best split.
  g = rng.standard_normal(len(ids), dtype=np.float32) + np.where(ids < 71, -0.25, 0.25).astype("f")
  hessian = rng.uniform(0.1, 1.0, len(ids)).astype("f")
  gl, hl = [np.bincount(ids, weights=a, minlength=256).cumsum() for a in (g, hessian)]
  gain = gl**2 / (hl + 1) + (gl[-1] - gl) ** 2 / (hl[-1] - hl + 1)
  yield "split", (ids, g, hessian), np.array([gain.argmax(), gain.max()])


def broadcast(xp, task, args):
  x, y, *v = args
  if task == "cce":
    z = x @ y
    z = z - z.max(1, keepdims=True)
    return (xp.log(xp.exp(z).sum(1)) - z[xp.arange(x.shape[0]), v[0]]).mean()
  if task == "split":
    # Fixed 256-bin, FP32 accumulation for both statistics.
    def histogram(weights):
      h = xp.zeros(256, dtype=weights.dtype)
      if hasattr(h, "at"):
        return h.at[x].add(weights)
      xp.add.at(h, x, weights)
      return h

    gl, hl = histogram(y).cumsum(), histogram(v[0]).cumsum()
    gain = gl**2 / (hl + 1) + (gl[-1] - gl) ** 2 / (hl[-1] - hl + 1)
    index = gain.argmax()
    return xp.stack((index.astype(gl.dtype), gain[index]))
  if task == "nn":
    return xp.argmin(xp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=1)
  if task == "rbf":
    s = -xp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
  else:
    s = xp.where(xp.tri(x.shape[-2], dtype=bool), (x @ y.swapaxes(-1, -2)) / 8, -float("inf"))
  p = xp.exp(s - s.max(-1, keepdims=True))
  return (p / p.sum(-1, keepdims=True)) @ v[0]


def cpu_adapters(task, args):
  import faiss
  from pykeops.numpy import Genred
  from scipy.spatial import cKDTree
  from sklearn.neighbors import NearestNeighbors

  import parakeet as pk
  from benchmarks.keops_pairwise import numpy_1nn
  from examples.algebraic_gallery import exact_1nn, rbf_kernel_regression

  x, y, *v = args
  if task == "nn":
    with pk.options(backend="threads", num_threads=4, blas=False):
      yield "pk-numpy", lambda: numpy_1nn(x, y)
      yield "pk-dsl", lambda: exact_1nn(x, y)
    faiss.omp_set_num_threads(4)
    flat = faiss.IndexFlatL2(x.shape[1])
    flat.add(y)
    yield "faiss-flat", lambda: flat.search(x, 1)[1].ravel()
    tree = cKDTree(y)
    yield "scipy-tree", lambda: tree.query(x, k=1, eps=0, workers=4)[1]
    brute = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=4).fit(y)
    yield "sklearn-brute", lambda: brute.kneighbors(x, return_distance=False).ravel()
  else:
    with pk.options(backend="threads", num_threads=4, blas=False):
      yield "pk-numpy", lambda: rbf_kernel_regression(x, y, v[0], np.float32(-1))
  aliases = [f"X=Vi({x.shape[1]})", f"Y=Vj({y.shape[1]})"]
  if task == "nn":
    reduce = Genred("SqDist(X,Y)", aliases, reduction_op="ArgMin", axis=1)
    yield "keops", lambda: reduce(x, y, backend="CPU").ravel()
  else:
    reduce = Genred("-SqDist(X,Y)", aliases + ["V=Vj(4)"], reduction_op="SumSoftMaxWeight", formula2="V", axis=1)
    yield "keops", lambda: reduce(*args, backend="CPU")
  yield "numpy", lambda: broadcast(np, task, args)


def adapters(provider, task, host):
  if provider == "cpu":
    for name, fn in cpu_adapters(task, host):
      yield name, fn, lambda out: None, np.asarray
    return
  if provider == "array":
    import cupy as cp

    import parakeet as pk
    from benchmarks.keops_pairwise import numpy_1nn
    from examples.algebraic_gallery import (
      causal_flash_attention,
      cut_cross_entropy,
      exact_1nn,
      gbdt_best_split,
      rbf_kernel_regression,
    )

    sync = cp.cuda.runtime.deviceSynchronize
    with pk.options(backend="cuda", blas=False, device_fallback="error", device_threshold=1, tensor_cores=False):
      args = pk.to_device(host, backend="cuda")
      for name, fn in {
        "cce": [("pk-numpy", pk.jit(cut_cross_entropy))],
        "split": [("pk-dsl", gbdt_best_split)],
        "nn": [("pk-numpy", numpy_1nn), ("pk-dsl", exact_1nn)],
        "rbf": [("pk-numpy", rbf_kernel_regression)],
        "causal": [("pk-numpy", causal_flash_attention)],
      }[task]:
        extra = {"rbf": (np.float32(-1),), "split": (256, np.float32(1))}.get(task, ())
        yield name, lambda: fn(*args, *extra), lambda out: sync(), pk.to_host
    args = tuple(map(cp.asarray, host))
    yield "cupy", lambda: broadcast(cp, task, args), lambda out: sync(), cp.asnumpy
    return

  if provider == "jax":
    import jax
    import jax.numpy as jnp

    assert jax.devices()[0].platform == "gpu"
    jax.config.update("jax_default_matmul_precision", "highest")
    args = tuple(map(jax.device_put, host))
    jax.block_until_ready(args)
    fn = jax.jit(lambda *a: broadcast(jnp, task, a))
    yield "jax-jit", lambda: fn(*args), jax.block_until_ready, np.asarray
    return

  import torch

  torch.set_float32_matmul_precision("highest")
  torch.backends.cuda.matmul.allow_tf32 = False
  args = tuple(torch.from_numpy(a).cuda() for a in host)
  sync = lambda out: torch.cuda.synchronize()
  cpu = lambda out: out.cpu().numpy()
  x, y, *v = args
  if provider == "faiss" and task == "nn":
    import faiss
    import faiss.contrib.torch_utils  # installs tensor interoperability

    assert faiss.get_num_gpus() > 0
    resources = faiss.StandardGpuResources()
    resources.setDefaultNullStreamAllDevices()
    index = faiss.GpuIndexFlatL2(resources, x.shape[1])
    index.add(y)
    yield "faiss-flat", lambda: index.search(x, 1)[1].ravel(), sync, cpu
  if provider == "keops" and task in ("nn", "rbf"):
    from pykeops.torch import Genred

    variables = [f"X=Vi({x.shape[1]})", f"Y=Vj({y.shape[1]})"]
    if task == "nn":
      nn = Genred("SqDist(X,Y)", variables, reduction_op="ArgMin", axis=1)
      yield "keops", lambda: nn(x, y, backend="GPU").ravel(), sync, cpu
    else:
      rbf = Genred("-SqDist(X,Y)", variables + ["V=Vj(4)"], reduction_op="SumSoftMaxWeight", formula2="V", axis=1)
      yield "keops", lambda: rbf(*args, backend="GPU"), sync, cpu
  if provider == "torch":
    if task == "cce":

      def fn(x, y, v):
        return torch.nn.functional.cross_entropy(x @ y, v)
    elif task == "split":

      def fn(ids, grads, hess):
        gl = torch.zeros(256, device=ids.device).scatter_add_(0, ids, grads).cumsum(0)
        hl = torch.zeros_like(gl).scatter_add_(0, ids, hess).cumsum(0)
        gain = gl.square() / (hl + 1) + (gl[-1] - gl).square() / (hl[-1] - hl + 1)
        i = gain.argmax()
        return torch.stack((i, gain.gather(0, i.reshape(1))[0]))
    elif task == "causal":

      def fn(x, y, v):
        return torch.nn.functional.scaled_dot_product_attention(x, y, v, is_causal=True)
    elif task == "nn":

      def fn(x, y):
        return torch.cdist(x, y).argmin(1)
    else:

      def fn(x, y, v):
        return torch.softmax(-torch.cdist(x, y).square(), -1) @ v

    yield (
      {"causal": "torch-sdpa", "split": "torch-scatter", "cce": "torch-cross-entropy"}.get(task, "torch-cdist"),
      lambda: fn(*args),
      sync,
      cpu,
    )
    compiled = torch.compile(fn, fullgraph=True)
    yield "torch-compile", lambda: compiled(*args), sync, cpu
    if task in ("nn", "rbf"):

      def formula(x, y, *v):
        d = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
        return d.argmin(1) if task == "nn" else torch.softmax(-d, -1) @ v[0]

      direct = torch.compile(formula, fullgraph=True)
      yield "torch-compile-broadcast", lambda: direct(*args), sync, cpu


def run(provider):
  unsupported = ()
  if provider == "array":
    from parakeet.errors import BackendUnavailable

    unsupported = (BackendUnavailable,)
  packages = {}
  for package in [
    "numpy",
    "cupy-cuda12x",
    "torch",
    "faiss-gpu-cu12",
    "pykeops",
    "jax",
    "jaxlib",
    "faiss-cpu",
    "scipy",
    "scikit-learn",
  ]:
    try:
      packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      pass
  result = {"packages": packages, "rows": []}
  if provider == "cpu":
    import platform

    result["cpu"] = next(
      (line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")),
      platform.machine(),
    )
    result["threads"] = 4
  else:
    result["gpu"] = subprocess.check_output(
      ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True
    ).strip()
  for task, host, ref in cases():
    if (provider == "faiss" and task != "nn") or (provider in ("keops", "cpu") and task not in ("nn", "rbf")):
      continue
    row = {"task": task, "shapes": [list(a.shape) for a in host], "provider": provider}
    try:
      for name, call, sync, to_host in adapters(provider, task, host):
        item = dict(row, method=name)
        try:
          if provider != "jax":
            sync(None)
          start = time.perf_counter()
          out = call()
          sync(out)
          item["first_call_s"] = time.perf_counter() - start
          actual = np.asarray(to_host(out))
          if task == "split":
            np.testing.assert_equal(actual[0], ref[0])
          if task == "nn":
            np.testing.assert_array_equal(actual, ref)
          else:
            np.testing.assert_allclose(actual, ref, atol=3e-5, rtol=5e-4)
          item["max_abs_error"] = float(np.abs(actual - ref).max())
          for _ in range(3):
            sync(call())
          device = None
          if provider == "array" and name.startswith("pk-"):
            from parakeet.backends.cuda import CudaDevice

            device = CudaDevice.get()
            counters = device.launches, device.uploads, device.downloads
          samples = []
          for _ in range(11 if provider == "cpu" else 21):
            start = time.perf_counter_ns()
            sync(call())
            samples.append((time.perf_counter_ns() - start) / 1e6)
          if device is not None:
            after = device.launches, device.uploads, device.downloads
            for key, old, new in zip(("launches", "uploads", "downloads"), counters, after):
              item[key + "_per_call"] = (new - old) / len(samples)
            if not item["launches_per_call"]:
              raise AssertionError("no CUDA kernel launched")
          item.update(median_ms=statistics.median(samples), samples_ms=samples)
        except unsupported as exc:
          if name == "pk-dsl" and task == "nn" and "temporary arrays inside the loop" in str(exc):
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


if __name__ == "__main__":
  import json
  import sys
  from pathlib import Path

  Path(sys.argv[2]).write_text(json.dumps(run(sys.argv[1])))
