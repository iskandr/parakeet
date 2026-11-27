"""Backend selection, GPU backends, threading and resource management."""

import gc
import os
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest

import parakeet
from parakeet import jit
from parakeet.testing import same

AVAILABLE = parakeet.available_backends()
CPU_BACKENDS = [b for b in ("c", "threads", "gcd", "openmp") if AVAILABLE.get(b) == "ok"]
needs_metal = pytest.mark.skipif(AVAILABLE.get("metal") != "ok", reason=f"metal: {AVAILABLE.get('metal')}")


def programs():
  """A set of functions exercising every kind of parallel loop."""

  def elementwise(x, y):
    return np.sqrt(x * x + y * y) - np.minimum(x, y)

  def reductions(x, y):
    return np.sum(x * y), x.max(), (x > y).sum(), x.min(), np.mean(x)

  def rows(X, C):
    return parakeet.allpairs(lambda a, b: np.sum((a - b) ** 2), X, C)

  def stencil(u):
    out = u.copy()
    out[1:-1, 1:-1] = (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]) * 0.25
    return out

  def axis_sums(X):
    return X.sum(axis=0), X.max(axis=1), X.mean(axis=1)

  def mixed_ints(i, x):
    return (i * 3 + 1) % 7, np.where(i % 2 == 0, x, -x), i.astype(np.float32) * x

  def gather(x, idx):
    return x[idx] * 2

  def inplace(x):
    x *= 2
    x[::2] += 1
    return x

  return [elementwise, reductions, rows, stencil, axis_sums, mixed_ints, gather, inplace]


def inputs(dtype):
  rng = np.random.default_rng(3)
  n = 60_000
  x = rng.random(n).astype(dtype)
  y = rng.random(n).astype(dtype)
  X = rng.random((1500, 24)).astype(dtype)
  C = rng.random((40, 24)).astype(dtype)
  u = rng.random((300, 200)).astype(dtype)
  i = rng.integers(0, 1000, n).astype(np.int32)
  idx = rng.integers(0, n, n)
  return {
        "elementwise": (x, y), "reductions": (x, y), "rows": (X, C), "stencil": (u,), "axis_sums": (X,),
        "mixed_ints": (i, x), "gather": (x, idx), "inplace": (x,),
    }  # fmt: skip


def run_everywhere(backend, dtype, rtol):
  args = inputs(dtype)
  for fn in programs():
    a = args[fn.__name__]
    expected = fn(*[v.copy() for v in a])
    got = jit(fn)(*[v.copy() for v in a], _backend=backend)
    diff = same(got, expected, rtol)
    assert diff is None, f"{backend}/{fn.__name__}: {diff}"


def aliased_arguments(backend, dtype):
  """Arguments of one GPU kernel that share memory must share a device buffer."""
  a = np.arange(500, dtype=dtype).reshape(50, 10)

  def dists(a, z):
    return parakeet.each(lambda row: np.sum((row - z) * (row - z)), a)

  def dists_view_first(z, a):
    return parakeet.each(lambda row: np.sum((row - z) * (row - z)), a)

  def both(a, b, n):
    a[:n] = b[:n] + a[:n] * 2

  with parakeet.options(device_threshold=1):
    for z in (a[0], a[3], a[::-1][0], a[:, 2][:10]):
      assert same(jit(dists)(a, z, _backend=backend), dists(a, z), 1e-6) is None
      assert same(jit(dists_view_first)(z, a, _backend=backend), dists(a, z), 1e-6) is None
    for lo in (0, 1, 7):
      x, y = np.arange(64, dtype=dtype), np.arange(64, dtype=dtype)
      both(x[lo : lo + 20], x[40:60], 20)
      jit(both)(y[lo : lo + 20], y[40:60], 20, _backend=backend)
      assert np.array_equal(x, y)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_cpu_backends_agree_with_numpy(backend):
  run_everywhere(backend, np.float64, 1e-10)
  run_everywhere(backend, np.float32, 2e-6)


@needs_metal
def test_metal_agrees_with_numpy():
  with warnings.catch_warnings():
    warnings.simplefilter("error", parakeet.FallbackWarning)  # float32 programs must not fall back
    run_everywhere("metal", np.float32, 2e-6)


@needs_metal
def test_metal_aliased_arguments():
  aliased_arguments("metal", np.float32)


@needs_metal
def test_metal_runs_kernels_on_the_device():
  @jit(backend="metal")
  def f(x):
    return x * np.float32(2) + np.float32(1)

  x = np.ones(100_000, dtype=np.float32)
  spec = f.specialize(x)
  assert len(spec.kernels) == 1 and "kernel void pk_kernel" in spec.kernels[0]
  plan = spec.module.kernels[0].device
  assert plan.compiled is None  # compiled lazily
  assert np.array_equal(f(x), x * 2 + 1)
  assert plan.compiled is not None  # ... and it really ran on the device
  small = np.ones(10, dtype=np.float32)  # below the device threshold: CPU
  assert np.array_equal(f(small), small * 2 + 1)


@needs_metal
def test_metal_float64_falls_back_with_a_warning():
  @jit(backend="metal")
  def f(x):
    return np.sin(x) * 2

  x = np.linspace(0, 1, 50_000)
  with pytest.warns(parakeet.FallbackWarning, match="float64"):
    r = f(x)
  assert np.allclose(r, np.sin(x) * 2)
  with parakeet.options(device_fallback="error"):
    g = jit(lambda x: np.cos(x) + 1, backend="metal")
    with pytest.raises(parakeet.BackendUnavailable, match="float64"):
      g(x)


@needs_metal
def test_metal_reports_runtime_errors():
  @jit(backend="metal")
  def f(x, idx):
    return x[idx] + np.float32(1)

  x = np.zeros(50_000, dtype=np.float32)
  with pytest.raises(IndexError, match="out of bounds"):
    f(x, np.arange(50_000) * 2)


@needs_metal
def test_metal_shared_memory_tiled_kernels():
  @jit(backend="metal")
  def matmul(a, b):
    return a @ b

  @jit(backend="metal")
  def fused_mlp(a, b, c):
    return np.tanh(a @ b + c)

  @jit(backend="metal")
  def fused_prologue_epilogue(x, y, bias):
    return np.tanh(parakeet.allpairs(lambda u, v: np.sum(np.sin(u) * np.cos(v)), x, y) + bias)

  @jit(backend="metal")
  def tropical(x, y):
    return parakeet.allpairs(lambda u, v: np.min(u + v), x, y)

  @jit(backend="metal")
  def large_tanh(x):
    return np.tanh(x)

  rng = np.random.default_rng(101)
  with parakeet.options(device_threshold=1, blas=False):
    # Regression test: Apple Metal built-in tanh overflows to NaN for |x| > ~44
    t_in = np.array([-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0], dtype=np.float32)
    np.testing.assert_allclose(large_tanh(t_in), np.tanh(t_in), rtol=1e-5, atol=1e-6)

    for m, k, n in ((1, 1, 1), (17, 33, 19), (65, 35, 67)):
      a = rng.standard_normal((m, k)).astype(np.float32)
      b = rng.standard_normal((k, n)).astype(np.float32)
      c = rng.standard_normal((m, n)).astype(np.float32)
      np.testing.assert_allclose(matmul(a, b), a @ b, rtol=1e-3, atol=1e-3)
      np.testing.assert_allclose(fused_mlp(a, b, c), np.tanh(a @ b + c), rtol=1e-3, atol=1e-3)

    x = rng.standard_normal((65, 35)).astype(np.float32)
    y = rng.standard_normal((67, 35)).astype(np.float32)
    bias = rng.standard_normal((65, 67)).astype(np.float32)
    np.testing.assert_allclose(
      fused_prologue_epilogue(x, y, bias), np.tanh(np.sin(x) @ np.cos(y).T + bias), rtol=1e-3, atol=1e-3
    )
    spec_ap = fused_prologue_epilogue.specialize(x, y, bias)
    assert len(spec_ap.kernels) == 1
    assert "threadgroup " in spec_ap.kernels[0] and "threadgroup_barrier" in spec_ap.kernels[0]

    np.testing.assert_allclose(tropical(x, y), (x[:, None, :] + y[None, :, :]).min(-1), rtol=1e-4)
    with pytest.raises(ValueError, match="no identity"):
      tropical(np.zeros((3, 0), dtype=np.float32), np.zeros((4, 0), dtype=np.float32))
    with pytest.raises(ValueError, match="broadcast"):
      tropical(np.zeros((3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))

    ba = rng.standard_normal((3, 33, 19)).astype(np.float32)
    bb = rng.standard_normal((3, 19, 35)).astype(np.float32)
    np.testing.assert_allclose(matmul(ba, bb), ba @ bb, rtol=1e-3, atol=1e-3)
    assert any("threadgroup " in k for k in matmul.specialize(ba, bb).kernels)

    # TargetProfile skips CPU tile_acc interchange for float32 on Metal (warps coalesce X[k*N+j])
    @jit(backend="metal")
    def col_sum_gpu(m):
      return np.sum(m, axis=0)

    cm = rng.standard_normal((32, 300)).astype(np.float32)
    np.testing.assert_allclose(col_sum_gpu(cm), cm.sum(axis=0), rtol=1e-4)
    assert "tile_acc" not in col_sum_gpu.specialize(cm).ir


def run_script(tmp_path, code, **env_extra):
  """Run ``code`` as a script in a fresh interpreter (compiled functions need a source file)."""
  script = tmp_path / "script.py"
  script.write_text(code)
  env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), *sys.path]), **env_extra)
  return subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=900)


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_cuda_kernels_run_in_the_simulator(tmp_path):
  """No NVIDIA hardware needed: the generated CUDA source is compiled as C++
  and its thread grid is executed on the CPU."""
  code = """
import numpy as np, warnings, parakeet
from parakeet import jit
from parakeet.testing import same
import tests.test_backends as tb
assert parakeet.available_backends()['cuda'] == 'ok'
with warnings.catch_warnings():
    warnings.simplefilter('error', parakeet.FallbackWarning)
    tb.run_everywhere('cuda', np.float64, 1e-10)
    tb.run_everywhere('cuda', np.float32, 2e-6)
    tb.aliased_arguments('cuda', np.float64)
f = jit(lambda x, idx: x[idx] * 2, backend='cuda')
x = np.zeros(50_000)
spec = f.specialize(x, np.arange(50_000))
assert '__global__ void pk_kernel' in spec.kernels[0]
try:
    f(x, np.arange(50_000) * 2)
except IndexError as e:
    assert 'out of bounds' in str(e)
else:
    raise SystemExit('expected IndexError')
assert spec.module.kernels[0].device.compiled is not None
print('simulated-cuda-ok')
"""
  out = run_script(tmp_path, code, PARAKEET_CUDA_SIMULATE="1")
  assert "simulated-cuda-ok" in out.stdout, out.stdout + out.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_cuda_shared_memory_tiled_kernels_in_simulator(tmp_path):
  """Verify cooperative __shared__ + __syncthreads() block-tiled CUDA kernels
  (matmul, fused prologue+contraction+epilogue, unseeded allpairs, 3D batched)."""
  code = """
import numpy as np, parakeet
from parakeet import allpairs, jit

@jit(backend='cuda')
def matmul(a, b):
    return a @ b

@jit(backend='cuda')
def fused_prologue_epilogue(x, y, bias):
    return np.tanh(allpairs(lambda u, v: np.sum(np.sin(u) * np.cos(v)), x, y) + bias)

@jit(backend='cuda')
def tropical(x, y):
    return allpairs(lambda u, v: np.min(u + v), x, y)

rng = np.random.default_rng(102)
with parakeet.options(device_threshold=1, blas=False):
    for dtype, tol in ((np.float32, 1e-3), (np.float64, 1e-10)):
        a = rng.standard_normal((19, 35)).astype(dtype)
        b = rng.standard_normal((35, 21)).astype(dtype)
        np.testing.assert_allclose(matmul(a, b), a @ b, rtol=tol, atol=tol)
        assert any('__shared__' in k and '__syncthreads()' in k for k in matmul.specialize(a, b).kernels)

    x = rng.standard_normal((19, 35)).astype(np.float32)
    y = rng.standard_normal((23, 35)).astype(np.float32)
    bias = rng.standard_normal((19, 23)).astype(np.float32)
    np.testing.assert_allclose(
        fused_prologue_epilogue(x, y, bias), np.tanh(np.sin(x) @ np.cos(y).T + bias), rtol=1e-3, atol=1e-3
    )
    spec_ap = fused_prologue_epilogue.specialize(x, y, bias)
    assert len(spec_ap.kernels) == 1 and '__shared__' in spec_ap.kernels[0] and '__syncthreads()' in spec_ap.kernels[0]

    np.testing.assert_allclose(tropical(x, y), (x[:, None, :] + y[None, :, :]).min(-1), rtol=1e-4)
    for bad_x, bad_y, match in (
        (np.zeros((3, 0), dtype=np.float32), np.zeros((4, 0), dtype=np.float32), 'no identity'),
        (np.zeros((3, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32), 'broadcast'),
    ):
        try:
            tropical(bad_x, bad_y)
        except ValueError as e:
            assert match in str(e), str(e)
        else:
            raise SystemExit(f'expected ValueError ({match})')

    ba = rng.standard_normal((3, 17, 19)).astype(np.float32)
    bb = rng.standard_normal((3, 19, 21)).astype(np.float32)
    np.testing.assert_allclose(matmul(ba, bb), ba @ bb, rtol=1e-3, atol=1e-3)
    assert any('__shared__' in k for k in matmul.specialize(ba, bb).kernels)

    # TargetProfile skips CPU tile_acc interchange on CUDA (warps coalesce X[k*N+j] directly)
    @jit(backend='cuda')
    def col_sum_cuda(m):
        return np.sum(m, axis=0)

    cm = rng.standard_normal((32, 300)).astype(np.float32)
    np.testing.assert_allclose(col_sum_cuda(cm), cm.sum(axis=0), rtol=1e-4)
    assert 'tile_acc' not in col_sum_cuda.specialize(cm).ir
print('cuda-shared-tile-ok')
"""
  out = run_script(tmp_path, code, PARAKEET_CUDA_SIMULATE="1")
  assert "cuda-shared-tile-ok" in out.stdout, out.stdout + out.stderr


def test_cuda_without_cupy_explains_what_to_install():
  if AVAILABLE.get("cuda") == "ok":
    pytest.skip("CUDA is available here")
  with pytest.raises(parakeet.BackendUnavailable, match="cupy|CuPy|CUDA"):
    jit(lambda x: x + 1, backend="cuda")(np.zeros(4))


def test_unknown_backend_and_setting_names():
  with pytest.raises(ValueError, match="unknown backend"):
    jit(lambda x: x, backend="quantum")
  with pytest.raises(AttributeError, match="no setting"):
    parakeet.config.bounds_chek = False
  with pytest.raises(ValueError, match="unknown backend"):
    parakeet.config.backend = "fortran"
  assert parakeet.config.backend in parakeet.config.as_dict()["backend"]


def test_options_context_manager():
  before = parakeet.config.num_threads
  with parakeet.options(num_threads=2, bounds_check=False):
    assert parakeet.config.num_threads == 2 and not parakeet.config.bounds_check
  assert parakeet.config.num_threads == before and parakeet.config.bounds_check


def test_compiled_functions_are_thread_safe():
  @jit
  def work(x, k):
    y = np.sqrt(x) * k
    return y.sum(), y

  x = np.arange(200_000, dtype=np.float64)
  errors = []

  def worker(k):
    try:
      for _ in range(20):
        total, y = work(x, float(k))
        assert np.isclose(total, np.sqrt(x).sum() * k)
        assert np.allclose(y[:5], np.sqrt(x[:5]) * k)
    except BaseException as e:  # noqa: BLE001
      errors.append(e)

  threads = [threading.Thread(target=worker, args=(k,)) for k in range(1, 9)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert not errors, errors


def test_concurrent_first_calls_compile_once():
  @jit
  def f(x):
    return x * 3 + 1

  results = []
  barrier = threading.Barrier(6)

  def worker():
    barrier.wait()
    results.append(f(np.arange(10.0)))

  threads = [threading.Thread(target=worker) for _ in range(6)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert len(results) == 6 and all(np.array_equal(r, np.arange(10.0) * 3 + 1) for r in results)
  assert len(f.specializations) == 1


def _rss_mb() -> float:
  import resource

  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


@pytest.mark.skipif(sys.platform == "win32", reason="uses the resource module")
def test_no_memory_leaks():
  @jit
  def churn(x, n):
    acc = x
    for i in range(n):
      tmp = np.sqrt(acc * acc + 1.0)  # a fresh array every iteration
      acc = tmp[::-1].copy() if i % 2 else tmp
    return acc, acc[1:], np.cumsum(acc)

  @jit
  def fails(x):
    y = x * 2.0
    z = np.cumsum(y)
    return z[len(x) + 5]

  x = np.random.default_rng(0).random(250_000)  # 2 MB per array
  churn(x, 4)
  gc.collect()
  base = _rss_mb()
  for _ in range(30):
    churn(x, 20)  # allocates ~40 arrays (80 MB) per call if nothing were freed
    with pytest.raises(IndexError):
      fails(x)
  gc.collect()
  grown = _rss_mb() - base
  assert grown < 60, f"resident memory grew by {grown:.0f} MB"


@pytest.mark.skipif(not hasattr(os, "fork") or "threads" not in CPU_BACKENDS, reason="needs fork and the thread pool")
def test_thread_pool_survives_fork(tmp_path):
  code = """
import os, warnings, numpy as np
from parakeet import jit
warnings.simplefilter("ignore")

@jit(backend="threads")
def total(x):
    return np.sum(np.sqrt(x))

x = np.arange(2_000_000, dtype=np.float64)
expect = np.sqrt(x).sum()
assert np.isclose(total(x), expect)      # starts the pool in the parent
pid = os.fork()
if pid == 0:
    os._exit(0 if np.isclose(total(x), expect) else 1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
assert np.isclose(total(x), expect)
print("fork-ok")
"""
  out = run_script(tmp_path, code)
  assert "fork-ok" in out.stdout, out.stdout + out.stderr


def test_results_outlive_their_function_and_inputs():
  @jit
  def make(n):
    return np.arange(n) * 2.0

  @jit
  def view(x):
    return x[::2], x.T

  a = make(1000)
  b = a[10:20]
  del a
  gc.collect()
  assert np.array_equal(b, np.arange(10, 20) * 2.0)
  x = np.arange(12.0).reshape(3, 4)
  v, t = view(x)
  del x
  gc.collect()
  assert v.shape == (2, 4) and t.shape == (4, 3) and t[1, 2] == 9.0


def test_cache_survives_across_processes(tmp_path):
  code = """
import numpy as np, time, parakeet
from parakeet import jit
@jit
def f(x): return np.sum(x * 2.5 - 1.25)
t = time.perf_counter(); r = f(np.arange(10.0)); print(r, time.perf_counter() - t)
"""
  cache = tmp_path / "cache"
  runs = [run_script(tmp_path, code, PARAKEET_CACHE_DIR=str(cache)) for _ in range(2)]
  for r in runs:
    assert r.returncode == 0, r.stderr
  v1, t1 = runs[0].stdout.split()
  v2, t2 = runs[1].stdout.split()
  assert v1 == v2 == str(float(np.sum(np.arange(10.0) * 2.5 - 1.25)))
  assert len(list(cache.glob("pk_mod_*"))) == 1  # second process reused the compiled module


def test_missing_compiler_is_explained(tmp_path):
  code = """
import numpy as np, parakeet
from parakeet import jit
parakeet.config.cc = '/nonexistent/compiler'
try:
    jit(lambda x: x + 1)(np.zeros(3))
except parakeet.BackendUnavailable as e:
    msg = str(e)
    assert 'C compiler' in msg and 'PARAKEET_CC' in msg and "backend='python'" in msg, msg
    print('explained')
"""
  out = run_script(tmp_path, code, PARAKEET_CACHE_DIR=str(tmp_path / "cache"))
  assert "explained" in out.stdout, out.stdout + out.stderr


def test_functions_without_source_are_explained():
  ns = {}
  exec("def f(x):\n    return x + 1\n", ns)  # noqa: S102
  with pytest.raises(parakeet.UnsupportedError, match="cannot find the source code"):
    jit(ns["f"])(1)


def resident_arrays(backend, dtype=np.float32):
  """Arrays stay on the device between the loops of one call, and host code
  in between still sees (and is seen by) what the device computed."""
  dev = parakeet.api.get_backend(backend).device()
  counts = {"up": 0, "down": 0}
  cls = type(dev)
  upload, download = cls.upload, cls.download

  def counting_upload(self, addr, nbytes, external, untouched=0):
    counts["up"] += 0 if untouched else 1  # a buffer the host never wrote is not copied
    return upload(self, addr, nbytes, external, untouched)

  def counting_download(self, d, addr, nbytes):
    counts["down"] += 1
    return download(self, d, addr, nbytes)

  @jit
  def chain(x, steps):
    for _ in range(steps):
      x = np.sin(x) * 0.5 + x * x * 0.25
    return x

  @jit
  def mixed(x):
    y = np.tanh(x) * 2.0  # device
    y[0] = 7.0  # the host stores into what the device just wrote
    z = y + 1.0  # the device has to see that store
    s = 0.0
    for i in range(4):
      s += z[i]  # the host reads what the device wrote
    w = np.sqrt(np.abs(z))
    return s, w, np.cumsum(w[:1000]), w[w > 1.6], np.argmax(w)

  x = np.random.default_rng(0).random(200_000).astype(dtype)
  cls.upload, cls.download = counting_upload, counting_download
  try:
    with parakeet.options(backend=backend, device_threshold=1, device_fallback="error"):
      before = dev.launches
      got = chain(x, 10)
      assert dev.launches - before == 10
      # one upload of the input and one download of the result, not ten of each
      # (Metal shares host memory instead of copying, and reports every buffer it wraps)
      assert counts["down"] == 1 and (counts["up"] == 1 if backend == "cuda" else counts["up"] <= 11), counts
      got_mixed = mixed(x)
  finally:
    cls.upload, cls.download = upload, download
  want = x.astype(np.float64)
  for _ in range(10):
    want = np.sin(want) * 0.5 + want * want * 0.25
  np.testing.assert_allclose(got, want, rtol=1e-4)
  y = np.tanh(x.astype(np.float64)) * 2.0
  y[0] = 7.0
  z = y + 1.0
  w = np.sqrt(np.abs(z))
  s, gw, gc, gf, ga = got_mixed
  assert np.isclose(s, z[:4].sum(), rtol=1e-5)
  np.testing.assert_allclose(gw, w, rtol=1e-5)
  np.testing.assert_allclose(gc, np.cumsum(w[:1000]), rtol=1e-4)
  np.testing.assert_allclose(gf, w[w > 1.6], rtol=1e-5)
  assert ga == np.argmax(gw)


@needs_metal
def test_metal_keeps_arrays_on_the_device():
  resident_arrays("metal")


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_cuda_simulator_keeps_arrays_on_the_device(tmp_path):
  code = """
import numpy as np
import tests.test_backends as tb
tb.resident_arrays('cuda', np.float32)
tb.resident_arrays('cuda', np.float64)
print('resident-ok')
"""
  proc = run_script(tmp_path, code, PARAKEET_CUDA_SIMULATE="1")
  assert proc.returncode == 0 and "resident-ok" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_cuda_split_k_wmma_in_simulator(tmp_path):
  """Verify Split-K WMMA codegen and exact numerical correctness for low-grid,
  deep-K weight-gradient GEMMs (e.g. dW = X.T @ dY)."""
  code = """
import numpy as np, parakeet
from parakeet import jit

@jit(backend='cuda')
def dw_gemm(x, dy):
    return x.T @ dy

rng = np.random.default_rng(77)
x = rng.standard_normal((512, 64)).astype(np.float32)
dy = rng.standard_normal((512, 64)).astype(np.float32)
with parakeet.options(device_threshold=1, blas=False, tensor_cores=True):
    spec = dw_gemm.specialize(x, dy)
    src = "\\n".join(spec.kernels)
    assert "nvcuda::wmma" in src, src
    assert "pk_split_k" in src and "atomicAdd" in src, src
    got = dw_gemm(x, dy)
want = x.T @ dy
np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
print('split-k-ok')
"""
  proc = run_script(tmp_path, code, PARAKEET_CUDA_SIMULATE="1")
  assert proc.returncode == 0 and "split-k-ok" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="NativeCudaRuntime uses POSIX pthreads/dlopen")
def test_native_cuda_rt_thread_isolation():
  """Verify that two concurrent host threads with distinct streams, kernel descriptors,
  buffer tables, and deferred error slots stay completely isolated in ``pk_cuda_rt``."""
  import ctypes
  import threading

  class Buf(ctypes.Structure):
    _fields_ = [
      ("data", ctypes.c_void_p),
      ("size", ctypes.c_int64),
      ("refs", ctypes.c_int64),
      ("flags", ctypes.c_int64),
    ]

  from parakeet.backends.cuda_rt import NativeCudaRuntime, PkCuKDesc

  rt = NativeCudaRuntime(init_driver=False)
  lib = rt.lib

  # Minimal in-process mock CUDA driver backed by host malloc/memcpy
  launched_records: list[tuple[int, int, int]] = []
  rec_lock = threading.Lock()
  drained = threading.Event()
  owners = []
  released = []
  alloc_ptrs: dict[int, ctypes.Array] = {}

  @ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
  )
  def mock_launch(fn, gx, gy, gz, bx, by, bz, shmem, stream, kargs, extra):
    with rec_lock:
      launched_records.append((int(fn or 0), int(stream or 0), int(gx)))
    return 0

  @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t)
  def mock_alloc(out_ptr, nbytes):
    buf = (ctypes.c_uint8 * max(nbytes, 1))()
    addr = ctypes.addressof(buf)
    with rec_lock:
      alloc_ptrs[addr] = buf
    out_ptr[0] = addr
    return 0

  @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64)
  def mock_free(dptr):
    with rec_lock:
      del alloc_ptrs[int(dptr)]
      released.append(dptr)
      if len(released) == 2:
        drained.set()
    return 0

  @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t, ctypes.c_void_p)
  def mock_memset(dptr, val, n, stream):
    ctypes.memset(int(dptr), int(val), int(n))
    return 0

  @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
  def mock_htod(dst, src, n, stream):
    ctypes.memmove(int(dst), int(src), int(n))
    return 0

  @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t)
  def mock_dtoh(dst, src, n):
    ctypes.memmove(int(dst), int(src), int(n))
    return 0

  @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
  def mock_sync(stream):
    return 0

  @ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64)
  def mock_fallback(k, ctx, total, nchunks):
    return 99

  rc = lib.pk_cu_init_custom(
    ctypes.cast(mock_launch, ctypes.c_void_p),
    ctypes.cast(mock_alloc, ctypes.c_void_p),
    ctypes.cast(mock_free, ctypes.c_void_p),
    ctypes.cast(mock_memset, ctypes.c_void_p),
    ctypes.cast(mock_htod, ctypes.c_void_p),
    ctypes.cast(mock_dtoh, ctypes.c_void_p),
    ctypes.cast(mock_sync, ctypes.c_void_p),
  )
  assert rc == 0

  barrier = threading.Barrier(2)
  errors: list[BaseException] = []
  results: dict[int, tuple[int, int, list[int]]] = {}

  def worker(tid: int):
    try:
      stream_id = 0x1000 + tid * 0x100
      fn_id = 0x2000 + tid * 0x100
      descs = (PkCuKDesc * 1)()
      descs[0].cu_fn = fn_id
      descs[0].is_map = 1
      descs[0].has_errors = 1 if tid == 1 else 0
      descs[0].block_threads = 64
      descs[0].n_scalars = 0
      descs[0].n_bufs = 2
      descs[0].buf_offsets[1] = ctypes.sizeof(ctypes.c_void_p)

      err_flags = (ctypes.c_int32 * 8)()
      err_infos = (ctypes.c_int64 * 32)()
      flag0 = ctypes.c_int32(0)
      info0 = (ctypes.c_int64 * 4)()

      lib.pk_cu_begin_call(
        stream_id,
        8,
        ctypes.cast(mock_fallback, ctypes.c_void_p),
        ctypes.addressof(descs),
        1,
        ctypes.addressof(flag0),
        ctypes.addressof(info0),
        ctypes.addressof(err_flags),
        ctypes.addressof(err_infos),
        8,
      )
      # Both threads overlap inside active calls simultaneously!
      barrier.wait()
      data = [(ctypes.c_uint8 * 64)() for _ in range(2)]
      bufs = [Buf(ctypes.addressof(a), 64, 1, 0) for a in data]
      context = (ctypes.c_void_p * 2)(*(ctypes.addressof(b) for b in bufs))
      for _ in range(3 + tid):
        assert lib.pk_cu_dispatch(0, ctypes.c_void_p(ctypes.addressof(context)), 128, 1) == 0
      barrier.wait()

      # One output outlives its originating thread. The other allocation goes
      # into the stream-bound pool and should be freed when the thread exits.
      ptr, cap = ctypes.c_uint64(), ctypes.c_int64()
      dirty, ctx, generation = ctypes.c_int32(), ctypes.c_void_p(), ctypes.c_uint64()
      assert lib.pk_cu_take_output(bufs[0].data, 64, ptr, cap, dirty, ctx, generation)
      owners.append(rt.make_owner(ptr.value, cap.value, ctx.value, generation.value))

      out_dirty = (ctypes.c_int32 * 1)()
      out_inval = (ctypes.c_int32 * 1)()
      out_stats = (ctypes.c_int64 * 7)()
      out_err_n = ctypes.c_int32(0)
      out_err_k = (ctypes.c_int32 * 8)()
      lib.pk_cu_end_call(out_dirty, out_inval, 0, out_stats, ctypes.byref(out_err_n), out_err_k)
      results[tid] = (int(out_stats[0]), int(out_err_n.value), [int(out_err_k[i]) for i in range(out_err_n.value)])
    except BaseException as e:  # noqa: BLE001
      errors.append(e)

  t0 = threading.Thread(target=worker, args=(0,))
  t1 = threading.Thread(target=worker, args=(1,))
  t0.start()
  t1.start()
  t0.join()
  t1.join()

  assert not errors, errors
  assert drained.wait(5) and len(alloc_ptrs) == 2
  owners.clear()
  assert not alloc_ptrs
  assert results[0] == (3, 0, [])
  assert results[1] == (4, 4, [0, 0, 0, 0])
  # Verify every launch matched its own thread's (fn_id, stream_id) pair!
  assert sum(1 for fn, st, _ in launched_records if fn == 0x2000 and st == 0x1000) == 3
  assert sum(1 for fn, st, _ in launched_records if fn == 0x2100 and st == 0x1100) == 4


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_cuda_double_buffered_wmma_and_warp_row_reduce_in_simulator(tmp_path):
  """Verify shape-adaptive double-buffered cp.async WMMA (NN/NT/TN/TT + 3D
  reshaped epilogue fusion) and 1-warp-per-row coalesced CUDA row reductions."""
  code = """
import numpy as np, parakeet
from parakeet import jit

@jit(backend='cuda')
def mm_all(a, b, bt, at):
    return a @ b, a @ bt.T, at.T @ b, at.T @ bt.T

@jit(backend='cuda')
def mlp_3d(x, w, bias):
    return np.tanh(x @ w + bias)

@jit(backend='cuda')
def ln_and_stats(x, g, b):
    mu = np.mean(x, axis=2, keepdims=True)
    var = np.mean((x - mu) ** np.float32(2.0), axis=2, keepdims=True)
    y = (x - mu) / np.sqrt(var + np.float32(1e-5)) * g + b
    return y, np.max(x, axis=2), np.min(x, axis=2)

rng = np.random.default_rng(19)
with parakeet.options(device_threshold=1, blas=False, tensor_cores=True, verify=True):
    a = rng.standard_normal((130, 35)).astype(np.float32) * 0.2
    b = rng.standard_normal((35, 65)).astype(np.float32) * 0.2
    bt = rng.standard_normal((65, 35)).astype(np.float32) * 0.2
    at = rng.standard_normal((35, 130)).astype(np.float32) * 0.2
    spec_mm = mm_all.specialize(a, b, bt, at)
    src_mm = "\\n".join(spec_mm.kernels)
    assert "pk_cp_async_16" in src_mm and "col_major" in src_mm, src_mm
    nn, nt, tn, tt = mm_all(a, b, bt, at)
    np.testing.assert_allclose(nn, a @ b, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(nt, a @ bt.T, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(tn, at.T @ b, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(tt, at.T @ bt.T, rtol=1e-3, atol=1e-3)

    x3 = rng.standard_normal((3, 17, 35)).astype(np.float32) * 0.2
    w3 = rng.standard_normal((35, 67)).astype(np.float32) * 0.2
    bias3 = rng.standard_normal((67,)).astype(np.float32) * 0.1
    spec_mlp = mlp_3d.specialize(x3, w3, bias3)
    assert len(spec_mlp.kernels) == 1 and "mma_sync" in spec_mlp.kernels[0], spec_mlp.kernels
    np.testing.assert_allclose(mlp_3d(x3, w3, bias3), np.tanh(x3 @ w3 + bias3), rtol=1e-3, atol=1e-3)

    x_ln = rng.standard_normal((4, 13, 65)).astype(np.float32) * 2.0 - 1.5
    g_ln = rng.standard_normal((65,)).astype(np.float32)
    b_ln = rng.standard_normal((65,)).astype(np.float32)
    spec_ln = ln_and_stats.specialize(x_ln, g_ln, b_ln)
    assert any("pk_shfl_down" in k for k in spec_ln.kernels), spec_ln.kernels
    y_g, mx_g, mn_g = ln_and_stats(x_ln, g_ln, b_ln)
    mu = np.mean(x_ln, axis=2, keepdims=True)
    var = np.mean((x_ln - mu) ** np.float32(2.0), axis=2, keepdims=True)
    y_ref = (x_ln - mu) / np.sqrt(var + np.float32(1e-5)) * g_ln + b_ln
    np.testing.assert_allclose(y_g, y_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mx_g, np.max(x_ln, axis=2), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mn_g, np.min(x_ln, axis=2), rtol=1e-5, atol=1e-5)
print("wmma-warp-ok")
"""
  proc = run_script(tmp_path, code, PARAKEET_CUDA_SIMULATE="1")
  assert proc.returncode == 0 and "wmma-warp-ok" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="the simulator needs a POSIX C++ toolchain")
def test_cuda_row_pipeline_adam_batch_atomic_split_and_dedup_in_simulator(tmp_path):
  """Verify multi-stage + epilogue WarpRowReducePlan (LayerNorm fwd/bwd),
  disjoint-union horizontal map batching (heterogeneous Adam), single-pass
  atomic split_gpu_reduce, alpha-canonical kernel dedup, and broadcast causal
  mask FlashAttention backward in the CUDA simulator."""
  code = """
import numpy as np, parakeet
from parakeet import jit, value_and_grad
from parakeet.backends.cuda import _canonicalize_cuda_kernel

def ln_loss(x, g, b):
    mu = np.mean(x, axis=2, keepdims=True)
    var = np.mean((x - mu) ** np.float32(2.0), axis=2, keepdims=True)
    y = (x - mu) / np.sqrt(var + np.float32(1e-5)) * g + b
    return np.sum(y * y)

_vg_ln = value_and_grad(ln_loss, wrt=(0, 1, 2))

@jit(backend='cuda')
def ln_vg_cuda(x, g, b):
    return _vg_ln(x, g, b)

@jit(backend='c')
def ln_vg_cpu(x, g, b):
    return _vg_ln(x, g, b)

@jit(backend='cuda')
def adam_multi(p0, g0, m0, v0, p1, g1, m1, v1, p2, g2, m2, v2, lr):
    b1, b2 = np.float32(0.9), np.float32(0.99)
    def upd(p, g, m, v):
        mn = b1 * m + (np.float32(1.0) - b1) * g
        vn = b2 * v + (np.float32(1.0) - b2) * g * g
        pn = p - (lr * mn) / (np.sqrt(vn) + np.float32(1e-8))
        return pn, mn, vn
    r0 = upd(p0, g0, m0, v0)
    r1 = upd(p1, g1, m1, v1)
    r2 = upd(p2, g2, m2, v2)
    return r0, r1, r2

def sdpa_loss(q, k, v, causal):
    scores = (q @ k.transpose((0, 1, 3, 2))) * np.float32(1.0 / np.sqrt(q.shape[3]))
    s = np.where(causal, scores, np.float32(-1e9))
    m = np.max(s, axis=3, keepdims=True)
    e = np.exp(s - m)
    w = e / np.sum(e, axis=3, keepdims=True)
    return np.sum((w @ v) ** np.float32(2.0))

_vg_sdpa = value_and_grad(sdpa_loss, wrt=(0, 1, 2))

@jit(backend='cuda')
def sdpa_vg_cuda(q, k, v, causal):
    return _vg_sdpa(q, k, v, causal)

rng = np.random.default_rng(77)
with parakeet.options(device_threshold=1, blas=False, verify=True):
    # 1. LayerNorm fwd + bwd: single fused WarpRowReducePlan kernels (pk_shfl_bcast) + single-pass atomic d_g/d_b
    x = rng.standard_normal((8, 16, 64)).astype(np.float32)
    g = rng.standard_normal((64,)).astype(np.float32)
    b = rng.standard_normal((64,)).astype(np.float32)
    spec_ln = ln_vg_cuda.specialize(x, g, b)
    assert any("pk_shfl_bcast" in k for k in spec_ln.kernels), spec_ln.kernels
    assert any("atomicAdd" in k for k in spec_ln.kernels), spec_ln.kernels
    l_g, (dx_g, dg_g, db_g) = ln_vg_cuda(x, g, b)
    l_c, (dx_c, dg_c, db_c) = ln_vg_cpu(x, g, b)
    np.testing.assert_allclose(float(l_g), float(l_c), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dx_g, dx_c, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dg_g, dg_c, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(db_g, db_c, rtol=1e-4, atol=1e-4)

    # 2. Heterogeneous-shape Adam batches into 1 disjoint-union CUDA kernel
    s0 = tuple(rng.standard_normal((17, 33)).astype(np.float32) for _ in range(4))
    s1 = tuple(rng.standard_normal((11, 29)).astype(np.float32) for _ in range(4))
    s2 = tuple(rng.standard_normal((53,)).astype(np.float32) for _ in range(4))
    s0 = (s0[0], s0[1], np.zeros_like(s0[0]), np.zeros_like(s0[0]))
    s1 = (s1[0], s1[1], np.zeros_like(s1[0]), np.zeros_like(s1[0]))
    s2 = (s2[0], s2[1], np.zeros_like(s2[0]), np.zeros_like(s2[0]))
    lr = np.float32(1e-3)
    spec_a = adam_multi.specialize(*s0, *s1, *s2, lr)
    assert len(spec_a.kernels) == 1, f"expected 1 batched Adam kernel, got {len(spec_a.kernels)}"
    r0, r1, r2 = adam_multi(*s0, *s1, *s2, lr)
    for (p, gg, m, v), (pn, mn, vn) in zip((s0, s1, s2), (r0, r1, r2)):
        mn_r = np.float32(0.9) * m + np.float32(0.1) * gg
        vn_r = np.float32(0.99) * v + np.float32(0.01) * gg * gg
        pn_r = p - (lr * mn_r) / (np.sqrt(vn_r) + np.float32(1e-8))
        np.testing.assert_allclose(pn, pn_r, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mn, mn_r, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(vn, vn_r, rtol=1e-5, atol=1e-5)

    # 3. Alpha-canonical CUDA kernel deduplication collapses IR UID differences
    k1 = 'extern "C" __global__ void pk_kernel(long long n_10, float* arr_11_d) { arr_11_d[0] = (float)n_10; }'
    k2 = 'extern "C" __global__ void pk_kernel(long long n_99, float* arr_42_d) { arr_42_d[0] = (float)n_99; }'
    assert _canonicalize_cuda_kernel(k1) == _canonicalize_cuda_kernel(k2)

    # 4. Batched multi-head SDPA with broadcast (1, 1, T, T) causal mask
    q = rng.standard_normal((2, 2, 16, 16)).astype(np.float32)
    k = rng.standard_normal((2, 2, 16, 16)).astype(np.float32)
    v = rng.standard_normal((2, 2, 16, 16)).astype(np.float32)
    causal = np.tril(np.ones((16, 16), dtype=bool))[None, None, :, :]
    sc = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(16))
    sm = np.where(causal, sc, np.float32(-1e9))
    em = np.exp(sm - np.max(sm, axis=3, keepdims=True))
    w_r = em / np.sum(em, axis=3, keepdims=True)
    do_r = np.float32(2.0) * (w_r @ v)
    dv_r = w_r.transpose(0, 1, 3, 2) @ do_r
    _, (dq_g, dk_g, dv_g) = sdpa_vg_cuda(q, k, v, causal)
    np.testing.assert_allclose(dv_g, dv_r, rtol=1e-4, atol=1e-4)
print("fusion-sim-ok")
"""
  proc = run_script(tmp_path, code, PARAKEET_CUDA_SIMULATE="1")
  assert proc.returncode == 0 and "fusion-sim-ok" in proc.stdout, proc.stdout + proc.stderr
