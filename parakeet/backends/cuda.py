"""NVIDIA GPU backend: parallel loops become CUDA kernels run through CuPy.

Kernels are compiled by CuPy's ``RawKernel`` (NVRTC), so the only requirements
are a CUDA-capable GPU, its driver, and the ``cupy`` package matching the
installed CUDA version - no ``nvcc`` and no build step.

Setting ``PARAKEET_CUDA_SIMULATE=1`` swaps the GPU for a *simulator* that
compiles the very same kernel source with the host C++ compiler and executes
its thread grid on the CPU.  It exists so the code generator can be tested on
machines without NVIDIA hardware; it is not a way to run CUDA code fast.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import struct
import threading

import numpy as np

from ..config import config
from ..errors import BackendUnavailable, InternalError
from ..ndtypes import BufType, ScalarType
from ..syntax import ir
from ..target import MatrixUnit, TargetProfile
from . import runtime_c, toolchain
from .cfamily import CFamilyEmitter, Unsupported, defined_vars
from .cuda_rt import MAX_BUFS, MAX_SCALARS, MAX_TILED_DIMS, PkCuKDesc, try_load_native_cuda_rt
from .device import (
  DeviceArray,
  DeviceBase,
  DevicePlanner,
  KernelPlan,
  _Resident,
  ctype_of,
  make_runtime_error,
  merge_regions,
  untouched_regions,
)
from .monoidal_gpu import generate_monoidal_gpu_kernel
from .tiled_gpu import BLOCK_THREADS, generate_tiled_gpu_kernel

BLOCK = 256

_CUDA_TYPES = {
    "bool": "bool", "int8": "signed char", "int16": "short", "int32": "int", "int64": "long long",
    "uint8": "unsigned char", "uint16": "unsigned short", "uint32": "unsigned int",
    "uint64": "unsigned long long", "float32": "float", "float64": "double",
}  # fmt: skip

PRELUDE = (
  r"""
#define int8_t signed char
#define int16_t short
#define int32_t int
#define int64_t long long
#define uint8_t unsigned char
#define uint16_t unsigned short
#define uint32_t unsigned int
#define uint64_t unsigned long long
#undef INT64_C
#undef UINT64_C
#undef INT64_MIN
#define INT64_C(x) (x##LL)
#define UINT64_C(x) (x##ULL)
#define INT64_MIN (-9223372036854775807LL - 1LL)
#undef INFINITY
#undef NAN
#define INFINITY (1e300 * 1e300)
#define NAN (INFINITY - INFINITY)
#define PK_INLINE __device__ __forceinline__
"""
  + runtime_c.HELPERS_INT
  + runtime_c.HELPERS_FLOAT_DEF
  + runtime_c.HELPERS_F32
  + runtime_c.HELPERS_F64
  + runtime_c.HELPERS_RNG
  + runtime_c.HELPERS_RNG_F64
  + r"""
__device__ void pk_fail(int* flag, long long* info, int id, long long a0, long long a1, long long a2, long long a3) {
  if (atomicCAS(flag, 0, id + 1) == 0) {
    info[0] = a0; info[1] = a1; info[2] = a2; info[3] = a3;
  }
}
"""
)

# Stand-ins for the CUDA runtime so a kernel can be compiled as ordinary C++.
_SIM_LOCK = threading.Lock()

SIM_SHIM = r"""
#include <cmath>
#include <cstring>
using std::isnan; using std::isinf; using std::isfinite; using std::signbit;
#define __device__ static
#define __global__
#define __forceinline__ inline
struct pk_dim3 { unsigned int x, y, z; };
/* not thread_local: a process may load thousands of kernels, and there are only a few
   hundred thread-local slots.  The simulator runs one launch at a time instead. */
static pk_dim3 blockIdx, blockDim, threadIdx;
static inline int atomicCAS(int* p, int compare, int value) {
  int old = *p;
  if (old == compare) *p = value;
  return old;
}
template <typename T>
static inline T atomicAdd(T* p, T value) {
  T old = *p;
  *p = old + value;
  return old;
}
static inline long long min(long long a, long long b) { return a < b ? a : b; }
"""

SIM_SHIM_TILED = r"""
#define PK_CUDA_SIMULATE 1
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
using std::isnan; using std::isinf; using std::isfinite; using std::signbit;
#define __device__ static
#define __global__
#define __forceinline__ inline
#define __shared__ static
struct pk_dim3 { unsigned int x, y, z; };
static pk_dim3 blockIdx, blockDim;
static thread_local pk_dim3 threadIdx;
static std::mutex pk_atomic_mtx;
static inline int atomicCAS(int* p, int compare, int value) {
  std::lock_guard<std::mutex> lk(pk_atomic_mtx);
  int old = *p;
  if (old == compare) *p = value;
  return old;
}
template <typename T>
static inline T atomicAdd(T* p, T value) {
  std::lock_guard<std::mutex> lk(pk_atomic_mtx);
  T old = *p;
  *p = old + value;
  return old;
}
struct pk_sim_barrier_t {
  std::mutex m;
  std::condition_variable cv;
  long long total = 0, count = 0, gen = 0;
  void init(long long n) { total = n; count = 0; gen = 0; }
  void wait() {
    std::unique_lock<std::mutex> lk(m);
    long long g = gen;
    if (++count == total) {
      count = 0;
      ++gen;
      cv.notify_all();
    } else {
      cv.wait(lk, [&] { return gen != g; });
    }
  }
};
static pk_sim_barrier_t pk_sim_barrier;
static pk_sim_barrier_t pk_sim_warp_barriers[8];
static inline void __syncthreads() { pk_sim_barrier.wait(); }
static inline void __syncwarp(unsigned int mask = 0xffffffffu) {
  (void)mask;
  pk_sim_warp_barriers[(threadIdx.x >> 5) & 7].wait();
}
static inline long long min(long long a, long long b) { return a < b ? a : b; }
namespace nvcuda { namespace wmma {
  struct matrix_a {};
  struct matrix_b {};
  struct accumulator {};
  struct row_major {};
  struct col_major {};
  namespace precision { struct tf32 {}; }
  enum layout_t { mem_row_major, mem_col_major };
  template <typename Use, int M, int N, int K, typename T, typename Layout = void>
  struct fragment {
    static constexpr int R = (std::is_same<Use, matrix_b>::value) ? K : M;
    static constexpr int C = (std::is_same<Use, matrix_a>::value) ? K : N;
    float x[R * C];
  };
  template <typename Frag>
  static inline void fill_fragment(Frag& f, float v) {
    for (int i = 0; i < Frag::R * Frag::C; ++i) f.x[i] = v;
  }
  template <typename Frag>
  static inline void load_matrix_sync(Frag& f, const float* p, unsigned int ldm) {
    for (int r = 0; r < Frag::R; ++r)
      for (int c = 0; c < Frag::C; ++c)
        f.x[r * Frag::C + c] = p[r * ldm + c];
  }
  template <int M, int N, int K, typename T, typename LA, typename LB>
  static inline void mma_sync(
      fragment<accumulator, M, N, K, float>& d,
      const fragment<matrix_a, M, N, K, T, LA>& a,
      const fragment<matrix_b, M, N, K, T, LB>& b,
      const fragment<accumulator, M, N, K, float>& c) {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        float acc = c.x[i * N + j];
        for (int k = 0; k < K; ++k) acc += a.x[i * K + k] * b.x[k * N + j];
        d.x[i * N + j] = acc;
      }
  }
  template <int M, int N, int K>
  static inline void store_matrix_sync(float* p, const fragment<accumulator, M, N, K, float>& f, unsigned int ldm, layout_t) {
    if ((threadIdx.x & 31u) == 0u) {
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
          p[r * ldm + c] = f.x[r * N + c];
    }
    __syncwarp();
  }
} }
"""


class CudaEmitter(CFamilyEmitter):
  def ctype(self, t: ScalarType) -> str:
    return _CUDA_TYPES[t.name]

  def int_literal(self, v: int, t: ScalarType) -> str:
    if t.name == "int64":
      if v == -(2**63):
        return "INT64_MIN"
      return f"{v}LL"
    if t.name == "uint64":
      return f"{v % 2**64}ULL"
    return f"(({self.ctype(t)}){v})"

  def rtcall(self, e: ir.RtCall) -> str:
    if not e.pure:
      raise Unsupported(f"the runtime function '{e.name}' is not available on the GPU")
    return f"pk_{e.name}({', '.join(self.expr(a) for a in e.args)})"

  def check(self, s: ir.Check, out: list[str], pad: str) -> None:
    eid = self.error_id(s.err)
    args = [f"(long long)({self.expr(a)})" for a in s.args[:4]]
    args += ["0LL"] * (4 - len(args))
    out.append(f"{pad}if (!({self.expr(s.cond)})) {{ pk_fail(pk_flag, pk_info, {eid}, {', '.join(args)}); return; }}")

  def atomic_store(self, s: ir.Store, ct: str, value: str, out: list[str], pad: str) -> None:
    if s.atomic == "add" and s.buf.type.elt.name in ("float32", "float64", "int32", "uint32"):
      out.append(f"{pad}atomicAdd(&{self.data(s.buf)}[{self.expr(s.index)}], {value});")
      return
    super().atomic_store(s, ct, value, out, pad)

  def release(self, v: ir.Var, out: list[str], pad: str) -> None:
    raise Unsupported("it allocates temporary arrays inside the loop")

  def alloc(self, s: ir.Alloc, out: list[str], pad: str) -> None:
    raise Unsupported("it allocates temporary arrays inside the loop")

  def assign_buf(self, s: ir.Assign, out: list[str], pad: str) -> None:
    raise Unsupported("it selects between arrays at run time inside the loop")

  def incref(self, s: ir.Incref, out: list[str], pad: str) -> None:
    raise Unsupported("it keeps references to temporary arrays inside the loop")


class CudaPlanner(DevicePlanner):
  name = "CUDA"

  def generate(self, loop: ir.ParLoop, free: list[ir.Var]) -> KernelPlan:
    em = CudaEmitter()
    if loop.monoidal_tile is not None:
      mon_plan = generate_monoidal_gpu_kernel(loop, free, "cuda", em, PRELUDE, profile=self.profile)
      if mon_plan is not None:
        return mon_plan
    if loop.shared_tile is not None:
      tiled_plan = generate_tiled_gpu_kernel(loop, free, "cuda", em, PRELUDE, profile=self.profile)
      if tiled_plan is not None:
        return tiled_plan
    defined = defined_vars(loop.body, into_parloops=True)
    scalars = [v for v in free if not isinstance(v.type, BufType)]
    bufs = [v for v in free if isinstance(v.type, BufType)]
    body: list[str] = []
    em.block(loop.body, body, 1)
    params = ["long long pk_total", "long long pk_csize"]
    params += [f"{em.ctype(v.type)} {v.name}" for v in scalars]
    params += ["int* pk_flag", "long long* pk_info"]
    params += [f"{em.ctype(v.type.elt)}* {v.name}_d" for v in bufs]
    lines = [PRELUDE, 'extern "C" __global__ void pk_kernel(' + ",\n    ".join(params) + ") {"]
    lines.append(f"  const long long {loop.chunk.name} = (long long)blockIdx.x * blockDim.x + threadIdx.x;")
    lines.append(f"  const long long {loop.start.name} = {loop.chunk.name} * pk_csize;")
    lines.append(f"  if ({loop.start.name} >= pk_total) return;")
    lines.append(f"  const long long {loop.stop.name} = min({loop.start.name} + pk_csize, pk_total);")
    for _uid, v in sorted(defined.items()):
      if v is loop.chunk or v is loop.start or v is loop.stop:
        continue
      if isinstance(v.type, BufType):
        raise Unsupported("it creates arrays inside the loop")
      init = "false" if v.type.kind == "b" else "0"
      lines.append(f"  {em.ctype(v.type)} {v.name} = {init};")
    lines += body
    lines.append("}")
    written = {s.buf.uid for s in ir.walk_stmts(loop.body) if isinstance(s, ir.Store)}
    return KernelPlan(
            loop=loop, source="\n".join(lines), scalars=scalars, bufs=bufs, errors=em.errors,
            written={v.uid for v in bufs if v.uid in written},
        )  # fmt: skip


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


class CuPyDriver:
  """Runs kernels on a real GPU through CuPy."""

  name = "cupy"

  def __init__(self) -> None:
    try:
      import cupy
    except ImportError as e:
      raise BackendUnavailable(
        "the CUDA backend needs CuPy, which is not installed. Install the wheel matching your "
        'CUDA version, e.g.: pip install "parakeet[cuda12]"  (or: pip install cupy-cuda12x)'
      ) from e
    try:
      count = cupy.cuda.runtime.getDeviceCount()
    except Exception as e:  # noqa: BLE001 - CuPy raises many different errors here
      raise BackendUnavailable(
        f"CuPy is installed but CUDA could not be initialised ({type(e).__name__}: {e}). "
        "Check that an NVIDIA driver is installed and that the CuPy wheel matches your CUDA version."
      ) from e
    if count < 1:
      raise BackendUnavailable("CuPy is installed but no CUDA device was found on this machine")
    self.cp = cupy
    self._compiled_cache: dict[str, object] = {}
    self.profile = self._query_profile(cupy, count)
    self.device_name = self.profile.name

  @staticmethod
  def _query_profile(cupy, count: int) -> TargetProfile:
    name = f"CUDA device 0 of {count}"
    sm_count = 0
    warp = 32
    smem = 48 * 1024
    l2 = 0
    max_threads = 1024
    integrated = False
    cc = (0, 0)
    try:
      props = cupy.cuda.runtime.getDeviceProperties(0)
      raw_name = props.get("name", b"")
      if raw_name:
        name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
      maj, minor = int(props.get("major", 0)), int(props.get("minor", 0))
      if maj > 0:
        cc = (maj, minor)
    except Exception:  # noqa: BLE001
      pass
    try:
      dev = cupy.cuda.Device(0)
      attrs = dev.attributes
      sm_count = int(attrs.get("MultiProcessorCount", 0))
      warp = int(attrs.get("WarpSize", 32)) or 32
      smem = int(attrs.get("MaxSharedMemoryPerBlock", 48 * 1024)) or 48 * 1024
      l2 = int(attrs.get("L2CacheSize", 0))
      max_threads = int(attrs.get("MaxThreadsPerBlock", 1024)) or 1024
      integrated = bool(attrs.get("Integrated", 0))
      if cc == (0, 0):
        cc_str = str(getattr(dev, "compute_capability", "") or "")
        if len(cc_str) >= 2 and cc_str.isdigit():
          cc = (int(cc_str[:-1]), int(cc_str[-1]))
        else:
          cc = (int(attrs.get("ComputeCapabilityMajor", 0)), int(attrs.get("ComputeCapabilityMinor", 0)))
    except Exception:  # noqa: BLE001
      pass
    mat = MatrixUnit("wmma", 16, 16, 8, ("float32",), ("float32",)) if cc >= (8, 0) else None
    return TargetProfile(
      kind="cuda",
      name=name,
      num_compute_units=sm_count,
      vector_width=warp,
      l1_bytes=smem,
      l2_bytes=l2,
      tile_bytes=config.tile_bytes,
      shared_mem_bytes=smem,
      max_threads_per_block=max_threads,
      unified_memory=integrated,
      supports_float64=True,
      compute_capability=cc,
      matrix_unit=mat,
      prefer_interchange=False,
    )

  def compile(self, plan: KernelPlan):
    cached = self._compiled_cache.get(plan.source)
    if cached is not None:
      return cached
    kernel = self.cp.RawKernel(plan.source, "pk_kernel")
    kernel.compile()  # surface compilation problems here, not at first launch
    self._compiled_cache[plan.source] = kernel
    return kernel

  def compile_module(self, plans: list[KernelPlan]) -> None:
    """Compile all unique uncompiled CUDA kernels of a module via batched
    (and multi-threaded across CPU cores for large modules) NVRTC ``RawModule``s."""
    todo: list[KernelPlan] = []
    seen_src: set[str] = set()
    for p in plans:
      if p.compiled is not None:
        continue
      cached = self._compiled_cache.get(p.source)
      if cached is not None:
        p.compiled = cached
        continue
      if p.source not in seen_src:
        seen_src.add(p.source)
        todo.append(p)
    if len(todo) <= 1:
      for p in todo:
        p.compiled = self.compile(p)
      return
    marker = 'extern "C" __global__ void pk_kernel('
    for p in todo:
      if p.source.find(marker) < 0:
        for q in todo:
          q.compiled = self.compile(q)
        return

    def _build_chunk(chunk: list[KernelPlan]) -> list[tuple[KernelPlan, object]]:
      needs_mma = any("#include <mma.h>" in p.source for p in chunk)
      parts: list[str] = []
      if needs_mma:
        parts.append("#include <mma.h>")
      parts.append(PRELUDE)
      for idx, p in enumerate(chunk):
        pos = p.source.find(marker)
        parts.append(p.source[pos:].replace(marker, f'extern "C" __global__ void pk_k{idx}(', 1))
      mod = self.cp.RawModule(code="\n".join(parts))
      mod.compile()
      return [(p, mod.get_function(f"pk_k{idx}")) for idx, p in enumerate(chunk)]

    n_workers = min(len(todo) // 4, os.cpu_count() or 1, 8) if len(todo) >= 8 else 1
    if n_workers <= 1:
      results = _build_chunk(todo)
    else:
      from concurrent.futures import ThreadPoolExecutor

      # Prime CuPy's lazily-cached CUDA path / NVRTC options on the main thread
      # before fanning out across worker threads (assemble_cupy_compiler_options
      # is not thread-safe on first call).
      if not getattr(self, "_nvrtc_primed", False):
        self.cp.RawKernel('extern "C" __global__ void pk_nvrtc_prime() {}', "pk_nvrtc_prime").compile()
        self._nvrtc_primed = True

      dev_id = int(self.cp.cuda.runtime.getDevice())
      chunks = [todo[i::n_workers] for i in range(n_workers)]

      def _worker(chunk: list[KernelPlan]):
        with self.cp.cuda.Device(dev_id):
          return _build_chunk(chunk)

      results = []
      with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for res in pool.map(_worker, chunks):
          results.extend(res)

    for p, fn in results:
      self._compiled_cache[p.source] = fn
      p.compiled = fn
    # Also attach compiled functions to any duplicate plans sharing the same source
    for p in plans:
      if p.compiled is None:
        p.compiled = self._compiled_cache.get(p.source)

  def to_device(self, host: np.ndarray):
    return self.cp.asarray(host)

  def zeros(self, n: int, dtype):
    return self.cp.zeros(n, dtype=dtype)

  def empty(self, nbytes: int):
    return self.cp.empty(nbytes, dtype=np.uint8)

  def launch(self, kernel, plan: KernelPlan, n: int, args: list) -> None:
    tiled_meta = plan.extra.get("tiled")
    if tiled_meta is not None:
      kernel((n,), (int(tiled_meta.get("threads", BLOCK_THREADS)),), tuple(args))
      return
    grid = (n + BLOCK - 1) // BLOCK
    kernel((grid,), (BLOCK,), tuple(args))

  def to_host(self, dev, host: np.ndarray) -> None:
    dev.get(out=host)  # straight into the host's memory, no intermediate array

  def read(self, dev) -> np.ndarray:
    return self.cp.asnumpy(dev)


class SimulatorDriver:
  """Compiles CUDA kernels as C++ and runs their thread grid on the CPU."""

  name = "simulator"
  device_name = "CUDA simulator (CPU)"

  def __init__(self) -> None:
    tc = toolchain.get_toolchain()
    if tc.kind == "msvc" or os.name == "nt":
      raise BackendUnavailable("the CUDA simulator is only available on POSIX systems with GCC or Clang")
    self.cxx = self._find_cxx(tc)
    self.profile = TargetProfile(
      kind="cuda",
      name=self.device_name,
      num_compute_units=4,
      vector_width=32,
      l1_bytes=48 * 1024,
      l2_bytes=4 * 1024 * 1024,
      tile_bytes=config.tile_bytes,
      shared_mem_bytes=48 * 1024,
      max_threads_per_block=1024,
      unified_memory=False,
      supports_float64=True,
      compute_capability=(8, 0),
      matrix_unit=MatrixUnit("wmma", 16, 16, 8, ("float32",), ("float32",)),
      prefer_interchange=False,
    )

  @staticmethod
  def _find_cxx(tc) -> str:
    cc = tc.cc[0]
    base = os.path.basename(cc)
    guesses = []
    if "clang" in base:
      guesses.append(cc.replace("clang", "clang++"))
    if "gcc" in base:
      guesses.append(cc.replace("gcc", "g++"))
    guesses += ["c++", "g++", "clang++"]
    for g in guesses:
      path = shutil.which(g) or (g if os.path.exists(g) else None)
      if path:
        return path
    raise BackendUnavailable("the CUDA simulator needs a C++ compiler (c++, g++ or clang++) on PATH")

  def compile(self, plan: KernelPlan):
    em = CudaEmitter()
    casts = ["*(long long*)a[0]", "*(long long*)a[1]"]
    k = 2
    for v in plan.scalars:
      casts.append(f"*({em.ctype(v.type)}*)a[{k}]")
      k += 1
    casts += [f"(int*)a[{k}]", f"(long long*)a[{k + 1}]"]
    k += 2
    for v in plan.bufs:
      casts.append(f"({em.ctype(v.type.elt)}*)a[{k}]")
      k += 1
    if plan.extra.get("tiled") is not None:
      trampoline = (
        '\nextern "C" __attribute__((visibility("default"))) '
        "void pk_sim_launch(long long grid, long long block, void** a) {\n"
        "  if (grid <= 0 || block <= 0) return;\n"
        "  blockDim.x = (unsigned int)block;\n"
        "  pk_sim_barrier.init(block);\n"
        "  for (int w = 0; w < 8; ++w) pk_sim_warp_barriers[w].init(block >= 32 ? 32 : block);\n"
        "  std::vector<std::thread> threads;\n"
        "  threads.reserve((size_t)block);\n"
        "  for (long long t = 0; t < block; t++) {\n"
        "    threads.emplace_back([&, t]() {\n"
        "      threadIdx.x = (unsigned int)t;\n"
        "      for (long long b = 0; b < grid; b++) {\n"
        "        pk_sim_barrier.wait();\n"
        "        if (t == 0) blockIdx.x = (unsigned int)b;\n"
        "        pk_sim_barrier.wait();\n"
        f"        pk_kernel({', '.join(casts)});\n"
        "        pk_sim_barrier.wait();\n"
        "      }\n"
        "    });\n"
        "  }\n"
        "  for (auto& th : threads) th.join();\n"
        "}\n"
      )
      source = SIM_SHIM_TILED + plan.source + trampoline
    else:
      trampoline = (
        '\nextern "C" __attribute__((visibility("default"))) '
        "void pk_sim_launch(long long grid, long long block, void** a) {\n"
        "  for (long long b = 0; b < grid; b++) {\n"
        "    for (long long t = 0; t < block; t++) {\n"
        "      blockIdx.x = (unsigned int)b; blockDim.x = (unsigned int)block; threadIdx.x = (unsigned int)t;\n"
        f"      pk_kernel({', '.join(casts)});\n"
        "    }\n  }\n}\n"
      )
      source = SIM_SHIM + plan.source + trampoline
    path = self._build(source, threaded=plan.extra.get("tiled") is not None)
    lib = toolchain.load(path)
    lib.pk_sim_launch.restype = None
    lib.pk_sim_launch.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    return lib

  def _build(self, source: str, threaded: bool = False):
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path

    tc = toolchain.get_toolchain()
    digest = hashlib.sha256((source + self.cxx).encode()).hexdigest()[:24]
    cdir = toolchain.cache_dir()
    out = cdir / f"pk_cudasim_{digest}{tc.ext}"
    if out.exists():
      return out
    with tempfile.TemporaryDirectory(prefix="pk_build_", dir=cdir) as d:
      src = Path(d) / "kernel.cpp"
      src.write_text(source)
      tmp = Path(d) / out.name
      cmd = [self.cxx, "-O1", "-std=c++14", "-fwrapv", "-w", "-shared", "-fPIC"]
      if threaded:
        cmd.append("-pthread")
      cmd += [str(src), "-o", str(tmp)]
      proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
      if proc.returncode != 0:
        kept = cdir / f"failed_{digest}.cpp"
        shutil.copyfile(src, kept)
        raise InternalError(
          "a generated CUDA kernel is not valid C++ (a Parakeet bug - please report it).\n"
          f"  source kept at: {kept}\n" + "\n".join(proc.stderr.splitlines()[:30])
        )
      os.replace(tmp, out)
    return out

  def to_device(self, host: np.ndarray):
    return host.copy()  # a separate "device" copy, like a real GPU

  def zeros(self, n: int, dtype):
    return np.zeros(n, dtype=dtype)

  def empty(self, nbytes: int):
    return np.full(nbytes, 0xA5, dtype=np.uint8)  # recognisable garbage: nothing may rely on it

  def launch(self, kernel, plan: KernelPlan, n: int, args: list) -> None:
    tiled_meta = plan.extra.get("tiled")
    if tiled_meta is not None:
      grid, block = n, int(tiled_meta.get("threads", BLOCK_THREADS))
    else:
      grid, block = (n + BLOCK - 1) // BLOCK, BLOCK
    keep = []
    ptrs = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
      if isinstance(a, np.ndarray):
        ptrs[i] = a.ctypes.data
      else:
        box = np.array([a])
        keep.append(box)
        ptrs[i] = box.ctypes.data
    with _SIM_LOCK:
      kernel.pk_sim_launch(grid, block, ptrs)

  def to_host(self, dev, host: np.ndarray) -> None:
    host[...] = dev

  def read(self, dev) -> np.ndarray:
    return dev


class _PkBuf(ctypes.Structure):
  _fields_ = [
    ("data", ctypes.c_void_p),
    ("nbytes", ctypes.c_int64),
    ("refcount", ctypes.c_int64),
    ("flags", ctypes.c_int64),
  ]


_STRUCT_FMT = {
  "bool": "?", "int8": "b", "int16": "h", "int32": "i", "int64": "q",
  "uint8": "B", "uint16": "H", "uint32": "I", "uint64": "Q",
  "float32": "f", "float64": "d",
}  # fmt: skip

_PKBUF_UNPACK = struct.Struct("<Qq8xq").unpack_from
_UNPACK_P = struct.Struct("<Q").unpack_from


def _build_fast_meta(plan: KernelPlan):
  """Precompute byte offsets into ``plan.host_struct`` for fast kernel dispatch."""
  cls = plan.host_struct
  ctx_size = ctypes.sizeof(cls)
  scalars = []
  for v in plan.scalars:
    off = getattr(cls, v.name).offset
    scalars.append((off, struct.Struct("<" + _STRUCT_FMT[v.type.name]).unpack_from, v.type.dtype.type))
  buf_offsets = tuple(getattr(cls, v.name).offset for v in plan.bufs)
  written_mask = tuple(v.uid in plan.written for v in plan.bufs)
  tiled_meta = plan.extra.get("tiled")
  tiled_spec = None
  if tiled_meta is not None:
    dims = []
    for kind, val in tiled_meta["shape"]:
      if kind == "c":
        dims.append((0, val))
      else:
        dims.append((1, getattr(cls, val).offset))
    sk_meta = tiled_meta.get("split_k")
    sk_spec = None
    if sk_meta is not None:
      k_kind, k_val = sk_meta["k_dim"]
      sk_spec = (
        0 if k_kind == "c" else 1,
        k_val if k_kind == "c" else getattr(cls, k_val).offset,
        sk_meta["out_buf_idx"],
      )
    tiled_spec = (tiled_meta["bm"], tiled_meta["bn"], tuple(dims), sk_spec)
  return (ctx_size, tuple(scalars), buf_offsets, written_mask, tiled_spec, plan.loop.kind == "map")


def _build_native_descs(spec) -> tuple[ctypes.Array, int] | None:
  """Build a C ``PkCuKDesc[]`` array for every kernel of ``spec.module`` once
  all device kernels have been compiled.  Returns ``None`` if any kernel is not
  yet compiled."""
  kernels = spec.module.kernels
  n = len(kernels)
  if n == 0:
    return None
  for k in kernels:
    p = k.device
    if p is not None and p.compiled is None:
      return None
  arr = (PkCuKDesc * n)()
  for idx, k in enumerate(kernels):
    p = k.device
    d = arr[idx]
    if p is None or p.failed:
      d.cpu_only = 1
      continue
    if p.compiled is None or p.host_struct is None:
      continue
    if len(p.scalars) > MAX_SCALARS or len(p.bufs) > MAX_BUFS:
      continue
    fn = p.compiled
    fn_ptr = getattr(fn, "ptr", None)
    if fn_ptr is None:
      kern = getattr(fn, "kernel", None)
      if kern is None and hasattr(fn, "compile"):
        fn.compile()
        kern = getattr(fn, "kernel", None)
      fn_ptr = getattr(kern, "ptr", None) if kern is not None else None
    if not fn_ptr:
      continue
    cls = p.host_struct
    d.cu_fn = int(fn_ptr)
    d.is_map = 1 if p.loop.kind == "map" else 0
    d.has_errors = 1 if p.errors else 0
    tiled_meta = p.extra.get("tiled")
    if tiled_meta is not None:
      shape = tiled_meta["shape"]
      if len(shape) > MAX_TILED_DIMS:
        d.cu_fn = 0
        continue
      d.is_tiled = 1
      d.block_threads = int(tiled_meta.get("threads", BLOCK_THREADS))
      d.bm = int(tiled_meta["bm"])
      d.bn = int(tiled_meta["bn"])
      d.n_dims = len(shape)
      for di, (kind, val) in enumerate(shape):
        if kind == "c":
          d.dim_is_var[di] = 0
          d.dim_val[di] = int(val)
        else:
          d.dim_is_var[di] = 1
          d.dim_val[di] = int(getattr(cls, val).offset)
      sk_meta = tiled_meta.get("split_k")
      if sk_meta is not None:
        d.split_k_enabled = 1
        k_kind, k_val = sk_meta["k_dim"]
        if k_kind == "c":
          d.split_k_dim_is_var = 0
          d.split_k_dim_val = int(k_val)
        else:
          d.split_k_dim_is_var = 1
          d.split_k_dim_val = int(getattr(cls, k_val).offset)
        d.split_k_out_buf = int(sk_meta["out_buf_idx"])
    else:
      d.is_tiled = 0
      d.block_threads = BLOCK
    d.n_scalars = len(p.scalars)
    for si, v in enumerate(p.scalars):
      d.scalar_offsets[si] = int(getattr(cls, v.name).offset)
    d.n_bufs = len(p.bufs)
    for bi, v in enumerate(p.bufs):
      d.buf_offsets[bi] = int(getattr(cls, v.name).offset)
      d.buf_written[bi] = 1 if v.uid in p.written else 0
  return arr, n


class CudaDevice(DeviceBase):
  name = "cuda"
  _instance: CudaDevice | None = None
  _lock = threading.Lock()

  @classmethod
  def get(cls) -> CudaDevice:
    with cls._lock:
      if cls._instance is None:
        cls._instance = cls()
      return cls._instance

  def __init__(self) -> None:
    simulate = os.environ.get("PARAKEET_CUDA_SIMULATE", "").strip().lower() not in ("", "0", "false", "no")
    self.driver = SimulatorDriver() if simulate else CuPyDriver()
    self.device_name = self.driver.device_name
    self.profile = self.driver.profile
    self.simulated = simulate
    self._err_cap = 256
    self._native = None if simulate else try_load_native_cuda_rt()
    super().__init__()
    self._py_callback_addr = self.callback_address
    if self._native is not None:
      self._native.lib.pk_cu_set_fallback(self._py_callback_addr)
      self.callback_address = self._native.dispatch_address

  def _err_bufs(self):
    """Per-host-thread GPU error flag/info buffers so concurrent threads never collide."""
    bufs = getattr(self._state, "err_bufs", None)
    if bufs is None:
      bufs = (
        self.driver.zeros(1, np.int32),
        self.driver.zeros(4, np.int64),
        self.driver.zeros(self._err_cap, np.int32),
        self.driver.zeros(self._err_cap * 4, np.int64),
      )
      self._state.err_bufs = bufs
    return bufs

  def begin_call(self, spec, arrays, pinned=None):
    token = super().begin_call(spec, arrays, pinned)
    if self._native is not None and len(self._state.stack) == 1:
      descs = getattr(spec.module, "_cu_descs", None)
      if descs is None:
        descs = _build_native_descs(spec)
        if descs is not None:
          spec.module._cu_descs = descs
      if descs is not None:
        descs_arr, n_kernels = descs
        cp = self.driver.cp
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        flag_buf, info_buf, err_flags, err_infos = self._err_bufs()
        lib = self._native.lib
        lib.pk_cu_begin_call(
          stream_ptr,
          int(self.profile.num_compute_units),
          self._py_callback_addr,
          ctypes.addressof(descs_arr),
          n_kernels,
          int(flag_buf.data.ptr),
          int(info_buf.data.ptr),
          int(err_flags.data.ptr),
          int(err_infos.data.ptr),
          self._err_cap,
        )
        pin_das = []
        if pinned:
          for pin_idx, (da, _writes) in enumerate(pinned):
            dev_buf = da.ensure_device()
            lib.pk_cu_add_pinned(da._addr, int(dev_buf.data.ptr), da._nbytes, pin_idx, 1 if da._dirty else 0)
            pin_das.append(da)
        self._state.native_token = token
        self._state.native_pinned = pin_das
        self._state.native_descs = descs_arr
    return token

  def promote(self, arr: np.ndarray) -> DeviceArray:
    if self._native is not None and getattr(self._state, "native_token", None) is not None:
      if arr.flags.c_contiguous:
        addr = arr.ctypes.data
        nbytes = arr.nbytes
        out_dptr = ctypes.c_uint64(0)
        out_cap = ctypes.c_int64(0)
        out_dirty = ctypes.c_int32(0)
        owner_ctx = ctypes.c_void_p()
        generation = ctypes.c_uint64()
        if self._native.lib.pk_cu_take_output(
          addr,
          nbytes,
          ctypes.byref(out_dptr),
          ctypes.byref(out_cap),
          ctypes.byref(out_dirty),
          ctypes.byref(owner_ctx),
          ctypes.byref(generation),
        ):
          cp = self.driver.cp
          owner = self._native.make_owner(out_dptr.value, out_cap.value, owner_ctx.value, generation.value)
          mem = cp.cuda.UnownedMemory(out_dptr.value, out_cap.value, owner)
          dev_buf = cp.ndarray((nbytes,), dtype=np.uint8, memptr=cp.cuda.MemoryPointer(mem, 0))
          return DeviceArray(arr, self, dev_buf, dirty=bool(out_dirty.value))
    return super().promote(arr)

  def end_call(self, token):
    if getattr(self._state, "native_token", None) is token:
      self._state.native_token = None
      pin_das = self._state.native_pinned
      self._state.native_pinned = ()
      self._state.native_descs = None
      n_pin = len(pin_das)
      out_dirty = (ctypes.c_int32 * max(n_pin, 1))()
      out_inval = (ctypes.c_int32 * max(n_pin, 1))()
      out_stats = (ctypes.c_int64 * 7)()
      out_err_count = ctypes.c_int32(0)
      out_err_kernels = (ctypes.c_int32 * self._err_cap)()
      self._native.lib.pk_cu_end_call(
        out_dirty, out_inval, n_pin, out_stats, ctypes.byref(out_err_count), out_err_kernels
      )
      py_used = bool(out_stats[3])
      self._last_native_fallback = (int(out_stats[4]), int(out_stats[5]), int(out_stats[6]))
      for i, da in enumerate(pin_das):
        if py_used:
          if out_dirty[i]:
            da._dirty = True
        else:
          da._dirty = bool(out_dirty[i])
        if out_inval[i]:
          da._dev_buf = None
      self.launches += int(out_stats[0])
      self.uploads += int(out_stats[1])
      self.downloads += int(out_stats[2])
      n_err = int(out_err_count.value)
      if n_err > 0:
        ks = token.spec.module.kernels
        for slot in range(n_err):
          p = ks[int(out_err_kernels[slot])].device
          if p is not None:
            self._state.pending_checks.append((slot, p))
    return super().end_call(token)

  def _check_pending_errors(self) -> None:
    pending = self._state.pending_checks
    if self._native is not None and getattr(self._state, "native_token", None) is not None and self._state.stack:
      out_err_kernels = (ctypes.c_int32 * self._err_cap)()
      n_c_err = int(self._native.lib.pk_cu_drain_pending_errs(out_err_kernels, self._err_cap))
      if n_c_err > 0:
        ks = self._state.stack[-1].spec.module.kernels
        for slot in range(n_c_err):
          p = ks[int(out_err_kernels[slot])].device
          if p is not None:
            pending.append((slot, p))
    if not pending:
      return
    n = len(pending)
    items = list(pending)
    pending.clear()
    _, _, err_flags, err_infos = self._err_bufs()
    flags = np.array(self.driver.read(err_flags[:n]), copy=True)
    if flags.any():
      err_flags[:n].fill(0)
      for slot, plan in items:
        code = int(flags[slot])
        if code:
          info = self.driver.read(err_infos[slot * 4 : slot * 4 + 4])
          raise make_runtime_error(plan.errors[code - 1], [int(x) for x in info])

  def flush(self, mid_call: bool = True) -> None:
    self._check_pending_errors()
    super().flush(mid_call=mid_call)

  def sync_buf(self, addr: int, nbytes: int, write: bool = False) -> None:
    if self._state.pending_checks:
      self._check_pending_errors()
    super().sync_buf(addr, nbytes, write=write)

  def planner(self, profile: TargetProfile | None = None) -> CudaPlanner:
    return CudaPlanner(self, profile or self.profile)

  @staticmethod
  def _host(addr: int, nbytes: int):
    return np.frombuffer((ctypes.c_char * nbytes).from_address(addr), dtype=np.uint8)

  def upload(self, addr: int, nbytes: int, external: bool, untouched: int = 0):
    if nbytes <= 0:
      return self.driver.to_device(np.zeros(16, dtype=np.uint8))
    if untouched == 4:
      return self.driver.zeros(nbytes, np.uint8)
    if untouched == 2:
      return self.driver.empty(nbytes)
    return self.driver.to_device(self._host(addr, nbytes))

  def download(self, dev, addr: int, nbytes: int) -> None:
    self.driver.to_host(dev, self._host(addr, nbytes))

  def run(self, plan: KernelPlan, ctx_addr: int, total: int, nchunks: int) -> None:
    drv = self.driver
    if plan.compiled is None:
      if hasattr(drv, "compile_module") and self._state.stack:
        mod_plans = [k.device for k in self._state.stack[-1].spec.module.kernels if k.device is not None]
        drv.compile_module(mod_plans)
      if plan.compiled is None:
        plan.compiled = drv.compile(plan)
    fast = plan.extra.get("_fast")
    if fast is None:
      fast = plan.extra["_fast"] = _build_fast_meta(plan)
    ctx_size, scalar_specs, buf_offsets, written_mask, tiled_spec, is_map = fast
    raw_ctx = ctypes.string_at(ctx_addr, ctx_size)
    split_k_out_idx = -1
    split_k_val = 1
    if tiled_spec is not None:
      bm, bn, dim_specs, sk_spec = tiled_spec
      dims = [val if kind == 0 else struct.unpack_from("<q", raw_ctx, val)[0] for kind, val in dim_specs]
      if any(d <= 0 for d in dims):
        return
      batch = math.prod(dims[:-2]) if len(dims) > 2 else 1
      base_blocks = batch * ((dims[-2] + bm - 1) // bm) * ((dims[-1] + bn - 1) // bn)
      if base_blocks <= 0:
        return
      n = base_blocks
      csize = 1
      if sk_spec is not None and self.profile.num_compute_units > 0 and base_blocks < self.profile.num_compute_units:
        k_kind, k_val, ob_idx = sk_spec
        k_dim = k_val if k_kind == 0 else struct.unpack_from("<q", raw_ctx, k_val)[0]
        if k_dim >= 256:
          sk = min(16, max(1, (self.profile.num_compute_units * 2) // base_blocks), k_dim // 64)
          if sk > 1:
            split_k_val = sk
            split_k_out_idx = ob_idx
            split_k_exact_bytes = batch * dims[-2] * dims[-1] * 4
    else:
      if is_map:
        n = min(total, 1 << 24)  # one GPU thread per element when possible
      else:
        n = nchunks
      csize = -(-total // n)
      n = -(-total // csize)
    args: list = [np.int64(total), np.int64(csize)]
    for off, unpack, np_type in scalar_specs:
      args.append(np_type(unpack(raw_ctx, off)[0]))
    flag_buf, info_buf, err_flags, err_infos = self._err_bufs()
    if plan.errors:
      pending = self._state.pending_checks
      if len(pending) >= self._err_cap:
        self._check_pending_errors()
      slot = len(pending)
      flag = err_flags[slot : slot + 1]
      info = err_infos[slot * 4 : slot * 4 + 4]
      pending.append((slot, plan))
    else:
      flag = flag_buf
      info = info_buf
    args += [flag, info]

    # Fast path for disjoint buffers (avoids sorting / merge_regions on 99.9% of launches!)
    n_bufs = len(buf_offsets)
    pk_ptrs = [_UNPACK_P(raw_ctx, off)[0] for off in buf_offsets]
    infos = [_PKBUF_UNPACK(ctypes.string_at(ptr, 32)) for ptr in pk_ptrs]
    disjoint = True
    for i in range(n_bufs):
      a0, s0, _ = infos[i]
      if s0 <= 0:
        disjoint = False
        break
      e0 = a0 + s0
      for j in range(i + 1, n_bufs):
        a1, s1, _ = infos[j]
        if a0 < a1 + s1 and a1 < e0:
          disjoint = False
          break
      if not disjoint:
        break

    if disjoint:
      pinned = self._state.pinned
      cache = self._state.cache
      for i in range(n_bufs):
        addr, nbytes, flags = infos[i]
        w = written_mask[i]
        if flags & 14:
          ctypes.c_int64.from_address(pk_ptrs[i] + 24).value = flags & ~14
        da = pinned.get(addr)
        if da is not None and da._nbytes == nbytes:
          dev_buf = da.ensure_device()
          if w:
            da._dirty = True
          args.append(dev_buf)
          continue
        hit = cache.get(addr)
        if hit is not None and hit.nbytes == nbytes:
          if w:
            hit.dirty = True
          args.append(hit.dev)
          continue
        unt = 4 if (flags & 4) else (2 if (flags & 2) else 0)
        if unt and hit is None:
          # Brand-new pk_alloc buffer: guaranteed disjoint from live pinned & cached buffers
          dev_buf = self.upload(addr, nbytes, False, unt)
          res = cache[addr] = _Resident(addr, nbytes, dev_buf)
          res.dirty = w or (unt == 4)
          args.append(dev_buf)
          continue
        # Slow acquire for cache miss (handles overlap with cached/pinned regions cleanly)
        devs = self.acquire([(addr, nbytes)], {0} if w else set(), [False], [unt])
        args.append(devs[0])
    else:
      pks = [_PkBuf.from_address(ptr) for ptr in pk_ptrs]
      regions, where = merge_regions([(pk.data or 0, pk.nbytes) for pk in pks])
      written = {r for w, (r, _) in zip(written_mask, where) if w}
      devs = self.acquire(regions, written, [False] * len(regions), untouched_regions(pks, where, len(regions)))
      for r, off in where:
        args.append(devs[r][off:] if off else devs[r])

    if disjoint and split_k_val > 1 and 0 <= split_k_out_idx < n_bufs:
      _, ob_nbytes, ob_flags = infos[split_k_out_idx]
      if (ob_flags & 6) or ob_nbytes == split_k_exact_bytes:
        out_dev = args[len(args) - n_bufs + split_k_out_idx]
        if not (ob_flags & 4):
          out_dev.fill(0)
        n = n * split_k_val
        args[1] = np.int64(split_k_val)

    drv.launch(plan.compiled, plan, n, args)


_ = ctype_of
