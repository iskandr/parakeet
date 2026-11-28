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
import re
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
from .ptx import bundle_ptx_module, emit_ptx_entry_body, verify_ptx
from .tiled_gpu import BLOCK_THREADS, generate_tiled_gpu_kernel, generate_warp_reduce_cuda_kernel

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
  + runtime_c.HELPERS_BITOPS
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
#ifndef PK_CUDA_SIMULATE
template <typename T>
__device__ __forceinline__ T pk_shfl_down(T v, int delta) {
  return __shfl_down_sync(0xffffffffu, v, delta);
}
template <typename T>
__device__ __forceinline__ T pk_shfl_bcast(T v, int src_lane) {
  return __shfl_sync(0xffffffffu, v, src_lane);
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void pk_cp_async_16(void* smem_ptr, const void* gmem_ptr) {
  unsigned int s;
  asm("{\n .reg .u64 smem64;\n cvta.to.shared.u64 smem64, %1;\n cvt.u32.u64 %0, smem64;\n}\n" : "=r"(s) : "l"(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_ptr) : "memory");
}
__device__ __forceinline__ void pk_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void pk_cp_async_wait() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
__device__ __forceinline__ unsigned int pk_f32_to_tf32(float v) {
  unsigned int r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(v));
  return r;
}
__device__ __forceinline__ void pk_mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
__device__ __forceinline__ void pk_mma_16x16x8_row_row(
    float c[8], const float* a_ptr, int a_ldm, const float* b_ptr, int b_ldm) {
  const int lane = (int)(threadIdx.x & 31u);
  const int grp = lane >> 2;
  const int tid = lane & 3;
  const unsigned int a0 = pk_f32_to_tf32(a_ptr[grp * a_ldm + tid]);
  const unsigned int a1 = pk_f32_to_tf32(a_ptr[(grp + 8) * a_ldm + tid]);
  const unsigned int a2 = pk_f32_to_tf32(a_ptr[grp * a_ldm + (tid + 4)]);
  const unsigned int a3 = pk_f32_to_tf32(a_ptr[(grp + 8) * a_ldm + (tid + 4)]);
  const unsigned int b00 = pk_f32_to_tf32(b_ptr[tid * b_ldm + grp]);
  const unsigned int b01 = pk_f32_to_tf32(b_ptr[(tid + 4) * b_ldm + grp]);
  const unsigned int b10 = pk_f32_to_tf32(b_ptr[tid * b_ldm + (grp + 8)]);
  const unsigned int b11 = pk_f32_to_tf32(b_ptr[(tid + 4) * b_ldm + (grp + 8)]);
  pk_mma_m16n8k8_tf32(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b00, b01, c[0], c[1], c[2], c[3]);
  pk_mma_m16n8k8_tf32(c[4], c[5], c[6], c[7], a0, a1, a2, a3, b10, b11, c[4], c[5], c[6], c[7]);
}
__device__ __forceinline__ void pk_mma_16x16x8_store(float* dst, const float c[8], int ldm) {
  const int lane = (int)(threadIdx.x & 31u);
  const int grp = lane >> 2;
  const int col = (lane & 3) * 2;
  dst[grp * ldm + col] = c[0];
  dst[grp * ldm + col + 1] = c[1];
  dst[(grp + 8) * ldm + col] = c[2];
  dst[(grp + 8) * ldm + col + 1] = c[3];
  dst[grp * ldm + col + 8] = c[4];
  dst[grp * ldm + col + 9] = c[5];
  dst[(grp + 8) * ldm + col + 8] = c[6];
  dst[(grp + 8) * ldm + col + 9] = c[7];
}
#else
__device__ __forceinline__ void pk_cp_async_16(void* smem_ptr, const void* gmem_ptr) {
  *(float4*)smem_ptr = *(const float4*)gmem_ptr;
}
__device__ __forceinline__ void pk_cp_async_commit() {}
__device__ __forceinline__ void pk_cp_async_wait() {}
__device__ __forceinline__ unsigned int pk_f32_to_tf32(float v) { return __float_as_uint(v); }
__device__ __forceinline__ void pk_mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int,
    float c0, float c1, float c2, float c3) {
  d0 = c0; d1 = c1; d2 = c2; d3 = c3;
}
__device__ __forceinline__ void pk_mma_16x16x8_row_row(float[8], const float*, int, const float*, int) {}
__device__ __forceinline__ void pk_mma_16x16x8_store(float*, const float[8], int) {}
#endif
#endif
"""
)

# Stand-ins for the CUDA runtime so a kernel can be compiled as ordinary C++.
_SIM_LOCK = threading.Lock()

SIM_SHIM = r"""
#define PK_CUDA_SIMULATE 1
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
#define __align__(n) alignas(n)
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
template <typename T>
static inline T pk_shfl_down(T v, int delta) {
  static T pk_shfl_buf[8][32];
  int w = (int)((threadIdx.x >> 5) & 7u);
  int l = (int)(threadIdx.x & 31u);
  pk_shfl_buf[w][l] = v;
  __syncwarp();
  T res = (l + delta < 32) ? pk_shfl_buf[w][l + delta] : v;
  __syncwarp();
  return res;
}
template <typename T>
static inline T pk_shfl_bcast(T v, int src_lane) {
  static T pk_bcast_buf[8][32];
  int w = (int)((threadIdx.x >> 5) & 7u);
  int l = (int)(threadIdx.x & 31u);
  pk_bcast_buf[w][l] = v;
  __syncwarp();
  T res = pk_bcast_buf[w][src_lane & 31];
  __syncwarp();
  return res;
}
static inline void pk_cp_async_16(void* smem_ptr, const void* gmem_ptr) {
  std::memcpy(smem_ptr, gmem_ptr, 16);
}
static inline void pk_cp_async_commit() {}
static inline void pk_cp_async_wait() {}
static inline long long min(long long a, long long b) { return a < b ? a : b; }
static inline unsigned int pk_f32_to_tf32(float v) {
  unsigned int u;
  std::memcpy(&u, &v, 4);
  return u;
}
static inline void pk_mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  static float sm_a[8][16][8];
  static float sm_b[8][8][8];
  int w = (int)((threadIdx.x >> 5) & 7u);
  int lane = (int)(threadIdx.x & 31u);
  int grp = lane >> 2;
  int tid = lane & 3;
  float fa0, fa1, fa2, fa3, fb0, fb1;
  std::memcpy(&fa0, &a0, 4); std::memcpy(&fa1, &a1, 4);
  std::memcpy(&fa2, &a2, 4); std::memcpy(&fa3, &a3, 4);
  std::memcpy(&fb0, &b0, 4); std::memcpy(&fb1, &b1, 4);
  sm_a[w][grp][tid] = fa0;
  sm_a[w][grp + 8][tid] = fa1;
  sm_a[w][grp][tid + 4] = fa2;
  sm_a[w][grp + 8][tid + 4] = fa3;
  sm_b[w][tid][grp] = fb0;
  sm_b[w][tid + 4][grp] = fb1;
  __syncwarp();
  float r0 = c0, r1 = c1, r2 = c2, r3 = c3;
  for (int k = 0; k < 8; ++k) {
    r0 += sm_a[w][grp][k]     * sm_b[w][k][tid * 2];
    r1 += sm_a[w][grp][k]     * sm_b[w][k][tid * 2 + 1];
    r2 += sm_a[w][grp + 8][k] * sm_b[w][k][tid * 2];
    r3 += sm_a[w][grp + 8][k] * sm_b[w][k][tid * 2 + 1];
  }
  __syncwarp();
  d0 = r0; d1 = r1; d2 = r2; d3 = r3;
}
static inline void pk_mma_16x16x8_row_row(
    float c[8], const float* a_ptr, int a_ldm, const float* b_ptr, int b_ldm) {
  const int lane = (int)(threadIdx.x & 31u);
  const int grp = lane >> 2;
  const int tid = lane & 3;
  const unsigned int a0 = pk_f32_to_tf32(a_ptr[grp * a_ldm + tid]);
  const unsigned int a1 = pk_f32_to_tf32(a_ptr[(grp + 8) * a_ldm + tid]);
  const unsigned int a2 = pk_f32_to_tf32(a_ptr[grp * a_ldm + (tid + 4)]);
  const unsigned int a3 = pk_f32_to_tf32(a_ptr[(grp + 8) * a_ldm + (tid + 4)]);
  const unsigned int b00 = pk_f32_to_tf32(b_ptr[tid * b_ldm + grp]);
  const unsigned int b01 = pk_f32_to_tf32(b_ptr[(tid + 4) * b_ldm + grp]);
  const unsigned int b10 = pk_f32_to_tf32(b_ptr[tid * b_ldm + (grp + 8)]);
  const unsigned int b11 = pk_f32_to_tf32(b_ptr[(tid + 4) * b_ldm + (grp + 8)]);
  pk_mma_m16n8k8_tf32(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b00, b01, c[0], c[1], c[2], c[3]);
  pk_mma_m16n8k8_tf32(c[4], c[5], c[6], c[7], a0, a1, a2, a3, b10, b11, c[4], c[5], c[6], c[7]);
}
static inline void pk_mma_16x16x8_store(float* dst, const float c[8], int ldm) {
  const int lane = (int)(threadIdx.x & 31u);
  const int grp = lane >> 2;
  const int col = (lane & 3) * 2;
  dst[grp * ldm + col] = c[0];
  dst[grp * ldm + col + 1] = c[1];
  dst[(grp + 8) * ldm + col] = c[2];
  dst[(grp + 8) * ldm + col + 1] = c[3];
  dst[grp * ldm + col + 8] = c[4];
  dst[grp * ldm + col + 9] = c[5];
  dst[(grp + 8) * ldm + col + 8] = c[6];
  dst[(grp + 8) * ldm + col + 9] = c[7];
}
"""


_KERNEL_MARKER = 'extern "C" __global__ void pk_kernel('
_PRELUDE_IDENTIFIERS = frozenset(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", PRELUDE))
_UID_TOKEN_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*_\d+(?:_d)?\b")


def _canonicalize_cuda_kernel(source: str) -> str:
  """Alpha-canonicalize IR variable UIDs inside a CUDA kernel body so structurally
  identical kernels across layers or tensors share one compiled NVRTC binary."""
  pos = source.find(_KERNEL_MARKER)
  if pos < 0:
    return source
  head = source[:pos]
  body = source[pos:]
  mapping: dict[str, str] = {}

  def _repl(m: re.Match[str]) -> str:
    tok = m.group(0)
    if tok in _PRELUDE_IDENTIFIERS:
      return tok
    out = mapping.get(tok)
    if out is None:
      out = f"pk_u{len(mapping)}"
      mapping[tok] = out
    return out

  return head + _UID_TOKEN_RE.sub(_repl, body)


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
    if s.stack and isinstance(s.count, ir.Const) and 1 <= int(s.count.value) <= 4096:
      if s.zero:
        z = "false" if s.var.type.elt.kind == "b" else "0"
        out.append(f"{pad}for (int _zi = 0; _zi < {int(s.count.value)}; _zi++) {s.var.name}_d[_zi] = {z};")
      return
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
    if loop.warp_reduce is not None:
      wr_plan = generate_warp_reduce_cuda_kernel(loop, free, em, PRELUDE)
      if wr_plan is not None:
        return wr_plan
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
    stack_bufs = {
      s.var.uid: int(s.count.value)
      for s in ir.walk_stmts(loop.body)
      if isinstance(s, ir.Alloc) and s.stack and isinstance(s.count, ir.Const) and 1 <= int(s.count.value) <= 4096
    }
    for _uid, v in sorted(defined.items()):
      if v is loop.chunk or v is loop.start or v is loop.stop:
        continue
      if isinstance(v.type, BufType):
        if v.uid in stack_bufs:
          lines.append(f"  {em.ctype(v.type.elt)} {v.name}_d[{stack_bufs[v.uid]}];")
          continue
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


class _CudaBinaryCache:
  """Persistent on-disk CUBIN/PTX cache stored inside ``toolchain.cache_dir()``.

  Keyed by ``(arch_tag, compiler_tag, sha256(source))`` with atomic
  ``tmp -> os.replace`` writes so repeated runs across processes skip NVRTC.
  """

  def __init__(self, arch_tag: str, compiler_tag: str, cache_root=None) -> None:
    self.arch_tag = arch_tag
    self.compiler_tag = compiler_tag
    self._cache_root = cache_root
    self.hits = 0
    self.misses = 0
    self.last_error: str | None = None

  def _dir(self):
    root = self._cache_root if self._cache_root is not None else toolchain.cache_dir()
    d = root / "cuda_cubin"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def key_for(self, code: str) -> str:
    import hashlib

    h = hashlib.sha256(f"{self.arch_tag}|{self.compiler_tag}|v1|{code}".encode()).hexdigest()[:24]
    return f"{self.arch_tag}_{h}"

  def get(self, key: str) -> bytes | None:
    path = self._dir() / f"{key}.cubin"
    try:
      data = path.read_bytes()
      if data:
        self.hits += 1
        return data
    except OSError:
      pass
    self.misses += 1
    return None

  def put(self, key: str, data: bytes) -> None:
    if not data:
      return
    d = self._dir()
    dst = d / f"{key}.cubin"
    if dst.exists():
      return
    tmp = d / f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
      tmp.write_bytes(data)
      os.replace(tmp, dst)
    except OSError:
      try:
        tmp.unlink(missing_ok=True)
      except OSError:
        pass


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
    self._loaded_modules: list[object] = []
    self._cache_lock = threading.Lock()
    self.profile = self._query_profile(cupy, count)
    self.device_name = self.profile.name
    maj, minor = self.profile.compute_capability
    arch_tag = f"sm_{maj}{minor}" if maj > 0 else "sm_70"
    nvrtc_tag = self._nvrtc_version_tag(cupy)
    self._binary_cache = _CudaBinaryCache(arch_tag, nvrtc_tag)
    self._nvcc_path = shutil.which("nvcc")
    self._nvcc_cache = _CudaBinaryCache(arch_tag, self._nvcc_version_tag(self._nvcc_path))
    self._ptx_arch = "sm_80" if (maj, minor) >= (8, 0) else ("sm_75" if (maj, minor) >= (7, 5) else "sm_70")
    self.ptx_compiled_count = 0
    self.nvrtc_compiled_count = 0
    self.nvcc_compiled_count = 0
    self.last_ptx_error: str | None = None
    self.last_nvcc_error: str | None = None

  @staticmethod
  def _nvcc_version_tag(nvcc_path: str | None) -> str:
    if not nvcc_path:
      return "nvcc_none"
    try:
      import subprocess

      out = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True, timeout=10, check=False).stdout
      m = re.search(r"release\s+(\d+\.\d+)", out)
      return f"nvcc_{m.group(1).replace('.', '_')}" if m else "nvcc_unknown"
    except Exception:  # noqa: BLE001
      return "nvcc_unknown"

  def _effective_compiler(self) -> str:
    mode = str(config.cuda_compiler).strip().lower()
    return mode if mode in ("nvrtc", "ptx", "nvcc") else "nvrtc"

  def _compile_nvcc_binary(self, code: str) -> bytes | None:
    """Compile zero-header CUDA C++ with ``nvcc --cubin -O3`` and cache on disk."""
    if not self._nvcc_path:
      self.last_nvcc_error = "nvcc executable not found on PATH"
      return None
    key = self._nvcc_cache.key_for(code)
    cached = self._nvcc_cache.get(key)
    if cached is not None:
      return cached
    try:
      import subprocess
      import tempfile
      from pathlib import Path

      maj, minor = self.profile.compute_capability
      arch = f"sm_{maj}{minor}" if maj > 0 else "sm_70"
      with tempfile.TemporaryDirectory(prefix="pk_nvcc_") as td:
        src = Path(td) / "pk_kernel.cu"
        out = Path(td) / "pk_kernel.cubin"
        src.write_text(code, encoding="utf-8")
        proc = subprocess.run(
          [self._nvcc_path, "--cubin", "-O3", f"-arch={arch}", "-w", str(src), "-o", str(out)],
          capture_output=True,
          text=True,
          timeout=180,
          check=False,
        )
        if proc.returncode != 0 or not out.exists():
          self.last_nvcc_error = (proc.stderr or proc.stdout or f"nvcc exit {proc.returncode}")[:800]
          return None
        binary = out.read_bytes()
      self._nvcc_cache.put(key, binary)
      return binary
    except Exception as e:  # noqa: BLE001
      self.last_nvcc_error = f"{type(e).__name__}: {e}"
      return None

  def _load_ptx_entries(self, entry_bodies: list[str], entry_names: list[str]):
    """Validate and load a batched PTX module directly via CUDA driver JIT."""
    try:
      ptx_text = bundle_ptx_module(entry_bodies, arch=self._ptx_arch)
      verify_ptx(ptx_text)
      mod = self._load_binary_module(ptx_text.encode("utf-8") + b"\x00")
      return [mod.get_function(name) for name in entry_names]
    except Exception as e:  # noqa: BLE001
      self.last_ptx_error = f"{type(e).__name__}: {e}"
      return None

  @staticmethod
  def _nvrtc_version_tag(cupy) -> str:
    try:
      ver = cupy.cuda.nvrtc.getVersion()
      return f"nvrtc_{ver[0]}_{ver[1]}"
    except Exception:  # noqa: BLE001
      return f"cupy_{getattr(cupy, '__version__', 'unknown')}"

  def _compile_nvrtc_binary(self, code: str) -> bytes | None:
    """Compile zero-header CUDA C++ directly through ``cupy.cuda.nvrtc`` to
    CUBIN (or PTX fallback), caching the compiled binary persistently on disk."""
    key = self._binary_cache.key_for(code)
    cached = self._binary_cache.get(key)
    if cached is not None:
      return cached
    try:
      nvrtc = self.cp.cuda.nvrtc
      maj, minor = self.profile.compute_capability
      arch_sm = f"sm_{maj}{minor}" if maj > 0 else None
      arch_compute = f"compute_{maj}{minor}" if maj > 0 else None
      prog = nvrtc.createProgram(code, "pk_kernel.cu", [], [])
      try:
        use_cubin = arch_sm is not None and hasattr(nvrtc, "getCUBIN")
        opts = (
          [f"--gpu-architecture={arch_sm}"]
          if use_cubin
          else ([f"--gpu-architecture={arch_compute}"] if arch_compute else [])
        )
        try:
          nvrtc.compileProgram(prog, opts)
        except Exception:
          if use_cubin and arch_compute is not None:
            # Older NVRTC that only accepts compute_XX -> compile to PTX instead
            nvrtc.destroyProgram(prog)
            prog = nvrtc.createProgram(code, "pk_kernel.cu", [], [])
            use_cubin = False
            nvrtc.compileProgram(prog, [f"--gpu-architecture={arch_compute}"])
          else:
            raise
        binary = nvrtc.getCUBIN(prog) if use_cubin else nvrtc.getPTX(prog)
        if isinstance(binary, str):
          binary = binary.encode("utf-8")
      finally:
        nvrtc.destroyProgram(prog)
      self._binary_cache.put(key, binary)
      return binary
    except Exception as e:  # noqa: BLE001
      self._binary_cache.last_error = f"{type(e).__name__}: {e}"
      return None

  def _load_binary_module(self, binary: bytes):
    mod = self.cp.cuda.Module()
    mod.load(binary)
    with self._cache_lock:
      self._loaded_modules.append(mod)
    return mod

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

  def compile(self, plan: KernelPlan, *, allow_ptx: bool = True):
    mode = self._effective_compiler()
    canon = _canonicalize_cuda_kernel(plan.source)
    cache_key = f"{mode}:{canon}"
    with self._cache_lock:
      cached = self._compiled_cache.get(cache_key)
    if cached is not None:
      return cached
    if mode == "ptx" and allow_ptx:
      ptx_body = emit_ptx_entry_body(plan, entry_name="pk_kernel")
      if ptx_body is not None:
        loaded = self._load_ptx_entries([ptx_body], ["pk_kernel"])
        if loaded is not None:
          kernel = loaded[0]
          with self._cache_lock:
            self._compiled_cache[cache_key] = kernel
            self.ptx_compiled_count += 1
          return kernel
    if mode == "nvcc":
      binary = self._compile_nvcc_binary(canon)
      if binary is not None:
        mod = self._load_binary_module(binary)
        kernel = mod.get_function("pk_kernel")
        with self._cache_lock:
          self._compiled_cache[cache_key] = kernel
          self.nvcc_compiled_count += 1
        return kernel
    binary = self._compile_nvrtc_binary(canon)
    if binary is not None:
      mod = self._load_binary_module(binary)
      kernel = mod.get_function("pk_kernel")
    else:
      kernel = self.cp.RawKernel(canon, "pk_kernel")
      kernel.compile()  # surface compilation problems here, not at first launch
    with self._cache_lock:
      self._compiled_cache[cache_key] = kernel
      self.nvrtc_compiled_count += 1
    return kernel

  def compile_module(self, plans: list[KernelPlan]) -> None:
    """Compile all alpha-unique uncompiled CUDA kernels of a module.

    Depending on ``config.cuda_compiler`` (``"ptx"``, ``"nvrtc"``, ``"nvcc"``):
    - ``"ptx"``: emits eligible kernels directly to PTX 7.8 in a single driver
      JIT load and routes remaining kernels through headerless NVRTC chunks.
    - ``"nvrtc"``: compiles all chunks via headerless NVRTC + CUBIN disk cache.
    - ``"nvcc"``: compiles chunks via ``nvcc --cubin -O3`` + CUBIN disk cache
      (falling back to NVRTC if ``nvcc`` is unavailable).
    """
    import hashlib

    mode = self._effective_compiler()
    todo: list[tuple[KernelPlan, str, str]] = []
    seen_keys: set[str] = set()
    plan_keys: list[tuple[KernelPlan, str]] = []
    for p in plans:
      if p.compiled is not None:
        continue
      canon = _canonicalize_cuda_kernel(p.source)
      ckey = f"{mode}:{canon}"
      plan_keys.append((p, ckey))
      cached = self._compiled_cache.get(ckey)
      if cached is not None:
        p.compiled = cached
        continue
      if ckey not in seen_keys:
        seen_keys.add(ckey)
        todo.append((p, canon, ckey))

    if not todo:
      for p, ckey in plan_keys:
        if p.compiled is None:
          p.compiled = self._compiled_cache.get(ckey)
      return

    # 1. Direct PTX batch for all PTX-eligible kernels (single driver JIT load)
    ptx_todo: list[tuple[KernelPlan, str, str, str, str]] = []
    c_todo: list[tuple[KernelPlan, str, str]] = []
    if mode == "ptx":
      for idx, (p, canon, ckey) in enumerate(todo):
        ename = f"pk_ptx_{idx}"
        body = emit_ptx_entry_body(p, entry_name=ename)
        if body is not None:
          ptx_todo.append((p, canon, ckey, ename, body))
        else:
          c_todo.append((p, canon, ckey))
    else:
      c_todo = list(todo)

    if ptx_todo:
      loaded = self._load_ptx_entries([b for _, _, _, _, b in ptx_todo], [n for _, _, _, n, _ in ptx_todo])
      if loaded is not None:
        with self._cache_lock:
          for (_, _, ckey, _, _), fn in zip(ptx_todo, loaded):
            self._compiled_cache[ckey] = fn
          self.ptx_compiled_count += len(ptx_todo)
      else:
        c_todo.extend((p, canon, ckey) for p, canon, ckey, _, _ in ptx_todo)

    if len(c_todo) <= 1:
      for p, _canon, _ckey in c_todo:
        p.compiled = self.compile(p, allow_ptx=False)
      with self._cache_lock:
        for p, ckey in plan_keys:
          if p.compiled is None:
            p.compiled = self._compiled_cache.get(ckey)
      return

    marker = _KERNEL_MARKER
    for _, canon, _ in c_todo:
      if canon.find(marker) < 0:
        for q, _, _ in c_todo:
          q.compiled = self.compile(q, allow_ptx=False)
        with self._cache_lock:
          for p, ck in plan_keys:
            if p.compiled is None:
              p.compiled = self._compiled_cache.get(ck)
        return

    def _build_chunk(chunk: list[tuple[KernelPlan, str, str]]) -> tuple[str, list[tuple[str, object]]]:
      parts: list[str] = [PRELUDE]
      for idx, (_, canon, _) in enumerate(chunk):
        pos = canon.find(marker)
        parts.append(canon[pos:].replace(marker, f'extern "C" __global__ void pk_k{idx}(', 1))
      code = "\n".join(parts)
      used_kind = "nvrtc"
      binary = None
      if mode == "nvcc":
        binary = self._compile_nvcc_binary(code)
        if binary is not None:
          used_kind = "nvcc"
      if binary is None:
        binary = self._compile_nvrtc_binary(code)
      if binary is not None:
        mod = self._load_binary_module(binary)
      else:
        mod = self.cp.RawModule(code=code)
        mod.compile()
      return used_kind, [(ckey, mod.get_function(f"pk_k{idx}")) for idx, (_, _, ckey) in enumerate(chunk)]

    # Sort canonically for deterministic chunk CUBIN disk caching across runs.
    sorted_todo = sorted(c_todo, key=lambda item: hashlib.sha256(item[1].encode()).hexdigest())
    max_workers = min(os.cpu_count() or 1, 8)
    n_workers = min(max(1, len(sorted_todo) // 4), max_workers) if len(sorted_todo) >= 8 else 1
    chunks = [sorted_todo[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if c]

    if len(chunks) <= 1:
      chunk_outs = [_build_chunk(chunks[0])] if chunks else []
    else:
      from concurrent.futures import ThreadPoolExecutor

      dev_id = int(self.cp.cuda.runtime.getDevice())

      def _worker(chunk: list[tuple[KernelPlan, str, str]]):
        with self.cp.cuda.Device(dev_id):
          return _build_chunk(chunk)

      with ThreadPoolExecutor(max_workers=min(len(chunks), max_workers)) as pool:
        chunk_outs = list(pool.map(_worker, chunks))

    with self._cache_lock:
      for used_kind, pairs in chunk_outs:
        for ckey, fn in pairs:
          self._compiled_cache[ckey] = fn
        if used_kind == "nvcc":
          self.nvcc_compiled_count += len(pairs)
        else:
          self.nvrtc_compiled_count += len(pairs)
      for p, ckey in plan_keys:
        if p.compiled is None:
          p.compiled = self._compiled_cache.get(ckey)

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
