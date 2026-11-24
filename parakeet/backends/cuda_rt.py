"""Native C CUDA Driver API fast-dispatch runtime (``cuLaunchKernel`` + GPU freelist).

Loaded dynamically via ``dlopen("libcuda.so.1")`` so no CUDA SDK headers or
``nvcc`` are required.  During steady-state GPU execution, ``pk_entry`` calls
:c:func:`pk_cu_dispatch` directly as ``rt->device`` — resolving resident
buffers in a C hash table, allocating/recycling temporaries from a stream-
ordered C freelist, computing Split-K grids, and invoking ``cuLaunchKernel``
with zero Python/GIL transitions per kernel launch.

Any cold-start (uncompiled kernel), overlapping/aliased buffer set, or
exception condition transparently falls back to the Python ``CudaDevice``
callback.
"""

from __future__ import annotations

import ctypes
import sys

from . import toolchain

MAX_SCALARS = 64
MAX_BUFS = 32
MAX_TILED_DIMS = 8
MAX_ERR_SLOTS = 256

CUDA_RT_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>

#define PK_EXPORT __attribute__((visibility("default")))
#define PK_MAX_SCALARS 64
#define PK_MAX_BUFS 32
#define PK_MAX_DIMS 8
#define PK_MAX_ERR_SLOTS 256
#define PK_NUM_BUCKETS 64
#define PK_BUCKET_CAP 256
#define PK_TBL_CAP 4096
#define PK_TBL_MASK (PK_TBL_CAP - 1)

typedef int CUresult;
typedef uint64_t CUdeviceptr;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUcontext;

typedef CUresult (*fn_cuLaunchKernel)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void** kernelParams, void** extra);
typedef CUresult (*fn_cuMemAlloc)(CUdeviceptr* dptr, size_t bytesize);
typedef CUresult (*fn_cuMemFree)(CUdeviceptr dptr);
typedef CUresult (*fn_cuMemsetD8Async)(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
typedef CUresult (*fn_cuMemcpyHtoDAsync)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
typedef CUresult (*fn_cuMemcpyDtoH)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
typedef CUresult (*fn_cuStreamSynchronize)(CUstream hStream);
typedef CUresult (*fn_cuCtxGetCurrent)(CUcontext* pctx);

typedef int32_t (*pk_py_fallback_fn)(int32_t kernel, void* ctx, int64_t total, int64_t nchunks);

typedef struct pk_buf {
  void* data;
  int64_t nbytes;
  volatile int64_t refcount;
  int64_t flags;
} pk_buf;

typedef struct {
  void* cu_fn;
  int32_t cpu_only;
  int32_t is_map;
  int32_t is_tiled;
  int32_t has_errors;
  int32_t block_threads;
  int32_t bm;
  int32_t bn;
  int32_t n_dims;
  int32_t dim_is_var[PK_MAX_DIMS];
  int64_t dim_val[PK_MAX_DIMS];
  int32_t split_k_enabled;
  int32_t split_k_dim_is_var;
  int64_t split_k_dim_val;
  int32_t split_k_out_buf;
  int32_t n_scalars;
  int32_t scalar_offsets[PK_MAX_SCALARS];
  int32_t n_bufs;
  int32_t buf_offsets[PK_MAX_BUFS];
  uint8_t buf_written[PK_MAX_BUFS];
} pk_cu_kdesc;

typedef struct {
  uint64_t host_addr;
  uint64_t dptr;
  int64_t nbytes;
  int64_t cap;
  int32_t pinned;
  int32_t dirty;
  int32_t invalidated;
  int32_t pin_idx;
  int32_t occupied;
} pk_tbl_entry;

typedef struct {
  CUdeviceptr ptrs[PK_BUCKET_CAP];
  int32_t count;
} pk_bucket;

/* Process-global driver table (immutable after pk_cu_init under g_init_mu) */
typedef struct {
  fn_cuLaunchKernel launch;
  fn_cuMemAlloc mem_alloc;
  fn_cuMemFree mem_free;
  fn_cuMemsetD8Async memset_async;
  fn_cuMemcpyHtoDAsync htod_async;
  fn_cuMemcpyDtoH dtoh;
  fn_cuStreamSynchronize stream_sync;
  fn_cuCtxGetCurrent ctx_get_current;
  pk_py_fallback_fn default_fallback;
  int32_t ready;
} pk_cu_drv;

/* Per-host-thread active call state + stream/context-bound freelist */
typedef struct {
  /* Size-bucketed GPU freelist bound to (pool_ctx, pool_stream) on this host thread */
  int32_t pool_inited;
  CUcontext pool_ctx;
  CUstream pool_stream;
  pk_bucket buckets[PK_NUM_BUCKETS];

  /* Active call state */
  int32_t active;
  int32_t py_cache_nonempty;
  int32_t sm_count;
  CUstream stream;
  pk_py_fallback_fn py_fallback;
  const pk_cu_kdesc* descs;
  int32_t n_kernels;

  CUdeviceptr flag_dptr;
  CUdeviceptr info_dptr;
  CUdeviceptr err_flags_dptr;
  CUdeviceptr err_infos_dptr;
  int32_t err_cap;
  int32_t pending_err_count;
  int32_t pending_err_kernels[PK_MAX_ERR_SLOTS];

  /* Diagnostics accumulated in C, synced to Python at end_call */
  int64_t launches;
  int64_t uploads;
  int64_t downloads;
  int64_t fallback_reason;
  int64_t fallback_kernel;
  int64_t fallback_rc;

  /* Open-addressed hash table of host_addr -> GPU buffer */
  pk_tbl_entry tbl[PK_TBL_CAP];
  int32_t used_slots[PK_TBL_CAP];
  int32_t n_used;
} pk_cu_rt;

static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;
static pk_cu_drv g_drv;
static __thread pk_cu_rt t_rt;

static inline uint32_t pk_hash_u64(uint64_t x) {
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  return (uint32_t)(x & PK_TBL_MASK);
}

/* Half-step power-of-2 bucketing: 2^k and 1.5*2^k for sizes >= 256B */
static inline int pk_bucket_idx(int64_t nbytes, int64_t* out_cap) {
  if (nbytes <= 256) { *out_cap = 256; return 0; }
  uint64_t u = (uint64_t)(nbytes - 1);
  int lg = 63 - __builtin_clzll(u); /* 2^lg < nbytes <= 2^(lg+1) */
  int64_t hi = (int64_t)(UINT64_C(1) << (lg + 1));
  int64_t mid = (int64_t)((UINT64_C(3) << (lg - 1)));
  int base = (lg - 7) * 2;
  int idx;
  if (nbytes <= mid) { *out_cap = mid; idx = base - 1; }
  else { *out_cap = hi; idx = base; }
  if (idx < 0) { *out_cap = 256; return 0; }
  if (idx >= PK_NUM_BUCKETS) return -1;
  return idx;
}

static void pk_pool_drain(pk_cu_rt* rt) {
  if (g_drv.stream_sync && rt->pool_inited) {
    g_drv.stream_sync(rt->pool_stream);
  }
  for (int b = 0; b < PK_NUM_BUCKETS; ++b) {
    for (int i = 0; i < rt->buckets[b].count; ++i) {
      if (rt->buckets[b].ptrs[i]) g_drv.mem_free(rt->buckets[b].ptrs[i]);
    }
    rt->buckets[b].count = 0;
  }
}

static CUdeviceptr pk_pool_alloc(pk_cu_rt* rt, int64_t nbytes, int64_t* out_cap) {
  int64_t cap = nbytes;
  int b = pk_bucket_idx(nbytes, &cap);
  if (b >= 0 && rt->buckets[b].count > 0) {
    *out_cap = cap;
    return rt->buckets[b].ptrs[--rt->buckets[b].count];
  }
  if (b < 0) cap = nbytes;
  CUdeviceptr dptr = 0;
  if (g_drv.mem_alloc(&dptr, (size_t)cap) != 0) return 0;
  *out_cap = cap;
  return dptr;
}

static void pk_pool_free(pk_cu_rt* rt, CUdeviceptr dptr, int64_t cap) {
  if (!dptr) return;
  int64_t want = cap;
  int b = pk_bucket_idx(cap, &want);
  if (b >= 0 && want == cap && rt->buckets[b].count < PK_BUCKET_CAP) {
    rt->buckets[b].ptrs[rt->buckets[b].count++] = dptr;
    return;
  }
  g_drv.mem_free(dptr);
}

static inline void pk_sync_and_dtoh(pk_cu_rt* rt, void* dst, CUdeviceptr src, size_t nbytes) {
  if (g_drv.stream_sync) g_drv.stream_sync(rt->stream);
  g_drv.dtoh(dst, src, nbytes);
}

static pk_tbl_entry* pk_tbl_find(pk_cu_rt* rt, uint64_t addr) {
  uint32_t h = pk_hash_u64(addr);
  for (int probe = 0; probe < PK_TBL_CAP; ++probe) {
    pk_tbl_entry* e = &rt->tbl[(h + (uint32_t)probe) & PK_TBL_MASK];
    if (e->occupied == 0) return NULL;
    if (e->occupied == 1 && e->host_addr == addr) return e;
  }
  return NULL;
}

static pk_tbl_entry* pk_tbl_insert(pk_cu_rt* rt, uint64_t addr) {
  if (rt->n_used >= PK_TBL_CAP / 2) return NULL;
  uint32_t h = pk_hash_u64(addr);
  int first_tomb = -1;
  for (int probe = 0; probe < PK_TBL_CAP; ++probe) {
    uint32_t idx = (h + (uint32_t)probe) & PK_TBL_MASK;
    pk_tbl_entry* e = &rt->tbl[idx];
    if (e->occupied == 0) {
      uint32_t slot = (first_tomb >= 0) ? (uint32_t)first_tomb : idx;
      e = &rt->tbl[slot];
      if (e->occupied == 0) rt->used_slots[rt->n_used++] = (int32_t)slot;
      e->occupied = 1;
      e->host_addr = addr;
      return e;
    }
    if (e->occupied == 2 && first_tomb < 0) first_tomb = (int)idx;
    if (e->occupied == 1 && e->host_addr == addr) return e;
  }
  return NULL;
}

static int pk_tbl_has_overlap(pk_cu_rt* rt, uint64_t addr, int64_t nbytes) {
  uint64_t end = addr + (uint64_t)nbytes;
  for (int k = 0; k < rt->n_used; ++k) {
    pk_tbl_entry* e = &rt->tbl[rt->used_slots[k]];
    if (e->occupied == 1 && !e->invalidated) {
      if (e->host_addr < end && addr < e->host_addr + (uint64_t)e->nbytes) return 1;
    }
  }
  return 0;
}

/* Download dirty non-pinned C-resident buffers to host and release them back to
   the pool so Python slow-path code sees an up-to-date host view. */
static void pk_cu_flush_to_host_internal(pk_cu_rt* rt, int include_pinned) {
  for (int k = 0; k < rt->n_used; ++k) {
    pk_tbl_entry* e = &rt->tbl[rt->used_slots[k]];
    if (e->occupied != 1) continue;
    if (e->pinned) {
      if (include_pinned && e->dirty && !e->invalidated && e->dptr && e->nbytes > 0) {
        pk_sync_and_dtoh(rt, (void*)(uintptr_t)e->host_addr, e->dptr, (size_t)e->nbytes);
        rt->downloads++;
        e->dirty = 0;
      }
      continue;
    }
    if (e->dirty && e->dptr && e->nbytes > 0) {
      pk_sync_and_dtoh(rt, (void*)(uintptr_t)e->host_addr, e->dptr, (size_t)e->nbytes);
      rt->downloads++;
      e->dirty = 0;
    }
    pk_pool_free(rt, e->dptr, e->cap);
    e->dptr = 0;
    e->occupied = 2;
  }
}

PK_EXPORT int32_t pk_cu_init(void) {
  pthread_mutex_lock(&g_init_mu);
  if (g_drv.ready) {
    pthread_mutex_unlock(&g_init_mu);
    return 0;
  }
  void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (!lib) lib = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!lib) {
    pthread_mutex_unlock(&g_init_mu);
    return 1;
  }
  g_drv.launch = (fn_cuLaunchKernel)dlsym(lib, "cuLaunchKernel");
  g_drv.mem_alloc = (fn_cuMemAlloc)dlsym(lib, "cuMemAlloc_v2");
  if (!g_drv.mem_alloc) g_drv.mem_alloc = (fn_cuMemAlloc)dlsym(lib, "cuMemAlloc");
  g_drv.mem_free = (fn_cuMemFree)dlsym(lib, "cuMemFree_v2");
  if (!g_drv.mem_free) g_drv.mem_free = (fn_cuMemFree)dlsym(lib, "cuMemFree");
  g_drv.memset_async = (fn_cuMemsetD8Async)dlsym(lib, "cuMemsetD8Async");
  g_drv.htod_async = (fn_cuMemcpyHtoDAsync)dlsym(lib, "cuMemcpyHtoDAsync_v2");
  if (!g_drv.htod_async) g_drv.htod_async = (fn_cuMemcpyHtoDAsync)dlsym(lib, "cuMemcpyHtoDAsync");
  g_drv.dtoh = (fn_cuMemcpyDtoH)dlsym(lib, "cuMemcpyDtoH_v2");
  if (!g_drv.dtoh) g_drv.dtoh = (fn_cuMemcpyDtoH)dlsym(lib, "cuMemcpyDtoH");
  g_drv.stream_sync = (fn_cuStreamSynchronize)dlsym(lib, "cuStreamSynchronize");
  g_drv.ctx_get_current = (fn_cuCtxGetCurrent)dlsym(lib, "cuCtxGetCurrent");
  if (!g_drv.launch || !g_drv.mem_alloc || !g_drv.mem_free || !g_drv.memset_async || !g_drv.htod_async || !g_drv.dtoh) {
    pthread_mutex_unlock(&g_init_mu);
    return 2;
  }
  g_drv.ready = 1;
  pthread_mutex_unlock(&g_init_mu);
  return 0;
}

PK_EXPORT int32_t pk_cu_init_custom(
    void* launch, void* mem_alloc, void* mem_free,
    void* memset_async, void* htod_async, void* dtoh, void* stream_sync) {
  pthread_mutex_lock(&g_init_mu);
  g_drv.launch = (fn_cuLaunchKernel)launch;
  g_drv.mem_alloc = (fn_cuMemAlloc)mem_alloc;
  g_drv.mem_free = (fn_cuMemFree)mem_free;
  g_drv.memset_async = (fn_cuMemsetD8Async)memset_async;
  g_drv.htod_async = (fn_cuMemcpyHtoDAsync)htod_async;
  g_drv.dtoh = (fn_cuMemcpyDtoH)dtoh;
  g_drv.stream_sync = (fn_cuStreamSynchronize)stream_sync;
  g_drv.ctx_get_current = NULL;
  g_drv.ready = 1;
  pthread_mutex_unlock(&g_init_mu);
  return 0;
}

PK_EXPORT void pk_cu_set_fallback(void* py_fallback) {
  pthread_mutex_lock(&g_init_mu);
  g_drv.default_fallback = (pk_py_fallback_fn)py_fallback;
  pthread_mutex_unlock(&g_init_mu);
}

PK_EXPORT void pk_cu_pool_free_block(uint64_t dptr, int64_t cap, void* owner_stream) {
  if (!g_drv.ready || !dptr) return;
  pk_cu_rt* rt = &t_rt;
  CUcontext cur_ctx = NULL;
  if (g_drv.ctx_get_current) g_drv.ctx_get_current(&cur_ctx);
  if (rt->pool_inited && rt->pool_stream == (CUstream)owner_stream && rt->pool_ctx == cur_ctx) {
    pk_pool_free(rt, (CUdeviceptr)dptr, cap);
    return;
  }
  if (g_drv.stream_sync) g_drv.stream_sync((CUstream)owner_stream);
  g_drv.mem_free((CUdeviceptr)dptr);
}

PK_EXPORT void pk_cu_begin_call(
    void* stream, int32_t sm_count, void* py_fallback,
    const void* descs, int32_t n_kernels,
    uint64_t flag_dptr, uint64_t info_dptr,
    uint64_t err_flags_dptr, uint64_t err_infos_dptr, int32_t err_cap) {
  pk_cu_rt* rt = &t_rt;
  CUcontext cur_ctx = NULL;
  if (g_drv.ctx_get_current) g_drv.ctx_get_current(&cur_ctx);
  if (rt->pool_inited && (rt->pool_stream != (CUstream)stream || rt->pool_ctx != cur_ctx)) {
    pk_pool_drain(rt);
  }
  rt->pool_inited = 1;
  rt->pool_stream = (CUstream)stream;
  rt->pool_ctx = cur_ctx;
  for (int k = 0; k < rt->n_used; ++k) rt->tbl[rt->used_slots[k]].occupied = 0;
  rt->n_used = 0;
  rt->active = 1;
  rt->py_cache_nonempty = 0;
  rt->sm_count = sm_count;
  rt->stream = (CUstream)stream;
  rt->py_fallback = (pk_py_fallback_fn)py_fallback;
  rt->descs = (const pk_cu_kdesc*)descs;
  rt->n_kernels = n_kernels;
  rt->flag_dptr = flag_dptr;
  rt->info_dptr = info_dptr;
  rt->err_flags_dptr = err_flags_dptr;
  rt->err_infos_dptr = err_infos_dptr;
  rt->err_cap = err_cap < PK_MAX_ERR_SLOTS ? err_cap : PK_MAX_ERR_SLOTS;
  rt->pending_err_count = 0;
  rt->launches = 0;
  rt->uploads = 0;
  rt->downloads = 0;
  rt->fallback_reason = 0;
  rt->fallback_kernel = -1;
  rt->fallback_rc = 0;
}

PK_EXPORT void pk_cu_add_pinned(
    uint64_t host_addr, uint64_t dptr, int64_t nbytes, int32_t pin_idx, int32_t initial_dirty) {
  pk_tbl_entry* e = pk_tbl_insert(&t_rt, host_addr);
  if (!e) return;
  e->dptr = dptr;
  e->nbytes = nbytes;
  e->cap = nbytes;
  e->pinned = 1;
  e->dirty = initial_dirty ? 1 : 0;
  e->invalidated = 0;
  e->pin_idx = pin_idx;
}

PK_EXPORT int32_t pk_cu_drain_pending_errs(int32_t* out_err_kernels, int32_t max_n) {
  pk_cu_rt* rt = &t_rt;
  int32_t n = rt->pending_err_count;
  if (n > max_n) n = max_n;
  for (int i = 0; i < n; ++i) out_err_kernels[i] = rt->pending_err_kernels[i];
  rt->pending_err_count = 0;
  return n;
}

PK_EXPORT int32_t pk_cu_take_output(uint64_t host_addr, int64_t nbytes, uint64_t* out_dptr, int64_t* out_cap, int32_t* out_dirty) {
  pk_tbl_entry* e = pk_tbl_find(&t_rt, host_addr);
  if (!e || e->pinned || e->nbytes < nbytes) return 0;
  *out_dptr = e->dptr;
  *out_cap = e->cap;
  *out_dirty = e->dirty;
  e->dptr = 0;
  e->occupied = 2;
  return 1;
}

/* End-of-call cleanup: retire remaining non-promoted buffers (downloading dirty
   ones to host), report pinned dirty/invalidated states, return counters &
   pending error count. */
PK_EXPORT void pk_cu_end_call(
    int32_t* out_pin_dirty, int32_t* out_pin_inval, int32_t n_pinned,
    int64_t* out_stats, int32_t* out_err_count, int32_t* out_err_kernels) {
  pk_cu_rt* rt = &t_rt;
  for (int k = 0; k < rt->n_used; ++k) {
    pk_tbl_entry* e = &rt->tbl[rt->used_slots[k]];
    if (e->occupied != 1) {
      e->occupied = 0;
      continue;
    }
    if (e->pinned) {
      if (e->pin_idx >= 0 && e->pin_idx < n_pinned) {
        out_pin_dirty[e->pin_idx] = e->dirty;
        out_pin_inval[e->pin_idx] = e->invalidated;
      }
      e->occupied = 0;
      continue;
    }
    if (e->dirty && e->dptr && e->nbytes > 0) {
      pk_sync_and_dtoh(rt, (void*)(uintptr_t)e->host_addr, e->dptr, (size_t)e->nbytes);
      rt->downloads++;
    }
    pk_pool_free(rt, e->dptr, e->cap);
    e->dptr = 0;
    e->occupied = 0;
  }
  rt->n_used = 0;
  rt->active = 0;
  out_stats[0] = rt->launches;
  out_stats[1] = rt->uploads;
  out_stats[2] = rt->downloads;
  out_stats[3] = rt->py_cache_nonempty;
  out_stats[4] = rt->fallback_reason;
  out_stats[5] = rt->fallback_kernel;
  out_stats[6] = rt->fallback_rc;
  *out_err_count = rt->pending_err_count;
  for (int i = 0; i < rt->pending_err_count; ++i) out_err_kernels[i] = rt->pending_err_kernels[i];
  rt->pending_err_count = 0;
}

static int pk_cu_check_errs_inline(pk_cu_rt* rt) {
  if (rt->pending_err_count <= 0) return 0;
  int32_t flags[PK_MAX_ERR_SLOTS];
  pk_sync_and_dtoh(rt, flags, rt->err_flags_dptr, (size_t)rt->pending_err_count * sizeof(int32_t));
  for (int i = 0; i < rt->pending_err_count; ++i) {
    if (flags[i] != 0) return 1; /* error happened! leave flags set for Python to raise */
  }
  rt->pending_err_count = 0;
  return 0;
}

static int32_t pk_cu_fallback_with_flush(
    pk_cu_rt* rt, int32_t kernel, void* ctx, int64_t total, int64_t nchunks, int reason, int rc) {
  if (rt->fallback_reason == 0) {
    rt->fallback_reason = reason;
    rt->fallback_kernel = kernel;
    rt->fallback_rc = rc;
  }
  if (rt->pending_err_count > 0 && pk_cu_check_errs_inline(rt)) {
    return rt->py_fallback(-1, ctx, 0, 0);
  }
  if (!rt->py_cache_nonempty) {
    /* Only flush non-pinned C buffers to host; Python already has pinned DeviceArrays on GPU! */
    pk_cu_flush_to_host_internal(rt, 0);
    rt->py_cache_nonempty = 1;
  }
  return rt->py_fallback(kernel, ctx, total, nchunks);
}

PK_EXPORT int32_t pk_cu_dispatch(int32_t kernel, void* ctx, int64_t total, int64_t nchunks) {
  pk_cu_rt* rt = &t_rt;
  if (!rt->active || !g_drv.ready) {
    pk_py_fallback_fn fb = rt->py_fallback ? rt->py_fallback : g_drv.default_fallback;
    return fb ? fb(kernel, ctx, total, nchunks) : 1;
  }

  /* Control messages: -1 flush, -2 forget, -3 sync read, -4 sync write */
  if (kernel < 0) {
    if (kernel == -2) {
      uint64_t addr = (uint64_t)(uintptr_t)ctx;
      pk_tbl_entry* e = pk_tbl_find(rt, addr);
      if (e && !e->pinned) {
        pk_pool_free(rt, e->dptr, e->cap);
        e->dptr = 0;
        e->occupied = 2;
      }
      if (rt->py_cache_nonempty) return rt->py_fallback(kernel, ctx, total, nchunks);
      return 0;
    }
    if (kernel == -3 || kernel == -4) {
      if (rt->pending_err_count > 0 && pk_cu_check_errs_inline(rt)) {
        /* Error detected: delegate to Python flush to raise the exact exception */
        return rt->py_fallback(-1, ctx, 0, 0);
      }
      uint64_t addr = (uint64_t)(uintptr_t)ctx;
      int64_t nbytes = total;
      if (nbytes > 0) {
        uint64_t end = addr + (uint64_t)nbytes;
        int is_write = (kernel == -4);
        int hit_pinned_write = 0;
        for (int k = 0; k < rt->n_used; ++k) {
          pk_tbl_entry* e = &rt->tbl[rt->used_slots[k]];
          if (e->occupied != 1 || e->invalidated) continue;
          if (e->host_addr < end && addr < e->host_addr + (uint64_t)e->nbytes) {
            if (e->dirty && e->dptr && e->nbytes > 0) {
              pk_sync_and_dtoh(rt, (void*)(uintptr_t)e->host_addr, e->dptr, (size_t)e->nbytes);
              rt->downloads++;
              e->dirty = 0;
            }
            if (is_write) {
              if (e->pinned) {
                e->invalidated = 1;
                hit_pinned_write = 1;
              } else {
                pk_pool_free(rt, e->dptr, e->cap);
                e->dptr = 0;
                e->occupied = 2;
              }
            }
          }
        }
        if (hit_pinned_write || rt->py_cache_nonempty) {
          return rt->py_fallback(kernel, ctx, total, nchunks);
        }
      }
      return 0;
    }
    /* kernel == -1: full flush */
    pk_cu_flush_to_host_internal(rt, 1);
    return rt->py_fallback(kernel, ctx, total, nchunks);
  }

  /* Normal kernel launch fast path */
  if (kernel < rt->n_kernels && rt->descs && rt->descs[kernel].cpu_only) {
    return 1;
  }
  if (kernel >= rt->n_kernels || !rt->descs || !rt->descs[kernel].cu_fn || rt->py_cache_nonempty) {
    return pk_cu_fallback_with_flush(rt, kernel, ctx, total, nchunks, rt->py_cache_nonempty ? 1 : 2, 0);
  }

  const pk_cu_kdesc* kd = &rt->descs[kernel];
  char* raw = (char*)ctx;
  int n_bufs = kd->n_bufs;
  pk_buf* pks[PK_MAX_BUFS];
  uint64_t addrs[PK_MAX_BUFS];
  int64_t sizes[PK_MAX_BUFS];
  int64_t orig_flags[PK_MAX_BUFS];

  int disjoint = 1;
  for (int i = 0; i < n_bufs; ++i) {
    pk_buf* pk = *(pk_buf**)(raw + kd->buf_offsets[i]);
    if (!pk || pk->nbytes <= 0 || !pk->data) { disjoint = 0; break; }
    pks[i] = pk;
    addrs[i] = (uint64_t)(uintptr_t)pk->data;
    sizes[i] = pk->nbytes;
    orig_flags[i] = pk->flags;
    uint64_t e0 = addrs[i] + (uint64_t)sizes[i];
    for (int j = 0; j < i; ++j) {
      if (addrs[i] < addrs[j] + (uint64_t)sizes[j] && addrs[j] < e0) {
        disjoint = 0;
        break;
      }
    }
    if (!disjoint) break;
  }
  if (!disjoint) {
    return pk_cu_fallback_with_flush(rt, kernel, ctx, total, nchunks, 3, 0);
  }

  /* Verify all buffers can be resolved in C before mutating anything */
  CUdeviceptr dptrs[PK_MAX_BUFS];
  pk_tbl_entry* found[PK_MAX_BUFS];
  for (int i = 0; i < n_bufs; ++i) {
    pk_tbl_entry* e = pk_tbl_find(rt, addrs[i]);
    if (e) {
      if (e->nbytes != sizes[i] || e->invalidated) {
        return pk_cu_fallback_with_flush(rt, kernel, ctx, total, nchunks, 4, 0);
      }
      found[i] = e;
    } else {
      if (!(orig_flags[i] & 6) && pk_tbl_has_overlap(rt, addrs[i], sizes[i])) {
        return pk_cu_fallback_with_flush(rt, kernel, ctx, total, nchunks, 5, 0);
      }
      found[i] = NULL;
    }
  }

  /* Acquire GPU pointers for every buffer */
  for (int i = 0; i < n_bufs; ++i) {
    int64_t fl = orig_flags[i];
    int w = kd->buf_written[i];
    if (fl & 14) pks[i]->flags = fl & ~14;
    pk_tbl_entry* e = found[i];
    if (e) {
      if (w) e->dirty = 1;
      dptrs[i] = e->dptr;
    } else {
      int64_t cap = 0;
      CUdeviceptr dptr = pk_pool_alloc(rt, sizes[i], &cap);
      if (!dptr) {
        return pk_cu_fallback_with_flush(rt, kernel, ctx, total, nchunks, 6, 0);
      }
      if (fl & 4) {
        g_drv.memset_async(dptr, 0, (size_t)sizes[i], rt->stream);
      } else if (!(fl & 2)) {
        g_drv.htod_async(dptr, (const void*)(uintptr_t)addrs[i], (size_t)sizes[i], rt->stream);
        rt->uploads++;
      }
      pk_tbl_entry* ne = pk_tbl_insert(rt, addrs[i]);
      if (!ne) {
        pk_pool_free(rt, dptr, cap);
        return pk_cu_fallback_with_flush(rt, kernel, ctx, total, nchunks, 7, 0);
      }
      ne->dptr = dptr;
      ne->nbytes = sizes[i];
      ne->cap = cap;
      ne->pinned = 0;
      ne->dirty = (w || (fl & 4)) ? 1 : 0;
      ne->invalidated = 0;
      ne->pin_idx = -1;
      dptrs[i] = dptr;
    }
  }

  /* Compute grid, block, k_total, k_csize (including Split-K WMMA) */
  int64_t k_total = total;
  int64_t k_csize = 1;
  int64_t grid = 1;
  int32_t block = kd->block_threads;

  if (kd->is_tiled) {
    int64_t dims[PK_MAX_DIMS];
    for (int d = 0; d < kd->n_dims; ++d) {
      dims[d] = kd->dim_is_var[d] ? *(int64_t*)(raw + kd->dim_val[d]) : kd->dim_val[d];
      if (dims[d] <= 0) return 0;
    }
    int64_t batch = 1;
    for (int d = 0; d < kd->n_dims - 2; ++d) batch *= dims[d];
    int64_t M = dims[kd->n_dims - 2];
    int64_t N = dims[kd->n_dims - 1];
    int64_t base_blocks = batch * ((M + kd->bm - 1) / kd->bm) * ((N + kd->bn - 1) / kd->bn);
    if (base_blocks <= 0) return 0;
    grid = base_blocks;
    k_csize = 1;
    if (kd->split_k_enabled && rt->sm_count > 0 && base_blocks < rt->sm_count) {
      int64_t K = kd->split_k_dim_is_var ? *(int64_t*)(raw + kd->split_k_dim_val) : kd->split_k_dim_val;
      int ob = kd->split_k_out_buf;
      if (K >= 256 && ob >= 0 && ob < n_bufs && ((orig_flags[ob] & 6) || sizes[ob] == batch * M * N * 4)) {
        int64_t sk = (rt->sm_count * 2) / base_blocks;
        int64_t max_k = K / 64;
        if (sk > max_k) sk = max_k;
        if (sk > 16) sk = 16;
        if (sk > 1) {
          if (!(orig_flags[ob] & 4)) {
            g_drv.memset_async(dptrs[ob], 0, (size_t)sizes[ob], rt->stream);
          }
          grid = base_blocks * sk;
          k_csize = sk;
        }
      }
    }
  } else {
    int64_t n = kd->is_map ? (total < (1LL << 24) ? total : (1LL << 24)) : nchunks;
    if (n <= 0) return 0;
    k_csize = (total + n - 1) / n;
    n = (total + k_csize - 1) / k_csize;
    grid = (n + block - 1) / block;
  }

  /* Assign deferred error slot if this kernel has Check statements */
  CUdeviceptr flag_ptr = rt->flag_dptr;
  CUdeviceptr info_ptr = rt->info_dptr;
  if (kd->has_errors) {
    if (rt->pending_err_count >= rt->err_cap) {
      if (pk_cu_check_errs_inline(rt)) {
        return rt->py_fallback(-1, ctx, 0, 0);
      }
    }
    int slot = rt->pending_err_count++;
    rt->pending_err_kernels[slot] = kernel;
    flag_ptr = rt->err_flags_dptr + (CUdeviceptr)((uint64_t)slot * sizeof(int32_t));
    info_ptr = rt->err_infos_dptr + (CUdeviceptr)((uint64_t)slot * 4 * sizeof(int64_t));
  }

  /* Pack kernelParams directly on stack (scalars read straight from pk_ctx_k!) */
  void* kargs[4 + PK_MAX_SCALARS + PK_MAX_BUFS];
  int pos = 0;
  kargs[pos++] = &k_total;
  kargs[pos++] = &k_csize;
  for (int j = 0; j < kd->n_scalars; ++j) {
    kargs[pos++] = (void*)(raw + kd->scalar_offsets[j]);
  }
  kargs[pos++] = &flag_ptr;
  kargs[pos++] = &info_ptr;
  for (int i = 0; i < n_bufs; ++i) {
    kargs[pos++] = &dptrs[i];
  }

  CUresult rc = g_drv.launch(kd->cu_fn, (unsigned int)grid, 1u, 1u, (unsigned int)block, 1u, 1u, 0u, rt->stream, kargs, NULL);
  if (rc != 0) {
    return pk_cu_fallback_with_flush(rt, kernel, ctx, total, nchunks, 8, (int)rc);
  }
  rt->launches++;
  return 0;
}
"""


class PkCuKDesc(ctypes.Structure):
  _fields_ = [
    ("cu_fn", ctypes.c_void_p),
    ("cpu_only", ctypes.c_int32),
    ("is_map", ctypes.c_int32),
    ("is_tiled", ctypes.c_int32),
    ("has_errors", ctypes.c_int32),
    ("block_threads", ctypes.c_int32),
    ("bm", ctypes.c_int32),
    ("bn", ctypes.c_int32),
    ("n_dims", ctypes.c_int32),
    ("dim_is_var", ctypes.c_int32 * MAX_TILED_DIMS),
    ("dim_val", ctypes.c_int64 * MAX_TILED_DIMS),
    ("split_k_enabled", ctypes.c_int32),
    ("split_k_dim_is_var", ctypes.c_int32),
    ("split_k_dim_val", ctypes.c_int64),
    ("split_k_out_buf", ctypes.c_int32),
    ("n_scalars", ctypes.c_int32),
    ("scalar_offsets", ctypes.c_int32 * MAX_SCALARS),
    ("n_bufs", ctypes.c_int32),
    ("buf_offsets", ctypes.c_int32 * MAX_BUFS),
    ("buf_written", ctypes.c_uint8 * MAX_BUFS),
  ]


class _CBlockOwner:
  """Keeps a C-freelist GPU allocation alive while wrapped in a CuPy ndarray
  and returns it straight to the C freelist on garbage collection (only when
  freed on the owning thread and stream; otherwise frees via ``cuMemFree``)."""

  __slots__ = ("_cap", "_dptr", "_free_fn", "_stream")

  def __init__(self, free_fn, dptr: int, cap: int, stream: int = 0) -> None:
    self._free_fn = free_fn
    self._dptr = dptr
    self._cap = cap
    self._stream = stream

  def __del__(self) -> None:
    try:
      self._free_fn(self._dptr, self._cap, self._stream)
    except Exception:  # noqa: BLE001 - interpreter shutdown
      pass


class NativeCudaRuntime:
  """Wrapper around the compiled ``pk_cuda_rt`` shared library."""

  def __init__(self, init_driver: bool = True) -> None:
    extra = ("-ldl", "-pthread") if sys.platform.startswith("linux") else ("-pthread",)
    path = toolchain.compile_shared(CUDA_RT_SOURCE, extra_flags=extra, tag="cuda_rt", opt_level=2)
    lib = toolchain.load(path)
    lib.pk_cu_init.restype = ctypes.c_int32
    lib.pk_cu_init.argtypes = []
    lib.pk_cu_init_custom.restype = ctypes.c_int32
    lib.pk_cu_init_custom.argtypes = [ctypes.c_void_p] * 7
    if init_driver and lib.pk_cu_init() != 0:
      raise RuntimeError("libcuda.so.1 driver symbols unavailable")
    lib.pk_cu_set_fallback.restype = None
    lib.pk_cu_set_fallback.argtypes = [ctypes.c_void_p]
    lib.pk_cu_pool_free_block.restype = None
    lib.pk_cu_pool_free_block.argtypes = [ctypes.c_uint64, ctypes.c_int64, ctypes.c_void_p]
    lib.pk_cu_begin_call.restype = None
    lib.pk_cu_begin_call.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int32,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_int32,
      ctypes.c_uint64,
      ctypes.c_uint64,
      ctypes.c_uint64,
      ctypes.c_uint64,
      ctypes.c_int32,
    ]
    lib.pk_cu_add_pinned.restype = None
    lib.pk_cu_add_pinned.argtypes = [
      ctypes.c_uint64,
      ctypes.c_uint64,
      ctypes.c_int64,
      ctypes.c_int32,
      ctypes.c_int32,
    ]
    lib.pk_cu_drain_pending_errs.restype = ctypes.c_int32
    lib.pk_cu_drain_pending_errs.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    lib.pk_cu_take_output.restype = ctypes.c_int32
    lib.pk_cu_take_output.argtypes = [
      ctypes.c_uint64,
      ctypes.c_int64,
      ctypes.POINTER(ctypes.c_uint64),
      ctypes.POINTER(ctypes.c_int64),
      ctypes.POINTER(ctypes.c_int32),
    ]
    lib.pk_cu_end_call.restype = None
    lib.pk_cu_end_call.argtypes = [
      ctypes.POINTER(ctypes.c_int32),
      ctypes.POINTER(ctypes.c_int32),
      ctypes.c_int32,
      ctypes.POINTER(ctypes.c_int64),
      ctypes.POINTER(ctypes.c_int32),
      ctypes.POINTER(ctypes.c_int32),
    ]
    self.lib = lib
    self.dispatch_address = ctypes.cast(lib.pk_cu_dispatch, ctypes.c_void_p).value or 0

  def make_owner(self, dptr: int, cap: int, stream: int = 0) -> _CBlockOwner:
    return _CBlockOwner(self.lib.pk_cu_pool_free_block, dptr, cap, stream)


def try_load_native_cuda_rt() -> NativeCudaRuntime | None:
  """Load the native C CUDA dispatch runtime on Linux with NVIDIA driver, or None."""
  if not sys.platform.startswith("linux"):
    return None
  try:
    return NativeCudaRuntime()
  except Exception:  # noqa: BLE001
    return None
