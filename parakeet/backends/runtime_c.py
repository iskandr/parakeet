"""The C runtime that is prepended to every generated module.

It is deliberately tiny and dependency-free: reference-counted buffers, error
reporting, NumPy-compatible integer/float helpers, a counter-based random
number generator and four interchangeable ways of running chunks in parallel
(serial, OpenMP, Grand Central Dispatch, POSIX threads).
"""

from __future__ import annotations

PAGE = 16384  # allocation granularity for large buffers (a multiple of every page size we target)

# ia[0..7]: nthreads, grain, parfor fn, device fn, gpu_min, rng key, device chunks, reserved
HEADER_SLOTS = 12


# ---------------------------------------------------------------------------
# Scalar helpers shared (as text) by the C, Metal and CUDA code generators.
# ---------------------------------------------------------------------------

HELPERS_INT = r"""
#define PK_DEF_SIGNED(T, N, UT, BITS) \
  PK_INLINE T pk_floordiv_##N(T a, T b) { \
    if (b == 0) return 0; \
    if (b == (T)-1) return (T)(0 - (UT)a); \
    T q = (T)(a / b); \
    if ((a % b != 0) && ((a < 0) != (b < 0))) q = (T)(q - 1); \
    return q; } \
  PK_INLINE T pk_mod_##N(T a, T b) { \
    if (b == 0 || b == (T)-1) return 0; \
    T r = (T)(a % b); \
    if (r != 0 && ((r < 0) != (b < 0))) r = (T)(r + b); \
    return r; } \
  PK_INLINE T pk_cmod_##N(T a, T b) { \
    if (b == 0 || b == (T)-1) return 0; \
    return (T)(a % b); } \
  PK_INLINE T pk_ipow_##N(T a, T b) { \
    UT r = 1, base = (UT)a; \
    if (b < 0) return (T)(a == 1 ? 1 : (a == (T)-1 ? ((b & 1) ? -1 : 1) : 0)); \
    while (b) { if (b & 1) r = (UT)(r * base); base = (UT)(base * base); b = (T)(b >> 1); } \
    return (T)r; } \
  PK_INLINE T pk_shl_##N(T a, T b) { \
    return (b < 0 || b >= BITS) ? (T)0 : (T)((UT)a << b); } \
  PK_INLINE T pk_shr_##N(T a, T b) { \
    return (b < 0 || b >= BITS) ? (T)(a < 0 ? -1 : 0) : (T)(a >> b); }

#define PK_DEF_UNSIGNED(T, N, BITS) \
  PK_INLINE T pk_floordiv_##N(T a, T b) { return b == 0 ? (T)0 : (T)(a / b); } \
  PK_INLINE T pk_mod_##N(T a, T b) { return b == 0 ? (T)0 : (T)(a % b); } \
  PK_INLINE T pk_cmod_##N(T a, T b) { return b == 0 ? (T)0 : (T)(a % b); } \
  PK_INLINE T pk_ipow_##N(T a, T b) { \
    T r = 1; \
    while (b) { if (b & 1) r = (T)(r * a); a = (T)(a * a); b = (T)(b >> 1); } \
    return r; } \
  PK_INLINE T pk_shl_##N(T a, T b) { return (b >= BITS) ? (T)0 : (T)(a << b); } \
  PK_INLINE T pk_shr_##N(T a, T b) { return (b >= BITS) ? (T)0 : (T)(a >> b); }

PK_DEF_SIGNED(int8_t, int8, uint8_t, 8)
PK_DEF_SIGNED(int16_t, int16, uint16_t, 16)
PK_DEF_SIGNED(int32_t, int32, uint32_t, 32)
PK_DEF_SIGNED(int64_t, int64, uint64_t, 64)
PK_DEF_UNSIGNED(uint8_t, uint8, 8)
PK_DEF_UNSIGNED(uint16_t, uint16, 16)
PK_DEF_UNSIGNED(uint32_t, uint32, 32)
PK_DEF_UNSIGNED(uint64_t, uint64, 64)
"""

HELPERS_BITOPS = r"""
#if defined(__METAL_VERSION__)
PK_INLINE int32_t pk_popcount_u32(uint32_t x) { return (int32_t)popcount(x); }
PK_INLINE int32_t pk_popcount_u64(uint64_t x) {
  return (int32_t)(popcount((uint32_t)x) + popcount((uint32_t)(x >> 32)));
}
PK_INLINE int32_t pk_clz_u32(uint32_t x) { return (int32_t)clz(x); }
PK_INLINE int32_t pk_clz_u64(uint64_t x) {
  uint32_t hi = (uint32_t)(x >> 32);
  return hi ? (int32_t)clz(hi) : (32 + (int32_t)clz((uint32_t)x));
}
PK_INLINE int32_t pk_ctz_u32(uint32_t x) { return (int32_t)ctz(x); }
PK_INLINE int32_t pk_ctz_u64(uint64_t x) {
  uint32_t lo = (uint32_t)x;
  return lo ? (int32_t)ctz(lo) : (32 + (int32_t)ctz((uint32_t)(x >> 32)));
}
PK_INLINE float pk_bf16_to_f32(uint16_t h) { return as_type<float>(((uint32_t)h) << 16); }
PK_INLINE uint16_t pk_f32_to_bf16(float f) {
  uint32_t u = as_type<uint32_t>(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#elif defined(__CUDA_ARCH__)
PK_INLINE int32_t pk_popcount_u32(uint32_t x) { return __popc(x); }
PK_INLINE int32_t pk_popcount_u64(uint64_t x) { return __popcll(x); }
PK_INLINE int32_t pk_clz_u32(uint32_t x) { return __clz((int)x); }
PK_INLINE int32_t pk_clz_u64(uint64_t x) { return __clzll((long long)x); }
PK_INLINE int32_t pk_ctz_u32(uint32_t x) { return x == 0 ? 32 : (__ffs((int)x) - 1); }
PK_INLINE int32_t pk_ctz_u64(uint64_t x) { return x == 0 ? 64 : (__ffsll((long long)x) - 1); }
PK_INLINE float pk_bf16_to_f32(uint16_t h) { return __uint_as_float(((uint32_t)h) << 16); }
PK_INLINE uint16_t pk_f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#elif defined(_MSC_VER) && !defined(__clang__)
PK_INLINE int32_t pk_popcount_u32(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  return (int32_t)((((x + (x >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24);
}
PK_INLINE int32_t pk_popcount_u64(uint64_t x) {
  return pk_popcount_u32((uint32_t)x) + pk_popcount_u32((uint32_t)(x >> 32));
}
PK_INLINE int32_t pk_clz_u32(uint32_t x) {
  int32_t n = 32;
  if (x >= 0x10000u) { n -= 16; x >>= 16; }
  if (x >= 0x100u) { n -= 8; x >>= 8; }
  if (x >= 0x10u) { n -= 4; x >>= 4; }
  if (x >= 0x4u) { n -= 2; x >>= 2; }
  if (x >= 0x2u) { n -= 1; x >>= 1; }
  return n - (int32_t)x;
}
PK_INLINE int32_t pk_clz_u64(uint64_t x) {
  uint32_t hi = (uint32_t)(x >> 32);
  return hi ? pk_clz_u32(hi) : (32 + pk_clz_u32((uint32_t)x));
}
PK_INLINE int32_t pk_ctz_u32(uint32_t x) {
  if (x == 0) return 32;
  int32_t n = 0;
  if ((x & 0x0000ffffu) == 0) { n += 16; x >>= 16; }
  if ((x & 0x000000ffu) == 0) { n += 8; x >>= 8; }
  if ((x & 0x0000000fu) == 0) { n += 4; x >>= 4; }
  if ((x & 0x00000003u) == 0) { n += 2; x >>= 2; }
  if ((x & 0x00000001u) == 0) { n += 1; }
  return n;
}
PK_INLINE int32_t pk_ctz_u64(uint64_t x) {
  uint32_t lo = (uint32_t)x;
  return lo ? pk_ctz_u32(lo) : (32 + pk_ctz_u32((uint32_t)(x >> 32)));
}
PK_INLINE float pk_bf16_to_f32(uint16_t h) {
  uint32_t u = ((uint32_t)h) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
PK_INLINE uint16_t pk_f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#else
PK_INLINE int32_t pk_popcount_u32(uint32_t x) { return (int32_t)__builtin_popcount(x); }
PK_INLINE int32_t pk_popcount_u64(uint64_t x) { return (int32_t)__builtin_popcountll((unsigned long long)x); }
PK_INLINE int32_t pk_clz_u32(uint32_t x) { return x == 0 ? 32 : (int32_t)__builtin_clz(x); }
PK_INLINE int32_t pk_clz_u64(uint64_t x) { return x == 0 ? 64 : (int32_t)__builtin_clzll((unsigned long long)x); }
PK_INLINE int32_t pk_ctz_u32(uint32_t x) { return x == 0 ? 32 : (int32_t)__builtin_ctz(x); }
PK_INLINE int32_t pk_ctz_u64(uint64_t x) { return x == 0 ? 64 : (int32_t)__builtin_ctzll((unsigned long long)x); }
PK_INLINE float pk_bf16_to_f32(uint16_t h) {
  uint32_t u = ((uint32_t)h) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
PK_INLINE uint16_t pk_f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
#endif

PK_INLINE uint8_t pk_bswap_u8(uint8_t x) { return x; }
PK_INLINE uint16_t pk_bswap_u16(uint16_t x) { return (uint16_t)((x >> 8) | (x << 8)); }
PK_INLINE uint32_t pk_bswap_u32(uint32_t x) {
  return ((x & 0x000000ffu) << 24) | ((x & 0x0000ff00u) << 8) |
         ((x & 0x00ff0000u) >> 8)  | ((x & 0xff000000u) >> 24);
}
PK_INLINE uint64_t pk_bswap_u64(uint64_t x) {
  return ((uint64_t)pk_bswap_u32((uint32_t)x) << 32) | (uint64_t)pk_bswap_u32((uint32_t)(x >> 32));
}

#define PK_DEF_BITOPS_SMALL(T, N, UT, BITS, BSWAP) \
  PK_INLINE int32_t pk_popcount_##N(T x) { return pk_popcount_u32((uint32_t)(UT)x); } \
  PK_INLINE int32_t pk_clz_##N(T x) { UT u = (UT)x; return u == 0 ? BITS : (pk_clz_u32((uint32_t)u) - (32 - BITS)); } \
  PK_INLINE int32_t pk_ctz_##N(T x) { UT u = (UT)x; return u == 0 ? BITS : pk_ctz_u32((uint32_t)u); } \
  PK_INLINE T pk_rotl_##N(T x, int32_t k) { \
    uint32_t r = ((uint32_t)k) & (BITS - 1); UT u = (UT)x; \
    return (T)(r == 0 ? u : (UT)((u << r) | (u >> (BITS - r)))); } \
  PK_INLINE T pk_rotr_##N(T x, int32_t k) { \
    uint32_t r = ((uint32_t)k) & (BITS - 1); UT u = (UT)x; \
    return (T)(r == 0 ? u : (UT)((u >> r) | (u << (BITS - r)))); } \
  PK_INLINE T pk_bswap_##N(T x) { return (T)BSWAP((UT)x); }

#define PK_DEF_BITOPS_64(T, N, UT) \
  PK_INLINE int32_t pk_popcount_##N(T x) { return pk_popcount_u64((UT)x); } \
  PK_INLINE int32_t pk_clz_##N(T x) { return pk_clz_u64((UT)x); } \
  PK_INLINE int32_t pk_ctz_##N(T x) { return pk_ctz_u64((UT)x); } \
  PK_INLINE T pk_rotl_##N(T x, int32_t k) { \
    uint32_t r = ((uint32_t)k) & 63u; UT u = (UT)x; \
    return (T)(r == 0 ? u : (UT)((u << r) | (u >> (64u - r)))); } \
  PK_INLINE T pk_rotr_##N(T x, int32_t k) { \
    uint32_t r = ((uint32_t)k) & 63u; UT u = (UT)x; \
    return (T)(r == 0 ? u : (UT)((u >> r) | (u << (64u - r)))); } \
  PK_INLINE T pk_bswap_##N(T x) { return (T)pk_bswap_u64((UT)x); }

PK_DEF_BITOPS_SMALL(int8_t, int8, uint8_t, 8, pk_bswap_u8)
PK_DEF_BITOPS_SMALL(uint8_t, uint8, uint8_t, 8, pk_bswap_u8)
PK_DEF_BITOPS_SMALL(int16_t, int16, uint16_t, 16, pk_bswap_u16)
PK_DEF_BITOPS_SMALL(uint16_t, uint16, uint16_t, 16, pk_bswap_u16)
PK_DEF_BITOPS_SMALL(int32_t, int32, uint32_t, 32, pk_bswap_u32)
PK_DEF_BITOPS_SMALL(uint32_t, uint32, uint32_t, 32, pk_bswap_u32)
PK_DEF_BITOPS_64(int64_t, int64, uint64_t)
PK_DEF_BITOPS_64(uint64_t, uint64, uint64_t)
"""

HELPERS_FLOAT_DEF = r"""
#define PK_DEF_FLOAT(T, N, FMOD, FLOOR, COPYSIGN) \
  PK_INLINE T pk_min_##N(T a, T b) { return (a != a) ? a : ((b != b) ? b : (a < b ? a : b)); } \
  PK_INLINE T pk_max_##N(T a, T b) { return (a != a) ? a : ((b != b) ? b : (a > b ? a : b)); } \
  PK_INLINE T pk_mod_##N(T a, T b) { \
    T m; \
    if (b == 0) return FMOD(a, b); \
    m = FMOD(a, b); \
    if (m != 0) { if ((b < 0) != (m < 0)) m += b; } else { m = COPYSIGN((T)0, b); } \
    return m; } \
  PK_INLINE T pk_floordiv_##N(T a, T b) { \
    T m, d, f; \
    if (b == 0) return a / b; \
    m = FMOD(a, b); \
    d = (a - m) / b; \
    if (m != 0 && ((b < 0) != (m < 0))) d -= (T)1; \
    if (d != 0) { f = FLOOR(d); if (d - f > (T)0.5) f += (T)1; return f; } \
    return COPYSIGN((T)0, a / b); }
"""

HELPERS_F32 = r"""
PK_DEF_FLOAT(float, float32, fmodf, floorf, copysignf)
PK_INLINE int64_t pk_f2i64f(float x) {
  return (x >= -9223372036854775808.0f && x < 9223372036854775808.0f) ? (int64_t)x : INT64_MIN;
}
PK_INLINE uint64_t pk_f2u64f(float x) {
  return (x >= 0.0f && x < 18446744073709551616.0f) ? (uint64_t)x : (uint64_t)pk_f2i64f(x);
}
"""

HELPERS_F64 = r"""
PK_DEF_FLOAT(double, float64, fmod, floor, copysign)
PK_INLINE int64_t pk_f2i64(double x) {
  return (x >= -9223372036854775808.0 && x < 9223372036854775808.0) ? (int64_t)x : INT64_MIN;
}
PK_INLINE uint64_t pk_f2u64(double x) {
  return (x >= 0.0 && x < 18446744073709551616.0) ? (uint64_t)x : (uint64_t)pk_f2i64(x);
}
"""

HELPERS_RNG = r"""
/* ---- counter-based random numbers (splitmix64 finaliser) ----------------- */

PK_INLINE uint64_t pk_mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}
PK_INLINE uint64_t pk_rand_u64(int64_t key, int64_t i) {
  return pk_mix64(pk_mix64((uint64_t)key) + (uint64_t)i * UINT64_C(0x9e3779b97f4a7c15));
}
PK_INLINE float pk_rand_f32(int64_t key, int64_t i) {
  return (float)(pk_rand_u64(key, i) >> 40) * (1.0f / 16777216.0f);
}
PK_INLINE int64_t pk_rand_int(int64_t key, int64_t i, int64_t lo, int64_t hi) {
  uint64_t span = (uint64_t)hi - (uint64_t)lo;
  if (span == 0) return lo;
  return (int64_t)((uint64_t)lo + pk_rand_u64(key, i) % span);
}
"""

HELPERS_RNG_F64 = r"""
PK_INLINE double pk_rand_f64(int64_t key, int64_t i) {
  return (double)(pk_rand_u64(key, i) >> 11) * (1.0 / 9007199254740992.0);
}
"""

_RNG_KEY = r"""
static int64_t pk_rng_key(pk_call* rt) {
  int64_t n = pk_atomic_add(&rt->rng_counter, 1);
  return (int64_t)pk_mix64(rt->rng_key + (uint64_t)n * UINT64_C(0xd1342543de82ef95));
}
"""

_PRELUDE_TEMPLATE = r"""
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_MSC_VER) && !defined(__clang__)
  #include <intrin.h>
  #include <malloc.h>
  #define PK_EXPORT __declspec(dllexport)
  #define PK_INLINE static __forceinline
  #define PK_OUTLINE static __declspec(noinline)
  #define PK_UNLIKELY(x) (x)
  PK_INLINE int64_t pk_atomic_add(volatile int64_t* p, int64_t v) {
    return _InterlockedExchangeAdd64((volatile __int64*)p, v) + v;
  }
  PK_INLINE bool pk_atomic_cas(volatile int32_t* p, int32_t expect, int32_t val) {
    return _InterlockedCompareExchange((volatile long*)p, (long)val, (long)expect) == (long)expect;
  }
  PK_INLINE void pk_atomic_add_float32(float* p, float v) {
    int32_t old_i, new_i;
    float old_f, new_f;
    do {
      old_i = *(volatile int32_t*)p;
      memcpy(&old_f, &old_i, 4);
      new_f = old_f + v;
      memcpy(&new_i, &new_f, 4);
    } while (_InterlockedCompareExchange((volatile long*)p, (long)new_i, (long)old_i) != (long)old_i);
  }
  PK_INLINE void pk_atomic_add_float64(double* p, double v) {
    int64_t old_i, new_i;
    double old_f, new_f;
    do {
      old_i = *(volatile int64_t*)p;
      memcpy(&old_f, &old_i, 8);
      new_f = old_f + v;
      memcpy(&new_i, &new_f, 8);
    } while (_InterlockedCompareExchange64((volatile __int64*)p, (__int64)new_i, (__int64)old_i) != (__int64)old_i);
  }
  PK_INLINE void pk_atomic_add_int32(int32_t* p, int32_t v) { _InterlockedExchangeAdd((volatile long*)p, (long)v); }
  PK_INLINE void pk_atomic_add_uint32(uint32_t* p, uint32_t v) { _InterlockedExchangeAdd((volatile long*)p, (long)v); }
#else
  #if defined(_WIN32)
    #include <malloc.h>
    #define PK_EXPORT __declspec(dllexport)
  #else
    #define PK_EXPORT __attribute__((visibility("default")))
  #endif
  #define PK_INLINE static inline __attribute__((always_inline, unused))
  /* called from very many places in generated code: inlining them there only
     makes the C compiler slow */
  #define PK_OUTLINE static __attribute__((noinline, unused))
  #define PK_UNLIKELY(x) __builtin_expect(!!(x), 0)
  PK_INLINE int64_t pk_atomic_add(volatile int64_t* p, int64_t v) { return __sync_add_and_fetch(p, v); }
  PK_INLINE bool pk_atomic_cas(volatile int32_t* p, int32_t expect, int32_t val) {
    return __sync_bool_compare_and_swap(p, expect, val);
  }
  PK_INLINE void pk_atomic_add_float32(float* p, float v) {
    uint32_t old_i, new_i;
    float old_f, new_f;
    do {
      old_i = __atomic_load_n((uint32_t*)p, __ATOMIC_RELAXED);
      memcpy(&old_f, &old_i, 4);
      new_f = old_f + v;
      memcpy(&new_i, &new_f, 4);
    } while (!__sync_bool_compare_and_swap((uint32_t*)p, old_i, new_i));
  }
  PK_INLINE void pk_atomic_add_float64(double* p, double v) {
    uint64_t old_i, new_i;
    double old_f, new_f;
    do {
      old_i = __atomic_load_n((uint64_t*)p, __ATOMIC_RELAXED);
      memcpy(&old_f, &old_i, 8);
      new_f = old_f + v;
      memcpy(&new_i, &new_f, 8);
    } while (!__sync_bool_compare_and_swap((uint64_t*)p, old_i, new_i));
  }
  PK_INLINE void pk_atomic_add_int32(int32_t* p, int32_t v) { __sync_fetch_and_add(p, v); }
  PK_INLINE void pk_atomic_add_uint32(uint32_t* p, uint32_t v) { __sync_fetch_and_add(p, v); }
#endif

#ifndef INFINITY
  #define INFINITY (1e300 * 1e300)
#endif
#ifndef NAN
  #define NAN (INFINITY - INFINITY)
#endif

/* ---- buffers ------------------------------------------------------------ */

#define PK_EXTERNAL 1
/* what the host has put into a buffer it allocated: nothing yet (the contents
   do not matter) or zeros.  A device need not copy such a buffer over. */
#define PK_FRESH 2
#define PK_ZEROED 4
#define PK_LAZY_ZERO 8
#define PK_UNTOUCHED (PK_FRESH | PK_ZEROED)
#define PK_PAGE 16384

typedef struct pk_buf {
  void* data;
  int64_t nbytes;
  volatile int64_t refcount;
  int64_t flags;
} pk_buf;

typedef struct pk_call {
  volatile int32_t err_code;
  int32_t err_id;
  int64_t err_args[4];
  int64_t nthreads;
  int64_t grain;
  int64_t gpu_min;
  int64_t device_chunks;
  void* parfor;
  void* device;
  uint64_t rng_key;
  volatile int64_t rng_counter;
  int64_t dev_live; /* arrays are on the device: flush before the host touches array memory */
  int64_t blas;    /* bit 0: there is one; bit 1: it takes 64-bit integers */
  void* gemm[2];   /* cblas_sgemm, cblas_dgemm */
} pk_call;

#define PK_ERR_MEMORY (-1)

PK_OUTLINE void pk_fail(pk_call* rt, int32_t id, int64_t a0, int64_t a1, int64_t a2, int64_t a3) {
  if (pk_atomic_cas(&rt->err_code, 0, 1)) {
    rt->err_id = id;
    rt->err_args[0] = a0; rt->err_args[1] = a1; rt->err_args[2] = a2; rt->err_args[3] = a3;
  }
}

static void* pk_aligned_alloc(size_t align, size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, align);
#else
  void* p = NULL;
  if (posix_memalign(&p, align, size) != 0) return NULL;
  return p;
#endif
}

static void pk_aligned_free(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

PK_OUTLINE pk_buf* pk_alloc(pk_call* rt, int64_t count, int64_t itemsize, int zero) {
  if (count < 0 || (itemsize > 0 && count > INT64_C(0x7fffffffffff) / itemsize)) {
    pk_fail(rt, PK_ERR_MEMORY, count, itemsize, 0, 0);
    return NULL;
  }
  int64_t nbytes = count * itemsize;
  size_t size = (size_t)(nbytes > 0 ? nbytes : 1);
  size_t align = 64;
  if (size >= PK_PAGE) { align = PK_PAGE; size = (size + PK_PAGE - 1) / PK_PAGE * PK_PAGE; }
  else { size = (size + 63) / 64 * 64; }
  pk_buf* b = (pk_buf*)malloc(sizeof(pk_buf));
  void* data = b ? pk_aligned_alloc(align, size) : NULL;
  if (!b || !data) {
    free(b);
    pk_fail(rt, PK_ERR_MEMORY, count, itemsize, 0, 0);
    return NULL;
  }
  if (zero && !rt->device) memset(data, 0, size);
  b->data = data;
  b->nbytes = (int64_t)size;
  b->refcount = 1;
  b->flags = zero ? (rt->device ? (PK_ZEROED | PK_LAZY_ZERO) : PK_ZEROED) : PK_FRESH;
  return b;
}

PK_OUTLINE void pk_ensure_zero(pk_buf* b) {
  if (b && (b->flags & PK_LAZY_ZERO)) {
    if (b->data && b->nbytes > 0) memset(b->data, 0, (size_t)b->nbytes);
    b->flags &= ~PK_LAZY_ZERO;
  }
}

PK_INLINE void pk_incref(pk_buf* b) {
  if (b) pk_atomic_add(&b->refcount, 1);
}

PK_OUTLINE void pk_drop(pk_buf* b) {
  if (b && pk_atomic_add(&b->refcount, -1) == 0 && !(b->flags & PK_EXTERNAL)) {
    pk_aligned_free(b->data);
    free(b);
  }
}

PK_OUTLINE void pk_release(pk_buf** pb) {
  pk_buf* b = *pb;
  if (b) { *pb = NULL; pk_drop(b); }
}

/* Called from Python when the last NumPy view of a result goes away. */
PK_EXPORT void pk_decref(void* b) { pk_drop((pk_buf*)b); }

PK_OUTLINE void pk_export(pk_call* rt, pk_buf* b, pk_buf** exts, int n_ext, int64_t* io) {
  int k;
  if (!b) b = pk_alloc(rt, 0, 1, 0);
  if (PK_UNLIKELY(b && (b->flags & PK_LAZY_ZERO))) pk_ensure_zero(b);
  for (k = 0; k < n_ext; k++) {
    if (b == exts[k]) { io[0] = 1; io[1] = k; io[2] = (int64_t)(intptr_t)b->data; return; }
  }
  io[0] = 0;
  io[1] = (int64_t)(intptr_t)b;
  io[2] = b ? (int64_t)(intptr_t)b->data : 0;
}

/* ---- scalar helpers ----------------------------------------------------- */
@@HELPERS@@

/* ---- chunked parallel loops --------------------------------------------- */

typedef void (*pk_body_fn)(void* ctx, int64_t chunk);

PK_OUTLINE int64_t pk_num_chunks(pk_call* rt, int64_t total, int reduce, int heavy, int on_device) {
  int64_t want, size;
  if (total <= 0) return 0;
  if (on_device && rt->device && total >= rt->gpu_min) {
    if (reduce && !heavy && total <= 512) return 1;
    want = rt->device_chunks > 0 ? rt->device_chunks : 4096;
  } else {
    int64_t grain = heavy ? 1 : (rt->grain > 0 ? rt->grain : 1);
    want = rt->nthreads <= 1 ? 1 : rt->nthreads * 4;
    if (want > total / grain) want = total / grain;
  }
  (void)reduce;
  if (want < 1) want = 1;
  if (want > total) want = total;
  size = (total + want - 1) / want;
  return (total + size - 1) / size;
}

/* C = op(A) op(B) through the BLAS found at run time: C is m x n and
   contiguous, A and B are row-major with leading dimensions lda and ldb,
   taken transposed when ta / tb are set. */
#define PK_GEMM_TYPES(NAME, INT, REAL) \
  typedef void (*NAME)(int, int, int, INT, INT, INT, REAL, const REAL*, INT, const REAL*, INT, REAL, REAL*, INT);
PK_GEMM_TYPES(pk_sgemm32, int, float)
PK_GEMM_TYPES(pk_sgemm64, int64_t, float)
PK_GEMM_TYPES(pk_dgemm32, int, double)
PK_GEMM_TYPES(pk_dgemm64, int64_t, double)

PK_OUTLINE int64_t pk_gemm(pk_call* rt, int64_t dbl, pk_buf* c, int64_t c_off, pk_buf* a, int64_t a_off,
                           int64_t lda, int64_t ta, pk_buf* b, int64_t b_off, int64_t ldb, int64_t tb,
                           int64_t m, int64_t n, int64_t k) {
  pk_ensure_zero(c); pk_ensure_zero(a); pk_ensure_zero(b);
  int64_t item = dbl ? 8 : 4;
  int64_t a_cols = ta ? m : k, b_cols = tb ? k : n;
  void* fn = rt->gemm[dbl ? 1 : 0];
  char* cp = (char*)c->data + c_off * item;
  if (m <= 0 || n <= 0) return 1;
  if (k <= 0) { memset(cp, 0, (size_t)(m * n * item)); return 1; }
  if (lda < a_cols) lda = a_cols;
  if (ldb < b_cols) ldb = b_cols;
  if (dbl) {
    const double* ap = (const double*)a->data + a_off; const double* bp = (const double*)b->data + b_off;
    if (rt->blas & 2) ((pk_dgemm64)fn)(101, ta ? 112 : 111, tb ? 112 : 111, m, n, k, 1.0, ap, lda, bp, ldb, 0.0, (double*)cp, n);
    else ((pk_dgemm32)fn)(101, ta ? 112 : 111, tb ? 112 : 111, (int)m, (int)n, (int)k, 1.0, ap, (int)lda, bp, (int)ldb, 0.0, (double*)cp, (int)n);
  } else {
    const float* ap = (const float*)a->data + a_off; const float* bp = (const float*)b->data + b_off;
    if (rt->blas & 2) ((pk_sgemm64)fn)(101, ta ? 112 : 111, tb ? 112 : 111, m, n, k, 1.0f, ap, lda, bp, ldb, 0.0f, (float*)cp, n);
    else ((pk_sgemm32)fn)(101, ta ? 112 : 111, tb ? 112 : 111, (int)m, (int)n, (int)k, 1.0f, ap, (int)lda, bp, (int)ldb, 0.0f, (float*)cp, (int)n);
  }
  return 1;
}

PK_INLINE int64_t pk_num_threads(pk_call* rt) { return rt->nthreads > 1 && rt->parfor ? rt->nthreads : 1; }

/* A loop goes to the device when the work in it is worth the launch: its
   size times the inner iterations each element is estimated to need. */
PK_INLINE int64_t pk_device_min(pk_call* rt, int64_t work) {
  /* ...and when the loop is wide enough: a device runs one slow thread per
     element, so a few dozen elements cannot use it however heavy each is. */
  int64_t width = rt->gpu_min < 1024 ? rt->gpu_min : 1024;
  int64_t need;
  if (work < 1) work = 1;
  need = (rt->gpu_min + work - 1) / work;
  return need > width ? need : width;
}

/* ---- hooks supplied by the Python side on every call --------------------- */

typedef void (*pk_parfor_fn)(pk_body_fn fn, void* ctx, int64_t nchunks, int64_t nthreads);
typedef int32_t (*pk_device_fn)(int32_t kernel, void* ctx, int64_t total, int64_t nchunks);

/* Arrays stay on the device between the loops of one call.  Before host code
   reads or writes array memory it brings the host copies up to date; before
   it frees a buffer it tells the device to drop its copy. */
PK_OUTLINE void pk_dev_flush(pk_call* rt) {
  rt->dev_live = 0;
  if (rt->device) ((pk_device_fn)rt->device)(-1, rt, 0, 0);
}
PK_OUTLINE void pk_dev_forget(pk_call* rt, pk_buf* b) {
  if (b && b->refcount == 1 && !(b->flags & PK_EXTERNAL) && rt->device) ((pk_device_fn)rt->device)(-2, b->data, 0, 0);
}
PK_OUTLINE void pk_dev_sync_buf(pk_call* rt, pk_buf* b, int write) {
  if (b && b->data && b->nbytes > 0 && rt->device)
    ((pk_device_fn)rt->device)(write ? -4 : -3, b->data, b->nbytes, 0);
}
#define PK_DEV_SYNC(rt) do { if (PK_UNLIKELY((rt)->dev_live)) pk_dev_flush(rt); } while (0)
#define PK_DEV_SYNC_READ(rt, b) do { if (PK_UNLIKELY((rt)->dev_live)) pk_dev_sync_buf(rt, b, 0); if (PK_UNLIKELY((b) && ((b)->flags & PK_LAZY_ZERO))) pk_ensure_zero(b); } while (0)
#define PK_DEV_SYNC_WRITE(rt, b) do { if (PK_UNLIKELY((rt)->dev_live)) pk_dev_sync_buf(rt, b, 1); if (PK_UNLIKELY((b) && ((b)->flags & PK_LAZY_ZERO))) pk_ensure_zero(b); } while (0)

PK_OUTLINE void pk_parallel_for(pk_call* rt, pk_body_fn fn, void* ctx, int64_t nchunks) {
  int64_t c;
  if (nchunks > 1 && rt->nthreads > 1 && rt->parfor) {
    ((pk_parfor_fn)rt->parfor)(fn, ctx, nchunks, rt->nthreads);
    return;
  }
  for (c = 0; c < nchunks; c++) fn(ctx, c);
}
"""

PRELUDE = _PRELUDE_TEMPLATE.replace(
  "@@HELPERS@@",
  "\n".join(
    [HELPERS_INT, HELPERS_BITOPS, HELPERS_FLOAT_DEF, HELPERS_F32, HELPERS_F64, HELPERS_RNG, HELPERS_RNG_F64, _RNG_KEY]
  ),
)


# ---------------------------------------------------------------------------
# The shared runtime library: one per process and threading strategy.  Modules
# receive a pointer to ``pk_rt_parallel_for`` on every call, so compiled code
# never links against a threading runtime itself.
# ---------------------------------------------------------------------------

RUNTIME = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(_WIN32)
  #define PK_EXPORT __declspec(dllexport)
#else
  #define PK_EXPORT __attribute__((visibility("default")))
#endif

typedef void (*pk_body_fn)(void* ctx, int64_t chunk);

static void pk_serial(pk_body_fn fn, void* ctx, int64_t nchunks) {
  int64_t c;
  for (c = 0; c < nchunks; c++) fn(ctx, c);
}

#if PK_PAR == 1  /* ---------------- OpenMP ---------------- */
#include <omp.h>
PK_EXPORT void pk_rt_parallel_for(pk_body_fn fn, void* ctx, int64_t nchunks, int64_t nthreads) {
  long long c;
  int nt = (int)(nthreads < nchunks ? nthreads : nchunks);
  if (nchunks <= 1 || nt <= 1 || omp_in_parallel()) { pk_serial(fn, ctx, nchunks); return; }
  #pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
  for (c = 0; c < nchunks; c++) fn(ctx, (int64_t)c);
}
PK_EXPORT const char* pk_rt_name(void) { return "openmp"; }

#elif PK_PAR == 2  /* ---------------- Grand Central Dispatch ---------------- */
#include <dispatch/dispatch.h>
typedef struct { pk_body_fn fn; void* ctx; } pk_gcd_job;
static void pk_gcd_tramp(void* job_, size_t i) {
  pk_gcd_job* job = (pk_gcd_job*)job_;
  job->fn(job->ctx, (int64_t)i);
}
PK_EXPORT void pk_rt_parallel_for(pk_body_fn fn, void* ctx, int64_t nchunks, int64_t nthreads) {
  pk_gcd_job job;
  if (nchunks <= 1 || nthreads <= 1) { pk_serial(fn, ctx, nchunks); return; }
  job.fn = fn; job.ctx = ctx;
  dispatch_apply_f((size_t)nchunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job, pk_gcd_tramp);
}
PK_EXPORT const char* pk_rt_name(void) { return "gcd"; }

#elif PK_PAR == 3  /* ---------------- persistent pthread pool ---------------- */
#include <pthread.h>
#define PK_MAX_THREADS 512

typedef struct { pk_body_fn fn; void* ctx; volatile int64_t next; int64_t nchunks; } pk_job;

static pthread_mutex_t pool_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t pool_owner = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pool_wake = PTHREAD_COND_INITIALIZER;
static pthread_cond_t pool_idle = PTHREAD_COND_INITIALIZER;
static pk_job* pool_job = NULL;
static int pool_tickets = 0, pool_running = 0, pool_threads = 0, pool_atfork = 0;
static __thread int pk_in_worker = 0;

static void pk_run_chunks(pk_job* job) {
  for (;;) {
    int64_t c = __sync_add_and_fetch(&job->next, 1) - 1;
    if (c >= job->nchunks) break;
    job->fn(job->ctx, c);
  }
}

static void* pk_worker(void* unused) {
  (void)unused;
  pk_in_worker = 1;
  pthread_mutex_lock(&pool_mu);
  for (;;) {
    pk_job* job;
    while (pool_tickets == 0) pthread_cond_wait(&pool_wake, &pool_mu);
    pool_tickets--;
    pool_running++;
    job = pool_job;
    pthread_mutex_unlock(&pool_mu);
    pk_run_chunks(job);
    pthread_mutex_lock(&pool_mu);
    if (--pool_running == 0) pthread_cond_broadcast(&pool_idle);
  }
  return NULL;
}

static void pk_after_fork_child(void) {
  /* worker threads do not survive fork(): start from a clean pool */
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
  memcpy(&pool_mu, &mu, sizeof(mu));
  memcpy(&pool_owner, &mu, sizeof(mu));
  memcpy(&pool_wake, &cv, sizeof(cv));
  memcpy(&pool_idle, &cv, sizeof(cv));
  pool_job = NULL;
  pool_tickets = pool_running = pool_threads = 0;
  pk_in_worker = 0;
}

PK_EXPORT void pk_rt_parallel_for(pk_body_fn fn, void* ctx, int64_t nchunks, int64_t nthreads) {
  pk_job job;
  int helpers;
  if (nchunks <= 1 || nthreads <= 1 || pk_in_worker || pthread_mutex_trylock(&pool_owner) != 0) {
    pk_serial(fn, ctx, nchunks);
    return;
  }
  job.fn = fn; job.ctx = ctx; job.next = 0; job.nchunks = nchunks;
  helpers = (int)((nthreads < nchunks ? nthreads : nchunks) - 1);
  if (helpers > PK_MAX_THREADS) helpers = PK_MAX_THREADS;
  pthread_mutex_lock(&pool_mu);
  if (!pool_atfork) { pool_atfork = 1; pthread_atfork(NULL, NULL, pk_after_fork_child); }
  while (pool_threads < helpers) {
    pthread_t t;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&t, &attr, pk_worker, NULL) != 0) { pthread_attr_destroy(&attr); break; }
    pthread_attr_destroy(&attr);
    pool_threads++;
  }
  if (helpers > pool_threads) helpers = pool_threads;
  pool_job = &job;
  pool_tickets = helpers;
  if (helpers > 0) pthread_cond_broadcast(&pool_wake);
  pthread_mutex_unlock(&pool_mu);

  pk_run_chunks(&job);

  pthread_mutex_lock(&pool_mu);
  pool_tickets = 0;  /* workers that have not started yet must not touch this job */
  while (pool_running > 0) pthread_cond_wait(&pool_idle, &pool_mu);
  pool_job = NULL;
  pthread_mutex_unlock(&pool_mu);
  pthread_mutex_unlock(&pool_owner);
}
PK_EXPORT const char* pk_rt_name(void) { return "threads"; }

#else  /* ---------------- serial ---------------- */
PK_EXPORT void pk_rt_parallel_for(pk_body_fn fn, void* ctx, int64_t nchunks, int64_t nthreads) {
  (void)nthreads;
  pk_serial(fn, ctx, nchunks);
}
PK_EXPORT const char* pk_rt_name(void) { return "serial"; }
#endif

/* self-test used when probing which strategies work on this machine */
static volatile int64_t pk_probe_hits = 0;
static void pk_probe_body(void* ctx, int64_t chunk) {
  (void)ctx; (void)chunk;
#if defined(_MSC_VER) && !defined(__clang__)
  _InterlockedExchangeAdd64((volatile __int64*)&pk_probe_hits, 1);
#else
  __sync_add_and_fetch(&pk_probe_hits, 1);
#endif
}
PK_EXPORT int32_t pk_rt_probe(void) {
  pk_probe_hits = 0;
  pk_rt_parallel_for(pk_probe_body, NULL, 64, 4);
  return pk_probe_hits == 64 ? 0 : 1;
}
"""

PAR_MODES = {"serial": 0, "openmp": 1, "gcd": 2, "threads": 3}
