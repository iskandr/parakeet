"""Queried hardware locality and compute descriptors for CPU, CUDA, and Metal targets.

A :class:`TargetProfile` is frozen and hashable so it participates directly in
specialization keys.  Middle-end passes (e.g. :mod:`parakeet.transforms.tiling`)
and GPU kernel emitters (:mod:`parakeet.backends.tiled_gpu`) query it to choose
cache/SRAM tile sizes, reduction interchange, reduction chunk counts, and
hardware matrix-unit lowering (Apple Metal ``simdgroup_matrix`` and NVIDIA CUDA
``nvcuda::wmma`` Tensor Cores).
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MatrixUnit:
  """Hardware matrix-multiply-accumulate unit on an accelerator."""

  kind: str  # "simdgroup_matrix" | "wmma" | "mxu"
  m: int  # M dimension of one hardware matrix tile (e.g. 8 on Metal, 16 on CUDA wmma)
  n: int  # N dimension of one hardware matrix tile
  k: int  # K dimension of one hardware matrix tile
  input_dtypes: tuple[str, ...] = ("float32",)
  acc_dtypes: tuple[str, ...] = ("float32",)


@dataclass(frozen=True)
class TargetProfile:
  """Queried hardware locality and compute profile for a compilation target."""

  kind: str = "cpu"  # "cpu" | "cuda" | "metal"
  name: str = "cpu"
  num_compute_units: int = 1  # CPU cores / CUDA SMs / GPU cores (0 if unknown)
  vector_width: int = 4  # float32 SIMD / warp / simdgroup lanes
  simd_regs: int = 16  # architectural SIMD register count on CPU
  l1_bytes: int = 32 * 1024  # per-core L1D (CPU) or L1/texture cache
  l2_bytes: int = 512 * 1024  # per-core L2 (CPU) or device L2 (GPU)
  tile_bytes: int = 256 * 1024  # cache tiling budget (from config.tile_bytes)
  shared_mem_bytes: int = 0  # CUDA __shared__ per block / Metal maxThreadgroupMemoryLength
  max_threads_per_block: int = 1  # max threads per CUDA block / Metal threadgroup
  unified_memory: bool = True  # True on CPU & Apple Silicon; False on discrete PCIe CUDA
  supports_float64: bool = True  # False on Apple Metal GPUs
  compute_capability: tuple[int, int] = (0, 0)  # CUDA (major, minor)
  matrix_unit: MatrixUnit | None = None
  prefer_interchange: bool = True  # CPU benefits from L1 column-strip interchange; GPUs coalesce across warps
  tensor_cores: bool = False  # enable TF32 WMMA on CUDA sm_80+ (Metal FP32 simdgroup_matrix is exact)

  @property
  def is_gpu(self) -> bool:
    return self.kind in ("cuda", "metal")

  def interchange_tile(self, elt_bytes: int = 8) -> int:
    """L1-sized contiguous column strip width for CPU reduction interchange."""
    # Keep accumulator strip comfortably inside 1/16 of L1D (power-of-two aligned, [128, 512]).
    target = max(128, min(512, (self.l1_bytes // max(elt_bytes * 16, 1)) // 64 * 64))
    return target if target >= 128 else 256

  def shared_budget(self) -> int:
    """Per-block SRAM budget for high occupancy on GPU targets."""
    if self.shared_mem_bytes <= 0:
      return 24 * 1024
    if self.kind == "cuda":
      # Use up to 48 KiB static __shared__ (half of 96-100 KiB opt-in or full 48 KiB default)
      return min(self.shared_mem_bytes, 48 * 1024)
    # Metal: 32 KiB max threadgroup memory; keep <= 24 KiB for 2+ threadgroups/core occupancy
    return min(max(self.shared_mem_bytes * 3 // 4, 16 * 1024), 24 * 1024)

  def reduce_chunks(self, configured: int) -> int:
    """Choose GPU reduction chunk count from hardware compute unit count when default."""
    if not self.is_gpu or configured != 8192 or self.num_compute_units <= 0:
      return configured
    # ~16 threadblocks per SM/CU gives high occupancy without excessive partial-sum overhead
    return max(256, min(8192, self.num_compute_units * 32))

  def with_options(self, tile_bytes: int, tensor_cores: bool = False) -> TargetProfile:
    return replace(self, tile_bytes=tile_bytes, tensor_cores=tensor_cores)

  @classmethod
  def for_device(
    cls,
    device_kind: str | None,
    tile_bytes: int,
    device=None,
    tensor_cores: bool = False,
  ) -> TargetProfile:
    if device is not None and getattr(device, "profile", None) is not None:
      return device.profile.with_options(tile_bytes=tile_bytes, tensor_cores=tensor_cores)
    if device_kind == "cuda":
      return cls(
        kind="cuda",
        name="cuda",
        num_compute_units=0,
        vector_width=32,
        shared_mem_bytes=48 * 1024,
        max_threads_per_block=1024,
        unified_memory=False,
        supports_float64=True,
        tile_bytes=tile_bytes,
        prefer_interchange=False,
        tensor_cores=tensor_cores,
      )
    if device_kind == "metal":
      return cls(
        kind="metal",
        name="metal",
        num_compute_units=0,
        vector_width=32,
        shared_mem_bytes=32 * 1024,
        max_threads_per_block=1024,
        unified_memory=True,
        supports_float64=False,
        tile_bytes=tile_bytes,
        prefer_interchange=False,
        tensor_cores=tensor_cores,
      )
    return query_cpu_profile(tile_bytes)


def _sysctl_int(keys: tuple[str, ...]) -> int:
  for key in keys:
    try:
      out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=3).stdout.strip()
      if out.isdigit() and int(out) > 0:
        return int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
      continue
  return 0


def _linux_cache_bytes(index: int) -> int:
  try:
    with open(f"/sys/devices/system/cpu/cpu0/cache/index{index}/size") as f:
      text = f.read().strip().upper()
    scale = {"K": 1024, "M": 1024 * 1024}.get(text[-1:], 1)
    return int(text.rstrip("KM")) * scale
  except (OSError, ValueError):
    return 0


def _detect_cpu_simd() -> tuple[int, int]:
  """Return ``(float32_vector_width, simd_register_count)`` for the host CPU."""
  arch = platform.machine().lower()
  if arch in ("arm64", "aarch64"):
    return 4, 32  # ARM64 NEON: 128-bit (4 x f32), 32 architectural vector registers
  if arch in ("x86_64", "amd64"):
    flags = ""
    if sys.platform == "darwin":
      if _sysctl_int(("hw.optional.avx512f",)) > 0:
        return 16, 32
      if _sysctl_int(("hw.optional.avx2_0",)) > 0:
        return 8, 16
    elif sys.platform.startswith("linux"):
      try:
        with open("/proc/cpuinfo") as f:
          for line in f:
            if line.startswith("flags"):
              flags = line
              break
      except OSError:
        pass
      if "avx512f" in flags:
        return 16, 32
      if "avx2" in flags:
        return 8, 16
    return 4, 16  # x86_64 baseline SSE2
  return 4, 16


@functools.lru_cache(maxsize=8)
def query_cpu_profile(tile_bytes: int = 256 * 1024) -> TargetProfile:
  """Query host CPU core count, L1D/L2 cache sizes, and SIMD width/register count."""
  cores = os.cpu_count() or 1
  vec_w, simd_regs = _detect_cpu_simd()
  l1 = 0
  l2 = 0
  name = platform.processor() or platform.machine() or "cpu"
  if sys.platform == "darwin":
    l1 = _sysctl_int(("hw.perflevel0.l1dcachesize", "hw.l1dcachesize"))
    l2 = _sysctl_int(("hw.perflevel0.l2cachesize", "hw.l2cachesize"))
    try:
      brand = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3
      ).stdout.strip()
      if brand:
        name = brand
    except (OSError, subprocess.SubprocessError):
      pass
  elif sys.platform.startswith("linux"):
    l1 = _linux_cache_bytes(0)
    l2 = _linux_cache_bytes(2)
  return TargetProfile(
    kind="cpu",
    name=name,
    num_compute_units=cores,
    vector_width=vec_w,
    simd_regs=simd_regs,
    l1_bytes=l1 if l1 > 0 else 32 * 1024,
    l2_bytes=l2 if l2 > 0 else max(tile_bytes * 2, 512 * 1024),
    tile_bytes=tile_bytes,
    shared_mem_bytes=0,
    max_threads_per_block=1,
    unified_memory=True,
    supports_float64=True,
    prefer_interchange=True,
  )


__all__ = [
  "MatrixUnit",
  "TargetProfile",
  "query_cpu_profile",
]
