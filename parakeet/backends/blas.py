"""Find a BLAS matrix product to call.

Parakeet's own matrix product is generated code (tiled, see
:mod:`parakeet.transforms.tiling`).  A hand-tuned BLAS is several times
faster, and one is nearly always already loaded: the one NumPy uses.  This
module looks for ``cblas_?gemm`` there, or in Apple's Accelerate, and hands
its address to compiled code; nothing is linked at build time.
"""

from __future__ import annotations

import ctypes
import functools
import glob
import os
import sys

# (symbol suffix, uses 64-bit integers)
_SPELLINGS = (
  ("scipy_cblas_{}gemm64_", True),
  ("cblas_{}gemm64_", True),
  ("scipy_cblas_{}gemm", False),
  ("cblas_{}gemm", False),
)


def _libraries() -> list[str]:
  import numpy

  root = os.path.dirname(numpy.__file__)
  found: list[str] = []
  for pattern in (".dylibs/*openblas*", "../numpy.libs/*openblas*", ".libs/*openblas*", "../numpy.libs/*blas*"):
    found += sorted(glob.glob(os.path.join(root, pattern)))
  if sys.platform == "darwin":
    found.append("/System/Library/Frameworks/Accelerate.framework/Accelerate")
  return found


@functools.lru_cache(maxsize=1)
def find() -> tuple[int, int, bool, str] | None:
  """``(address of sgemm, address of dgemm, 64-bit integers?, library)``, or None."""
  if os.environ.get("PARAKEET_BLAS", "1").strip().lower() in ("0", "false", "no", "off"):
    return None
  if sys.platform == "win32" and os.environ.get("PARAKEET_BLAS", "") != "force":
    # A product with constant small shapes came back as zeros from NumPy's
    # OpenBLAS on Windows (and only there); until that is understood, Windows
    # uses Parakeet's own loops.  PARAKEET_BLAS=force turns the BLAS on anyway.
    return None
  for path in _libraries():
    try:
      lib = ctypes.CDLL(path)
    except OSError:
      continue
    for spelling, wide in _SPELLINGS:
      try:
        s = ctypes.cast(getattr(lib, spelling.format("s")), ctypes.c_void_p).value
        d = ctypes.cast(getattr(lib, spelling.format("d")), ctypes.c_void_p).value
      except AttributeError:
        continue
      if s and d:
        _keep.append(lib)
        return s, d, wide, path
  return None


_keep: list = []


def available() -> bool:
  return find() is not None


def header() -> list[int]:
  """The three call-header slots describing the BLAS: flags, sgemm, dgemm."""
  hit = find()
  if hit is None:
    return [0, 0, 0]
  s, d, wide, _ = hit
  return [1 | (2 if wide else 0), s, d]
