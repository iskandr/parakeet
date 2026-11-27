"""Finite shape-bucket and power-of-2 divisibility abstract domain.

Models non-negative integer dimensions with ``DimInfo(lo, hi, align)``:

* exact small constants ``0, 1, 2, 3, 4, 8, 16, 32, 64``,
* power-of-2 intervals ``[5, 7], [9, 15], [17, 31], [33, 63], [65, 127], [128, +inf)``, and
* largest known power-of-2 divisor ``align`` in ``{0, 1, 2, 4, 8, 16, 32, 64, 128}``
  (where ``0`` denotes exact zero).

All large aligned dimensions (``128, 256, 512, 1024, 4096, ...``) collapse to a
single bucket ``DimInfo(128, None, 128)``, preventing recompilation churn while
still letting middle-end passes and GPU codegen prove tile-size selection,
divisibility (interior ``f^k`` vs straggler ``f`` tiles), and 16-byte vector
alignment soundly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_ALIGN = 128


def pow2_align(n: int) -> int:
  """Largest power of 2 dividing ``n`` (capped at :data:`MAX_ALIGN`), or ``0`` if ``n == 0``."""
  if n == 0:
    return 0
  v = abs(n)
  low = v & -v
  return MAX_ALIGN if low > MAX_ALIGN else low


def _combine_align_gcd(a: int, b: int) -> int:
  if a == 0:
    return b
  if b == 0:
    return a
  return min(a, b)


def _combine_align_lcm(a: int, b: int) -> int:
  if a == 0 or b == 0:
    return 0
  return max(a, b)


@dataclass(frozen=True, slots=True)
class DimInfo:
  """Sound abstract summary of a non-negative integer expression."""

  lo: int
  hi: int | None
  align: int

  @property
  def exact(self) -> int | None:
    return self.lo if self.hi == self.lo else None

  @property
  def is_top(self) -> bool:
    return self.lo == 0 and self.hi is None and self.align == 1

  def divisible_by(self, k: int) -> bool:
    """True if every concrete value in this abstract domain is divisible by ``k``."""
    if k <= 0:
      return False
    if k == 1:
      return True
    if self.align == 0:
      return True
    if self.hi == self.lo:
      return (self.lo % k) == 0
    return (self.align % k) == 0

  def matches(self, value: int) -> bool:
    """True if concrete runtime dimension ``value`` satisfies all facts in ``self``."""
    if value < self.lo:
      return False
    if self.hi is not None and value > self.hi:
      return False
    if self.align == 0:
      return value == 0
    return (value % self.align) == 0

  def meet(self, other: DimInfo) -> DimInfo:
    """Sound intersection when two expressions are proven equal."""
    lo = max(self.lo, other.lo)
    if self.hi is None:
      hi = other.hi
    elif other.hi is None:
      hi = self.hi
    else:
      hi = min(self.hi, other.hi)
    align = _combine_align_lcm(self.align, other.align)
    if hi is not None and hi < lo:
      # Contradictory/unreachable path: fall back conservatively to self.
      return self
    if hi == lo:
      align = pow2_align(lo)
    return DimInfo(lo, hi, align)

  def join(self, other: DimInfo) -> DimInfo:
    """Sound union across branches."""
    lo = min(self.lo, other.lo)
    hi = None if (self.hi is None or other.hi is None) else max(self.hi, other.hi)
    align = _combine_align_gcd(self.align, other.align)
    return DimInfo(lo, hi, align)

  def add(self, other: DimInfo) -> DimInfo:
    lo = self.lo + other.lo
    hi = None if (self.hi is None or other.hi is None) else self.hi + other.hi
    align = pow2_align(lo) if hi == lo else _combine_align_gcd(self.align, other.align)
    return DimInfo(lo, hi, align)

  def sub(self, other: DimInfo) -> DimInfo:
    lo = 0 if other.hi is None else max(0, self.lo - other.hi)
    hi = None if self.hi is None else max(0, self.hi - other.lo)
    align = pow2_align(lo) if hi == lo else _combine_align_gcd(self.align, other.align)
    return DimInfo(lo, hi, align)

  def mul(self, other: DimInfo) -> DimInfo:
    lo = self.lo * other.lo
    hi = None if (self.hi is None or other.hi is None) else self.hi * other.hi
    if self.align == 0 or other.align == 0:
      align = 0
    else:
      align = min(MAX_ALIGN, self.align * other.align)
    if hi == lo:
      align = pow2_align(lo)
    return DimInfo(lo, hi, align)

  def floordiv_const(self, k: int) -> DimInfo:
    if k <= 0:
      return TOP
    lo = self.lo // k
    hi = None if self.hi is None else self.hi // k
    if hi == lo:
      align = pow2_align(lo)
    elif (k & (k - 1)) == 0 and self.align >= k:
      align = max(1, self.align // k)
    else:
      align = 1
    return DimInfo(lo, hi, align)

  def imin(self, other: DimInfo) -> DimInfo:
    lo = min(self.lo, other.lo)
    if self.hi is None:
      hi = other.hi
    elif other.hi is None:
      hi = self.hi
    else:
      hi = min(self.hi, other.hi)
    align = pow2_align(lo) if hi == lo else _combine_align_gcd(self.align, other.align)
    return DimInfo(lo, hi, align)

  def imax(self, other: DimInfo) -> DimInfo:
    lo = max(self.lo, other.lo)
    hi = None if (self.hi is None or other.hi is None) else max(self.hi, other.hi)
    align = pow2_align(lo) if hi == lo else _combine_align_gcd(self.align, other.align)
    return DimInfo(lo, hi, align)

  @staticmethod
  def from_const(c: int) -> DimInfo:
    if c < 0:
      return TOP
    return DimInfo(c, c, pow2_align(c))


TOP = DimInfo(0, None, 1)


def classify_dim(n: int) -> DimInfo:
  """Map a concrete array dimension ``n`` to its finite abstract bucket."""
  if n < 0:
    return TOP
  if n <= 4 or n in (8, 16, 32, 64):
    return DimInfo(n, n, pow2_align(n))
  a = pow2_align(n)
  if n < 8:
    return DimInfo(5, 7, a)
  if n < 16:
    return DimInfo(9, 15, a)
  if n < 32:
    return DimInfo(17, 31, a)
  if n < 64:
    return DimInfo(33, 63, a)
  if n < 128:
    return DimInfo(65, 127, a)
  return DimInfo(128, None, a)


def describe_dim(info: DimInfo) -> str:
  """Compact human-readable description of a :class:`DimInfo`."""
  if info.hi == info.lo:
    return str(info.lo)
  rng = f">={info.lo}" if info.hi is None else f"{info.lo}..{info.hi}"
  return f"{rng}" if info.align <= 1 else f"{rng}%{info.align}==0"


__all__ = [
  "MAX_ALIGN",
  "TOP",
  "DimInfo",
  "classify_dim",
  "describe_dim",
  "math",
  "pow2_align",
]
