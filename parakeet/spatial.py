"""CSR block-sparse attention masks (:class:`BlockMask`) for Law 5 sparse tiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _csr_to_csc(row_ptr: np.ndarray, col_idx: np.ndarray, num_k_blocks: int) -> tuple[np.ndarray, np.ndarray]:
  """Transpose a CSR block graph into CSC ``(col_ptr, row_idx)`` for atomic-free backward ``dK``/``dV``."""
  n_q = len(row_ptr) - 1
  counts = np.zeros(num_k_blocks, dtype=np.int64)
  if len(col_idx) > 0:
    np.add.at(counts, col_idx, 1)
  col_ptr = np.zeros(num_k_blocks + 1, dtype=np.int64)
  np.cumsum(counts, out=col_ptr[1:])
  row_idx = np.empty(len(col_idx), dtype=np.int64)
  cursor = col_ptr[:-1].copy()
  for qb in range(n_q):
    s, e = int(row_ptr[qb]), int(row_ptr[qb + 1])
    for p in range(s, e):
      kb = int(col_idx[p])
      row_idx[cursor[kb]] = qb
      cursor[kb] += 1
  return col_ptr, row_idx


@dataclass(frozen=True)
class BlockMask:
  """Immutable CSR + CSC block-sparse mask for Law 5 block-sparse FlashAttention.

  ``row_ptr`` (length ``num_q_blocks + 1``) and ``col_idx`` list the active key
  block indices for each query block of size ``block_q x block_k``.
  ``col_ptr`` and ``row_idx`` store the transposed CSC adjacency for atomic-free
  Law 2 ``dK``/``dV`` backward passes.
  """

  row_ptr: Any
  col_idx: Any
  col_ptr: Any
  row_idx: Any
  block_q: int
  block_k: int

  @classmethod
  def from_csr(cls, row_ptr, col_idx, num_k_blocks: int, block_q: int = 32, block_k: int = 32) -> BlockMask:
    rp = np.ascontiguousarray(row_ptr, dtype=np.int64)
    ci = np.ascontiguousarray(col_idx, dtype=np.int64)
    cp, ri = _csr_to_csc(rp, ci, int(num_k_blocks))
    return cls(rp, ci, cp, ri, int(block_q), int(block_k))

  @classmethod
  def from_dense_mask(cls, mask: np.ndarray, block_q: int = 32, block_k: int = 32) -> BlockMask:
    """Compress a 2D boolean mask ``(M, N)`` into a CSR :class:`BlockMask`."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
      raise ValueError(f"from_dense_mask expects a 2D boolean mask, got shape {m.shape}")
    M, N = m.shape
    bq, bk = int(block_q), int(block_k)
    nq = (M + bq - 1) // bq
    nk = (N + bk - 1) // bk
    row_ptr = np.zeros(nq + 1, dtype=np.int64)
    cols: list[int] = []
    for qb in range(nq):
      q0, q1 = qb * bq, min((qb + 1) * bq, M)
      for kb in range(nk):
        k0, k1 = kb * bk, min((kb + 1) * bk, N)
        if np.any(m[q0:q1, k0:k1]):
          cols.append(kb)
      row_ptr[qb + 1] = len(cols)
    return cls.from_csr(row_ptr, np.asarray(cols, dtype=np.int64), nk, bq, bk)

  @classmethod
  def banded(
    cls, m: int, n: int, band: int, block_q: int = 32, block_k: int = 32, *, causal: bool = False
  ) -> BlockMask:
    """Build a banded (optionally causal) CSR :class:`BlockMask` without allocating an ``M x N`` mask."""
    bq, bk = int(block_q), int(block_k)
    nq = (m + bq - 1) // bq
    nk = (n + bk - 1) // bk
    row_ptr = np.zeros(nq + 1, dtype=np.int64)
    cols: list[int] = []
    for qb in range(nq):
      q_lo = qb * bq
      q_hi = min((qb + 1) * bq - 1, m - 1)
      k_min = max(0, q_lo - band)
      k_max = min(n - 1, q_hi if causal else q_hi + band)
      if k_min <= k_max:
        kb_lo = k_min // bk
        kb_hi = k_max // bk
        cols.extend(range(kb_lo, kb_hi + 1))
      row_ptr[qb + 1] = len(cols)
    return cls.from_csr(row_ptr, np.asarray(cols, dtype=np.int64), nk, bq, bk)

  @classmethod
  def from_3d_cutoff(
    cls, coords_q: np.ndarray, coords_k: np.ndarray, cutoff: float, block_q: int = 32, block_k: int = 32
  ) -> BlockMask:
    """Build an exact 3D AABB-pruned CSR :class:`BlockMask` for pairs within ``cutoff``."""
    cq = np.asarray(coords_q, dtype=np.float64)
    ck = np.asarray(coords_k, dtype=np.float64)
    M, N = cq.shape[0], ck.shape[0]
    bq, bk = int(block_q), int(block_k)
    nq = (M + bq - 1) // bq
    nk = (N + bk - 1) // bk
    r2 = float(cutoff) * float(cutoff)

    def _aabbs(c, bsz, nb):
      mn = np.empty((nb, 3), dtype=np.float64)
      mx = np.empty((nb, 3), dtype=np.float64)
      has_nan = np.zeros(nb, dtype=bool)
      for t in range(nb):
        sl = c[t * bsz : min((t + 1) * bsz, c.shape[0])]
        if np.isnan(sl).any():
          has_nan[t] = True
          mn[t] = -np.inf
          mx[t] = np.inf
        else:
          mn[t] = sl.min(axis=0)
          mx[t] = sl.max(axis=0)
      return mn, mx, has_nan

    qmin, qmax, qnan = _aabbs(cq, bq, nq)
    kmin, kmax, knan = _aabbs(ck, bk, nk)
    row_ptr = np.zeros(nq + 1, dtype=np.int64)
    cols: list[int] = []
    for qb in range(nq):
      if qnan[qb]:
        cols.extend(range(nk))
      else:
        dx = np.maximum(0.0, np.maximum(qmin[qb, 0] - kmax[:, 0], kmin[:, 0] - qmax[qb, 0]))
        dy = np.maximum(0.0, np.maximum(qmin[qb, 1] - kmax[:, 1], kmin[:, 1] - qmax[qb, 1]))
        dz = np.maximum(0.0, np.maximum(qmin[qb, 2] - kmax[:, 2], kmin[:, 2] - qmax[qb, 2]))
        active = knan | (dx * dx + dy * dy + dz * dz <= r2)
        cols.extend(np.flatnonzero(active).tolist())
      row_ptr[qb + 1] = len(cols)
    return cls.from_csr(row_ptr, np.asarray(cols, dtype=np.int64), nk, bq, bk)

  @property
  def num_q_blocks(self) -> int:
    rp = self.row_ptr._host if getattr(type(self.row_ptr), "__parakeet_device_array__", False) else self.row_ptr
    return max(0, int(rp.shape[0]) - 1)

  @property
  def num_k_blocks(self) -> int:
    cp = self.col_ptr._host if getattr(type(self.col_ptr), "__parakeet_device_array__", False) else self.col_ptr
    return max(0, int(cp.shape[0]) - 1)

  @property
  def nnz_blocks(self) -> int:
    ci = self.col_idx._host if getattr(type(self.col_idx), "__parakeet_device_array__", False) else self.col_idx
    return int(ci.shape[0])
