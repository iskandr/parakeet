"""First-class CSR ragged arrays (``values`` + ``offsets``) for variable-length data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .syntax.records import record_type


def _leading_len(v: Any) -> int:
  if isinstance(v, RaggedArray):
    return len(v)
  if isinstance(v, np.ndarray):
    return int(v.shape[0])
  if getattr(type(v), "__parakeet_device_array__", False):
    return int(v._host.shape[0])
  if rec := record_type(v):
    vals = rec.values(v)
    return _leading_len(vals[0]) if vals else 0
  return len(v)


def _slice_leading(v: Any, start: int, stop: int):
  if isinstance(v, RaggedArray):
    sub_off = np.asarray(v.offsets[start : stop + 1], dtype=np.int64)
    base = int(sub_off[0])
    end = int(sub_off[-1])
    return RaggedArray(_slice_leading(v.values, base, end), sub_off - base)
  if isinstance(v, np.ndarray):
    return v[start:stop]
  if getattr(type(v), "__parakeet_device_array__", False):
    return v.numpy()[start:stop]
  if rec := record_type(v):
    return rec.build(_slice_leading(x, start, stop) for x in rec.values(v))
  return v[start:stop]


@dataclass(frozen=True)
class RaggedArray:
  """Immutable variable-length sequence container in CSR layout.

  ``values`` holds all elements concatenated along axis 0 (an ``np.ndarray``,
  :class:`~parakeet.DeviceArray`, nested :class:`RaggedArray`, or a frozen
  record/``NamedTuple`` of arrays sharing the same leading length).
  ``offsets`` is a 1D integer array of length ``num_segments + 1`` with
  ``offsets[0] == 0`` and ``offsets[i] <= offsets[i + 1]``.
  """

  values: Any
  offsets: Any

  @classmethod
  def from_sequences(cls, sequences, dtype=None) -> RaggedArray:
    """Pack a Python iterable of variable-length sequences into a :class:`RaggedArray`."""
    seqs = list(sequences)
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    if not seqs:
      dt = np.float32 if dtype is None else np.dtype(dtype)
      return cls(np.empty((0,), dtype=dt), offsets)
    arrays = [np.asarray(s, dtype=dtype) for s in seqs]
    non_empty = [a for a in arrays if a.shape[0] > 0]
    if non_empty:
      dt = np.result_type(*(a.dtype for a in non_empty)) if dtype is None else np.dtype(dtype)
      trailing = non_empty[0].shape[1:]
      norm = [a.astype(dt, copy=False) if a.shape[0] > 0 else np.empty((0, *trailing), dtype=dt) for a in arrays]
      values = np.concatenate(norm, axis=0)
    else:
      dt = arrays[0].dtype if dtype is None else np.dtype(dtype)
      trailing = arrays[0].shape[1:] if arrays[0].ndim > 1 else ()
      values = np.empty((0, *trailing), dtype=dt)
    for i, a in enumerate(arrays):
      offsets[i + 1] = offsets[i] + a.shape[0]
    return cls(values, offsets)

  @classmethod
  def from_lengths(cls, values, lengths) -> RaggedArray:
    """Build a :class:`RaggedArray` from flat ``values`` and per-segment ``lengths``."""
    lens = np.asarray(lengths, dtype=np.int64)
    if lens.ndim != 1:
      raise ValueError(f"lengths must be 1D, got shape {lens.shape}")
    if np.any(lens < 0):
      raise ValueError("lengths must be non-negative")
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    if int(offsets[-1]) > _leading_len(values):
      raise ValueError(f"sum of lengths ({int(offsets[-1])}) exceeds len(values) ({_leading_len(values)})")
    return cls(values, offsets)

  @property
  def num_segments(self) -> int:
    off = self.offsets._host if getattr(type(self.offsets), "__parakeet_device_array__", False) else self.offsets
    return max(0, int(off.shape[0]) - 1)

  @property
  def lengths(self) -> np.ndarray:
    off = (
      self.offsets.numpy()
      if getattr(type(self.offsets), "__parakeet_device_array__", False)
      else np.asarray(self.offsets)
    )
    return off[1:] - off[:-1]

  def __len__(self) -> int:
    return self.num_segments

  def __getitem__(self, idx):
    off = (
      self.offsets.numpy()
      if getattr(type(self.offsets), "__parakeet_device_array__", False)
      else np.asarray(self.offsets)
    )
    n = len(off) - 1
    if isinstance(idx, slice):
      start, stop, step = idx.indices(n)
      if step != 1:
        raise ValueError("RaggedArray slicing only supports step == 1")
      sub_off = np.asarray(off[start : stop + 1], dtype=np.int64)
      base = int(sub_off[0])
      end = int(sub_off[-1])
      return RaggedArray(_slice_leading(self.values, base, end), sub_off - base)
    i = int(idx)
    if i < 0:
      i += n
    if not 0 <= i < n:
      raise IndexError(f"RaggedArray index {idx} out of range for {n} segments")
    return _slice_leading(self.values, int(off[i]), int(off[i + 1]))

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def to_padded(self, fill=0, *, return_mask: bool = False):
    """Materialize a dense padded array of shape ``(num_segments, max_len, *trailing)``."""
    vals = (
      self.values.numpy() if getattr(type(self.values), "__parakeet_device_array__", False) else np.asarray(self.values)
    )
    off = (
      self.offsets.numpy()
      if getattr(type(self.offsets), "__parakeet_device_array__", False)
      else np.asarray(self.offsets)
    )
    n = len(off) - 1
    lens = off[1:] - off[:-1] if n > 0 else np.zeros(0, dtype=np.int64)
    max_len = int(lens.max()) if n > 0 else 0
    out = np.full((n, max_len, *vals.shape[1:]), fill, dtype=vals.dtype)
    mask = np.zeros((n, max_len), dtype=bool)
    for i in range(n):
      s, e = int(off[i]), int(off[i + 1])
      L = e - s
      if L > 0:
        out[i, :L] = vals[s:e]
        mask[i, :L] = True
    return (out, mask) if return_mask else out
