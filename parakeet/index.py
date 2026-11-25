"""Compiled read-only lookup indices: :class:`SortedIndex` and :class:`FrozenHashTable`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _mix64_u64(z: np.ndarray) -> np.ndarray:
  z = np.asarray(z, dtype=np.uint64)
  with np.errstate(over="ignore"):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _mix64_scalar(k: int) -> int:
  mask = (1 << 64) - 1
  z = int(k) & mask
  z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
  z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
  return (z ^ (z >> 31)) & mask


@dataclass(frozen=True)
class SortedIndex:
  """Immutable sorted key->value index supporting O(log N) ``get`` and ``range`` lookups.

  Flattens cleanly to plain array leaves at the ``@jit`` boundary and runs
  natively on CPU, CUDA, and Metal GPUs.
  """

  keys: Any
  values: Any

  @classmethod
  def from_unsorted(cls, keys, values) -> SortedIndex:
    k = np.ascontiguousarray(keys)
    v = np.ascontiguousarray(values)
    if k.ndim != 1:
      raise ValueError(f"SortedIndex keys must be 1D, got shape {k.shape}")
    if len(k) != len(v):
      raise ValueError(f"SortedIndex keys ({len(k)}) and values ({len(v)}) must have equal length")
    order = np.argsort(k, kind="stable")
    return cls(np.ascontiguousarray(k[order]), np.ascontiguousarray(v[order]))

  def __len__(self) -> int:
    k = self.keys._host if getattr(type(self.keys), "__parakeet_device_array__", False) else self.keys
    return int(k.shape[0])

  def __contains__(self, key) -> bool:
    k = self.keys.numpy() if getattr(type(self.keys), "__parakeet_device_array__", False) else np.asarray(self.keys)
    pos = int(np.searchsorted(k, key, side="left"))
    return pos < len(k) and bool(k[pos] == key)

  def get(self, key, default=0):
    k = self.keys.numpy() if getattr(type(self.keys), "__parakeet_device_array__", False) else np.asarray(self.keys)
    v = (
      self.values.numpy() if getattr(type(self.values), "__parakeet_device_array__", False) else np.asarray(self.values)
    )
    pos = int(np.searchsorted(k, key, side="left"))
    if pos < len(k) and bool(k[pos] == key):
      return v[pos]
    return default

  def range(self, lo, hi) -> tuple[int, int]:
    """Return ``(start, stop)`` bounds of entries with ``lo <= key < hi``."""
    k = self.keys.numpy() if getattr(type(self.keys), "__parakeet_device_array__", False) else np.asarray(self.keys)
    return int(np.searchsorted(k, lo, side="left")), int(np.searchsorted(k, hi, side="left"))


@dataclass(frozen=True)
class FrozenHashTable:
  """Immutable open-addressing linear-probing hash table for O(1) lookups inside ``@jit``.

  ``keys`` is a power-of-2 length 1D ``int64`` or ``uint64`` array with unused
  slots filled with ``empty_key``.  ``values`` has matching length.
  """

  keys: Any
  values: Any
  empty_key: int

  @classmethod
  def from_items(cls, keys, values, empty_key: int | None = None, *, load_factor: float = 0.5) -> FrozenHashTable:
    k = np.ascontiguousarray(keys)
    v = np.ascontiguousarray(values)
    if k.ndim != 1:
      raise ValueError(f"FrozenHashTable keys must be 1D, got shape {k.shape}")
    if len(k) != len(v):
      raise ValueError(f"FrozenHashTable keys ({len(k)}) and values ({len(v)}) must have equal length")
    if k.dtype not in (np.int64, np.uint64):
      k = k.astype(np.int64)
    if empty_key is None:
      empty_key = (1 << 64) - 1 if k.dtype == np.uint64 else -(1 << 63)
    if len(k) > 0 and np.any(k == k.dtype.type(empty_key)):
      raise ValueError(f"empty_key sentinel {empty_key} appears in input keys; pass an explicit unused empty_key")

    n = len(k)
    min_cap = max(2, int(np.ceil(max(1, n) / max(0.1, min(0.75, load_factor)))))
    cap = 1
    while cap < min_cap:
      cap <<= 1
    mask = cap - 1

    table_k = np.full(cap, k.dtype.type(empty_key), dtype=k.dtype)
    table_v = np.zeros((cap, *v.shape[1:]), dtype=v.dtype)
    empty_val = k.dtype.type(empty_key)
    hashes = _mix64_u64(k.view(np.uint64))
    for i in range(n):
      ki = k[i]
      slot = int(hashes[i]) & mask
      while table_k[slot] != empty_val and table_k[slot] != ki:
        slot = (slot + 1) & mask
      table_k[slot] = ki
      table_v[slot] = v[i]
    return cls(table_k, table_v, k.dtype.type(empty_key))

  def __len__(self) -> int:
    k = self.keys._host if getattr(type(self.keys), "__parakeet_device_array__", False) else self.keys
    return int(k.shape[0])

  def __contains__(self, key) -> bool:
    k = self.keys.numpy() if getattr(type(self.keys), "__parakeet_device_array__", False) else np.asarray(self.keys)
    mask = len(k) - 1
    slot = _mix64_scalar(int(key)) & mask
    ek = k.dtype.type(self.empty_key)
    kk = k.dtype.type(key)
    for _ in range(len(k)):
      cur = k[slot]
      if cur == kk:
        return True
      if cur == ek:
        return False
      slot = (slot + 1) & mask
    return False

  def get(self, key, default=0):
    k = self.keys.numpy() if getattr(type(self.keys), "__parakeet_device_array__", False) else np.asarray(self.keys)
    v = (
      self.values.numpy() if getattr(type(self.values), "__parakeet_device_array__", False) else np.asarray(self.values)
    )
    mask = len(k) - 1
    slot = _mix64_scalar(int(key)) & mask
    ek = k.dtype.type(self.empty_key)
    kk = k.dtype.type(key)
    for _ in range(len(k)):
      cur = k[slot]
      if cur == kk:
        return v[slot]
      if cur == ek:
        return default
      slot = (slot + 1) & mask
    return default
