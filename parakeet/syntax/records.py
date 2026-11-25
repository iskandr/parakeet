"""Fixed Python record structure carried alongside flattened numerical leaves."""

from dataclasses import dataclass, fields, is_dataclass

import numpy as np


@dataclass(frozen=True)
class Record:
  cls: type
  names: tuple[str, ...]

  def values(self, obj):
    if self.cls is dict:
      return tuple(obj[name] for name in self.names)
    return tuple(getattr(obj, name) for name in self.names)

  def build(self, values):
    if self.cls is dict:
      return dict(zip(self.names, values, strict=True))
    if issubclass(self.cls, tuple):
      return tuple.__new__(self.cls, values)
    obj = object.__new__(self.cls)
    for name, value in zip(self.names, values, strict=True):
      object.__setattr__(obj, name, value)
    return obj


class StructuredView:
  """Compile-time view of a structured NumPy array as named field arrays."""

  __slots__ = ("dtype", "names", "_fields", "rank")

  def __init__(self, dtype: np.dtype, names: tuple[str, ...], field_vals: tuple, rank: int):
    self.dtype = dtype
    self.names = names
    self._fields = tuple(field_vals)
    self.rank = rank

  def field(self, name: str):
    return self._fields[self.names.index(name)]

  @property
  def shape(self) -> tuple:
    return self._fields[0].shape[: self.rank] if self._fields else ()


class StructuredRow:
  """Single element of a :class:`StructuredView`, supporting ``row['x']`` and ``row.x``."""

  __slots__ = ("dtype", "names", "_fields")

  def __init__(self, dtype: np.dtype, names: tuple[str, ...], field_vals: tuple):
    self.dtype = dtype
    self.names = names
    self._fields = tuple(field_vals)

  def field(self, name: str):
    return self._fields[self.names.index(name)]


@dataclass(frozen=True)
class StructuredRecord:
  dtype: np.dtype
  names: tuple[str, ...]
  rank: int

  @property
  def cls(self) -> type:
    return StructuredView

  def values(self, obj):
    if isinstance(obj, np.ndarray):
      return tuple(obj[name] for name in self.names)
    return obj._fields

  def build(self, values):
    vals = tuple(values)
    if vals and all(isinstance(v, np.ndarray) for v in vals):
      shape = vals[0].shape[: self.rank]
      out = np.empty(shape, dtype=self.dtype)
      for name, arr in zip(self.names, vals, strict=True):
        out[name] = arr
      return out
    return StructuredView(self.dtype, self.names, vals, self.rank)


@dataclass(frozen=True)
class StructuredRowRecord:
  dtype: np.dtype
  names: tuple[str, ...]

  @property
  def cls(self) -> type:
    return StructuredRow

  def values(self, obj):
    return obj._fields

  def build(self, values):
    vals = tuple(values)
    if vals and not any(hasattr(v, "type") or hasattr(v, "elt") for v in vals):
      out = np.empty((), dtype=self.dtype)
      for name, val in zip(self.names, vals, strict=True):
        out[name] = val
      return out[()]
    return StructuredRow(self.dtype, self.names, vals)


def record_type(obj):
  if isinstance(obj, StructuredView):
    return StructuredRecord(obj.dtype, obj.names, obj.rank)
  if isinstance(obj, StructuredRow):
    return StructuredRowRecord(obj.dtype, obj.names)
  cls = type(obj)
  if cls is dict and all(type(key) is str for key in obj):
    return Record(dict, tuple(obj))
  if isinstance(obj, tuple) and isinstance(names := getattr(cls, "_fields", None), tuple):
    return Record(cls, names)
  if (
    not isinstance(obj, type)
    and is_dataclass(obj)
    and cls.__dataclass_params__.frozen
    and (
      not cls.__module__.startswith("parakeet.")
      or (cls.__module__ == "parakeet.ragged" and cls.__name__ == "RaggedArray")
      or (cls.__module__ == "parakeet.index" and cls.__name__ in ("SortedIndex", "FrozenHashTable"))
    )
  ):
    return Record(cls, tuple(f.name for f in fields(obj)))
  return None
