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


@dataclass(frozen=True)
class ComplexTypeVal:
  """Compile-time complex dtype (``np.complex64``, ``np.complex128``, or Python ``complex``)."""

  elt: object  # Float32 or Float64 (ScalarType)
  weak: bool = False

  @property
  def dtype(self) -> np.dtype:
    return np.dtype(np.complex64) if self.elt.name == "float32" else np.dtype(np.complex128)

  @property
  def name(self) -> str:
    return self.dtype.name

  @property
  def itemsize(self) -> int:
    return self.dtype.itemsize


class ComplexArray:
  """Unboxed compile-time complex array represented as ``(re, im)`` float arrays."""

  __slots__ = ("re", "im", "_dtype", "_rank")

  def __init__(self, re, im, dtype: np.dtype | None = None, rank: int | None = None):
    self.re = re
    self.im = im
    self._dtype = dtype
    self._rank = rank

  @property
  def elt(self):
    return self.re.elt

  @property
  def rank(self) -> int:
    if self._rank is not None:
      return self._rank
    return self.re.rank

  @property
  def shape(self) -> tuple:
    return self.re.shape

  @property
  def dtype(self) -> np.dtype:
    if self._dtype is not None:
      return self._dtype
    return np.dtype(np.complex64) if self.re.elt.name == "float32" else np.dtype(np.complex128)


class ComplexScalar:
  """Unboxed compile-time complex scalar represented as ``(re, im)`` float expressions."""

  __slots__ = ("re", "im", "python_complex", "_dtype")

  def __init__(self, re, im, python_complex: bool = False, dtype: np.dtype | None = None):
    self.re = re
    self.im = im
    self.python_complex = python_complex
    self._dtype = dtype

  @property
  def elt(self):
    return self.re.type

  @property
  def dtype(self) -> np.dtype:
    if self._dtype is not None:
      return self._dtype
    return np.dtype(np.complex64) if self.re.type.name == "float32" else np.dtype(np.complex128)


@dataclass(frozen=True)
class ComplexArrayRecord:
  dtype: np.dtype
  rank: int

  @property
  def cls(self) -> type:
    return ComplexArray

  @property
  def names(self) -> tuple[str, str]:
    return ("re", "im")

  def values(self, obj):
    if isinstance(obj, np.ndarray):
      return (obj.real, obj.imag)
    return (obj.re, obj.im)

  def build(self, values):
    re, im = values
    if isinstance(re, np.ndarray) and isinstance(im, np.ndarray):
      out = np.empty(re.shape, dtype=self.dtype)
      out.real = re
      out.imag = im
      return out
    return ComplexArray(re, im, dtype=self.dtype, rank=self.rank)


@dataclass(frozen=True)
class ComplexScalarRecord:
  dtype: np.dtype
  python_complex: bool = False

  @property
  def cls(self) -> type:
    return ComplexScalar

  @property
  def names(self) -> tuple[str, str]:
    return ("re", "im")

  def values(self, obj):
    if isinstance(obj, ComplexScalar):
      return (obj.re, obj.im)
    if self.dtype == np.complex64:
      return (np.float32(obj.real), np.float32(obj.imag))
    if self.python_complex:
      return (float(obj.real), float(obj.imag))
    return (np.float64(obj.real), np.float64(obj.imag))

  def build(self, values):
    re, im = values
    if isinstance(re, (int, float, np.number)) and isinstance(im, (int, float, np.number)):
      c = complex(float(re), float(im))
      return c if self.python_complex else self.dtype.type(c)
    return ComplexScalar(re, im, python_complex=self.python_complex, dtype=self.dtype)


def record_type(obj):
  if isinstance(obj, ComplexArray):
    return ComplexArrayRecord(obj.dtype, obj.rank)
  if isinstance(obj, ComplexScalar):
    return ComplexScalarRecord(obj.dtype, python_complex=obj.python_complex)
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
    and (
      not cls.__module__.startswith("parakeet.")
      or (cls.__module__ == "parakeet.ragged" and cls.__name__ == "RaggedArray")
      or (cls.__module__ == "parakeet.index" and cls.__name__ in ("SortedIndex", "FrozenHashTable"))
      or (cls.__module__ == "parakeet.spatial" and cls.__name__ == "BlockMask")
      or getattr(cls, "__parakeet_record__", False)
    )
  ):
    return Record(cls, tuple(f.name for f in fields(obj)))
  if not isinstance(obj, type) and getattr(cls, "__parakeet_record__", False):
    return Record(cls, tuple(obj.__dict__.keys()))
  return None
