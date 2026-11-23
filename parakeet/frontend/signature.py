"""Classify Python values into *type keys* plus a flat list of runtime leaves.

A type key captures everything the compiler specialises on (dtypes, ranks,
tuple structure, identities of functions, ...).  The leaves are the values that
are passed to compiled code at call time (scalars and arrays).  The same
routine is used for explicit arguments, default values, closure cells and the
globals a function reads.
"""

from __future__ import annotations

import threading
import types as pytypes
from typing import Any

import numpy as np

from ..ndtypes import (
  Float32,
  ScalarType,
  WeakBool,
  WeakFloat,
  WeakInt,
  from_dtype,
  is_supported_dtype,
)

Key = tuple

_NONE_KEY: Key = ("n",)
_ELLIPSIS_KEY: Key = ("e",)
_WEAK_INT: Key = ("s", WeakInt)
_WEAK_FLOAT: Key = ("s", WeakFloat)
_WEAK_BOOL: Key = ("s", WeakBool)

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


_active = threading.local()


class Ref:
  """Hashable identity wrapper for static Python objects inside keys."""

  __slots__ = ("obj",)

  def __init__(self, obj: Any):
    self.obj = obj

  def __hash__(self) -> int:
    return id(self.obj)

  def __eq__(self, other: object) -> bool:
    return isinstance(other, Ref) and other.obj is self.obj

  def __repr__(self) -> str:
    return f"Ref({getattr(self.obj, '__name__', self.obj)!r})"


class Unsupported(TypeError):
  """Raised by :func:`classify` for values Parakeet cannot accept."""


# dtype -> ScalarType for dtypes that can be passed without conversion
_DIRECT: dict[np.dtype, ScalarType] = {}
for _name in ("bool int8 int16 int32 int64 uint8 uint16 uint32 uint64 float32 float64").split():
  _DIRECT[np.dtype(_name)] = ScalarType(_name)

#: Python functions that have a staged implementation (filled in by
#: :mod:`parakeet.lib`); these must never be compiled from their own source
STAGED: dict = {}

#: one representative function object per code object, so that a
#: ``("fn", code, ...)`` key can be turned back into source
CODE_FUNCTIONS: dict[pytypes.CodeType, pytypes.FunctionType] = {}


def type_value(obj: Any) -> ScalarType | type | None:
  """Normalise a dtype-like object; ``None`` if ``obj`` is not one."""
  if obj is bool or obj is int or obj is float:
    return obj
  if isinstance(obj, np.dtype) or (isinstance(obj, type) and issubclass(obj, np.generic)):
    try:
      return from_dtype(obj)
    except TypeError as e:
      raise Unsupported(str(e)) from None
  return None


def _convert_array(arr: np.ndarray) -> np.ndarray:
  dt = arr.dtype
  if dt.name == "float16":
    return arr.astype(np.float32)
  if dt.kind in "biuf" and dt.name in ("bool",) + tuple(d.name for d in _DIRECT):
    # right dtype but wrong byte order or alignment
    return np.ascontiguousarray(arr, dtype=dt.newbyteorder("="))
  if dt.kind == "f" and dt.itemsize > 8:
    raise Unsupported(
      f"arrays of dtype {dt.name} (extended precision) are not supported; convert with .astype(np.float64)"
    )
  if dt.kind == "c":
    raise Unsupported(
      f"complex arrays (dtype {dt.name}) are not supported; pass the real and imaginary parts separately"
    )
  if dt.kind == "O":
    raise Unsupported(
      "object arrays are not supported (this often means a ragged nested list or a list mixing arrays and scalars)"
    )
  raise Unsupported(f"arrays of dtype {dt.name} are not supported")


def classify(value: Any, leaves: list, fixups: list | None = None) -> Key:
  """Return the type key of ``value`` and append its runtime leaves to ``leaves``.

  ``fixups`` (if given) collects ``(original, converted)`` array pairs for
  inputs that had to be copied into a supported layout, so the caller can copy
  mutations back.
  """
  t = type(value)
  if getattr(t, "__parakeet_device_array__", False):
    host = value._host
    elt = _DIRECT.get(host.dtype)
    if elt is None:
      return classify(value.numpy(), leaves, fixups)
    leaves.append(value)
    shape = host.shape
    return (
      "a",
      elt,
      len(shape),
      "C" if host.flags.c_contiguous else "S",
      tuple([d == 1 for d in shape]),
    )
  if t is np.ndarray:
    elt = _DIRECT.get(value.dtype)
    if elt is None or not value.flags.aligned or not value.dtype.isnative:
      converted = _convert_array(value)
      if fixups is not None:
        fixups.append((value, converted))
      value = converted
      elt = _DIRECT[value.dtype]
    c_contig = value.flags.c_contiguous
    if not c_contig:
      itemsize = value.itemsize
      for s in value.strides:
        if s % itemsize:
          # e.g. a field of a structured array: copy into a regular layout
          converted = np.ascontiguousarray(value)
          if fixups is not None:
            fixups.append((value, converted))
          value = converted
          c_contig = True
          break
    leaves.append(value)
    shape = value.shape
    return (
      "a",
      elt,
      len(shape),
      "C" if c_contig else "S",
      tuple([d == 1 for d in shape]),
    )
  if t is float:
    leaves.append(value)
    return _WEAK_FLOAT
  if t is int:
    if not _INT64_MIN <= value <= _INT64_MAX:
      raise Unsupported(f"integer {value} does not fit in 64 bits")
    leaves.append(value)
    return _WEAK_INT
  if t is bool:
    leaves.append(value)
    return _WEAK_BOOL
  if t is tuple:
    return ("t", *[classify(v, leaves, fixups) for v in value])
  if value is None:
    return _NONE_KEY
  if isinstance(value, np.generic):
    if value.dtype.name == "float16":
      leaves.append(np.float32(value))
      return ("s", Float32)
    if not is_supported_dtype(value.dtype):
      raise Unsupported(f"scalars of type {type(value).__name__} are not supported")
    leaves.append(value)
    return ("s", _DIRECT[value.dtype])
  if isinstance(value, np.ndarray):
    if isinstance(value, np.ma.MaskedArray):
      raise Unsupported("masked arrays are not supported; pass .data and .mask separately")
    return classify(np.asarray(value), leaves, fixups)
  if t is str:
    return ("str", value)
  if t is list:
    try:
      arr = np.asarray(value)
    except ValueError as e:
      raise Unsupported(f"cannot convert list argument to an array: {e}") from None
    if arr.size == 0 and arr.dtype == np.float64 and arr.ndim == 1:
      arr = np.empty(0, dtype=np.float64)
    return classify(arr, leaves, fixups)
  if isinstance(value, tuple):  # namedtuple and friends
    return ("t", *[classify(v, leaves, fixups) for v in value])
  if t is slice:
    return (
      "sl",
      classify(value.start, leaves, fixups),
      classify(value.stop, leaves, fixups),
      classify(value.step, leaves, fixups),
    )
  if value is Ellipsis:
    return _ELLIPSIS_KEY
  if t is range:
    leaves.extend((value.start, value.stop, value.step))
    return ("r",)
  tv = type_value(value)
  if tv is not None:
    return ("ty", tv)
  if hasattr(value, "__parakeet_fn__"):  # a @jit wrapper: use the wrapped function
    return classify(value.__parakeet_fn__, leaves, fixups)
  if t is pytypes.FunctionType:
    if value in STAGED:
      return ("o", Ref(value))
    code = value.__code__
    cells = value.__closure__
    if not cells:
      return ("fn", code, id(value.__globals__), Ref(value))
    if code not in CODE_FUNCTIONS:
      CODE_FUNCTIONS[code] = value
    active = _active.__dict__.setdefault("functions", set())
    if id(value) in active:  # a closure that (indirectly) refers to itself
      return ("fn", code, id(value.__globals__), Ref(value))
    active.add(id(value))
    try:
      cell_keys = []
      for name, cell in zip(code.co_freevars, cells):
        try:
          contents = cell.cell_contents
        except ValueError:
          cell_keys.append((name, ("unbound",)))
          continue
        cell_keys.append((name, classify(contents, leaves, fixups)))
    finally:
      active.discard(id(value))
    return ("fn", code, id(value.__globals__), tuple(cell_keys))
  if isinstance(value, pytypes.MethodType):
    raise Unsupported(
      f"bound methods ({value.__qualname__}) cannot be compiled; "
      "use a plain function and pass the object's arrays explicitly"
    )
  if callable(value) or isinstance(value, pytypes.ModuleType):
    return ("o", Ref(value))
  # last resort: anything that can present itself as a numeric array
  if hasattr(value, "__array__") or hasattr(value, "__array_interface__"):
    try:
      arr = np.asarray(value)
    except Exception as e:  # noqa: BLE001 - foreign __array__ can raise anything
      raise Unsupported(
        f"could not convert {type(value).__module__}.{type(value).__name__} to a "
        f"NumPy array ({e}); move it to host memory first (e.g. tensor.cpu())"
      ) from None
    if arr.dtype.kind in "biuf":
      return classify(arr, leaves, fixups)
  raise Unsupported(
    f"values of type {type(value).__module__}.{type(value).__qualname__} cannot be passed to compiled code"
  )


def describe(key: Key) -> str:
  """Human-readable rendering of a type key (for messages and ``repr``)."""
  tag = key[0]
  if tag == "s":
    t = key[1]
    return {"b": "bool", "i": "int", "u": "int", "f": "float"}[t.kind] if t.weak else t.name
  if tag == "c":
    return f"{describe(('s', key[1]))}={key[2]!r}"
  if tag == "a":
    return f"array[{key[1].name}, {key[2]}d{'' if key[3] == 'C' else ', strided'}]"
  if tag == "t":
    return "(" + ", ".join(describe(k) for k in key[1:]) + ("," if len(key) == 2 else "") + ")"
  if tag == "n":
    return "None"
  if tag == "str":
    return repr(key[1])
  if tag == "fn":
    return f"<function {key[1].co_name}>"
  if tag == "ty":
    return f"<type {getattr(key[1], 'name', None) or key[1].__name__}>"
  if tag == "o":
    return repr(key[1])
  if tag == "sl":
    return "slice"
  if tag == "r":
    return "range"
  if tag == "e":
    return "..."
  return str(key)


def count_leaves(key: Key) -> int:
  tag = key[0]
  if tag in ("s", "a"):
    return 1
  if tag == "r":
    return 3
  if tag in ("t",):
    return sum(count_leaves(k) for k in key[1:])
  if tag == "sl":
    return sum(count_leaves(k) for k in key[1:])
  if tag == "fn" and isinstance(key[3], tuple):
    return sum(count_leaves(k) for _, k in key[3])
  return 0


def with_static_leaves(key: Key, leaves: list, static: frozenset[int], _pos: list | None = None) -> Key:
  """Rewrite ``key`` so that the scalar leaves whose indices are in ``static``
  become compile-time constants (``("c", type, value)``).  An entry
  ``(index, k)`` does the same for dimension ``k`` of the array leaf ``index``."""
  pos = _pos if _pos is not None else [0]
  tag = key[0]
  if tag == "s":
    i = pos[0]
    pos[0] += 1
    if i in static:
      v = leaves[i]
      return ("c", key[1], v.item() if isinstance(v, np.generic) else v)
    return key
  if tag == "a":
    i = pos[0]
    pos[0] += 1
    fixed = tuple(int(leaves[i].shape[k]) if (i, k) in static else None for k in range(key[2]))
    if any(d is not None for d in fixed):
      return (*key[:5], fixed)
    return key
  if tag == "r":
    i = pos[0]
    pos[0] += 3
    return key
  if tag in ("t", "sl"):
    return (tag, *[with_static_leaves(k, leaves, static, pos) for k in key[1:]])
  if tag == "fn" and isinstance(key[3], tuple):
    cells = tuple((n, with_static_leaves(k, leaves, static, pos)) for n, k in key[3])
    return (tag, key[1], key[2], cells)
  return key
