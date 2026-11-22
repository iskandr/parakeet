"""Scalar types and NumPy-compatible type promotion.

Parakeet follows NumPy 2 (NEP 50) promotion.  Python ``int``/``float``/``bool``
values are *weak*: they adapt to the other operand (``float32_array * 2.0``
stays ``float32``), exactly as in NumPy.  Weakness is tracked in the type so the
result dtypes of compiled code match what NumPy would have produced.
"""

from __future__ import annotations

import functools

import numpy as np

__all__ = [
    "ScalarType",
    "BufType",
    "Bool",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "WeakBool",
    "WeakInt",
    "WeakFloat",
    "from_dtype",
    "is_supported_dtype",
    "join",
]

_C_NAMES = {
    "bool": "bool",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
}


class ScalarType:
    """An element type: one of NumPy's bool/int/uint/float dtypes, maybe weak."""

    __slots__ = ("name", "weak", "dtype", "kind", "bits", "_strong")
    _cache: dict[tuple[str, bool], ScalarType] = {}

    def __new__(cls, name: str, weak: bool = False) -> ScalarType:
        key = (name, weak)
        self = cls._cache.get(key)
        if self is None:
            if name not in _C_NAMES:
                raise ValueError(f"unsupported scalar type {name!r}")
            if weak and name not in ("bool", "int64", "float64"):
                raise ValueError(f"{name} cannot be weak")
            self = object.__new__(cls)
            self.name = name
            self.weak = weak
            self.dtype = np.dtype(name)
            self.kind = self.dtype.kind  # 'b', 'i', 'u' or 'f'
            self.bits = self.dtype.itemsize * 8
            cls._cache[key] = self
        return self

    # interned, so identity comparison is equality
    def __repr__(self) -> str:
        return f"{self.name}{'?' if self.weak else ''}"

    def __reduce__(self):
        return (ScalarType, (self.name, self.weak))

    @property
    def strong(self) -> ScalarType:
        return ScalarType(self.name) if self.weak else self

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def is_float(self) -> bool:
        return self.kind == "f"

    @property
    def is_int(self) -> bool:
        return self.kind in "iu"

    @property
    def is_bool(self) -> bool:
        return self.kind == "b"

    @property
    def is_signed(self) -> bool:
        return self.kind in "if"

    @property
    def ctype(self) -> str:
        return _C_NAMES[self.name]

    def python_type(self) -> type:
        return {"b": bool, "i": int, "u": int, "f": float}[self.kind]


class BufType:
    """Type of a variable holding a reference to a buffer of ``elt`` values."""

    __slots__ = ("elt",)
    _cache: dict[ScalarType, BufType] = {}

    def __new__(cls, elt: ScalarType) -> BufType:
        elt = elt.strong
        self = cls._cache.get(elt)
        if self is None:
            self = object.__new__(cls)
            self.elt = elt
            cls._cache[elt] = self
        return self

    def __repr__(self) -> str:
        return f"buf[{self.elt}]"


Bool = ScalarType("bool")
Int8 = ScalarType("int8")
Int16 = ScalarType("int16")
Int32 = ScalarType("int32")
Int64 = ScalarType("int64")
UInt8 = ScalarType("uint8")
UInt16 = ScalarType("uint16")
UInt32 = ScalarType("uint32")
UInt64 = ScalarType("uint64")
Float32 = ScalarType("float32")
Float64 = ScalarType("float64")
WeakBool = ScalarType("bool", weak=True)
WeakInt = ScalarType("int64", weak=True)
WeakFloat = ScalarType("float64", weak=True)

_SUPPORTED = frozenset(_C_NAMES)


def is_supported_dtype(dtype) -> bool:
    try:
        dt = np.dtype(dtype)
    except TypeError:
        return False
    return dt.name in _SUPPORTED and dt.isnative


def from_dtype(dtype) -> ScalarType:
    """Convert anything ``np.dtype`` accepts into a (strong) :class:`ScalarType`."""
    dt = np.dtype(dtype)
    if dt.name == "float16":  # no half-float code generation: widen
        return Float32
    if dt.name not in _SUPPORTED:
        raise TypeError(f"dtype {dt.name!r} is not supported by Parakeet (supported: {', '.join(sorted(_SUPPORTED))})")
    return ScalarType(dt.name)


def weak_of_kind(kind: str) -> ScalarType:
    return {"b": WeakBool, "i": WeakInt, "u": WeakInt, "f": WeakFloat}[kind]


_KIND_RANK = {"b": 0, "i": 1, "u": 1, "f": 2}


@functools.cache
def join(a: ScalarType, b: ScalarType) -> ScalarType:
    """Least common type of two values flowing into the same variable."""
    if a is b:
        return a
    if a.weak and b.weak:
        return a if _KIND_RANK[a.kind] >= _KIND_RANK[b.kind] else b
    if a.weak or b.weak:
        w, s = (a, b) if a.weak else (b, a)
        if _KIND_RANK[w.kind] <= _KIND_RANK[s.kind]:
            return s
        return from_dtype(np.result_type(s.dtype, w.dtype))
    return from_dtype(np.promote_types(a.dtype, b.dtype))


@functools.cache
def resolve_ufunc(
    ufunc: np.ufunc, in_types: tuple[ScalarType, ...]
) -> tuple[tuple[ScalarType, ...], tuple[ScalarType, ...]]:
    """Ask NumPy which loop ``ufunc`` would pick for these operand types.

    Returns ``(cast_input_types, output_types)``; raises ``TypeError`` (with
    NumPy's own message) when no loop exists.
    """
    # Python bools are passed as the bool dtype: it is NumPy's lowest type, so
    # promotion treats it exactly like a weak value
    spec = tuple(t.python_type() if (t.weak and t.kind != "b") else t.dtype for t in in_types)
    resolved = ufunc.resolve_dtypes(spec + (None,) * ufunc.nout)
    ins = tuple(from_dtype(d) for d in resolved[: ufunc.nin])
    outs = tuple(from_dtype(d) for d in resolved[ufunc.nin :])
    return ins, outs
