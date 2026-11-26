"""Compile-time values manipulated by the frontend.

While a function is elaborated every Python expression evaluates to one of:

* an :class:`~parakeet.syntax.ir.Expr` - a runtime scalar,
* an :class:`ArrayVal` - an array, either *concrete* (a strided view of a
  buffer) or *delayed* (a function from index to element, fused into whatever
  consumes it),
* a Python ``tuple`` / ``slice`` / ``None`` / ``str`` holding further values,
* a function-like value (:class:`PyFunc`, :class:`Closure`, a library entry),
* helper values such as :class:`RangeVal` or :class:`TypeVal`.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..ndtypes import ScalarType
from .ir import Expr, Scope, Var

INF_COST = 10**9
CHEAP_COST = 24  # delayed arrays at most this expensive may be re-evaluated freely


@dataclass(eq=False, slots=True)
class Root:
  """An allocation (or input array) a view may point into."""

  scope: Scope | None  # None: memory owned by the caller (an input array)
  label: str = ""
  param: Any = None  # the ir.ArrayParam for inputs


class ArrayVal:
  """An n-dimensional array known to the compiler.

  Concrete arrays have ``buf``/``offset``/``strides`` (strides in elements).
  Delayed arrays have ``fn``: called with one index expression per dimension
  it returns the element (and may emit statements into the current block).
  """

  __slots__ = (
        "elt", "shape", "buf", "offset", "strides", "fn", "roots", "cost", "contig",
        "depth", "epoch", "block", "origin", "uses", "reads", "inj", "base",
    )  # fmt: skip

  def __init__(
    self,
    elt: ScalarType,
    shape: tuple[Expr, ...],
    *,
    buf: Var | None = None,
    offset: Expr | None = None,
    strides: tuple[Expr, ...] | None = None,
    fn: Callable[[tuple[Expr, ...]], Expr] | None = None,
    roots: frozenset[Root] | None = frozenset(),
    cost: int = 1,
    contig: bool = False,
    reads: tuple[ArrayVal, ...] | None = None,
  ):
    self.elt = elt.strong
    self.shape = tuple(shape)
    self.buf = buf
    self.offset = offset
    self.strides = None if strides is None else tuple(strides)
    self.fn = fn
    self.roots = roots  # None means "could alias anything"
    self.cost = cost
    self.contig = contig
    # concrete arrays this (delayed) value reads, when known exactly
    self.reads = reads
    # bookkeeping for delayed arrays, filled in by the elaborator
    self.depth = 0
    self.epoch = 0
    self.block: Any = None
    self.origin: Any = None
    self.uses = 0
    self.inj = False  # concrete: every index addresses a different element
    self.base: ArrayVal | None = None  # the contiguous array this is a view of, when known

  @property
  def rank(self) -> int:
    return len(self.shape)

  @property
  def concrete(self) -> bool:
    return self.buf is not None

  def __repr__(self) -> str:
    kind = "array" if self.concrete else "delayed"
    return f"<{kind} {self.elt.name} rank={self.rank}>"


@dataclass(eq=False, slots=True)
class TypeVal:
  """A dtype used as a value (``np.float32``, ``x.dtype``, ``float``...)."""

  type: ScalarType  # weak for the Python builtins bool/int/float


@dataclass(eq=False, slots=True)
class PyFunc:
  """A Python function whose source will be inlined."""

  fn: Any  # the function object (any instance sharing the code object)
  cells: dict[str, Any] | None = None  # closure variables as compile-time values


@dataclass(eq=False, slots=True)
class Closure:
  """A ``def`` or ``lambda`` appearing inside compiled code."""

  node: ast.FunctionDef | ast.Lambda
  frame: Any
  defaults: list[Any]
  kw_defaults: dict[str, Any]
  name: str


@dataclass(eq=False, slots=True)
class BoundMethod:
  """``obj.name`` before it is called."""

  obj: Any
  name: str


@dataclass(eq=False, slots=True)
class Partial:
  """A function with some leading arguments already supplied."""

  fn: Any
  args: tuple
  kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False, slots=True)
class RangeVal:
  start: Expr
  stop: Expr
  step: Expr


@dataclass(eq=False, slots=True)
class ZipVal:
  items: tuple
  strict: Expr


@dataclass(eq=False, slots=True)
class EnumerateVal:
  item: Any
  start: Expr


@dataclass(eq=False, slots=True)
class ListVal:
  """Compile-time mutable Python list in straight-line or unrolled code."""

  items: list[Any]
  origin: Any = None
  path: Any = None
  loop_depth: int = 0
  branch_depth: int = 0


@dataclass(eq=False, slots=True)
class VecVal:
  """CPU runtime growable homogeneous list backed by ``(buf, length, cap)``."""

  buf: Var
  length: Var
  cap: Var
  elt: ScalarType
  slot: Var
  home: Scope


@dataclass(eq=False, slots=True)
class FixedListVal:
  """Fixed-capacity thread-local stack/list backed by a stack-allocated buffer
  (zero heap malloc — native on CPU, CUDA, and Metal)."""

  buf: Var
  length: Var
  cap: int
  elt: ScalarType


@dataclass(eq=False, slots=True)
class ChainVal:
  """Zero-allocation ``itertools.chain(*iterables)``."""

  items: tuple


@dataclass(eq=False, slots=True)
class ProductVal:
  """Zero-allocation ``itertools.product(*iterables)``."""

  items: tuple


@dataclass(eq=False, slots=True)
class BatchedVal:
  """Zero-allocation ``itertools.batched(iterable, n)``."""

  item: Any
  n: Expr


@dataclass(eq=False, slots=True)
class Undefined:
  """Placeholder for a name that is not bound on every path."""

  name: str
  reason: str = "is not defined on every path reaching this point"


def describe(v: Any) -> str:
  """Short description of a value for error messages."""
  if isinstance(v, Expr):
    t = v.type
    if isinstance(t, ScalarType):
      if t.weak:
        return {"b": "bool", "i": "int", "u": "int", "f": "float"}[t.kind]
      return t.name
    return "buffer"
  if isinstance(v, ArrayVal):
    return f"array[{v.elt.name}, {v.rank}d]"
  if isinstance(v, tuple):
    return "(" + ", ".join(describe(x) for x in v) + ("," if len(v) == 1 else "") + ")"
  if v is None:
    return "None"
  if isinstance(v, TypeVal):
    return f"dtype {v.type.name}"
  if isinstance(v, (PyFunc, Closure)):
    name = v.fn.__name__ if isinstance(v, PyFunc) else v.name
    return f"function {name}"
  if isinstance(v, BoundMethod):
    return f"method {v.name} of {describe(v.obj)}"
  if isinstance(v, RangeVal):
    return "range"
  if isinstance(v, ListVal):
    return f"list[{len(v.items)}]"
  if isinstance(v, VecVal):
    return f"list[{v.elt.name}]"
  if isinstance(v, FixedListVal):
    return f"FixedList[{v.elt.name}, {v.cap}]"
  if isinstance(v, (ChainVal, ProductVal, BatchedVal)):
    return type(v).__name__
  if isinstance(v, str):
    return f"str {v!r}"
  if isinstance(v, slice):
    return "slice"
  if isinstance(v, Undefined):
    return f"undefined variable {v.name}"
  name = getattr(v, "__name__", None)
  return f"{type(v).__name__} {name}" if name else type(v).__name__
