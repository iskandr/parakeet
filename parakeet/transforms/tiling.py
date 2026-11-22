"""Tiling: run matmul-shaped maps a cache-sized block at a time.

A *matmul-shaped* operation is a map over pairs ``(i, j)`` whose element is a
reduction over data selected by ``i`` and data selected by ``j``: a matrix
product, all-pairs distances, a tropical product, the assignment step of
k-means.  Evaluated row by row, every row of the result sweeps over all of
the ``j`` operand, which stops fitting in cache long before the arrays are
large.  Two rewrites fix that, and neither changes what is computed:

*tiling*
    the ``j`` loop is cut into tiles and hoisted outside the ``i`` loop, so a
    tile of the ``j`` operand is reused for every ``i`` while it is still in
    cache.  Legal for any map: its elements are independent.

*jamming*
    the reductions of a small block of elements (``UNROLL_I`` rows by
    ``UNROLL_J`` columns) are fused into a single loop with one accumulator
    each, so every operand element that was loaded is used several times.
    Each element still folds its own terms in the original order.

The tile size is a heuristic, not a search: as many ``j`` rows as fit in
``config.tile_bytes`` (half of a core's L2 cache unless set).

Nothing here knows about dot products.  :func:`probe` runs the element
function once and asks whether it consists of a single reduction plus
straight-line code; if so :func:`tiled_map` evaluates the map in tiles.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..errors import InternalError
from ..syntax import ops
from ..syntax.ir import Assign, Const, Expr, Store, Var
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, idiv, imax, imin, imul, isub
from ..syntax.values import ArrayVal

UNROLL_I = 2
UNROLL_J = 4
MIN_STATIC_SIZE = 256  # maps known to be smaller than this are not worth tiling


class Capture:
  """One reduction whose loop a :class:`Jam` will emit."""

  __slots__ = ("shape", "elem", "combine", "init", "acc_type", "assoc_op", "what", "pos", "red", "context")


class Jam:
  """Collects the reductions of the elements being elaborated instead of
  letting each emit its own loop (installed as ``el.jam``)."""

  def __init__(self, block: list):
    self.block = block
    self.captures: list[Capture] = []
    self.declined = False

  def capture(self, el, shape, elem, combine, init, acc_type, assoc_op, what: str) -> Var | None:
    """Take over a reduction about to be emitted; None to leave it alone."""
    if len(shape) != 1 or el.block is not self.block or self.captures:
      self.declined = True
      return None
    c = Capture()
    c.shape, c.elem, c.combine, c.init = shape, elem, combine, init
    c.acc_type, c.assoc_op, c.what = acc_type, assoc_op, what
    c.pos = len(el.block)
    c.red = Var("red", acc_type)
    c.context = el.capture()
    self.captures.append(c)
    return c.red


def probe(el, shape: tuple[Expr, ...], fn: Callable[[tuple], Any], probe_idx: tuple[Var, ...]) -> Expr | None:
  """If the map of ``fn`` over ``shape`` is worth tiling, the length of its
  elements' reduction (as an expression valid outside the loops)."""
  from ..lib.core import _defined_vars, block_weight, outer_expr
  from ..syntax.values import INF_COST

  if not el.options.tiling or len(shape) < 2 or el.jam is not None:
    return None
  if all(isinstance(d, Const) for d in shape) and math.prod(d.value for d in shape) < MIN_STATIC_SIZE:
    return None
  el.loop_depth += 1
  try:
    with el.trial() as blk:
      el.jam = jam = Jam(blk)
      try:
        r = fn(probe_idx)
      finally:
        el.jam = None
      if jam.declined or len(jam.captures) != 1 or not isinstance(r, Expr) or block_weight(blk) >= INF_COST:
        return None
      inner = _defined_vars(blk) | {v.uid for v in probe_idx}
      return outer_expr(el, jam.captures[0].shape[0], inner)
  finally:
    el.loop_depth -= 1


def tile_size(el, length: Expr, itemsize: int) -> Expr:
  """How many ``j`` rows of ``length`` elements make one tile."""
  budget = max(el.options.tile_bytes // max(itemsize, 1), 1)
  rows = idiv(Const(budget, I), imax(length, 1))
  whole = imul(idiv(rows, UNROLL_J), UNROLL_J)
  size = el.tmp(imax(whole, UNROLL_J), "tile")
  if isinstance(size, Var):
    el.nonneg.add(size.uid)
  return size


def tiled_map(el, shape: tuple[Expr, ...], fn: Callable[[tuple], Any], elt, length: Expr, what: str) -> ArrayVal:
  """Evaluate ``out[..., j] = fn((..., j))`` into a new array, in tiles of
  the last dimension.  The dimensions before it are one long row index."""
  lead, n_j = tuple(shape[:-1]), shape[-1]
  n_i = el.tmp(ops.iprod(lead), "rows")
  if isinstance(n_i, Var):
    el.nonneg.add(n_i.uid)
  out = el.alloc_array(elt, shape, label="out")
  element = fn

  def fn(ij):
    i, j = ij
    return element(((i,) if len(lead) == 1 else el.unravel(i, lead)) + (j,))

  reduce_length = el.tmp(length, "len")
  tile = tile_size(el, reduce_length, elt.itemsize)

  def index(e, bound: Expr) -> Expr:
    v = el.tmp(e, "i")
    if isinstance(v, Var):
      el.nonneg.add(v.uid)
      el.upper[v.uid] = bound
    return v

  def store(i: Expr, j: Expr, v) -> None:
    el.emit(Store(out.buf, iadd(imul(i, n_j), j), cast(el.as_scalar(v, f"result of {what}"), elt)))

  def single(i: Expr, j: Expr) -> None:
    store(i, j, fn((i, j)))

  def jammed(i: Expr, j: Expr) -> None:
    """The block of elements ``[i, i + UNROLL_I) x [j, j + UNROLL_J)``."""
    parts = []
    for p in range(UNROLL_I):
      for q in range(UNROLL_J):
        ip, jq = index(iadd(i, p), n_i), index(iadd(j, q), n_j)
        blk: list = []
        with el.in_block(blk):
          el.jam = jam = Jam(blk)
          try:
            v = fn((ip, jq))
          finally:
            el.jam = None
          if len(jam.captures) != 1:
            raise InternalError(f"tiling {what}: an element stopped being a single reduction")
          store(ip, jq, v)
        parts.append((blk, jam.captures[0]))
    for blk, c in parts:
      el.block.extend(blk[: c.pos])
    accs = []
    first = parts[0][1]
    start: Expr = ZERO
    for _, c in parts:
      acc = Var("acc", c.acc_type)
      el.mutables.add(acc.uid)
      accs.append(acc)
      if c.init is not None:
        el.emit(Assign(acc, cast(c.init, c.acc_type)))
      else:
        el.check(cmp("gt", c.shape[0], 0), "ValueError", f"zero-size array to {c.what}, which has no identity")
        with el.restored(c.context):
          el.emit(Assign(acc, cast(el.as_scalar(c.elem((ZERO,)), "element"), c.acc_type)))
        start = ONE

    def step(k: Var) -> None:
      el.upper[k.uid] = first.shape[0]
      for (_, c), acc in zip(parts, accs):
        with el.restored(c.context):
          x = c.elem((k,))
          el.emit(Assign(acc, cast(el.as_scalar(c.combine(acc, x), "reduction result"), c.acc_type)))

    foldable = first.assoc_op in ("add", "mul") and all(c.assoc_op == first.assoc_op for _, c in parts)
    el.for_range(
      start, first.shape[0], step, simd=True, name="k", reduction=(first.assoc_op, accs) if foldable else None
    )
    for (_, c), acc in zip(parts, accs):
      el.emit(Assign(c.red, acc))
    for blk, c in parts:
      el.block.extend(blk[c.pos :])

  def span(row: Expr, lo: Expr, hi: Expr) -> None:
    """Columns ``[lo, hi)`` of one row, an element at a time."""
    el.for_range(lo, hi, lambda j: single(row, index(j, n_j)), name="j")

  def rows(r0: Expr, r1: Expr) -> None:
    """Complete rows ``[r0, r1)``, a tile of columns at a time."""
    i_jam = el.tmp(iadd(r0, imul(idiv(isub(r1, r0), UNROLL_I), UNROLL_I)), "i_jam")

    def tile_body(j0: Var) -> None:
      el.nonneg.add(j0.uid)
      j1 = el.tmp(imin(n_j, iadd(j0, tile)), "j1")
      j_jam = el.tmp(iadd(j0, imul(idiv(isub(j1, j0), UNROLL_J), UNROLL_J)), "j_jam")

      def block_rows(i: Var) -> None:
        el.upper[i.uid] = n_i
        el.for_range(j0, j_jam, lambda j: jammed(i, j), step=Const(UNROLL_J, I), name="j")

        def leftover(j: Var) -> None:
          el.upper[j.uid] = n_j
          for p in range(UNROLL_I):
            single(index(iadd(i, p), n_i), j)

        el.for_range(j_jam, j1, leftover, name="j")

      el.for_range(r0, i_jam, block_rows, step=Const(UNROLL_I, I), name="i")
      el.for_range(i_jam, r1, lambda i: span(index(i, n_i), j0, j1), name="i")

    el.for_range(ZERO, n_j, tile_body, step=tile, name="j0")

  def body(chunk, start: Expr, stop: Expr) -> None:
    """Flat positions ``[start, stop)``: the end of a row, whole rows, the
    beginning of a row."""
    r0 = index(idiv(start, n_j), n_i)
    c0 = el.tmp(isub(start, imul(r0, n_j)), "c0")
    lead_end = el.tmp(ops.select(cmp("gt", c0, 0), imin(stop, imul(iadd(r0, 1), n_j)), start), "lead_end")
    span(r0, c0, iadd(c0, isub(lead_end, start)))
    first_row = el.tmp(ops.select(cmp("gt", c0, 0), iadd(r0, 1), r0), "first_row")
    last_row = el.tmp(idiv(stop, n_j), "last_row")  # may be one past the end
    for v in (first_row, lead_end, last_row):
      if isinstance(v, Var):
        el.nonneg.add(v.uid)
    # a chunk inside a single row has no complete rows (and first_row > last_row)
    rows(first_row, el.tmp(imax(first_row, last_row), "rows_end"))
    tail_start = el.tmp(imax(lead_end, imul(last_row, n_j)), "tail_start")
    span(last_row, isub(tail_start, imul(last_row, n_j)), isub(stop, imul(last_row, n_j)))

  total = el.tmp(imul(n_i, n_j), "total")
  # an empty row would make the row arithmetic divide by zero; there is nothing to do then anyway
  from ..syntax.ir import If

  work: list = []
  with el.in_block(work):
    # where it cannot run in parallel it is still a loop of one chunk: the
    # backend compiles that as a function of its own, and identical ones once
    el.par_loop(total, body, nchunks=None if el.can_parallel(shape) else ONE, base_depth=2, work=reduce_length)
  el.emit(If(cmp("gt", total, 0), work, []))
  el.record_store(out)
  return out


__all__ = ["Jam", "UNROLL_I", "UNROLL_J", "probe", "tile_size", "tiled_map"]
