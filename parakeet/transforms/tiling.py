"""Tiling: run matmul-shaped maps a cache-sized block at a time.

A *matmul-shaped* operation is a map whose element is a reduction over data
selected by the map's index: a matrix product, all-pairs distances, a
tropical product, the assignment step of k-means, a stack of any of those.
At the operator level that is a pattern, not a guess::

    map i.., j over (.., m):          # any rank >= 2
      straight-line code
      reduce k.. over (K..):          # one reduction, of any rank,
        fold(...), fold(...)          #   with any number of folds
      straight-line code              # the "epilogue": sqrt, a store, ...

Evaluated row by row, every row of the result sweeps over all of the ``j``
operand, which stops fitting in cache long before the arrays are large.  Two
rewrites fix that, and neither changes what is computed:

*tiling*
    the ``j`` loop is cut into tiles and hoisted outside the loops before
    it, so a tile of the ``j`` operand is reused for every row while it is
    still in cache.  Legal for any map: its elements are independent.

*jamming*
    the reductions of a small block of elements (``UNROLL_I`` rows by
    ``UNROLL_J`` columns) become one reduction with a fold per element, so
    every operand element that was loaded is used several times.  Each
    element still folds its own terms in the original order.  This is
    sibling fusion (see :mod:`.fusion`) applied to copies of the map's body.

The tile size is a heuristic, not a search: as many ``j`` rows as fit in
``config.tile_bytes`` (half of a core's L2 cache unless set).

Nothing here knows about dot products, and nothing is elaborated twice: the
body of the map is an IR block, and a tile is made of copies of it.
"""

from __future__ import annotations

import math

from ..syntax import ir, ops
from ..syntax.clone import Cloner, defined
from ..syntax.ir import Assign, Const, Expr, ForRange, If, IndexMap, IndexReduce, Scope, Var
from ..syntax.ops import ONE, ZERO, I, cmp, iadd, idiv, imax, imin, imul, iprod, isub
from .lowering import Loops, outer_expr, pure_defs

UNROLL_I = 2
UNROLL_J = 4
MIN_STATIC_SIZE = 256  # maps known to be smaller than this are not worth tiling

STRAIGHT = (ir.Assign, ir.Store, ir.Check, ir.Comment)


class Shape:
  """What :func:`match` found: the map is ``before; reduce; after``."""

  __slots__ = ("after", "before", "extent", "reduce")


def match(s: ir.Stmt, defs: dict[int, Expr]) -> Shape | None:
  """Is ``s`` a map whose element is one reduction and straight-line code,
  over an index space that does not depend on the element?"""
  if not isinstance(s, IndexMap) or s.ordered or len(s.shape) < 2 or s.scope.owned:
    return None
  if all(isinstance(d, Const) for d in s.shape) and math.prod(d.value for d in s.shape) < MIN_STATIC_SIZE:
    return None
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  where = [k for k, t in enumerate(body) if isinstance(t, IndexReduce)]
  if len(where) != 1 or not all(isinstance(t, STRAIGHT) for k, t in enumerate(body) if k != where[0]):
    return None
  red = body[where[0]]
  if red.where is not None:
    return None
  if any(isinstance(t.var.type, ir.BufType) for t in body if isinstance(t, Assign)):
    return None
  inner = set(defined(s.body)) | {v.uid for v in s.index}
  extent = [outer_expr(defs, d, inner) for d in red.shape]
  if any(d is None for d in extent):
    return None  # ragged: elements fold different numbers of terms
  m = Shape()
  m.before, m.reduce, m.after, m.extent = body[: where[0]], red, body[where[0] + 1 :], tuple(extent)
  return m


def element(lp: Loops, s: IndexMap, m: Shape, at: tuple[Expr, ...]) -> None:
  """One element of the map, as it was written."""
  c = Cloner({v.uid: e for v, e in zip(s.index, at)})
  lp.block.extend(c.block(m.before + [m.reduce] + m.after))


def jam(lp: Loops, s: IndexMap, m: Shape, ats: list[tuple[Expr, ...]]) -> None:
  """The elements at ``ats`` with their reductions fused into one."""
  red = m.reduce
  index = tuple(Var("k", I) for _ in red.index)
  scope = Scope("loop")
  body: list = []
  folds: list = []
  tails: list = []
  for at in ats:
    c = Cloner({v.uid: e for v, e in zip(s.index, at)})
    c.m.update({v.uid: k for v, k in zip(red.index, index)})
    for v in defined(m.before + [red] + m.after).values():
      c.var(v)
    lp.block.extend(c.stmt(t) for t in m.before)
    body.extend(c.stmt(t) for t in red.body)
    folds.extend(c.fold(f) for f in red.folds)
    scope.owned.extend(c.scope(red.scope).owned)
    tails.append([c.stmt(t) for t in m.after])
  lp.emit(IndexReduce(m.extent, index, body, folds, scope, red.home, False, red.loc))
  for tail in tails:
    lp.block.extend(tail)


def loop(lp: Loops, start: Expr, stop: Expr, fill, step: Expr = ONE, name: str = "i") -> None:
  v = Var(name, I)
  blk: list = []
  scope = Scope("loop")
  fill(lp.sub(blk, scope), v)
  lp.emit(ForRange(v, start, stop, step, blk, scope))


def tile_size(lp: Loops, length: Expr, itemsize: int, tile_bytes: int) -> Expr:
  """How many ``j`` rows of ``length`` elements make one tile."""
  budget = max(tile_bytes // max(itemsize, 1), 1)
  rows = idiv(Const(budget, I), imax(length, 1))
  whole = imul(idiv(rows, UNROLL_J), UNROLL_J)
  size = lp.tmp(imax(whole, UNROLL_J), "tile")
  if isinstance(size, Const):  # the printed IR says so either way
    v = Var("tile", I)
    lp.emit(Assign(v, size))
    size = v
  return size


def tiled(s: IndexMap, m: Shape, defs: dict[int, Expr], owner: Scope, tile_bytes: int) -> list:
  """The loops of ``s``, in tiles of its last dimension.  The dimensions
  before it are one long row index."""
  out: list = []
  lp = Loops(out, defs, owner)
  lead = tuple(lp.tmp(d, "n") for d in s.shape[:-1])
  n_j = lp.tmp(s.shape[-1], "n")
  n_i = lp.tmp(iprod(lead), "rows")
  length = lp.tmp(iprod(m.extent), "len")
  itemsize = m.reduce.folds[0].a.type.bits // 8
  tile = tile_size(lp, length, itemsize, tile_bytes)

  def at(inner: Loops, i: Expr, j: Expr) -> tuple[Expr, ...]:
    return ((inner.tmp(i, "i"),) if len(lead) == 1 else inner.unravel(inner.tmp(i, "i"), lead)) + (inner.tmp(j, "j"),)

  def single(inner: Loops, i: Expr, j: Expr) -> None:
    element(inner, s, m, at(inner, i, j))

  def span(inner: Loops, row: Expr, lo: Expr, hi: Expr) -> None:
    """Columns ``[lo, hi)`` of one row, an element at a time."""
    loop(inner, lo, hi, lambda body, j: single(body, row, j), name="j")

  def rows(inner: Loops, r0: Expr, r1: Expr) -> None:
    """Complete rows ``[r0, r1)``, a tile of columns at a time."""
    i_jam = inner.tmp(iadd(r0, imul(idiv(isub(r1, r0), UNROLL_I), UNROLL_I)), "i_jam")

    def tile_body(t: Loops, j0: Var) -> None:
      j1 = t.tmp(imin(n_j, iadd(j0, tile)), "j1")
      j_jam = t.tmp(iadd(j0, imul(idiv(isub(j1, j0), UNROLL_J), UNROLL_J)), "j_jam")

      def block_rows(b: Loops, i: Var) -> None:
        def block(e: Loops, j: Var) -> None:
          jam(e, s, m, [at(e, iadd(i, p), iadd(j, q)) for p in range(UNROLL_I) for q in range(UNROLL_J)])

        def leftover(e: Loops, j: Var) -> None:
          for p in range(UNROLL_I):
            single(e, iadd(i, p), j)

        loop(b, j0, j_jam, block, step=Const(UNROLL_J, I), name="j")
        loop(b, j_jam, j1, leftover, name="j")

      loop(t, r0, i_jam, block_rows, step=Const(UNROLL_I, I), name="i")
      loop(t, i_jam, r1, lambda b, i: span(b, i, j0, j1), name="i")

    loop(inner, ZERO, n_j, tile_body, step=tile, name="j0")

  def chunk(inner: Loops, _chunk, start: Expr, stop: Expr) -> None:
    """Flat positions ``[start, stop)``: the end of a row, whole rows, the
    beginning of a row."""
    r0 = inner.tmp(idiv(start, n_j), "r0")
    c0 = inner.tmp(isub(start, imul(r0, n_j)), "c0")
    lead_end = inner.tmp(ops.select(cmp("gt", c0, 0), imin(stop, imul(iadd(r0, 1), n_j)), start), "lead_end")
    span(inner, r0, c0, iadd(c0, isub(lead_end, start)))
    first_row = inner.tmp(ops.select(cmp("gt", c0, 0), iadd(r0, 1), r0), "first_row")
    last_row = inner.tmp(idiv(stop, n_j), "last_row")  # may be one past the end
    # a chunk inside a single row has no complete rows (and first_row > last_row)
    rows(inner, first_row, inner.tmp(imax(first_row, last_row), "rows_end"))
    tail_start = inner.tmp(imax(lead_end, imul(last_row, n_j)), "tail_start")
    span(inner, last_row, isub(tail_start, imul(last_row, n_j)), isub(stop, imul(last_row, n_j)))

  total = lp.tmp(imul(n_i, n_j), "total")
  work: list = []
  # where it cannot run in parallel it is still a loop of one chunk: the
  # backend compiles that as a function of its own
  lp.sub(work).parallel(total, chunk, nchunks=None if s.parallel else ONE, loc=s.loc, work=length, heavy=True)
  # an empty row would make the row arithmetic divide by zero; there is nothing to do then anyway
  lp.emit(If(cmp("gt", total, 0), work, []))
  return out


def tile_block(block: list, defs: dict[int, Expr], owner: Scope, tile_bytes: int) -> list:
  out: list = []
  for s in block:
    if isinstance(s, If):
      s.then[:] = tile_block(s.then, defs, owner, tile_bytes)
      s.orelse[:] = tile_block(s.orelse, defs, owner, tile_bytes)
    elif isinstance(s, IndexReduce):
      s.body[:] = tile_block(s.body, defs, s.scope, tile_bytes)
    elif isinstance(s, (IndexMap, ForRange, ir.While, ir.ParLoop)):
      s.body[:] = tile_block(s.body, defs, s.scope, tile_bytes)
    m = match(s, defs)
    if m is None:
      out.append(s)
    else:
      out.extend(tiled(s, m, defs, owner, tile_bytes))  # type: ignore[arg-type]
  return out


def tiling(fn: ir.Function, options) -> None:
  """TiledMap: tile and jam every matmul-shaped map of ``fn``."""
  if options.tiling:
    fn.body[:] = tile_block(fn.body, pure_defs(fn.body), fn.scope, options.tile_bytes)


__all__ = ["UNROLL_I", "UNROLL_J", "jam", "match", "tile_size", "tiled", "tiling"]
