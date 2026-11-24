"""Simplification at the operator level.

The C compiler removes dead scalar code well enough, but it will not remove
a parallel loop whose result nobody reads: by the time it sees one, the loop
is a call through a function pointer.  So dead operators are removed here,
where they are still statements with visible results:

*dead folds*
    a fold of an ``IndexReduce`` whose result is never read is dropped; a
    reduction left without folds is dropped whole.
*dead maps, scans, filters and allocations*
    a pure ``IndexMap`` or ``IndexScan`` that only writes to freshly allocated
    buffers nobody reads (or returns), an ``IndexFilter`` whose output and
    count are never read, and allocations left with neither readers nor
    writers are all removed.

Anything that could raise (a fold without an ``init``, a ``Check``, a write
to an input array, an impure runtime call) is kept.
"""

from __future__ import annotations

from ..syntax import ir
from ..syntax.ir import Expr, IndexFilter, IndexMap, IndexReduce, IndexScan, IndexScatter, Var

EFFECTS = (
  ir.Store,
  ir.Check,
  ir.Alloc,
  ir.Incref,
  ir.Release,
  ir.IndexMap,
  ir.IndexFilter,
  ir.IndexScan,
  ir.IndexScatter,
  ir.ParLoop,
  ir.While,
)
NON_STORE_EFFECTS = tuple(t for t in EFFECTS if t is not ir.Store)


def _note_exprs(e, out: set[int]) -> None:
  if isinstance(e, Expr):
    out.update(x.uid for x in ir.walk_expr(e) if isinstance(x, Var))
  elif isinstance(e, (tuple, list)):
    for x in e:
      _note_exprs(x, out)


def read_vars(fn: ir.Function) -> set[int]:
  """Every variable some statement or result of ``fn`` reads."""
  out: set[int] = set()
  for s in ir.walk_stmts(fn.body):
    _note_exprs(ir.stmt_exprs(s), out)
  for r in fn.results:
    if isinstance(r, Expr):
      _note_exprs(r, out)
    else:
      for name in getattr(type(r), "__dataclass_fields__", ()):
        _note_exprs(getattr(r, name), out)
  return out


def buffer_liveness(fn: ir.Function) -> tuple[set[int], set[int], set[int]]:
  """Return ``(local_allocs, read_bufs, stored_bufs)`` across ``fn``.

  ``read_bufs`` collects every buffer variable that is loaded, returned,
  reference-counted, assigned to another variable, or passed to a runtime call
  — i.e. anything other than being allocated, stored into, or released."""
  allocs: set[int] = set()
  read_bufs: set[int] = set()
  stored_bufs: set[int] = set()
  for s in ir.walk_stmts(fn.body):
    if isinstance(s, ir.Alloc):
      allocs.add(s.var.uid)
    elif isinstance(s, ir.Store):
      stored_bufs.add(s.buf.uid)
      _note_exprs(s.index, read_bufs)
      _note_exprs(s.value, read_bufs)
    elif isinstance(s, IndexScatter):
      stored_bufs.add(s.buf.uid)
      for e in (*s.shape, s.dest_size, s.target, s.value, *(() if s.where is None else (s.where,))):
        _note_exprs(e, read_bufs)
    else:
      _note_exprs(ir.stmt_exprs(s), read_bufs)
  for r in fn.results:
    if isinstance(r, Expr):
      _note_exprs(r, read_bufs)
    else:
      for name in getattr(type(r), "__dataclass_fields__", ()):
        _note_exprs(getattr(r, name), read_bufs)
  return allocs, read_bufs, stored_bufs


def has_effects(block: list) -> bool:
  """True if ``block`` performs any observable effect (stores, checks,
  allocations, nested operators/loops, or impure runtime calls)."""
  return any(
    isinstance(s, EFFECTS)
    or (isinstance(s, ir.Assign) and any(isinstance(x, ir.RtCall) and not x.pure for x in ir.walk_expr(s.value)))
    for s in ir.walk_stmts(block)
  )


def pure_writes_only(block: list, dead_bufs: set[int]) -> bool:
  """True if ``block`` has no observable effects other than stores into ``dead_bufs``."""
  for s in ir.walk_stmts(block):
    if isinstance(s, ir.Store):
      if s.buf.uid not in dead_bufs:
        return False
    elif isinstance(s, NON_STORE_EFFECTS):
      return False
    elif isinstance(s, IndexReduce):
      if any(f.init is None or has_effects(f.combine) for f in s.folds):
        return False
    elif isinstance(s, ir.Assign):
      if any(isinstance(x, ir.RtCall) and not x.pure for x in ir.walk_expr(s.value)):
        return False
  return True


def drop_buffers(fn: ir.Function, gone: set[int], drop_allocs: bool = False) -> None:
  """Drop scope ownership, explicit releases (and optionally allocations) for
  eliminated temporary buffers."""
  if not gone:
    return
  fn.scope.owned[:] = [v for v in fn.scope.owned if v.uid not in gone]
  for s in ir.walk_stmts(fn.body):
    sc = getattr(s, "scope", None)
    if sc is not None:
      sc.owned[:] = [v for v in sc.owned if v.uid not in gone]

  def drop_stmts(block: list) -> None:
    block[:] = [
      s
      for s in block
      if not (
        (isinstance(s, ir.Release) and s.var.uid in gone)
        or (drop_allocs and isinstance(s, ir.Alloc) and s.var.uid in gone)
      )
    ]
    for s in block:
      for b in ir.sub_blocks(s):
        drop_stmts(b)

  drop_stmts(fn.body)


def sweep(block: list, live: set[int], dead_bufs: set[int], stored_bufs: set[int], gone: set[int]) -> tuple[list, bool]:
  """Remove dead operators and unused allocations from ``block`` in one pass."""
  out: list = []
  changed = False
  for s in block:
    for b in ir.sub_blocks(s):
      b[:], inner = sweep(b, live, dead_bufs, stored_bufs, gone)
      changed |= inner
    if isinstance(s, IndexReduce):
      keep = [
        f
        for f in s.folds
        if f.result.uid in live
        or any(c.result.uid in live for c in f.carries)
        or f.init is None
        or has_effects(f.combine)
      ]
      if not keep and not has_effects(s.body) and not has_effects(s.pre):
        changed = True
        continue
      if keep and len(keep) < len(s.folds):
        s.folds[:] = keep
        changed = True
    elif isinstance(s, IndexMap):
      if not s.scope.owned and pure_writes_only(s.body, dead_bufs):
        changed = True
        continue
    elif isinstance(s, IndexFilter):
      if s.out.uid not in live and s.count.uid not in live and not has_effects(s.body):
        gone.add(s.out.uid)
        changed = True
        continue
    elif isinstance(s, IndexScan):
      if (
        not s.scope.owned
        and not has_effects(s.body)
        and not has_effects(s.fold.combine)
        and pure_writes_only(s.emit, dead_bufs)
      ):
        changed = True
        continue
    elif isinstance(s, IndexScatter):
      if not s.scope.owned and s.buf.uid in dead_bufs and pure_writes_only(s.body, dead_bufs):
        changed = True
        continue
    elif isinstance(s, ir.Alloc):
      if s.var.uid in dead_bufs and s.var.uid not in stored_bufs:
        gone.add(s.var.uid)
        changed = True
        continue
    out.append(s)
  return out, changed


def dead_code(fn: ir.Function, options) -> None:
  """Remove dead operators and allocations nobody reads."""
  gone: set[int] = set()
  for _ in range(8):  # removing one dead operator can make its producers dead too
    live = read_vars(fn)
    allocs, read_bufs, stored_bufs = buffer_liveness(fn)
    dead_bufs = allocs - read_bufs
    fn.body[:], changed = sweep(fn.body, live, dead_bufs, stored_bufs, gone)
    if gone:
      drop_buffers(fn, gone)
      gone.clear()
    if not changed:
      break


__all__ = ["buffer_liveness", "dead_code", "drop_buffers", "has_effects", "pure_writes_only", "read_vars", "sweep"]
