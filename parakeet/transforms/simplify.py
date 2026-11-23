"""Simplification at the operator level.

The C compiler removes dead scalar code well enough, but it will not remove
a parallel loop whose result nobody reads: by the time it sees one, the loop
is a call through a function pointer.  So dead operators are removed here,
where they are still statements with visible results.

*dead folds*
    a fold of an ``IndexReduce`` whose result is never read is dropped; a
    reduction left without folds is dropped whole.

A fold that could raise is kept: one that starts from the first element
(``min`` of an empty array is an error), and any reduction whose body checks
something or writes to an array.
"""

from __future__ import annotations

from ..syntax import ir
from ..syntax.ir import Expr, IndexReduce, Var

EFFECTS = (
  ir.Store,
  ir.Check,
  ir.Alloc,
  ir.Incref,
  ir.Release,
  ir.IndexMap,
  ir.IndexFilter,
  ir.IndexScan,
  ir.ParLoop,
  ir.While,
)


def read_vars(fn: ir.Function) -> set[int]:
  """Every variable some statement or result of ``fn`` reads."""
  out: set[int] = set()

  def note(e) -> None:
    if isinstance(e, Expr):
      out.update(x.uid for x in ir.walk_expr(e) if isinstance(x, Var))
    elif isinstance(e, (tuple, list)):
      for x in e:
        note(x)

  for s in ir.walk_stmts(fn.body):
    note(ir.stmt_exprs(s))
  for r in fn.results:
    if isinstance(r, Expr):
      note(r)
    else:
      for name in getattr(type(r), "__dataclass_fields__", ()):
        note(getattr(r, name))
  return out


def has_effects(block: list) -> bool:
  return any(
    isinstance(s, EFFECTS)
    or (isinstance(s, ir.Assign) and any(isinstance(x, ir.RtCall) and not x.pure for x in ir.walk_expr(s.value)))
    for s in ir.walk_stmts(block)
  )


def sweep(block: list, live: set[int]) -> tuple[list, bool]:
  out: list = []
  changed = False
  for s in block:
    for b in ir.sub_blocks(s):
      b[:], inner = sweep(b, live)
      changed |= inner
    if isinstance(s, IndexReduce):
      keep = [f for f in s.folds if f.result.uid in live or f.init is None or has_effects(f.combine)]
      if not keep and not has_effects(s.body) and not has_effects(s.pre):
        changed = True
        continue
      if keep and len(keep) < len(s.folds):
        s.folds[:] = keep
        changed = True
    out.append(s)
  return out, changed


def dead_code(fn: ir.Function, options) -> None:
  """Remove reductions nobody reads."""
  for _ in range(8):  # a dead reduction may have been the only reader of another
    fn.body[:], changed = sweep(fn.body, read_vars(fn))
    if not changed:
      break


__all__ = ["dead_code", "read_vars"]
