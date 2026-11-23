"""Fusion rules that are passes over the operator statements.

Vertical fusion (a map feeding a map, a map feeding a reduction) still
happens as array expressions are elaborated: an array that is not needed in
memory never becomes a statement at all.  What is here is the rule that
cannot be done that way, because it joins two statements neither of which
feeds the other.

*sibling fusion* (horizontal fusion)::

    a = reduce i over S: fold(f, x(i))        r = reduce i over S:
    b = reduce i over S: fold(g, y(i))   =>     fold(f, x(i)), fold(g, y(i))

Two reductions over the same index space, the second not using the result
of the first, become one loop with two accumulators: ``x.min(), x.max()``
reads ``x`` once.  Tiling uses the same rule on copies of one reduction
(see :func:`parakeet.transforms.tiling.jam`).
"""

from __future__ import annotations

from ..syntax import ir
from ..syntax.clone import Cloner, defined
from ..syntax.ir import Assign, Const, Expr, IndexReduce, Var
from .lowering import pure_defs


def resolved(e: Expr, defs: dict[int, Expr], budget: int = 20) -> Expr:
  """``e`` with single-assignment integer variables replaced by their definitions."""
  if isinstance(e, Var):
    d = defs.get(e.uid)
    return e if d is None or budget <= 0 else resolved(d, defs, budget - 1)
  if isinstance(e, ir.Prim):
    return ir.Prim(e.op, tuple(resolved(a, defs, budget - 1) for a in e.args), e.type)
  if isinstance(e, ir.Cast):
    return ir.Cast(resolved(e.arg, defs, budget - 1), e.type)
  return e


def key(e: Expr) -> object:
  """Structural identity of an expression."""
  if isinstance(e, Var):
    return ("v", e.uid)
  if isinstance(e, Const):
    return ("c", e.value, e.type)
  if isinstance(e, ir.Prim):
    return ("p", e.op, tuple(key(a) for a in e.args))
  if isinstance(e, ir.Cast):
    return ("t", e.type, key(e.arg))
  return ("?", id(e))


def uses(stmts: list, uids: set[int]) -> bool:
  for s in ir.walk_stmts(stmts):
    for e in ir.stmt_exprs(s):
      if any(isinstance(x, Var) and x.uid in uids for x in ir.walk_expr(e)):
        return True
    if isinstance(s, ir.Store) and s.buf.uid in uids:
      return True
  return False


def pure(e: Expr) -> bool:
  return not any(isinstance(x, ir.RtCall) and not x.pure for x in ir.walk_expr(e))


def simd_op(r: IndexReduce) -> str | None:
  """The operation a backend may vectorise the fold by, if there is one."""
  op = r.folds[0].op
  return op if op in ("add", "mul") and all(f.op == op and not f.combine for f in r.folds) else None


def compatible(a: IndexReduce, b: IndexReduce, defs: dict[int, Expr]) -> bool:
  if len(a.shape) != len(b.shape) or a.parallel != b.parallel:
    return False
  if (a.folds[0].init is None) != (b.folds[0].init is None):
    return False
  # do not trade a vectorised loop for a fused one that is not
  if simd_op(a) != simd_op(b):
    return False
  return all(key(resolved(x, defs)) == key(resolved(y, defs)) for x, y in zip(a.shape, b.shape))


def absorb(a: IndexReduce, b: IndexReduce) -> None:
  """Move the folds of ``b`` into ``a``."""
  c = Cloner({v.uid: v for v in defined([b]).values()})  # same names...
  c.m.update({v.uid: i for v, i in zip(b.index, a.index)})  # ...but the index of ``a``
  a.body.extend(c.stmt(t) for t in b.body)
  a.folds.extend(c.fold(f) for f in b.folds)
  a.scope.owned.extend(c.scope(b.scope).owned)


def fuse_block(block: list, defs: dict[int, Expr]) -> list:
  for s in block:
    for b in ir.sub_blocks(s):
      b[:] = fuse_block(b, defs)
  out: list = []
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    if not isinstance(s, IndexReduce):
      out.append(s)
      continue
    # ``s`` may sink past scalar definitions that do not need its result,
    # to meet the next reduction
    while True:
      results = {f.result.uid for f in s.folds}
      between: list = []
      j = k
      while j < len(block) and isinstance(block[j], (Assign, ir.Comment)):
        t = block[j]
        if isinstance(t, Assign) and (isinstance(t.var.type, ir.BufType) or not pure(t.value) or uses([t], results)):
          break
        between.append(t)
        j += 1
      nxt = block[j] if j < len(block) else None
      if not isinstance(nxt, IndexReduce) or uses([nxt], results) or not compatible(s, nxt, defs):
        break
      if any(f.init is not None and uses([Assign(f.result, f.init)], results) for f in nxt.folds):
        break
      out.extend(between)
      absorb(s, nxt)
      k = j + 1
    out.append(s)
  return out


def fuse_siblings(fn: ir.Function, options) -> None:
  """Sibling fusion of the reductions of ``fn``."""
  fn.body[:] = fuse_block(fn.body, pure_defs(fn.body))


__all__ = ["fuse_siblings"]
