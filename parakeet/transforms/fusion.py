"""Fusion rules that are passes over the operator statements.

Vertical fusion (a map feeding a map, a map feeding a reduction) still
happens as array expressions are elaborated: an array that is not needed in
memory never becomes a statement at all.  What is here is the rule that
cannot be done that way, because it joins two statements neither of which
feeds the other, and the rule for a filter, whose result has to exist before
anyone can know how long it is.

*sibling fusion* (horizontal fusion)::

    a = reduce i over S: fold(f, x(i))        r = reduce i over S:
    b = reduce i over S: fold(g, y(i))   =>     fold(f, x(i)), fold(g, y(i))

Two reductions over the same index space, the second not using the result
of the first, become one loop with two accumulators: ``x.min(), x.max()``
reads ``x`` once.  Tiling uses the same rule on copies of one reduction
(see :func:`parakeet.transforms.tiling.jam`).

*FilterReduce*::

    idx, n = filter i over S: keep(i, where=p(i))
    r = reduce k over n: fold(f, x(idx[k]))        =>
    r = reduce i over S where p(i): fold(f, x(i))

A reduction over what a filter kept, when nothing else looks at the kept
indices, folds the survivors as they are found: no index array, no counting
pass.  ``np.sum(x[x > 0])`` is one loop.  If something else needs the number
of survivors (``np.mean``), that is one more fold.
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
  if len(a.shape) != len(b.shape) or a.parallel != b.parallel or a.where is not None or b.where is not None:
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


# ---------------------------------------------------------------------------
# FilterReduce
# ---------------------------------------------------------------------------


def strip(e: Expr) -> Expr:
  while isinstance(e, ir.Cast):
    e = e.arg
  return e


class Survivors(Cloner):
  """Rewrites a reduction over a filter's output into terms of the filter's
  own index: ``idx[k]`` is ``i``, and ``i`` is known to be in ``[0, n)``."""

  def __init__(self, f: ir.IndexFilter, r: IndexReduce, defs: dict[int, Expr]):
    super().__init__({v.uid: v for v in defined([r]).values()})
    self.f, self.k, self.defs = f, r.index[0], defs
    self.inrange: set[int] = {f.index.uid}  # variables equal to the filter's index
    self.ok = True

  def expr(self, e: Expr) -> Expr:
    if isinstance(e, ir.Load) and e.buf is self.f.out:
      if e.index is not self.k:
        self.ok = False
      return ir.Cast(self.f.index, e.type) if e.type is not self.f.index.type else self.f.index
    if isinstance(e, Var):
      if e is self.k or e is self.f.out:
        self.ok = False  # the position among the survivors is not known any more
      return e
    if isinstance(e, ir.Prim):
      args = tuple(self.expr(a) for a in e.args)
      if e.op == "select" and self.is_negative_test(args[0], args[2]):
        return args[2]  # wrapping a negative index: this one is not
      return ir.Prim(e.op, args, e.type)
    if isinstance(e, ir.Cast):
      return ir.Cast(self.expr(e.arg), e.type)
    if isinstance(e, ir.Load):
      return ir.Load(e.buf, self.expr(e.index))
    if isinstance(e, ir.RtCall):
      return ir.RtCall(e.name, tuple(self.expr(a) for a in e.args), e.type, e.pure)
    return e

  def known(self, e: Expr) -> bool:
    e = strip(e)
    return isinstance(e, Var) and e.uid in self.inrange

  def is_negative_test(self, c: Expr, otherwise: Expr) -> bool:
    return (
      isinstance(c, ir.Prim)
      and c.op == "lt"
      and self.known(c.args[0])
      and isinstance(c.args[1], Const)
      and c.args[1].value == 0
      and self.known(otherwise)
    )

  def proven(self, c: Expr) -> bool:
    """Is ``c`` the bounds check ``0 <= i < n`` of the filter's own index?"""
    if not (isinstance(c, ir.Prim) and c.op == "and"):
      return False
    lo, hi = c.args
    return (
      isinstance(lo, ir.Prim) and lo.op == "ge" and self.known(lo.args[0])
      and isinstance(hi, ir.Prim) and hi.op == "lt" and self.known(hi.args[0])
      and key(resolved(hi.args[1], self.defs)) == key(resolved(self.f.n, self.defs))
    )  # fmt: skip

  def body(self, block: list) -> list:
    out: list = []
    for s in block:
      t = self.stmt(s)
      if isinstance(t, Assign) and self.known(t.value):
        self.inrange.add(t.var.uid)
      if isinstance(t, ir.Check) and self.proven(t.cond):
        continue
      out.append(t)
    return out


def filter_reduce(f: ir.IndexFilter, r: IndexReduce, readers: dict[int, int], defs: dict[int, Expr]):
  """The fused statement, or None if the rule does not apply."""
  from .lowering import identity

  if len(r.shape) != 1 or r.where is not None or any(g.init is None for g in r.folds):
    return None
  if r.shape[0] is not f.count:
    return None
  c = Survivors(f, r, defs)
  body = c.body(r.body)
  folds = [c.fold(g) for g in r.folds]
  if not c.ok:
    return None
  # nothing but this reduction may have looked at the kept indices
  exprs = list(ir.stmt_exprs(r)) + [e for b in ir.sub_blocks(r) for s in ir.walk_stmts(b) for e in ir.stmt_exprs(s)]
  inside = sum(1 for e in exprs for x in ir.walk_expr(e) if x is f.out)
  if readers.get(f.out.uid, 0) != inside:
    return None
  if readers.get(f.count.uid, 0) > 1:  # someone else wants to know how many survived
    a, b = Var("acc", f.count.type), Var("x", f.count.type)
    folds.append(
      ir.Fold(
        f.count, Const(1, f.count.type), a, b, [], ir.Prim("add", (a, b), f.count.type), Const(0, f.count.type), "add"
      )
    )
  parallel = f.parallel and all(
    g.op is not None and not g.combine and identity(g.op, g.a.type) is not None for g in folds
  )
  scope = c.scope(r.scope)
  scope.owned.extend(f.scope.owned)
  return IndexReduce((f.n,), (f.index,), body, folds, scope, r.home, parallel, r.loc, list(f.body), f.keep)


def count_readers(fn: ir.Function) -> dict[int, int]:
  out: dict[int, int] = {}

  def note(e) -> None:
    if isinstance(e, Expr):
      for x in ir.walk_expr(e):
        if isinstance(x, Var):
          out[x.uid] = out.get(x.uid, 0) + 1
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


def forget(fn: ir.Function, gone: set[int]) -> None:
  """Buffers that are no longer allocated are no longer anyone's to release."""

  def scopes():
    yield fn.scope
    for s in ir.walk_stmts(fn.body):
      sc = getattr(s, "scope", None)
      if sc is not None:
        yield sc

  for sc in scopes():
    sc.owned[:] = [v for v in sc.owned if v.uid not in gone]

  def sweep(block: list) -> None:
    block[:] = [s for s in block if not (isinstance(s, ir.Release) and s.var.uid in gone)]
    for s in block:
      for b in ir.sub_blocks(s):
        sweep(b)

  sweep(fn.body)


def filter_block(block: list, readers: dict[int, int], defs: dict[int, Expr], gone: set[int]) -> list:
  for s in block:
    for b in ir.sub_blocks(s):
      b[:] = filter_block(b, readers, defs, gone)
  out: list = []
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    nxt = block[k] if k < len(block) else None
    if isinstance(s, ir.IndexFilter) and isinstance(nxt, IndexReduce):
      fused = filter_reduce(s, nxt, readers, defs)
      if fused is not None:
        out.append(fused)
        gone.add(s.out.uid)
        k += 1
        continue
    out.append(s)
  return out


def fuse_filters(fn: ir.Function, options) -> None:
  """FilterReduce: reductions over what a filter kept."""
  gone: set[int] = set()
  fn.body[:] = filter_block(fn.body, count_readers(fn), pure_defs(fn.body), gone)
  if gone:
    forget(fn, gone)


def fuse_siblings(fn: ir.Function, options) -> None:
  """Sibling fusion of the reductions of ``fn``."""
  fn.body[:] = fuse_block(fn.body, pure_defs(fn.body))


__all__ = ["fuse_filters", "fuse_siblings"]
