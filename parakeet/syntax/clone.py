"""Copies of IR blocks with fresh variables.

A pass that needs the body of an operator more than once (the first element
of a fold, the neighbouring elements of a tile) instantiates it: variables
the block defines get new names, variables named in ``subst`` are replaced
by the given expressions, everything else is shared.
"""

from __future__ import annotations

from . import ir
from .ir import Cast, Const, Expr, Load, Null, Prim, RtCall, Var


def subst_expr(e: Expr, m: dict[int, Expr]) -> Expr:
  """``e`` with variables replaced according to ``m`` (uid -> expression)."""
  if isinstance(e, Var):
    return m.get(e.uid, e)
  if isinstance(e, (Const, Null)):
    return e
  if isinstance(e, Prim):
    args = tuple(subst_expr(a, m) for a in e.args)
    return e if all(x is y for x, y in zip(args, e.args)) else Prim(e.op, args, e.type)
  if isinstance(e, Cast):
    arg = subst_expr(e.arg, m)
    return e if arg is e.arg else Cast(arg, e.type)
  if isinstance(e, Load):
    buf, index = subst_expr(e.buf, m), subst_expr(e.index, m)
    return e if buf is e.buf and index is e.index else Load(buf, index)  # type: ignore[arg-type]
  if isinstance(e, RtCall):
    args = tuple(subst_expr(a, m) for a in e.args)
    return e if all(x is y for x, y in zip(args, e.args)) else RtCall(e.name, args, e.type, e.pure)
  return e


def defined(block: list[ir.Stmt]) -> dict[int, Var]:
  """Every variable a block (and the blocks inside it) assigns or binds."""
  out: dict[int, Var] = {}
  for s in ir.walk_stmts(block):
    for v in ir.stmt_defs(s):
      out[v.uid] = v
  return out


class Cloner:
  def __init__(self, subst: dict[int, Expr] | None = None, home: ir.Scope | None = None):
    self.m: dict[int, Expr] = dict(subst or {})
    self.scopes: dict[int, ir.Scope] = {}
    self.home = home

  def var(self, v: Var) -> Var:
    """The copy of a variable the block defines."""
    hit = self.m.get(v.uid)
    if hit is None:
      hit = self.m[v.uid] = Var(v.name.rsplit("_", 1)[0], v.type)
    return hit  # type: ignore[return-value]

  def expr(self, e: Expr) -> Expr:
    return subst_expr(e, self.m)

  def scope(self, sc: ir.Scope) -> ir.Scope:
    hit = self.scopes.get(id(sc))
    if hit is None:
      hit = self.scopes[id(sc)] = ir.Scope(sc.kind, sc.parent)
      hit.owned = [self.var(v) for v in sc.owned]
    return hit

  def block(self, block: list[ir.Stmt]) -> list[ir.Stmt]:
    for v in defined(block).values():
      self.var(v)
    return [self.stmt(s) for s in block]

  def fold(self, f: ir.Fold) -> ir.Fold:
    return ir.Fold(
      self.var(f.result), self.expr(f.value), self.var(f.a), self.var(f.b), [self.stmt(s) for s in f.combine],
      self.expr(f.combined), None if f.init is None else self.expr(f.init), f.op, f.empty,
    )  # fmt: skip

  def stmt(self, s: ir.Stmt) -> ir.Stmt:
    x, b = self.expr, lambda blk: [self.stmt(t) for t in blk]
    if isinstance(s, ir.Assign):
      return ir.Assign(self.var(s.var), x(s.value))
    if isinstance(s, ir.Store):
      return ir.Store(x(s.buf), x(s.index), x(s.value))  # type: ignore[arg-type]
    if isinstance(s, ir.If):
      return ir.If(x(s.cond), b(s.then), b(s.orelse))
    if isinstance(s, ir.ForRange):
      red = s.reduction
      if red is not None:
        accs = red[1] if isinstance(red[1], (list, tuple)) else [red[1]]
        red = (red[0], [x(a) for a in accs])
      return ir.ForRange(self.var(s.var), x(s.start), x(s.stop), x(s.step), b(s.body), self.scope(s.scope), s.simd, red)
    if isinstance(s, ir.While):
      return ir.While(b(s.body), self.scope(s.scope))
    if isinstance(s, (ir.Break, ir.Continue, ir.Comment)):
      return s
    if isinstance(s, ir.Alloc):
      return ir.Alloc(self.var(s.var), x(s.count), s.zero)
    if isinstance(s, ir.Incref):
      return ir.Incref(x(s.buf))
    if isinstance(s, ir.Release):
      return ir.Release(x(s.var))  # type: ignore[arg-type]
    if isinstance(s, ir.ReleaseScope):
      return ir.ReleaseScope(self.scope(s.scope))
    if isinstance(s, ir.Check):
      return ir.Check(x(s.cond), s.err, tuple(x(a) for a in s.args))
    if isinstance(s, ir.ParLoop):
      return ir.ParLoop(
        x(s.total), x(s.nchunks), self.var(s.chunk), self.var(s.start), self.var(s.stop), b(s.body),
        self.scope(s.scope), s.kind, s.loc, s.heavy, None if s.work is None else x(s.work),
      )  # fmt: skip
    if isinstance(s, ir.IndexMap):
      return ir.IndexMap(
        tuple(x(d) for d in s.shape), tuple(self.var(v) for v in s.index), b(s.body), self.scope(s.scope),
        s.ordered, s.parallel, s.loc,
      )  # fmt: skip
    if isinstance(s, ir.IndexReduce):
      return ir.IndexReduce(
        tuple(x(d) for d in s.shape), tuple(self.var(v) for v in s.index), b(s.body), [self.fold(f) for f in s.folds],
        self.scope(s.scope), self.home or s.home, s.parallel, s.loc, b(s.pre), None if s.where is None else x(s.where),
      )  # fmt: skip
    if isinstance(s, ir.IndexFilter):
      return ir.IndexFilter(
        x(s.n), self.var(s.index), b(s.body), x(s.keep), self.var(s.out), self.var(s.count), self.scope(s.scope),
        s.parallel, s.loc,
      )  # fmt: skip
    raise TypeError(f"cannot clone {type(s).__name__}")


def clone_block(block: list[ir.Stmt], subst: dict[int, Expr] | None = None) -> tuple[list[ir.Stmt], Cloner]:
  """A copy of ``block`` with fresh names for what it defines."""
  c = Cloner(subst)
  return c.block(block), c


__all__ = ["Cloner", "clone_block", "defined", "subst_expr"]
