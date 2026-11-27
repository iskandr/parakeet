"""Unified middle-end dataflow, affine index, alias, and cost analyses.

Provides three compact, composable abstractions shared across normalization,
fusion, tiling, and lowering—replacing duplicated bounded tree-walkers with
one canonical representation:

* :class:`AffineExpr` & :class:`IndexFacts`: scoped union-find for proven
  scalar equalities (from ``Assign`` and ``Check(a == b)``) plus canonical
  multivariate polynomial normal form for integer index expressions.
  Handles ``i * n + j == j + n * i``, ``x + 0 == x``, stride queries,
  loop-invariant lifting, and provably-in-bounds row-major coordinate
  decomposition.
* :class:`BufferInfo`: single-pass buffer root-allocation, read-only param,
  escape (``fn.results`` / ``Incref``), reader-count, and alias analysis.
* :func:`op_cost`: lightweight middle-end cost model for deciding when
  multi-consumer fusion of a cheap producer is profitable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..shape_lattice import TOP as DIM_TOP
from ..shape_lattice import DimInfo
from ..syntax import ir, ops
from ..syntax.ir import Assign, Const, Expr, Load, Prim, Var

# ---------------------------------------------------------------------------
# Canonical multivariate polynomial / affine form for integer expressions
# ---------------------------------------------------------------------------

# A Factor is either ("v", canonical_uid) or ("o", opaque_expr_key).
# A Monomial is a sorted tuple of Factors (empty tuple () == constant 1).
Factor = tuple
Monomial = tuple[Factor, ...]

CHEAP_FUSE_COST = 8  # matches frontend CHEAP_COST intuition: arith/loads yes, exp/sin/loops no
LOOP_COST = 10_000

_EXPENSIVE_PRIMS = frozenset(
  {
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "cbrt",
    "pow",
    "erf",
    "hypot",
    "fmod",
  }
)


@dataclass(frozen=True, slots=True)
class AffineExpr:
  """Canonical polynomial normal form ``sum(coeff * prod(factors))`` over integer
  expressions (with nonlinear sub-expressions represented as opaque factors)."""

  terms: tuple[tuple[Monomial, int], ...]

  @staticmethod
  def const(v: int) -> AffineExpr:
    return AffineExpr((((), v),) if v else ())

  @staticmethod
  def factor(f: Factor) -> AffineExpr:
    return AffineExpr((((f,), 1),))

  @staticmethod
  def from_dict(d: dict[Monomial, int]) -> AffineExpr:
    return AffineExpr(tuple(sorted((m, c) for m, c in d.items() if c != 0)))

  def as_dict(self) -> dict[Monomial, int]:
    return dict(self.terms)

  def is_const(self) -> int | None:
    if not self.terms:
      return 0
    if len(self.terms) == 1 and self.terms[0][0] == ():
      return self.terms[0][1]
    return None

  def add(self, other: AffineExpr, sign: int = 1) -> AffineExpr:
    d = self.as_dict()
    for m, c in other.terms:
      nc = d.get(m, 0) + sign * c
      if nc:
        d[m] = nc
      else:
        d.pop(m, None)
    return AffineExpr.from_dict(d)

  def neg(self) -> AffineExpr:
    return AffineExpr(tuple((m, -c) for m, c in self.terms))

  def mul(self, other: AffineExpr) -> AffineExpr:
    if not self.terms or not other.terms:
      return AffineExpr(())
    d: dict[Monomial, int] = {}
    for m1, c1 in self.terms:
      for m2, c2 in other.terms:
        m = tuple(sorted(m1 + m2))
        nc = d.get(m, 0) + c1 * c2
        if nc:
          d[m] = nc
        else:
          d.pop(m, None)
    return AffineExpr.from_dict(d)


# ---------------------------------------------------------------------------
# IndexFacts: scoped equalities, bounds, strides, and row-major decomposition
# ---------------------------------------------------------------------------


class IndexFacts:
  """Scoped symbolic facts over integer expressions and checked equalities.

  Fork with :meth:`fork` when entering a nested block or branch so local
  definitions and checked equalities never leak across sibling branches.
  """

  def __init__(
    self,
    defs: dict[int, Expr] | None = None,
    vars_by_uid: dict[int, Var] | None = None,
    bounds: dict[int, Expr] | None = None,
    guarded_eqs: list[tuple[frozenset[object], Expr, Expr]] | None = None,
    expr_eqs: set[tuple[object, object]] | None = None,
    dim_facts: dict[int, tuple[DimInfo, frozenset[tuple]]] | None = None,
    used_guards: set[tuple] | None = None,
  ) -> None:
    self.defs: dict[int, Expr] = {} if defs is None else dict(defs)
    self.vars_by_uid: dict[int, Var] = {} if vars_by_uid is None else dict(vars_by_uid)
    self.bounds: dict[int, Expr] = {} if bounds is None else dict(bounds)
    self.guarded_eqs: list[tuple[frozenset[object], Expr, Expr]] = [] if guarded_eqs is None else list(guarded_eqs)
    self.expr_eqs: set[tuple[object, object]] = set() if expr_eqs is None else set(expr_eqs)
    self.dim_facts: dict[int, tuple[DimInfo, frozenset[tuple]]] = {} if dim_facts is None else dict(dim_facts)
    self.used_guards: set[tuple] = set() if used_guards is None else used_guards
    self._poly_cache: dict[int, AffineExpr] = {}

  @classmethod
  def from_function(cls, fn: ir.Function) -> IndexFacts:
    """Seed facts from single-assignment integer definitions and array shape
    buckets in ``fn``."""
    counts: dict[int, int] = {}
    values: dict[int, Expr] = {}
    vmap: dict[int, Var] = {}
    dfacts: dict[int, tuple[DimInfo, frozenset[tuple]]] = {}
    for p in fn.params:
      if isinstance(p, Var):
        vmap[p.uid] = p
      else:
        vmap[p.offset.uid] = p.offset
        if p.strides is None:
          # C-contiguous array parameter always has offset == 0 (already in type_key)
          dfacts[p.offset.uid] = (DimInfo(0, 0, 0), frozenset())
        for d in p.shape:
          vmap[d.uid] = d
          entry = fn.dim_facts.get(d.uid)
          if entry is not None:
            leaf_idx, axis, info = entry
            dfacts[d.uid] = (info, frozenset({(leaf_idx, axis, info)}))
        if p.strides:
          for st in p.strides:
            vmap[st.uid] = st
    for s in ir.walk_stmts(fn.body):
      for v in ir.stmt_defs(s):
        vmap[v.uid] = v
      if isinstance(s, Assign) and not isinstance(s.var.type, ir.BufType) and s.var.type.kind in "iu":
        uid = s.var.uid
        counts[uid] = counts.get(uid, 0) + 1
        values[uid] = s.value
    pure = {u: values[u] for u, c in counts.items() if c == 1 and is_pure_expr(values[u])}
    facts = cls(pure, vmap, dim_facts=dfacts, used_guards=fn.dim_guards)
    for s in fn.body:
      facts.record_stmt(s)
    return facts

  def fork(self) -> IndexFacts:
    return IndexFacts(
      self.defs,
      self.vars_by_uid,
      self.bounds,
      self.guarded_eqs,
      self.expr_eqs,
      self.dim_facts,
      self.used_guards,
    )

  def has_dim_facts(self) -> bool:
    return bool(self.dim_facts)

  def dim_info(
    self,
    e: Expr,
    extra: dict[int, DimInfo] | None = None,
    extra_defs: dict[int, Expr] | None = None,
    budget: int = 24,
  ) -> tuple[DimInfo, frozenset[tuple]]:
    """Soundly evaluate ``e`` in the :class:`DimInfo` shape/divisibility domain,
    returning ``(info, guard_deps)`` without committing guards yet."""
    if budget <= 0:
      return DIM_TOP, frozenset()
    if isinstance(e, Const):
      if not isinstance(e.value, bool) and isinstance(e.value, int) and e.value >= 0:
        return DimInfo.from_const(int(e.value)), frozenset()
      return DIM_TOP, frozenset()
    if isinstance(e, Var):
      if extra is not None and e.uid in extra:
        return extra[e.uid], frozenset()
      direct = self.dim_facts.get(e.uid)
      d = extra_defs.get(e.uid) if (extra_defs is not None and e.uid in extra_defs) else self.defs.get(e.uid)
      if d is not None:
        d_info, d_deps = self.dim_info(d, extra, extra_defs, budget - 1)
        if direct is not None:
          return direct[0].meet(d_info), direct[1] | d_deps
        return d_info, d_deps
      if direct is not None:
        return direct
      return DIM_TOP, frozenset()
    if isinstance(e, ir.Cast) and not isinstance(e.type, ir.BufType) and e.type.kind in "iu":
      if not isinstance(e.arg.type, ir.BufType) and e.arg.type.kind in "iu":
        return self.dim_info(e.arg, extra, extra_defs, budget - 1)
      return DIM_TOP, frozenset()
    if isinstance(e, Prim):
      if len(e.args) == 2:
        a_info, a_deps = self.dim_info(e.args[0], extra, extra_defs, budget - 1)
        b_info, b_deps = self.dim_info(e.args[1], extra, extra_defs, budget - 1)
        deps = a_deps | b_deps
        if e.op == "add":
          return a_info.add(b_info), deps
        if e.op == "sub":
          return a_info.sub(b_info), deps
        if e.op == "mul":
          return a_info.mul(b_info), deps
        if e.op == "floordiv":
          k = b_info.exact
          if k is not None and k > 0:
            return a_info.floordiv_const(k), deps
          return DIM_TOP, frozenset()
        if e.op == "min":
          return a_info.imin(b_info), deps
        if e.op == "max":
          return a_info.imax(b_info), deps
    return DIM_TOP, frozenset()

  def commit_dim_guards(self, deps: frozenset[tuple]) -> None:
    if deps:
      self.used_guards.update(deps)

  def require_dim_info(self, e: Expr) -> DimInfo:
    """Evaluate ``dim_info(e)`` and commit its contributing input guards when
    the result is strictly more informative than ``TOP``."""
    info, deps = self.dim_info(e)
    if not info.is_top and deps:
      self.used_guards.update(deps)
    return info

  def stage_index_div4(
    self,
    stage_body: list[ir.Stmt],
    raw_idx: Expr,
    fast_var: Var,
    slow_var: Var,
  ) -> bool:
    """Return True if ``raw_idx`` is provably divisible by 4 whenever
    ``fast_var`` is a multiple of 4 (committing contributing guards if True)."""
    local_defs = {s.var.uid: s.value for s in stage_body if isinstance(s, Assign)}
    extra = {
      fast_var.uid: DimInfo(0, None, 4),
      slow_var.uid: DimInfo(0, None, 1),
    }
    info, deps = self.dim_info(raw_idx, extra=extra, extra_defs=local_defs)
    if info.divisible_by(4):
      self.commit_dim_guards(deps)
      return True
    return False

  def note_var(self, v: Var) -> None:
    self.vars_by_uid[v.uid] = v

  def bind_index(self, v: Var, bound: Expr) -> None:
    """Record that loop index ``v`` ranges over ``[0, bound)``."""
    self.vars_by_uid[v.uid] = v
    self.bounds[v.uid] = bound

  def _canon_var(self, v: Var, budget: int = 20) -> Expr:
    self.vars_by_uid[v.uid] = v
    cur: Expr = v
    steps = 0
    while isinstance(cur, Var) and cur.uid in self.defs and steps < budget:
      nxt = self.defs[cur.uid]
      if isinstance(nxt, Var):
        self.vars_by_uid[nxt.uid] = nxt
        cur = nxt
        steps += 1
      elif isinstance(nxt, Const):
        return nxt
      else:
        break
    return cur

  def record_equality(self, a: Expr, b: Expr) -> None:
    """Record a proven equality ``a == b`` (oriented acyclic: higher uid -> lower uid or Const)."""
    ra = self.resolve(a)
    rb = self.resolve(b)
    if isinstance(ra, Var) and isinstance(rb, Const) and not isinstance(ra.type, ir.BufType) and ra.type.kind in "iu":
      if ra.uid not in self.defs:
        self.defs[ra.uid] = rb
        self.vars_by_uid[ra.uid] = ra
        self._poly_cache.clear()
    elif isinstance(rb, Var) and isinstance(ra, Const) and not isinstance(rb.type, ir.BufType) and rb.type.kind in "iu":
      if rb.uid not in self.defs:
        self.defs[rb.uid] = ra
        self.vars_by_uid[rb.uid] = rb
        self._poly_cache.clear()
    elif (
      isinstance(ra, Var)
      and isinstance(rb, Var)
      and ra.uid != rb.uid
      and not isinstance(ra.type, ir.BufType)
      and not isinstance(rb.type, ir.BufType)
      and ra.type.kind in "iu"
      and rb.type.kind in "iu"
    ):
      hi, lo = (ra, rb) if ra.uid > rb.uid else (rb, ra)
      if hi.uid not in self.defs:
        self.defs[hi.uid] = lo
        self.vars_by_uid[hi.uid] = hi
        self.vars_by_uid[lo.uid] = lo
        self._poly_cache.clear()
        if hi.uid in self.dim_facts:
          hi_info, hi_deps = self.dim_facts[hi.uid]
          if lo.uid in self.dim_facts:
            lo_info, lo_deps = self.dim_facts[lo.uid]
            self.dim_facts[lo.uid] = (lo_info.meet(hi_info), lo_deps | hi_deps)
          else:
            self.dim_facts[lo.uid] = (hi_info, hi_deps)
    elif (
      isinstance(a, Var)
      and isinstance(rb, (Var, Const))
      and not isinstance(a.type, ir.BufType)
      and a.type.kind in "iu"
      and not isinstance(ra, (Var, Const))
    ):
      # A Check(a == rb) where a was previously bound to a complex expr (e.g. select) simplifies a -> rb
      self.defs[a.uid] = rb
      self.vars_by_uid[a.uid] = a
      if isinstance(rb, Var):
        self.vars_by_uid[rb.uid] = rb
      self._poly_cache.clear()
    elif (
      isinstance(b, Var)
      and isinstance(ra, (Var, Const))
      and not isinstance(b.type, ir.BufType)
      and b.type.kind in "iu"
      and not isinstance(rb, (Var, Const))
    ):
      self.defs[b.uid] = ra
      self.vars_by_uid[b.uid] = b
      if isinstance(ra, Var):
        self.vars_by_uid[ra.uid] = ra
      self._poly_cache.clear()
    else:
      ka, kb = self.expr_key(a), self.expr_key(b)
      if ka != kb:
        self.expr_eqs.add((ka, kb))
        self.expr_eqs.add((kb, ka))

  def record_stmt(self, s: ir.Stmt) -> None:
    """Absorb integer definitions and checked equalities from ``s``."""
    if (
      isinstance(s, Assign)
      and not isinstance(s.var.type, ir.BufType)
      and s.var.type.kind in "iu"
      and is_pure_expr(s.value)
    ):
      self.vars_by_uid[s.var.uid] = s.var
      if s.var.uid not in self.defs:
        self.defs[s.var.uid] = s.value
        self._poly_cache.clear()
    elif isinstance(s, ir.Check) and isinstance(s.cond, Prim) and s.cond.op == "eq" and len(s.cond.args) == 2:
      self.record_equality(s.cond.args[0], s.cond.args[1])
    elif isinstance(s, ir.If) and not s.orelse and s.then and all(isinstance(t, ir.Check) for t in s.then):
      # Guarded hoisted check: ``if prod(dims) > 0: check (a == b)``
      guard_dims = self._positive_product_dims(s.cond)
      if guard_dims is not None:
        for t in s.then:
          if isinstance(t, ir.Check) and isinstance(t.cond, Prim) and t.cond.op == "eq" and len(t.cond.args) == 2:
            self.guarded_eqs.append((guard_dims, t.cond.args[0], t.cond.args[1]))

  def activate_nonempty(self, shape: tuple[Expr, ...]) -> None:
    """Activate any guarded ``if prod(dims) > 0: check (a == b)`` equalities
    implied when entering an operator over non-empty ``shape``."""
    if not self.guarded_eqs:
      return
    shape_keys = frozenset(self.expr_key(d) for d in shape)
    for guard_dims, a, b in self.guarded_eqs:
      if guard_dims <= shape_keys:
        self.record_equality(a, b)

  def _positive_product_dims(self, cond: Expr) -> frozenset[object] | None:
    """If ``cond`` is ``prod(d_0, ..., d_k) > 0``, return the set of keys ``{key(d_i)}``."""
    if not (isinstance(cond, Prim) and cond.op == "gt" and len(cond.args) == 2):
      return None
    if not (isinstance(cond.args[1], Const) and cond.args[1].value == 0):
      return None
    dims: list[object] = []

    def collect(e: Expr) -> None:
      if isinstance(e, Prim) and e.op == "mul" and len(e.args) == 2:
        collect(e.args[0])
        collect(e.args[1])
      elif isinstance(e, Const) and e.value > 0:
        return
      else:
        dims.append(self.expr_key(e))

    collect(cond.args[0])
    return frozenset(dims) if dims else None

  def resolve(self, e: Expr, budget: int = 20) -> Expr:
    """Substitute single-assignment integer definitions and fold trivial identities."""
    if isinstance(e, Var):
      self.vars_by_uid[e.uid] = e
      d = self.defs.get(e.uid)
      return e if d is None or budget <= 0 else self.resolve(d, budget - 1)
    if isinstance(e, Prim):
      args = tuple(self.resolve(a, budget - 1) for a in e.args)
      if len(args) == 2:
        a, b = args
        if e.op == "add":
          if isinstance(a, Const) and a.value == 0:
            return b
          if isinstance(b, Const) and b.value == 0:
            return a
          if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.value + b.value, e.type)
        elif e.op == "sub":
          if isinstance(b, Const) and b.value == 0:
            return a
          if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.value - b.value, e.type)
        elif e.op == "mul":
          if isinstance(a, Const) and a.value == 1:
            return b
          if isinstance(b, Const) and b.value == 1:
            return a
          if (isinstance(a, Const) and a.value == 0) or (isinstance(b, Const) and b.value == 0):
            return Const(0, e.type)
          if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.value * b.value, e.type)
      return Prim(e.op, args, e.type)
    if isinstance(e, ir.Cast):
      inner = self.resolve(e.arg, budget - 1)
      return inner if inner.type is e.type else ir.Cast(inner, e.type)
    return e

  def affine(self, e: Expr, budget: int = 20) -> AffineExpr:
    """Normalize integer expression ``e`` to canonical :class:`AffineExpr`."""
    if isinstance(e, Const) and not isinstance(e.value, bool) and isinstance(e.value, int):
      return AffineExpr.const(int(e.value))
    if isinstance(e, Var):
      self.vars_by_uid[e.uid] = e
      cached = self._poly_cache.get(e.uid)
      if cached is not None:
        return cached
      d = self.defs.get(e.uid)
      if d is not None and budget > 0:
        res = self.affine(d, budget - 1)
        self._poly_cache[e.uid] = res
        return res
      return AffineExpr.factor(("v", e.uid))
    if isinstance(e, ir.Cast) and not isinstance(e.type, ir.BufType) and e.type.kind in "iu":
      if not isinstance(e.arg.type, ir.BufType) and e.arg.type.kind in "iu":
        return self.affine(e.arg, budget - 1)
    if isinstance(e, Prim) and budget > 0:
      if e.op == "add" and len(e.args) == 2:
        return self.affine(e.args[0], budget - 1).add(self.affine(e.args[1], budget - 1), 1)
      if e.op == "sub" and len(e.args) == 2:
        return self.affine(e.args[0], budget - 1).add(self.affine(e.args[1], budget - 1), -1)
      if e.op == "neg" and len(e.args) == 1:
        return self.affine(e.args[0], budget - 1).neg()
      if e.op == "mul" and len(e.args) == 2:
        return self.affine(e.args[0], budget - 1).mul(self.affine(e.args[1], budget - 1))
    return AffineExpr.factor(("o", self._struct_key(e, budget)))

  def _struct_key(self, e: Expr, budget: int = 20) -> object:
    if isinstance(e, Var):
      self.vars_by_uid[e.uid] = e
      d = self.defs.get(e.uid)
      return ("v", e.uid) if d is None or budget <= 0 else self._struct_key(d, budget - 1)
    if isinstance(e, Const):
      return ("c", e.value, e.type)
    if isinstance(e, Prim):
      return ("p", e.op, tuple(self._struct_key(a, budget - 1) for a in e.args))
    if isinstance(e, ir.Cast):
      return ("t", e.type, self._struct_key(e.arg, budget - 1))
    if isinstance(e, Load):
      return ("l", e.buf.uid, self._struct_key(e.index, budget - 1))
    if isinstance(e, ir.RtCall):
      return ("r", e.name, e.type, tuple(self._struct_key(a, budget - 1) for a in e.args))
    return ("?", id(e))

  def expr_key(self, e: Expr) -> object:
    """Canonical equality key for ``e`` (polynomial-normal for integers, structural otherwise)."""
    if not isinstance(e.type, ir.BufType) and e.type.kind in "iu":
      return ("aff", self.affine(e).terms)
    return self._struct_key(e)

  def same_expr(self, a: Expr, b: Expr) -> bool:
    ka, kb = self.expr_key(a), self.expr_key(b)
    if ka == kb:
      return True
    if not self.expr_eqs:
      return False
    if (ka, kb) in self.expr_eqs:
      return True
    # Transitive closure over recorded checked expression equalities
    seen = {ka}
    stack = [ka]
    while stack:
      cur = stack.pop()
      for x, y in self.expr_eqs:
        if x == cur and y not in seen:
          if y == kb:
            return True
          seen.add(y)
          stack.append(y)
    return False

  def same_shape(self, a: tuple[Expr, ...], b: tuple[Expr, ...]) -> bool:
    return len(a) == len(b) and all(self.same_expr(x, y) for x, y in zip(a, b))

  def same_iteration_shape(self, a: tuple[Expr, ...], b: tuple[Expr, ...]) -> bool:
    """True if two loop shapes execute the exact same index space.

    Extends :meth:`same_shape` with sound factor cancellation: if ``a`` and ``b``
    have the same rank, match on every axis except one, and ``prod(a) == prod(b)``
    is proven, then either a shared axis is 0 (both loops execute 0 times) or all
    shared axes are positive (proving the remaining axis equal)."""
    if self.same_shape(a, b):
      return True
    if len(a) != len(b) or len(a) < 2:
      return False
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if not self.same_expr(x, y)]
    return len(diffs) == 1 and self.same_expr(ops.iprod(a), ops.iprod(b))

  def _rebuild_expr(self, poly: AffineExpr) -> Expr | None:
    """Rebuild an :class:`Expr` from an :class:`AffineExpr` if every factor is a known Var."""
    if not poly.terms:
      return ops.ZERO
    acc: Expr | None = None
    for mon, coeff in poly.terms:
      term: Expr | None = None
      for kind, payload in mon:
        if kind != "v" or payload not in self.vars_by_uid:
          return None
        v = self.vars_by_uid[payload]
        term = v if term is None else ops.imul(term, v)
      if term is None:
        part: Expr = Const(coeff, ops.I)
      elif coeff == 1:
        part = term
      elif coeff == -1:
        part = ops.isub(ops.ZERO, term)
      else:
        part = ops.imul(Const(coeff, ops.I), term)
      acc = part if acc is None else ops.iadd(acc, part)
    return ops.ZERO if acc is None else acc

  def stride_of(self, e: Expr, var: Var) -> Expr | None:
    """Affine coefficient (stride) of ``var`` in integer expression ``e``, or ``None`` if nonlinear."""
    cv = self._canon_var(var)
    if not isinstance(cv, Var):
      return ops.ZERO
    target = ("v", cv.uid)
    poly = self.affine(e)
    out: dict[Monomial, int] = {}
    for mon, coeff in poly.terms:
      cnt = mon.count(target)
      if cnt > 1:
        return None
      if cnt == 1:
        idx = mon.index(target)
        rem = mon[:idx] + mon[idx + 1 :]
        out[rem] = out.get(rem, 0) + coeff
      else:
        # Check whether an opaque factor inside ``mon`` mentions ``var``
        for kind, payload in mon:
          if kind == "o" and _key_mentions_uid(payload, cv.uid):
            return None
    return self._rebuild_expr(AffineExpr.from_dict(out))

  def lift_outside(self, e: Expr, inner: set[int], budget: int = 40) -> Expr | None:
    """Rewrite ``e`` without variables in ``inner`` by substituting pure definitions;
    return ``None`` if ``e`` genuinely depends on ``inner``."""
    if isinstance(e, Const):
      return e
    if isinstance(e, Var):
      if e.uid not in inner:
        return e
      d = self.defs.get(e.uid)
      if d is None or budget <= 0:
        return None
      return self.lift_outside(d, inner, budget - 1)
    if isinstance(e, Prim):
      args = [self.lift_outside(a, inner, budget - 1) for a in e.args]
      if any(a is None for a in args):
        return None
      return ops.prim(e.op, tuple(args), e.type)  # type: ignore[arg-type]
    if isinstance(e, ir.Cast):
      a = self.lift_outside(e.arg, inner, budget - 1)
      return None if a is None else ops.cast(a, e.type)
    return None

  def _is_in_bounds(self, coord: Expr, dim: Expr) -> bool:
    """Conservatively prove ``0 <= coord < dim`` using recorded loop bounds and affine identities."""
    p_coord = self.affine(coord)
    p_dim = self.affine(dim)
    # 1. Constant coordinate: 0 <= c < dim (for Const dim, or c == 0 for symbolic dim inside non-empty consumer)
    c_val = p_coord.is_const()
    d_val = p_dim.is_const()
    if c_val is not None:
      if c_val < 0:
        return False
      if d_val is not None:
        return c_val < d_val
      return c_val == 0
    # 2. Single loop variable with optional constant shift: v + off (or (dim - 1) - v)
    var_terms = [(m, c) for m, c in p_coord.terms if m != ()]
    off = dict(p_coord.terms).get((), 0)
    if len(var_terms) == 1 and len(var_terms[0][0]) == 1 and var_terms[0][0][0][0] == "v":
      v_uid = var_terms[0][0][0][1]
      coeff = var_terms[0][1]
      bound = self.bounds.get(v_uid)
      if bound is None:
        return False
      p_bound = self.affine(bound)
      if coeff == 1 and off >= 0:
        # 0 <= v + off < dim  <==  bound + off <= dim  <==  dim - (bound + off) >= 0 (const >= 0)
        diff = p_dim.add(p_bound.add(AffineExpr.const(off), 1), -1).is_const()
        return diff is not None and diff >= 0
    # 3. Reversed index: (dim - 1 - off) - v where v < bound and bound + off <= dim
    if len(var_terms) >= 1:
      # Check if dim - 1 - coord == v + k with k >= 0 and bound + k <= dim
      rev = p_dim.add(AffineExpr.const(-1), 1).add(p_coord, -1)
      rev_vars = [(m, c) for m, c in rev.terms if m != ()]
      rev_off = dict(rev.terms).get((), 0)
      if len(rev_vars) == 1 and len(rev_vars[0][0]) == 1 and rev_vars[0][0][0][0] == "v":
        v_uid = rev_vars[0][0][0][1]
        if rev_vars[0][1] == 1 and rev_off >= 0 and v_uid in self.bounds:
          diff = p_dim.add(self.affine(self.bounds[v_uid]).add(AffineExpr.const(rev_off), 1), -1).is_const()
          return diff is not None and diff >= 0
    return False

  def decompose_row_major(
    self, idx: Expr, shape: tuple[Expr, ...], require_bounds: bool = True
  ) -> tuple[Expr, ...] | None:
    """Decompose flat row-major index ``idx`` into coordinates ``(c_0, ..., c_{r-1})``
    such that ``row_major(c, shape) == idx``.

    When ``require_bounds`` is True (default for rank >= 2), every coordinate is
    provably in ``[0, shape[k])`` so flat reshapes or out-of-range strides cannot
    be mis-decomposed."""
    rank = len(shape)
    if rank == 0:
      return () if self.affine(idx).is_const() == 0 else None
    if rank == 1:
      # In 1D, row_major((idx,), (N,)) is the identity: any valid Load(buf, idx)
      # in the consumer already addresses a valid element of the 1D producer.
      return (idx,)

    # Compute row-major strides S_k = prod(shape[k+1:])
    strides: list[AffineExpr] = [AffineExpr.const(1)] * rank
    for k in range(rank - 2, -1, -1):
      strides[k] = strides[k + 1].mul(self.affine(shape[k + 1]))

    rem = self.affine(idx)
    coords: list[Expr] = []
    for k in range(rank - 1):
      sk = strides[k]
      if len(sk.terms) != 1:
        return None
      s_mon, s_coeff = sk.terms[0]
      if s_coeff == 0:
        return None
      # Extract terms of ``rem`` divisible by s_mon (and s_coeff)
      q_terms: dict[Monomial, int] = {}
      left_terms: dict[Monomial, int] = {}
      for mon, coeff in rem.terms:
        q_mon = _monomial_div(mon, s_mon)
        if q_mon is not None and coeff % s_coeff == 0:
          q_terms[q_mon] = q_terms.get(q_mon, 0) + (coeff // s_coeff)
        else:
          left_terms[mon] = left_terms.get(mon, 0) + coeff
      # For constant stride (s_mon == ()), avoid pulling a constant offset into c_k
      # unless no symbolic stride handles it; only strip constant term if dim is const.
      if s_mon == ():
        # Constant strides like (M, 4): handle via non-constant terms divisible by s_coeff
        q_const = q_terms.pop((), 0)
        if q_const:
          left_terms[()] = left_terms.get((), 0) + q_const * s_coeff
      q_expr = self._rebuild_expr(AffineExpr.from_dict(q_terms))
      if q_expr is None:
        return None
      coords.append(q_expr)
      rem = AffineExpr.from_dict(left_terms)

    last_expr = self._rebuild_expr(rem)
    if last_expr is None:
      return None
    coords.append(last_expr)

    # Verify exact row-major reconstruction
    recon = self.affine(coords[0])
    for c, d in zip(coords[1:], shape[1:]):
      recon = recon.mul(self.affine(d)).add(self.affine(c), 1)
    if recon.terms != self.affine(idx).terms:
      return None
    if require_bounds and not all(self._is_in_bounds(c, d) for c, d in zip(coords, shape)):
      return None
    return tuple(coords)


def _monomial_div(num: Monomial, den: Monomial) -> Monomial | None:
  """Return ``num / den`` as a sorted Monomial if ``den`` divides ``num``, else ``None``."""
  if not den:
    return num
  rem = list(num)
  for f in den:
    try:
      rem.remove(f)
    except ValueError:
      return None
  return tuple(rem)


def _key_mentions_uid(k: object, uid: int) -> bool:
  if not isinstance(k, tuple):
    return False
  if len(k) == 2 and k[0] == "v":
    return k[1] == uid
  return any(_key_mentions_uid(x, uid) for x in k)


# ---------------------------------------------------------------------------
# BufferInfo: alias, root allocation, escape, and reader analysis
# ---------------------------------------------------------------------------


def is_pure_expr(e: Expr) -> bool:
  return not any(isinstance(x, ir.RtCall) and not x.pure for x in ir.walk_expr(e))


def buffer_accesses(stmts: list) -> tuple[set[int], set[int]]:
  """Return ``(reads, writes)`` buffer uids touched anywhere in ``stmts``."""
  reads: set[int] = set()
  writes: set[int] = set()
  for s in ir.walk_stmts(stmts):
    if isinstance(s, ir.Store):
      writes.add(s.buf.uid)
      for e in (s.index, s.value):
        for x in ir.walk_expr(e):
          if isinstance(x, Load):
            reads.add(x.buf.uid)
    elif isinstance(s, ir.IndexScatter):
      writes.update(u.buf.uid for u in s.updates)
      reads.update(u.buf.uid for u in s.updates if u.op != "assign")
      for e in (
        *s.shape,
        s.dest_size,
        s.target,
        *(u.value for u in s.updates),
        *(() if s.where is None else (s.where,)),
      ):
        for x in ir.walk_expr(e):
          if isinstance(x, Load):
            reads.add(x.buf.uid)
          elif isinstance(x, Var) and isinstance(x.type, ir.BufType):
            reads.add(x.uid)
    elif isinstance(s, ir.IndexExpand):
      writes.add(s.offsets.uid)
      writes.update(buf.uid for buf, _ in s.values)
      for e in (s.n, s.count, *(val for _, val in s.values)):
        for x in ir.walk_expr(e):
          if isinstance(x, Load):
            reads.add(x.buf.uid)
          elif isinstance(x, Var) and isinstance(x.type, ir.BufType):
            reads.add(x.uid)
    else:
      for e in ir.stmt_exprs(s):
        for x in ir.walk_expr(e):
          if isinstance(x, Load):
            reads.add(x.buf.uid)
          elif isinstance(x, Var) and isinstance(x.type, ir.BufType):
            reads.add(x.uid)
  return reads, writes


@dataclass
class BufferInfo:
  """Single-pass summary of buffer allocations, roots, escapes, stores, and reads."""

  local_allocs: set[int] = field(default_factory=set)
  root_alloc: dict[int, int] = field(default_factory=dict)
  readonly_params: set[int] = field(default_factory=set)
  escapes: set[int] = field(default_factory=set)
  store_counts: dict[int, int] = field(default_factory=dict)
  readers: dict[int, int] = field(default_factory=dict)

  @classmethod
  def analyze(cls, fn: ir.Function) -> BufferInfo:
    info = cls()
    for p in fn.params:
      if isinstance(p, ir.ArrayParam) and not p.writes:
        info.readonly_params.add(p.buf.uid)

    def note_expr(e) -> None:
      if isinstance(e, Expr):
        for x in ir.walk_expr(e):
          if isinstance(x, Var):
            info.readers[x.uid] = info.readers.get(x.uid, 0) + 1
      elif isinstance(e, (tuple, list)):
        for x in e:
          note_expr(x)

    for s in ir.walk_stmts(fn.body):
      if isinstance(s, ir.Alloc):
        info.local_allocs.add(s.var.uid)
        info.root_alloc[s.var.uid] = s.var.uid
      elif isinstance(s, Assign) and isinstance(s.var.type, ir.BufType) and isinstance(s.value, Var):
        root = info.root_alloc.get(s.value.uid)
        if root is not None:
          info.root_alloc[s.var.uid] = root
      elif isinstance(s, ir.Store):
        info.store_counts[s.buf.uid] = info.store_counts.get(s.buf.uid, 0) + 1
        note_expr(s.index)
        note_expr(s.value)
        continue
      elif isinstance(s, ir.IndexScatter):
        for u in s.updates:
          info.store_counts[u.buf.uid] = info.store_counts.get(u.buf.uid, 0) + 1
        for e in (
          *s.shape,
          s.dest_size,
          s.target,
          *(u.value for u in s.updates),
          *(() if s.where is None else (s.where,)),
        ):
          note_expr(e)
        continue
      elif isinstance(s, ir.IndexExpand):
        info.local_allocs.add(s.offsets.uid)
        info.root_alloc[s.offsets.uid] = s.offsets.uid
        info.store_counts[s.offsets.uid] = info.store_counts.get(s.offsets.uid, 0) + 1
        for buf, val in s.values:
          info.local_allocs.add(buf.uid)
          info.root_alloc[buf.uid] = buf.uid
          info.store_counts[buf.uid] = info.store_counts.get(buf.uid, 0) + 1
          note_expr(val)
        note_expr(s.n)
        note_expr(s.count)
        continue
      elif isinstance(s, ir.Incref) and isinstance(s.buf, Var):
        info.escapes.add(s.buf.uid)
      note_expr(ir.stmt_exprs(s))

    for r in fn.results:
      if isinstance(r, Expr):
        note_expr(r)
        if isinstance(r, Var) and isinstance(r.type, ir.BufType):
          info.escapes.add(r.uid)
      else:
        for name in getattr(type(r), "__dataclass_fields__", ()):
          val = getattr(r, name)
          note_expr(val)
        if isinstance(r, ir.ArrayResult) and isinstance(r.buf, Var):
          info.escapes.add(r.buf.uid)
    return info

  def may_alias(self, first: set[int], second: set[int]) -> bool:
    """True if any buffer in ``first`` could alias a buffer in ``second``.

    Distinct local ``Alloc`` roots never alias each other or any external
    parameter; non-local parameters conservatively may alias each other."""
    if not first or not second:
      return False
    r1 = {self.root_alloc.get(u, u) for u in first}
    r2 = {self.root_alloc.get(u, u) for u in second}
    if r1 & r2:
      return True
    ext1 = [u for u in r1 if u not in self.local_allocs]
    ext2 = [u for u in r2 if u not in self.local_allocs]
    return bool(ext1 and ext2)


# ---------------------------------------------------------------------------
# Lightweight middle-end cost model
# ---------------------------------------------------------------------------


def expr_cost(e: Expr) -> int:
  """Approximate scalar compute cost of ``e`` (used for multi-consumer fusion)."""
  cost = 0
  for x in ir.walk_expr(e):
    if isinstance(x, Load):
      cost += 2
    elif isinstance(x, Prim):
      cost += 12 if x.op in _EXPENSIVE_PRIMS else 1
    elif isinstance(x, ir.RtCall):
      cost += 12
  return cost


def op_cost(stmts: list[ir.Stmt]) -> int:
  """Approximate cost of a straight-line operator body; loops/operators return ``LOOP_COST``."""
  total = 0
  for s in stmts:
    if isinstance(s, (ir.Comment, ir.ReleaseScope)):
      continue
    if isinstance(s, Assign):
      total += expr_cost(s.value)
    elif isinstance(s, ir.Store):
      total += expr_cost(s.index) + expr_cost(s.value)
    elif isinstance(s, ir.Check):
      total += expr_cost(s.cond)
    else:
      return LOOP_COST
  return total


# ---------------------------------------------------------------------------
# Law 5: Affine Tile-Interval Bounds & Predicate Propagation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TileInterval:
  """Closed integer interval ``[lo, hi]`` over symbolic IR expressions."""

  lo: Expr
  hi: Expr


def _bool_and(a: Expr, b: Expr) -> Expr:
  from ..ndtypes import Bool

  if isinstance(a, Const) and not a.value:
    return Const(False, Bool)
  if isinstance(b, Const) and not b.value:
    return Const(False, Bool)
  if isinstance(a, Const) and a.value:
    return b
  if isinstance(b, Const) and b.value:
    return a
  return ops.prim("and", (a, b), Bool)


def _bool_or(a: Expr, b: Expr) -> Expr:
  from ..ndtypes import Bool

  if isinstance(a, Const) and a.value:
    return Const(True, Bool)
  if isinstance(b, Const) and b.value:
    return Const(True, Bool)
  if isinstance(a, Const) and not a.value:
    return b
  if isinstance(b, Const) and not b.value:
    return a
  return ops.prim("or", (a, b), Bool)


def _simplify_cmp(op: str, lhs: Expr, rhs: Expr, facts: IndexFacts) -> Expr:
  from ..ndtypes import Bool

  diff = facts.affine(lhs).add(facts.affine(rhs), -1).is_const()
  if diff is not None:
    if op == "le":
      return Const(diff <= 0, Bool)
    if op == "lt":
      return Const(diff < 0, Bool)
    if op == "ge":
      return Const(diff >= 0, Bool)
    if op == "gt":
      return Const(diff > 0, Bool)
    if op == "eq":
      return Const(diff == 0, Bool)
    if op == "ne":
      return Const(diff != 0, Bool)
  return ops.cmp(op, lhs, rhs)


def eval_interval(
  e: Expr,
  var_intervals: dict[int, TileInterval],
  facts: IndexFacts,
  budget: int = 24,
) -> TileInterval | None:
  """Compute a sound symbolic closed interval ``[lo, hi]`` for integer ``e``
  given tile intervals for index variables (e.g. ``i in [i0, i0 + BM - 1]``)."""
  if budget <= 0:
    return None
  if isinstance(e, Const) and not isinstance(e.value, bool) and isinstance(e.value, int):
    return TileInterval(e, e)
  if isinstance(e, Var):
    iv = var_intervals.get(e.uid)
    if iv is not None:
      return iv
    d = facts.defs.get(e.uid)
    if d is not None:
      return eval_interval(d, var_intervals, facts, budget - 1)
    return TileInterval(e, e)
  if isinstance(e, ir.Cast) and not isinstance(e.type, ir.BufType) and e.type.kind in "iu":
    return eval_interval(e.arg, var_intervals, facts, budget - 1)
  if isinstance(e, Prim):
    if e.op == "neg" and len(e.args) == 1:
      iv = eval_interval(e.args[0], var_intervals, facts, budget - 1)
      return None if iv is None else TileInterval(ops.isub(ops.ZERO, iv.hi), ops.isub(ops.ZERO, iv.lo))
    if e.op == "add" and len(e.args) == 2:
      a = eval_interval(e.args[0], var_intervals, facts, budget - 1)
      b = eval_interval(e.args[1], var_intervals, facts, budget - 1)
      if a is None or b is None:
        return None
      return TileInterval(facts.resolve(ops.iadd(a.lo, b.lo)), facts.resolve(ops.iadd(a.hi, b.hi)))
    if e.op == "sub" and len(e.args) == 2:
      a = eval_interval(e.args[0], var_intervals, facts, budget - 1)
      b = eval_interval(e.args[1], var_intervals, facts, budget - 1)
      if a is None or b is None:
        return None
      return TileInterval(facts.resolve(ops.isub(a.lo, b.hi)), facts.resolve(ops.isub(a.hi, b.lo)))
    if e.op == "mul" and len(e.args) == 2:
      a = eval_interval(e.args[0], var_intervals, facts, budget - 1)
      b = eval_interval(e.args[1], var_intervals, facts, budget - 1)
      if a is None or b is None:
        return None
      ca = facts.affine(e.args[0]).is_const()
      cb = facts.affine(e.args[1]).is_const()
      if ca is not None:
        c = Const(ca, ops.I)
        if ca >= 0:
          return TileInterval(facts.resolve(ops.imul(c, b.lo)), facts.resolve(ops.imul(c, b.hi)))
        return TileInterval(facts.resolve(ops.imul(c, b.hi)), facts.resolve(ops.imul(c, b.lo)))
      if cb is not None:
        c = Const(cb, ops.I)
        if cb >= 0:
          return TileInterval(facts.resolve(ops.imul(a.lo, c)), facts.resolve(ops.imul(a.hi, c)))
        return TileInterval(facts.resolve(ops.imul(a.hi, c)), facts.resolve(ops.imul(a.lo, c)))
      return None
    if e.op == "min" and len(e.args) == 2:
      a = eval_interval(e.args[0], var_intervals, facts, budget - 1)
      b = eval_interval(e.args[1], var_intervals, facts, budget - 1)
      if a is None or b is None:
        return None
      return TileInterval(ops.imin(a.lo, b.lo), ops.imin(a.hi, b.hi))
    if e.op == "max" and len(e.args) == 2:
      a = eval_interval(e.args[0], var_intervals, facts, budget - 1)
      b = eval_interval(e.args[1], var_intervals, facts, budget - 1)
      if a is None or b is None:
        return None
      return TileInterval(ops.imax(a.lo, b.lo), ops.imax(a.hi, b.hi))
  return None


def tile_predicate_guards(
  cond: Expr,
  var_intervals: dict[int, TileInterval],
  facts: IndexFacts,
  budget: int = 24,
) -> tuple[Expr, Expr]:
  """Derive symbolic tile-level guards ``(always_false, always_true)`` for
  boolean predicate ``cond`` over tile intervals ``var_intervals`` (Law 5).

  * If ``always_false`` evaluates to True for a tile ``(i0, j0)``, every point
    in the tile fails ``cond`` -> skip the entire tile.
  * If ``always_true`` evaluates to True for a tile ``(i0, j0)``, every point
    in the tile satisfies ``cond`` -> elide per-element masking inside the tile.
  """
  from ..ndtypes import Bool

  false_c = Const(False, Bool)
  true_c = Const(True, Bool)
  if budget <= 0:
    return (false_c, false_c)
  if isinstance(cond, Const):
    return (false_c, true_c) if cond.value else (true_c, false_c)
  if isinstance(cond, Var):
    d = facts.defs.get(cond.uid)
    if d is not None:
      return tile_predicate_guards(d, var_intervals, facts, budget - 1)
    return (false_c, false_c)
  if isinstance(cond, ir.Cast) and cond.type is Bool:
    return tile_predicate_guards(cond.arg, var_intervals, facts, budget - 1)
  if not isinstance(cond, Prim):
    return (false_c, false_c)

  op = cond.op
  if op == "not" and len(cond.args) == 1:
    af, at = tile_predicate_guards(cond.args[0], var_intervals, facts, budget - 1)
    return (at, af)
  if op in ("and", "bitand") and len(cond.args) == 2:
    af1, at1 = tile_predicate_guards(cond.args[0], var_intervals, facts, budget - 1)
    af2, at2 = tile_predicate_guards(cond.args[1], var_intervals, facts, budget - 1)
    return (_bool_or(af1, af2), _bool_and(at1, at2))
  if op in ("or", "bitor") and len(cond.args) == 2:
    af1, at1 = tile_predicate_guards(cond.args[0], var_intervals, facts, budget - 1)
    af2, at2 = tile_predicate_guards(cond.args[1], var_intervals, facts, budget - 1)
    return (_bool_and(af1, af2), _bool_or(at1, at2))

  # Rewrite abs(x) <= W / abs(x) < W / abs(x) >= W / abs(x) > W into exact linear clauses
  if op in ("le", "lt", "ge", "gt") and len(cond.args) == 2:
    lhs, rhs = cond.args
    if isinstance(lhs, Prim) and lhs.op == "abs" and len(lhs.args) == 1:
      x = lhs.args[0]
      nx = ops.isub(ops.ZERO, x)
      if op in ("le", "lt"):
        c1 = ops.cmp(op, x, rhs)
        c2 = ops.cmp(op, nx, rhs)
        return tile_predicate_guards(ops.prim("and", (c1, c2), Bool), var_intervals, facts, budget - 1)
      c1 = ops.cmp(op, x, rhs)
      c2 = ops.cmp(op, nx, rhs)
      return tile_predicate_guards(ops.prim("or", (c1, c2), Bool), var_intervals, facts, budget - 1)
    if isinstance(rhs, Prim) and rhs.op == "abs" and len(rhs.args) == 1:
      flip = {"le": "ge", "lt": "gt", "ge": "le", "gt": "lt"}[op]
      return tile_predicate_guards(ops.cmp(flip, rhs, lhs), var_intervals, facts, budget - 1)

  if op in ("le", "lt", "ge", "gt", "eq", "ne") and len(cond.args) == 2:
    lhs, rhs = cond.args
    a = eval_interval(lhs, var_intervals, facts, budget - 1)
    b = eval_interval(rhs, var_intervals, facts, budget - 1)
    if a is None or b is None:
      return (false_c, false_c)
    if op == "le":
      return (_simplify_cmp("gt", a.lo, b.hi, facts), _simplify_cmp("le", a.hi, b.lo, facts))
    if op == "lt":
      return (_simplify_cmp("ge", a.lo, b.hi, facts), _simplify_cmp("lt", a.hi, b.lo, facts))
    if op == "ge":
      return (_simplify_cmp("lt", a.hi, b.lo, facts), _simplify_cmp("ge", a.lo, b.hi, facts))
    if op == "gt":
      return (_simplify_cmp("le", a.hi, b.lo, facts), _simplify_cmp("gt", a.lo, b.hi, facts))
    if op == "eq":
      af = _bool_or(_simplify_cmp("lt", a.hi, b.lo, facts), _simplify_cmp("lt", b.hi, a.lo, facts))
      at = _bool_and(
        _bool_and(_simplify_cmp("eq", a.lo, a.hi, facts), _simplify_cmp("eq", b.lo, b.hi, facts)),
        _simplify_cmp("eq", a.lo, b.lo, facts),
      )
      return (af, at)
    if op == "ne":
      at = _bool_or(_simplify_cmp("lt", a.hi, b.lo, facts), _simplify_cmp("lt", b.hi, a.lo, facts))
      af = _bool_and(
        _bool_and(_simplify_cmp("eq", a.lo, a.hi, facts), _simplify_cmp("eq", b.lo, b.hi, facts)),
        _simplify_cmp("eq", a.lo, b.lo, facts),
      )
      return (af, at)
  return (false_c, false_c)


__all__ = [
  "CHEAP_FUSE_COST",
  "LOOP_COST",
  "AffineExpr",
  "BufferInfo",
  "IndexFacts",
  "TileInterval",
  "buffer_accesses",
  "eval_interval",
  "expr_cost",
  "is_pure_expr",
  "op_cost",
  "tile_predicate_guards",
]
