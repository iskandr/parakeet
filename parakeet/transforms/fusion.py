"""Fusion rules that are passes over the operator statements.

*vertical fusion* (thesis Fig. 6.1, at the operator level)::

    Map -> Map       epilogue fusion (e.g. ``np.tanh(a @ b + c)`` into a tile)
    Map -> Filter    predicate fusion (no intermediate boolean mask array)
    Scan -> Map      ``Map(g, Scan(f, +, e, X)) => Scan(f, +, g o e, X)``
    Map -> Reduce    producer map fused directly into reduction

*filter fusion* (thesis Fig. 6.1)::

    Filter -> Filter   ``Filter(q, Filter(p, X)) => Filter(p and q, X)``
    Filter -> Reduce   ``Reduce(g, +, Filter(f, p, X)) => FilterReduce``

*sibling fusion* (thesis Fig. 6.3, horizontal fusion)::

    Reduce + Reduce    adjacent reductions over the same index space share one loop
    Map + Map          adjacent independent maps over the same shape share one loop
"""

from __future__ import annotations

from ..ndtypes import Bool
from ..syntax import ir, ops
from ..syntax.clone import Cloner, defined
from ..syntax.ir import Assign, Const, Expr, IndexFilter, IndexMap, IndexReduce, IndexScan, Var
from .analysis import (
  CHEAP_FUSE_COST,
  BufferInfo,
  IndexFacts,
  buffer_accesses,
  expr_cost,
  is_pure_expr,
  op_cost,
)
from .normalize import split_invariant_map, squeeze_unit_map
from .simplify import drop_buffers, has_effects


def resolved(e: Expr, defs: dict[int, Expr], budget: int = 20) -> Expr:
  """``e`` with single-assignment integer variables replaced by their definitions
  and trivial additive/multiplicative identities folded."""
  return IndexFacts(defs).resolve(e, budget)


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


def same_shape(a: tuple[Expr, ...], b: tuple[Expr, ...], defs: dict[int, Expr] | IndexFacts) -> bool:
  facts = defs if isinstance(defs, IndexFacts) else IndexFacts(defs)
  return facts.same_shape(a, b)


def record_check_equality(s: ir.Stmt, defs: dict[int, Expr] | IndexFacts) -> None:
  """Record proven scalar integer equalities from ``check (a == b)``."""
  if isinstance(defs, IndexFacts):
    defs.record_stmt(s)
    return
  facts = IndexFacts(defs)
  facts.record_stmt(s)
  defs.update(facts.defs)


def uses(stmts: list, uids: set[int]) -> bool:
  for s in ir.walk_stmts(stmts):
    for e in ir.stmt_exprs(s):
      if any(isinstance(x, Var) and x.uid in uids for x in ir.walk_expr(e)):
        return True
    if isinstance(s, ir.Store) and s.buf.uid in uids:
      return True
  return False


def pure(e: Expr) -> bool:
  return is_pure_expr(e)


def buffers_may_alias(first: set[int], second: set[int], local_allocs: set[int] | BufferInfo) -> bool:
  """True if any buffer in ``first`` could alias a buffer in ``second``."""
  if isinstance(local_allocs, BufferInfo):
    return local_allocs.may_alias(first, second)
  if not first or not second:
    return False
  if first & second:
    return True
  return any(u not in local_allocs for u in first) and any(u not in local_allocs for u in second)


def function_buffer_info(fn: ir.Function) -> tuple[set[int], dict[int, int]]:
  """Return ``(local_allocs, store_counts)`` across ``fn``."""
  info = BufferInfo.analyze(fn)
  return info.local_allocs, info.store_counts


def count_readers(fn: ir.Function) -> dict[int, int]:
  """Number of non-store reads of every variable across ``fn``."""
  return BufferInfo.analyze(fn).readers


def count_reads_in(stmts: list, var: Var) -> int:
  """Count non-store reads of ``var`` anywhere inside ``stmts``."""
  count = 0
  for s in ir.walk_stmts(stmts):
    exprs = (s.index, s.value) if isinstance(s, ir.Store) else ir.stmt_exprs(s)
    for e in exprs:
      for x in ir.walk_expr(e):
        if x is var:
          count += 1
  return count


# ---------------------------------------------------------------------------
# vertical fusion (Map->Map, Map->Filter, Scan->Map, Map->Reduce)
# ---------------------------------------------------------------------------


def local_int_defs(block: list, base: dict[int, Expr]) -> dict[int, Expr]:
  """Extend ``base`` with single-assignment integer definitions local to ``block``."""
  out = dict(base)
  for s in block:
    if isinstance(s, Assign) and not isinstance(s.var.type, ir.BufType) and s.var.type.kind in "iu":
      out[s.var.uid] = s.value
  return out


class InlineBufferLoad(Cloner):
  """Clone a consumer body, replacing every ``Load(buf, idx)`` at the
  producer's store index with ``value`` (rejecting any other use of ``buf``)."""

  def __init__(
    self, subst: dict[int, Expr], buf: Var, expected_index_key, value: Var, defs: dict[int, Expr] | IndexFacts
  ):
    super().__init__(subst)
    self.buf = buf
    self.expected_index_key = expected_index_key
    self.value = value
    self.facts = defs if isinstance(defs, IndexFacts) else IndexFacts(defs)
    self.ok = True
    self.replaced = 0

  def expr(self, e: Expr) -> Expr:
    if isinstance(e, ir.Load) and e.buf is self.buf:
      idx = self.expr(e.index)
      if self.facts.expr_key(idx) == self.expected_index_key:
        self.replaced += 1
        return ir.Cast(self.value, e.type) if e.type is not self.value.type else self.value
      self.ok = False
      return e
    if isinstance(e, Var) and e is self.buf:
      self.ok = False
      return e
    if isinstance(e, ir.Prim):
      return ir.Prim(e.op, tuple(self.expr(a) for a in e.args), e.type)
    if isinstance(e, ir.Cast):
      return ir.Cast(self.expr(e.arg), e.type)
    if isinstance(e, ir.Load):
      return ir.Load(e.buf, self.expr(e.index))
    if isinstance(e, ir.RtCall):
      return ir.RtCall(e.name, tuple(self.expr(a) for a in e.args), e.type, e.pure)
    return super().expr(e)


class AffineInlineBufferLoad(Cloner):
  """Clone a consumer body, replacing every ``Load(buf, idx)`` whose flat index
  decomposes via :meth:`IndexFacts.decompose_row_major` into provably in-bounds
  producer coordinates ``(c_0, ..., c_{d-1})`` by cloning the producer's body at
  those coordinates.

  Subsumes exact elementwise fusion, transpose fusion (``cos(b) -> b.T``),
  shifted slice/stencil fusion (``y[1:] - y[:-1]``), and broadcast reads."""

  def __init__(
    self,
    producer: IndexMap,
    prefix: list[ir.Stmt],
    value: Expr,
    buf: Var,
    facts: IndexFacts,
    subst: dict[int, Expr] | None = None,
  ):
    super().__init__(subst or {})
    self.producer = producer
    self.prefix = prefix
    self.value = value
    self.buf = buf
    self.facts = facts
    self.pending: list[ir.Stmt] = []
    self.ok = True
    self.replaced = 0

  def _inline_at(self, coords: tuple[Expr, ...], want_type) -> Expr:
    cl = Cloner({p.uid: c for p, c in zip(self.producer.index, coords)})
    for v in defined(self.prefix).values():
      cl.var(v)
    inlined_prefix = [cl.stmt(t) for t in self.prefix]
    inlined_val = cl.expr(self.value)
    elem = Var("v", self.buf.type.elt)
    self.pending.extend(inlined_prefix)
    self.pending.append(Assign(elem, inlined_val))
    self.replaced += 1
    return ir.Cast(elem, want_type) if want_type is not elem.type else elem

  def expr(self, e: Expr) -> Expr:
    if isinstance(e, ir.Load) and e.buf is self.buf:
      idx = self.expr(e.index)
      coords = self.facts.decompose_row_major(idx, self.producer.shape, require_bounds=True)
      if coords is None:
        self.ok = False
        return e
      return self._inline_at(coords, e.type)
    if isinstance(e, Var) and e is self.buf:
      self.ok = False
      return e
    if isinstance(e, ir.Prim):
      return ir.Prim(e.op, tuple(self.expr(a) for a in e.args), e.type)
    if isinstance(e, ir.Cast):
      return ir.Cast(self.expr(e.arg), e.type)
    if isinstance(e, ir.Load):
      return ir.Load(e.buf, self.expr(e.index))
    if isinstance(e, ir.RtCall):
      return ir.RtCall(e.name, tuple(self.expr(a) for a in e.args), e.type, e.pure)
    return super().expr(e)

  def block(self, stmts: list) -> list:
    out: list = []
    saved_facts = self.facts
    self.facts = saved_facts.fork()
    for s in stmts:
      if isinstance(s, IndexReduce):
        sub_facts = self.facts.fork()
        for v, d in zip(s.index, s.shape):
          sub_facts.bind_index(v, d)
        prev_facts = self.facts
        self.facts = sub_facts
        new_body = self.block(s.body)
        self.pending = []
        new_folds = [self.fold(f) for f in s.folds]
        if self.pending:
          new_body.extend(self.pending)
          self.pending = []
        self.facts = prev_facts
        out.append(
          IndexReduce(
            tuple(self.expr(d) for d in s.shape),
            tuple(self.var(v) for v in s.index),
            new_body,
            new_folds,
            self.scope(s.scope),
            s.home,
            s.parallel,
            s.loc,
            self.block(s.pre),
            None if s.where is None else self.expr(s.where),
          )
        )
        continue
      if isinstance(s, IndexMap):
        sub_facts = self.facts.fork()
        for v, d in zip(s.index, s.shape):
          sub_facts.bind_index(v, d)
        prev_facts = self.facts
        self.facts = sub_facts
        new_body = self.block(s.body)
        self.facts = prev_facts
        out.append(
          IndexMap(
            tuple(self.expr(d) for d in s.shape),
            tuple(self.var(v) for v in s.index),
            new_body,
            self.scope(s.scope),
            s.ordered,
            s.parallel,
            s.loc,
          )
        )
        continue
      self.pending = []
      ns = self.stmt(s)
      if self.pending:
        out.extend(self.pending)
        self.pending = []
      self.facts.record_stmt(ns)
      out.append(ns)
    self.facts = saved_facts
    return out


def _is_row_major_store(store_index: Expr, producer: IndexMap, facts: IndexFacts) -> bool:
  """True if ``store_index`` equals ``row_major(producer.index, producer.shape)``."""
  if not producer.index:
    return False
  acc: Expr = producer.index[0]
  for idx_v, dim in zip(producer.index[1:], producer.shape[1:]):
    acc = ops.iadd(ops.imul(acc, dim), idx_v)
  return facts.same_expr(store_index, acc)


def _affine_fusion_profitable(
  producer: IndexMap,
  consumer_shape: tuple[Expr, ...],
  prefix: list[ir.Stmt],
  value: Expr,
  total_consumer_reads: int,
  facts: IndexFacts,
) -> bool:
  """True if inlining ``producer`` via affine reindexing will not duplicate expensive work."""
  # Same total element count (e.g. transpose (K, N) -> (N, K)) or sub-slice of same rank with 1 read:
  # each producer element runs at most once, so any cost is profitable!
  if total_consumer_reads == 1:
    if facts.same_expr(ops.iprod(producer.shape), ops.iprod(consumer_shape)):
      return True
    if len(producer.shape) == len(consumer_shape):
      # Check if every consumer dim <= producer dim (e.g. slice y[1:])
      all_le = True
      for cd, pd in zip(consumer_shape, producer.shape):
        diff = facts.affine(pd).add(facts.affine(cd), -1).is_const()
        if diff is None or diff < 0:
          all_le = False
          break
      if all_le:
        return True
  # Otherwise (broadcast or multi-load stencil), only fuse cheap, loop-free producers
  if any(isinstance(t, (ir.Check, *ir.OPERATORS, ir.ForRange, ir.While)) for t in ir.walk_stmts(prefix)):
    return False
  return op_cost(prefix) + expr_cost(value) <= CHEAP_FUSE_COST


def match_single_output_map(
  s: ir.Stmt, defs: dict[int, Expr] | IndexFacts, allocs: set[int], store_counts: dict[int, int]
):
  """Match an ``IndexMap`` that writes a freshly allocated buffer once at the
  end of its body.

  Returns ``(prefix, buf, store_index, index_key, value)`` or ``None``."""
  if not isinstance(s, IndexMap) or s.ordered or s.scope.owned:
    return None
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  if not body or not isinstance(body[-1], ir.Store):
    return None
  store = body[-1]
  buf = store.buf
  if buf.uid not in allocs or store_counts.get(buf.uid, 0) != 1:
    return None
  prefix = body[:-1]
  if uses(prefix, {buf.uid}):
    return None
  facts = defs.fork() if isinstance(defs, IndexFacts) else IndexFacts(defs)
  for t in prefix:
    facts.record_stmt(t)
  return prefix, buf, store.index, facts.expr_key(store.index), store.value


def match_single_output_scan(
  s: ir.Stmt, defs: dict[int, Expr] | IndexFacts, allocs: set[int], store_counts: dict[int, int]
):
  """Match an ``IndexScan`` whose ``emit`` block writes a freshly allocated
  buffer once at its end.

  Returns ``(emit_prefix, buf, index_key, value)`` or ``None``."""
  if not isinstance(s, IndexScan) or s.scope.owned:
    return None
  emit = [t for t in s.emit if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  if not emit or not isinstance(emit[-1], ir.Store):
    return None
  store = emit[-1]
  buf = store.buf
  if buf.uid not in allocs or store_counts.get(buf.uid, 0) != 1:
    return None
  emit_prefix = emit[:-1]
  if uses(s.body + emit_prefix, {buf.uid}):
    return None
  facts = defs.fork() if isinstance(defs, IndexFacts) else IndexFacts(defs)
  for t in s.body + emit_prefix:
    facts.record_stmt(t)
  return emit_prefix, buf, facts.expr_key(store.index), store.value


def fuse_vertical_pair(
  producer: ir.Stmt,
  consumer: ir.Stmt,
  readers: dict[int, int],
  defs: dict[int, Expr] | IndexFacts,
  allocs: set[int] | BufferInfo,
  store_counts: dict[int, int],
  options,
  allow_partial: bool = False,
):
  """Try to vertically fuse ``producer`` into ``consumer``.

  Returns ``(fused_stmt, eliminated_buf_uid)`` or ``None``."""
  facts = defs if isinstance(defs, IndexFacts) else IndexFacts(defs)
  binfo = allocs if isinstance(allocs, BufferInfo) else None
  alloc_set = binfo.local_allocs if binfo is not None else allocs

  # ---- Map -> Map / Filter / Reduce -----------------------------------------
  map_match = match_single_output_map(producer, facts, alloc_set, store_counts)
  if map_match is not None:
    assert isinstance(producer, IndexMap)
    prefix, buf, store_index, index_key, value = map_match
    if binfo is not None and buf.uid in binfo.escapes:
      return None
    consumer_reads = count_reads_in([consumer], buf)
    if consumer_reads == 0:
      return None
    if not allow_partial and readers.get(buf.uid, 0) != consumer_reads:
      return None
    producer_reads, _ = buffer_accesses([producer])
    _, consumer_writes = buffer_accesses([consumer])
    if buffers_may_alias(producer_reads, consumer_writes, allocs):
      return None

    if isinstance(consumer, IndexMap) and not consumer.ordered:
      # Fast path: exact same-shape Map -> Map (computes producer once per iteration
      # and reuses ``elem`` across any number of loads at that same index, e.g. layer_norm)
      if not allow_partial and facts.same_shape(producer.shape, consumer.shape):
        elem = Var("v", buf.type.elt)
        consumer_body = [t for t in consumer.body if not (isinstance(t, ir.ReleaseScope) and t.scope is consumer.scope)]
        c_facts = facts.fork()
        for t in consumer_body:
          c_facts.record_stmt(t)
        subst = {v.uid: u for v, u in zip(consumer.index, producer.index)}
        rewriter = InlineBufferLoad(subst, buf, index_key, elem, c_facts)
        new_consumer_body = rewriter.block(consumer_body)
        if rewriter.ok and rewriter.replaced == consumer_reads:
          producer.body[:] = prefix + [Assign(elem, value)] + new_consumer_body + [ir.ReleaseScope(producer.scope)]
          producer.scope.owned.extend(rewriter.scope(consumer.scope).owned)
          producer.parallel = producer.parallel and consumer.parallel
          return producer, buf.uid

      # Generalized affine Map -> Map: transpose, shifted slice/stencil, broadcast, multi-load.
      # Never inline into a Load nested inside an inner loop/reduction of consumer (that would
      # multiply execution count and undo intentional layout copies like ``bt = b.T.copy()``).
      nested_buf_read = any(
        count_reads_in([t], buf) > 0
        for t in consumer.body
        if isinstance(t, (IndexReduce, IndexMap, ir.ForRange, ir.While))
      )
      if (
        not nested_buf_read
        and not any(isinstance(t, ir.Check) for t in ir.walk_stmts(prefix))
        and _is_row_major_store(store_index, producer, facts)
        and _affine_fusion_profitable(
          producer, consumer.shape, prefix, value, readers.get(buf.uid, consumer_reads), facts
        )
      ):
        c_facts = facts.fork()
        c_facts.activate_nonempty(consumer.shape)
        for v, d in zip(consumer.index, consumer.shape):
          c_facts.bind_index(v, d)
        consumer_body = [t for t in consumer.body if not (isinstance(t, ir.ReleaseScope) and t.scope is consumer.scope)]
        subst = {v.uid: v for v in defined([consumer]).values()}
        aff = AffineInlineBufferLoad(producer, prefix, value, buf, c_facts, subst)
        new_body = aff.block(consumer_body)
        if aff.ok and aff.replaced == consumer_reads:
          consumer.body[:] = new_body + [ir.ReleaseScope(consumer.scope)]
          return consumer, buf.uid

    if (
      isinstance(consumer, IndexFilter) and len(producer.shape) == 1 and facts.same_shape(producer.shape, (consumer.n,))
    ):
      # Map -> Filter: only inline straight-line predicates (filter runs its body twice)
      if not all(isinstance(t, (Assign, ir.Check, ir.Comment)) for t in prefix):
        return None
      elem = Var("v", buf.type.elt)
      inliner = Cloner({producer.index[0].uid: consumer.index})
      inlined_prefix = inliner.block(prefix)
      inlined_value = inliner.expr(value)
      p_facts = facts.fork()
      for t in inlined_prefix:
        p_facts.record_stmt(t)
      expected_key = p_facts.expr_key(inliner.expr(store_index))
      c_facts = facts.fork()
      for t in consumer.body:
        c_facts.record_stmt(t)
      rewriter = InlineBufferLoad(
        {v.uid: v for v in defined([consumer]).values()}, buf, expected_key, elem, c_facts.defs
      )
      new_consumer_body = [rewriter.stmt(t) for t in consumer.body]
      new_keep = rewriter.expr(consumer.keep)
      if not rewriter.ok or rewriter.replaced != consumer_reads:
        return None
      consumer.body[:] = inlined_prefix + [Assign(elem, inlined_value)] + new_consumer_body
      consumer.keep = new_keep
      return consumer, buf.uid

    if isinstance(consumer, IndexReduce) and consumer.where is None:
      # Map -> Reduce: keep matmul-shaped maps intact when tiling is enabled
      if options.tiling and any(isinstance(t, IndexReduce) for t in ir.walk_stmts(prefix)):
        return None
      if not allow_partial and facts.same_shape(producer.shape, consumer.shape):
        elem = Var("v", buf.type.elt)
        inliner = Cloner({u.uid: v for u, v in zip(producer.index, consumer.index)})
        inlined_prefix = inliner.block(prefix)
        inlined_value = inliner.expr(value)
        p_facts = facts.fork()
        for t in inlined_prefix:
          p_facts.record_stmt(t)
        expected_key = p_facts.expr_key(inliner.expr(store_index))
        c_facts = facts.fork()
        for t in consumer.body:
          c_facts.record_stmt(t)
        rewriter = InlineBufferLoad({v.uid: v for v in defined([consumer]).values()}, buf, expected_key, elem, c_facts)
        new_consumer_body = [rewriter.stmt(t) for t in consumer.body]
        new_folds = [rewriter.fold(f) for f in consumer.folds]
        if rewriter.ok and rewriter.replaced == consumer_reads:
          consumer.body[:] = inlined_prefix + [Assign(elem, inlined_value)] + new_consumer_body
          consumer.folds[:] = new_folds
          return consumer, buf.uid
      elif (
        not any(isinstance(t, ir.Check) for t in ir.walk_stmts(prefix))
        and _is_row_major_store(store_index, producer, facts)
        and _affine_fusion_profitable(
          producer, consumer.shape, prefix, value, readers.get(buf.uid, consumer_reads), facts
        )
      ):
        c_facts = facts.fork()
        c_facts.activate_nonempty(consumer.shape)
        for v, d in zip(consumer.index, consumer.shape):
          c_facts.bind_index(v, d)
        subst = {v.uid: v for v in defined([consumer]).values()}
        aff = AffineInlineBufferLoad(producer, prefix, value, buf, c_facts, subst)
        new_body = aff.block(consumer.body)
        aff.pending = []
        new_folds = [aff.fold(f) for f in consumer.folds]
        if aff.pending:
          new_body.extend(aff.pending)
        if aff.ok and aff.replaced == consumer_reads:
          consumer.body[:] = new_body
          consumer.folds[:] = new_folds
          return consumer, buf.uid

  # ---- Scan -> Map ----------------------------------------------------------
  scan_match = match_single_output_scan(producer, facts, alloc_set, store_counts)
  if scan_match is not None and isinstance(consumer, IndexMap) and not consumer.ordered and len(consumer.shape) == 1:
    assert isinstance(producer, IndexScan)
    if not facts.same_shape((producer.n,), consumer.shape):
      return None
    emit_prefix, buf, index_key, value = scan_match
    if binfo is not None and buf.uid in binfo.escapes:
      return None
    consumer_reads = count_reads_in([consumer], buf)
    if consumer_reads == 0 or readers.get(buf.uid, 0) != consumer_reads:
      return None
    producer_reads, _ = buffer_accesses([producer])
    _, consumer_writes = buffer_accesses([consumer])
    if buffers_may_alias(producer_reads, consumer_writes, allocs):
      return None
    elem = Var("v", buf.type.elt)
    consumer_body = [t for t in consumer.body if not (isinstance(t, ir.ReleaseScope) and t.scope is consumer.scope)]
    c_facts = facts.fork()
    for t in consumer_body:
      c_facts.record_stmt(t)
    rewriter = InlineBufferLoad({consumer.index[0].uid: producer.index}, buf, index_key, elem, c_facts)
    new_consumer_body = rewriter.block(consumer_body)
    if not rewriter.ok or rewriter.replaced != consumer_reads:
      return None
    producer.emit[:] = emit_prefix + [Assign(elem, value)] + new_consumer_body
    producer.scope.owned.extend(rewriter.scope(consumer.scope).owned)
    return producer, buf.uid

  return None


def combine_nested_map(outer: IndexMap, defs: dict[int, Expr] | IndexFacts) -> IndexMap | None:
  """``CombineNestedMaps`` (thesis Fig. 6.2): flatten ``IndexMap(S1, i -> IndexMap(S2, j -> body))``
  into ``IndexMap(S1 ++ S2, (i, j) -> body)`` when ``S2`` does not depend on ``i``."""
  if outer.scope.owned:
    return None
  body = [t for t in outer.body if not (isinstance(t, ir.ReleaseScope) and t.scope is outer.scope)]
  if not body or not isinstance(body[-1], IndexMap):
    return None
  inner = body[-1]
  prefix = body[:-1]
  if outer.ordered != inner.ordered:
    return None
  if not all(
    isinstance(t, (Assign, ir.Comment))
    and (not isinstance(t, Assign) or (not isinstance(t.var.type, ir.BufType) and pure(t.value)))
    for t in prefix
  ):
    return None
  outer_locals = set(defined(prefix)) | {v.uid for v in outer.index}
  facts = defs.fork() if isinstance(defs, IndexFacts) else IndexFacts(defs)
  for t in prefix:
    facts.record_stmt(t)
  lifted_shape = [facts.lift_outside(d, outer_locals) for d in inner.shape]
  if any(d is None for d in lifted_shape):
    return None
  inner_body = [t for t in inner.body if not (isinstance(t, ir.ReleaseScope) and t.scope is inner.scope)]
  outer.scope.owned.extend(inner.scope.owned)
  return IndexMap(
    outer.shape + tuple(lifted_shape),  # type: ignore[arg-type]
    outer.index + inner.index,
    prefix + inner_body + [ir.ReleaseScope(outer.scope)],
    outer.scope,
    outer.ordered,
    outer.parallel or inner.parallel,
    outer.loc,
  )


def _is_hoistable_check_stmt(t: ir.Stmt, blocked_uids: set[int]) -> bool:
  if isinstance(t, ir.Check):
    return pure(t.cond) and all(pure(a) for a in t.args) and not uses([t], blocked_uids)
  if isinstance(t, ir.If) and not t.orelse and t.then and all(isinstance(x, ir.Check) for x in t.then):
    return (
      pure(t.cond)
      and all(pure(x.cond) and all(pure(a) for a in x.args) for x in t.then)
      and not uses([t], blocked_uids)
    )
  return False


def _try_fuse_multi_consumers(
  producer: IndexMap,
  rest: list[ir.Stmt],
  binfo: BufferInfo,
  facts: IndexFacts,
  options,
) -> tuple[list[ir.Stmt], int, int] | None:
  """If ``producer`` is a cheap, pure, non-escaping single-output map read by
  several adjacent consumers in ``rest``, fuse it into every consumer and
  eliminate its temporary buffer.

  Returns ``(replacement_stmts, consumed_count, dead_buf_uid)`` or ``None``."""
  map_match = match_single_output_map(producer, facts, binfo.local_allocs, binfo.store_counts)
  if map_match is None:
    return None
  prefix, buf, store_index, _, value = map_match
  if buf.uid in binfo.escapes or binfo.readers.get(buf.uid, 0) <= 1:
    return None
  if any(isinstance(t, (ir.Check, *ir.OPERATORS, ir.ForRange, ir.While)) for t in ir.walk_stmts(prefix)):
    return None
  if op_cost(prefix) + expr_cost(value) > CHEAP_FUSE_COST:
    return None
  if not _is_row_major_store(store_index, producer, facts):
    return None

  total_reads = binfo.readers.get(buf.uid, 0)
  producer_reads, _ = buffer_accesses([producer])
  producer_defs = set(defined([producer]))
  found_reads = 0
  replaced: list[ir.Stmt] = []
  step_facts = facts.fork()
  j = 0
  while j < len(rest) and found_reads < total_reads:
    t = rest[j]
    j += 1
    if isinstance(t, (IndexMap, IndexReduce, IndexFilter)) and count_reads_in([t], buf) > 0:
      c_reads = count_reads_in([t], buf)
      # Clone consumer before attempting partial fusion so failed multi-consumer attempts never mutate IR
      t_copy = Cloner({v.uid: v for v in defined([t]).values()}).stmt(t)
      res = fuse_vertical_pair(
        producer, t_copy, binfo.readers, step_facts, binfo, binfo.store_counts, options, allow_partial=True
      )
      if res is None:
        return None
      fused, _ = res
      replaced.append(fused)
      found_reads += c_reads
      _, w = buffer_accesses([fused])
      if binfo.may_alias(producer_reads, w):
        if found_reads < total_reads:
          return None
      continue
    if uses([t], {buf.uid} | producer_defs):
      return None
    _, w = buffer_accesses([t])
    if binfo.may_alias(producer_reads, w):
      return None
    step_facts.record_stmt(t)
    replaced.append(t)
  if found_reads != total_reads:
    return None
  return replaced, j, buf.uid


def vertical_block(
  block: list,
  binfo: BufferInfo,
  facts: IndexFacts,
  gone: set[int],
  options,
) -> tuple[list, bool]:
  """Walk ``block`` flattening nested maps and fusing adjacent producer-consumer
  operators (including affine transpose/slice/stencil and cheap multi-consumer fusion)."""
  changed = False
  local_facts = facts.fork()
  normalized: list = []
  for s in block:
    local_facts.record_stmt(s)
    sub_facts = local_facts.fork()
    if isinstance(s, (IndexMap, IndexReduce)):
      sub_facts.activate_nonempty(s.shape)
    for b in ir.sub_blocks(s):
      b[:], inner = vertical_block(b, binfo, sub_facts, gone, options)
      changed |= inner
    if isinstance(s, IndexMap):
      squeezed = squeeze_unit_map(s)
      if squeezed is not None:
        s = squeezed
        changed = True
      split_out = split_invariant_map(s)
      if split_out is not None:
        normalized.extend(split_out)
        changed = True
        continue
      flat = combine_nested_map(s, local_facts)
      if flat is not None:
        s = flat
        changed = True
    normalized.append(s)
  block = normalized
  out: list = []
  pair_facts = facts.fork()
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    pair_facts.record_stmt(s)
    if not isinstance(s, (IndexMap, IndexScan)):
      out.append(s)
      continue
    # Step over hoistable statements (allocs, pure scalar assigns, comments,
    # and pure checks / guarded checks when s cannot raise and only writes fresh buffers)
    j = k
    between: list = []
    producer_defs = set(defined([s]))
    _, producer_writes = buffer_accesses([s])
    producer_cannot_raise = producer_writes <= binfo.local_allocs and not any(
      isinstance(x, ir.Check) or (isinstance(x, IndexReduce) and any(f.empty is not None for f in x.folds))
      for x in ir.walk_stmts([s])
    )
    step_facts = pair_facts.fork()
    while j < len(block):
      t = block[j]
      if isinstance(t, ir.Comment):
        between.append(t)
      elif isinstance(t, ir.Alloc) and not uses([t], producer_defs | producer_writes):
        between.append(t)
      elif producer_cannot_raise and _is_hoistable_check_stmt(t, producer_defs | producer_writes):
        step_facts.record_stmt(t)
        between.append(t)
      elif (
        isinstance(t, Assign)
        and not isinstance(t.var.type, ir.BufType)
        and pure(t.value)
        and not uses([t], producer_defs | producer_writes)
      ):
        step_facts.record_stmt(t)
        between.append(t)
      else:
        break
      j += 1
    nxt = block[j] if j < len(block) else None
    if nxt is not None and isinstance(nxt, (IndexMap, IndexFilter, IndexReduce)):
      res = fuse_vertical_pair(s, nxt, binfo.readers, step_facts, binfo, binfo.store_counts, options)
      if res is not None:
        fused, dead_buf = res
        out.extend(between)
        pair_facts = step_facts
        out.append(fused)
        gone.add(dead_buf)
        k = j + 1
        changed = True
        continue
    if isinstance(s, IndexMap):
      multi = _try_fuse_multi_consumers(s, block[k:], binfo, pair_facts, options)
      if multi is not None:
        repl, consumed, dead_buf = multi
        out.extend(repl)
        gone.add(dead_buf)
        k += consumed
        changed = True
        continue
    out.append(s)
  return out, changed


def fuse_vertical(fn: ir.Function, options) -> None:
  """Vertical fusion across operator statements: CombineNestedMaps, affine
  Map->Map (including epilogues, transposes, stencils, and cheap multi-consumer
  fusion), Map->Filter, Scan->Map, and Map->Reduce."""
  if not options.fusion:
    return
  for _ in range(4):
    gone: set[int] = set()
    binfo = BufferInfo.analyze(fn)
    facts = IndexFacts.from_function(fn)
    fn.body[:], changed = vertical_block(fn.body, binfo, facts, gone, options)
    if gone:
      drop_buffers(fn, gone, drop_allocs=True)
    if not changed:
      break


# ---------------------------------------------------------------------------
# Filter-Filter and FilterReduce
# ---------------------------------------------------------------------------


def strip(e: Expr) -> Expr:
  while isinstance(e, ir.Cast):
    e = e.arg
  return e


class Survivors(Cloner):
  """Rewrites a consumer of a filter's output into terms of the filter's own
  index: ``idx[k]`` is ``i``, and ``i`` is known to be in ``[0, n)``."""

  def __init__(self, f: IndexFilter, consumer_stmts: list, k_var: Var, defs: dict[int, Expr]):
    super().__init__({v.uid: v for v in defined(consumer_stmts).values()})
    self.f, self.k, self.defs = f, k_var, defs
    self.inrange: set[int] = {f.index.uid}  # variables equal to the filter's index
    self.ok = True

  def is_survivor_index(self, e: Expr) -> bool:
    """True if ``e`` is ``self.k`` or ``self.k * select(f.count == 1, 0, 1)``
    (which equals ``self.k`` for every ``0 <= self.k < f.count``)."""
    if e is self.k:
      return True
    r = resolved(e, self.defs)
    if r is self.k:
      return True
    if isinstance(r, ir.Prim) and r.op == "mul" and len(r.args) == 2:
      a, b = r.args
      if b is self.k:
        a, b = b, a
      if a is self.k and isinstance(b, ir.Prim) and b.op == "select" and len(b.args) == 3:
        cond, t_val, f_val = b.args
        if (
          isinstance(cond, ir.Prim)
          and cond.op == "eq"
          and len(cond.args) == 2
          and (
            (cond.args[0] is self.f.count and isinstance(cond.args[1], Const) and cond.args[1].value == 1)
            or (cond.args[1] is self.f.count and isinstance(cond.args[0], Const) and cond.args[0].value == 1)
          )
          and isinstance(t_val, Const)
          and t_val.value == 0
          and isinstance(f_val, Const)
          and f_val.value == 1
        ):
          return True
    return False

  def expr(self, e: Expr) -> Expr:
    if isinstance(e, ir.Load) and e.buf is self.f.out:
      if not self.is_survivor_index(e.index):
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


class CollapseFilterChain(Cloner):
  """After fusing ``first`` into ``second`` so ``second.out`` holds original
  indices in ``[0, first.n)`` directly, rewrite remaining consumers so
  ``first.out[norm(second.out[m])]`` collapses to ``second.out[m]``."""

  def __init__(self, first: IndexFilter, second: IndexFilter, rest: list, defs: dict[int, Expr]):
    super().__init__({v.uid: v for v in defined(rest).values()})
    self.first, self.second, self.defs = first, second, defs
    self.from_second: set[int] = set()  # variables holding a value loaded from second.out
    self.ok = True

  def is_second_index(self, e: Expr) -> bool:
    e = strip(e)
    return isinstance(e, Var) and e.uid in self.from_second

  def is_second_load(self, e: Expr) -> bool:
    e = strip(e)
    return isinstance(e, ir.Load) and e.buf is self.second.out

  def is_negative_wrap(self, c: Expr, otherwise: Expr) -> bool:
    return (
      isinstance(c, ir.Prim)
      and c.op == "lt"
      and self.is_second_index(c.args[0])
      and isinstance(c.args[1], Const)
      and c.args[1].value == 0
      and self.is_second_index(otherwise)
    )

  def is_first_count_check(self, c: Expr) -> bool:
    if not (isinstance(c, ir.Prim) and c.op == "and"):
      return False
    lo, hi = c.args
    return (
      isinstance(lo, ir.Prim) and lo.op == "ge" and self.is_second_index(lo.args[0])
      and isinstance(hi, ir.Prim) and hi.op == "lt" and self.is_second_index(hi.args[0])
      and key(resolved(hi.args[1], self.defs)) == key(resolved(self.first.count, self.defs))
    )  # fmt: skip

  def expr(self, e: Expr) -> Expr:
    if isinstance(e, ir.Load) and e.buf is self.first.out:
      idx = self.expr(e.index)
      if self.is_second_index(idx):
        return ir.Cast(idx, e.type) if e.type is not idx.type else idx
      self.ok = False
      return e
    if isinstance(e, Var) and (e is self.first.out or e is self.first.count):
      self.ok = False
      return e
    if isinstance(e, ir.Prim):
      if e.op == "select" and self.is_negative_wrap(e.args[0], e.args[2]):
        return self.expr(e.args[2])
      return ir.Prim(e.op, tuple(self.expr(a) for a in e.args), e.type)
    if isinstance(e, ir.Cast):
      return ir.Cast(self.expr(e.arg), e.type)
    if isinstance(e, ir.Load):
      return ir.Load(e.buf, self.expr(e.index))
    if isinstance(e, ir.RtCall):
      return ir.RtCall(e.name, tuple(self.expr(a) for a in e.args), e.type, e.pure)
    return super().expr(e)

  def rewrite_block(self, block: list) -> list:
    """Rewrite ``block`` in place of the removed intermediate filter buffer."""
    out: list = []
    for s in block:
      if isinstance(s, ir.Check) and self.is_first_count_check(s.cond):
        continue
      if isinstance(s, Assign):
        rhs = self.expr(s.value)
        if self.is_second_load(s.value) or self.is_second_index(rhs):
          self.from_second.add(s.var.uid)
        out.append(Assign(self.var(s.var), rhs))
        continue
      if isinstance(s, ir.If):
        out.append(ir.If(self.expr(s.cond), self.rewrite_block(s.then), self.rewrite_block(s.orelse)))
        continue
      if isinstance(s, IndexMap):
        out.append(
          IndexMap(
            tuple(self.expr(d) for d in s.shape),
            tuple(self.var(v) for v in s.index),
            self.rewrite_block(s.body),
            self.scope(s.scope),
            s.ordered,
            s.parallel,
            s.loc,
          )
        )
        continue
      if isinstance(s, IndexReduce):
        out.append(
          IndexReduce(
            tuple(self.expr(d) for d in s.shape),
            tuple(self.var(v) for v in s.index),
            self.rewrite_block(s.body),
            [self.fold(f) for f in s.folds],
            self.scope(s.scope),
            self.home or s.home,
            s.parallel,
            s.loc,
            self.rewrite_block(s.pre),
            None if s.where is None else self.expr(s.where),
          )
        )
        continue
      if isinstance(s, IndexFilter):
        out.append(
          IndexFilter(
            self.expr(s.n),
            self.var(s.index),
            self.rewrite_block(s.body),
            self.expr(s.keep),
            self.var(s.out),
            self.var(s.count),
            self.scope(s.scope),
            s.parallel,
            s.loc,
          )
        )
        continue
      if isinstance(s, IndexScan):
        out.append(
          IndexScan(
            self.expr(s.n),
            self.var(s.index),
            self.rewrite_block(s.body),
            self.fold(s.fold),
            self.rewrite_block(s.emit),
            self.scope(s.scope),
            s.parallel,
            s.loc,
          )
        )
        continue
      out.append(self.stmt(s))
    return out


def filter_filter(first: IndexFilter, second: IndexFilter, rest: list, readers: dict[int, int], defs: dict[int, Expr]):
  """``Filter(q, Filter(p, X)) => Filter(p and q, X)`` with short-circuiting."""
  if second.n is not first.count:
    return None
  # All reads of first.out and first.count in the whole function must be inside second + rest
  out_reads_here = count_reads_in([second] + rest, first.out)
  count_reads_here = count_reads_in([second] + rest, first.count)
  if readers.get(first.out.uid, 0) != out_reads_here or readers.get(first.count.uid, 0) != count_reads_here:
    return None
  # second's own body cannot depend on first.count
  if uses(second.body, {first.count.uid}) or uses([Assign(second.count, second.keep)], {first.count.uid}):
    return None
  survivors = Survivors(first, [second], second.index, defs)
  second_body = survivors.body(second.body)
  second_keep = survivors.expr(second.keep)
  if not survivors.ok:
    return None
  collapser = CollapseFilterChain(first, second, rest, defs)
  new_rest = collapser.rewrite_block(rest)
  if not collapser.ok:
    return None
  keep_both = Var("keep", Bool)
  fused_body = list(first.body) + [
    Assign(keep_both, Const(False, Bool)),
    ir.If(first.keep, second_body + [Assign(keep_both, second_keep)], []),
  ]
  scope = survivors.scope(second.scope)
  scope.owned.extend(first.scope.owned)
  fused = IndexFilter(
    first.n,
    first.index,
    fused_body,
    keep_both,
    second.out,
    second.count,
    scope,
    first.parallel and second.parallel,
    second.loc,
  )
  return fused, new_rest


def filter_signature(f: IndexFilter, defs: dict[int, Expr]):
  """Structural alpha-invariant key for a pure ``IndexFilter`` (or ``None`` if effectful)."""
  if f.scope.owned or has_effects(f.body):
    return None
  local_idx: dict[int, int] = {f.index.uid: 0}

  def ek(e: Expr):
    if isinstance(e, Var):
      if e.uid in local_idx:
        return ("loc", local_idx[e.uid], e.type)
      return ("var", e.uid)
    if isinstance(e, Const):
      return ("c", e.value, e.type)
    if isinstance(e, ir.Load):
      return ("ld", e.buf.uid, ek(e.index))
    if isinstance(e, ir.Prim):
      return ("p", e.op, e.type, tuple(ek(a) for a in e.args))
    if isinstance(e, ir.Cast):
      return ("t", e.type, ek(e.arg))
    if isinstance(e, ir.RtCall):
      return ("rt", e.name, e.pure, e.type, tuple(ek(a) for a in e.args))
    return ("?", id(e))

  body_keys = []
  for st in f.body:
    if isinstance(st, ir.Comment):
      continue
    if isinstance(st, Assign) and not isinstance(st.var.type, ir.BufType) and pure(st.value):
      vk = ek(st.value)
      slot = local_idx[st.var.uid] = len(local_idx)
      body_keys.append(("assign", slot, st.var.type, vk))
    elif isinstance(st, ir.ReleaseScope) and st.scope is f.scope:
      continue
    else:
      return None
  return (key(resolved(f.n, defs)), tuple(body_keys), ek(f.keep))


def _is_trivial_self_broadcast_check(s: ir.Stmt) -> bool:
  """True for ``check ((x == x) or ...)`` left after deduplicating identical filters."""
  if not isinstance(s, ir.Check):
    return False
  c = s.cond
  while isinstance(c, ir.Prim) and c.op == "or" and len(c.args) == 2:
    if isinstance(c.args[0], ir.Prim) and c.args[0].op == "eq" and key(c.args[0].args[0]) == key(c.args[0].args[1]):
      return True
    c = c.args[1]
  return isinstance(c, ir.Prim) and c.op == "eq" and key(c.args[0]) == key(c.args[1])


class _FoldIdenticalSelect(Cloner):
  """Substitute ``subst`` and fold ``select(cond, x, x) -> x``."""

  def expr(self, e: Expr) -> Expr:
    out = super().expr(e)
    if isinstance(out, ir.Prim) and out.op == "select" and len(out.args) == 3 and key(out.args[1]) == key(out.args[2]):
      return out.args[1]
    return out


def dedup_filters(
  block: list, readers: dict[int, int], defs: dict[int, Expr], allocs: set[int], gone: set[int]
) -> tuple[list, bool]:
  """Eliminate duplicate pure ``IndexFilter``s with identical predicates (e.g.
  ``mask = x > 0; np.sum(x[mask] * w[mask])``) so downstream ``FilterReduce``
  sees a single filter."""
  changed = False
  local_defs = dict(defs)
  seen: dict[object, tuple[IndexFilter, set[int]]] = {}
  out: list = []
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    record_check_equality(s, local_defs)
    if isinstance(s, IndexFilter):
      sig = filter_signature(s, local_defs)
      rest = block[k:]
      can_dedup = (
        sig is not None
        and sig in seen
        and readers.get(s.out.uid, 0) == count_reads_in(rest, s.out)
        and readers.get(s.count.uid, 0) == count_reads_in(rest, s.count)
        and not any(isinstance(t, ir.Incref) and t.buf is s.out for t in ir.walk_stmts(rest))
      )
      if can_dedup:
        prev, _ = seen[sig]
        # Replace s.out -> prev.out and s.count -> prev.count in the rest of block
        subst = {s.out.uid: prev.out, s.count.uid: prev.count}
        rew = _FoldIdenticalSelect(subst)
        for v in defined(rest).values():
          rew.m[v.uid] = v
        new_rest = []
        for t in rest:
          nt = rew.stmt(t)
          if _is_trivial_self_broadcast_check(nt):
            continue
          new_rest.append(nt)
        gone.add(s.out.uid)
        out.extend(new_rest)
        return out, True
      if sig is not None:
        reads, _ = buffer_accesses(s.body)
        seen[sig] = (s, reads)
      out.append(s)
      continue
    # Invalidate cached filters if this statement may write to any buffer they read
    _, writes = buffer_accesses([s])
    if writes:
      for sig in [sg for sg, (_, reads) in seen.items() if buffers_may_alias(reads, writes, allocs)]:
        del seen[sig]
    out.append(s)
  return out, changed


def filter_reduce(
  f: IndexFilter,
  r: IndexReduce,
  readers: dict[int, int],
  defs: dict[int, Expr],
  dead_count_reads: int = 0,
):
  """Fuse an ``IndexFilter`` followed by an ``IndexReduce`` over its survivors
  into one guarded ``IndexReduce`` with no intermediate index array."""
  from .lowering import identity

  if len(r.shape) != 1 or r.where is not None or any(g.init is None for g in r.folds):
    return None
  if resolved(r.shape[0], defs) is not f.count:
    return None
  # f.count is only computed at the end of the fused loop, so r's body/inits cannot read it
  if uses(r.body, {f.count.uid}) or any(
    uses([Assign(g.result, g.init)], {f.count.uid}) for g in r.folds if g.init is not None
  ):
    return None
  rewriter = Survivors(f, [r], r.index[0], defs)
  body = rewriter.body(r.body)
  folds = [rewriter.fold(g) for g in r.folds]
  if not rewriter.ok:
    return None
  # nothing but this reduction may have looked at the kept indices
  inside = count_reads_in([r], f.out)
  if readers.get(f.out.uid, 0) != inside:
    return None
  remaining_count_reads = readers.get(f.count.uid, 0) - dead_count_reads
  if remaining_count_reads > 1:  # someone else wants to know how many survived
    a, b = Var("acc", f.count.type), Var("x", f.count.type)
    folds.append(
      ir.Fold(
        f.count, Const(1, f.count.type), a, b, [], ir.Prim("add", (a, b), f.count.type), Const(0, f.count.type), "add"
      )
    )
  parallel = f.parallel and all(
    g.op is not None and not g.combine and identity(g.op, g.a.type) is not None for g in folds
  )
  scope = rewriter.scope(r.scope)
  scope.owned.extend(f.scope.owned)
  return IndexReduce((f.n,), (f.index,), body, folds, scope, r.home, parallel, r.loc, list(f.body), f.keep)


def filter_block(
  block: list, readers: dict[int, int], defs: dict[int, Expr], allocs: set[int], gone: set[int]
) -> tuple[list, bool]:
  """Walk ``block`` deduplicating identical filters and collapsing
  ``Filter->Filter`` and ``Filter->Reduce`` pairs."""
  changed = False
  local_defs = dict(defs)
  for s in block:
    record_check_equality(s, local_defs)
    for b in ir.sub_blocks(s):
      b[:], inner = filter_block(b, readers, local_defs, allocs, gone)
      changed |= inner
  block, deduped = dedup_filters(block, readers, defs, allocs, gone)
  if deduped:
    return block, True
  out: list = []
  walk_defs = dict(defs)
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    record_check_equality(s, walk_defs)
    if not isinstance(s, IndexFilter):
      out.append(s)
      continue
    nxt = block[k] if k < len(block) else None
    if isinstance(nxt, IndexFilter):
      res = filter_filter(s, nxt, block[k + 1 :], readers, walk_defs)
      if res is not None:
        fused, new_rest = res
        out.append(fused)
        out.extend(new_rest)
        gone.add(s.out.uid)
        return out, True
    # Step over pure checks/comments/scalar assigns between IndexFilter and IndexReduce
    j = k
    hoist: list = []
    bridge_assigns: list[Assign] = []
    step_defs = dict(walk_defs)
    while j < len(block):
      t = block[j]
      if isinstance(t, ir.Comment):
        hoist.append(t)
      elif (
        isinstance(t, ir.Check)
        and pure(t.cond)
        and all(pure(a) for a in t.args)
        and not uses([t], {s.out.uid, s.count.uid})
      ):
        record_check_equality(t, step_defs)
        hoist.append(t)
      elif (
        isinstance(t, Assign)
        and not isinstance(t.var.type, ir.BufType)
        and pure(t.value)
        and not uses([t], {s.out.uid})
      ):
        if uses([t], {s.count.uid}):
          bridge_assigns.append(t)
        else:
          hoist.append(t)
        if t.var.type.kind in "iu":
          step_defs[t.var.uid] = t.value
      else:
        break
      j += 1
    nxt = block[j] if j < len(block) else None
    if isinstance(nxt, IndexReduce):
      # Bridge assigns (e.g. dim = nnz, bmul = select(nnz == 1, 0, 1)) that are only read
      # by nxt (and eliminated by Survivors) do not keep s.count alive!
      rest_after = block[j + 1 :]
      dead_bridges = [a for a in bridge_assigns if count_reads_in(rest_after, a.var) == 0]
      dead_count_reads = (
        sum(count_reads_in([a], s.count) for a in dead_bridges) if len(dead_bridges) == len(bridge_assigns) else 0
      )
      # Also account for r.shape[0] being an alias (e.g. dim_15) rather than s.count directly
      if len(dead_bridges) == len(bridge_assigns) and nxt.shape and nxt.shape[0] is not s.count:
        dead_count_reads = max(dead_count_reads - 1, 0)
      fused = filter_reduce(s, nxt, readers, step_defs, dead_count_reads=dead_count_reads)
      if fused is not None:
        out.extend(hoist)
        walk_defs = step_defs
        # Keep any bridge assigns still referenced after nxt
        if len(dead_bridges) < len(bridge_assigns):
          out.append(fused)
          out.extend(a for a in bridge_assigns if a not in dead_bridges)
        else:
          out.append(fused)
        gone.add(s.out.uid)
        k = j + 1
        changed = True
        continue
    out.append(s)
  return out, changed


def fuse_filters(fn: ir.Function, options) -> None:
  """Filter deduplication, Filter-Filter, and FilterReduce fusion."""
  if not options.fusion:
    return
  for _ in range(4):
    gone: set[int] = set()
    binfo = BufferInfo.analyze(fn)
    facts = IndexFacts.from_function(fn)
    fn.body[:], changed = filter_block(fn.body, binfo.readers, facts.defs, binfo.local_allocs, gone)
    if gone:
      drop_buffers(fn, gone, drop_allocs=True)
    if not changed:
      break


# ---------------------------------------------------------------------------
# sibling fusion (horizontal fusion of reductions and maps)
# ---------------------------------------------------------------------------


def simd_op(r: IndexReduce) -> str | None:
  """The operation a backend may vectorise the fold by, if there is one."""
  op = r.folds[0].op
  return op if op in ("add", "mul") and all(f.op == op and not f.combine for f in r.folds) else None


def compatible(a: IndexReduce, b: IndexReduce, defs: dict[int, Expr] | IndexFacts) -> bool:
  """True if two ``IndexReduce`` statements can safely share one loop."""
  if len(a.shape) != len(b.shape) or a.parallel != b.parallel or a.where is not None or b.where is not None:
    return False
  if (a.folds[0].init is None) != (b.folds[0].init is None):
    return False
  if has_effects(a.body) or has_effects(b.body):
    return False
  # do not trade a vectorised loop for a fused one that is not
  if simd_op(a) != simd_op(b):
    return False
  return same_shape(a.shape, b.shape, defs)


def absorb(a: IndexReduce, b: IndexReduce) -> None:
  """Move the folds of ``b`` into ``a``."""
  cloner = Cloner({v.uid: v for v in defined([b]).values()})  # same names...
  cloner.m.update({v.uid: i for v, i in zip(b.index, a.index)})  # ...but the index of ``a``
  a.body.extend(cloner.stmt(t) for t in b.body)
  a.folds.extend(cloner.fold(f) for f in b.folds)
  a.scope.owned.extend(cloner.scope(b.scope).owned)


def contains_reduce(s: IndexMap) -> bool:
  """True if ``s`` wraps an ``IndexReduce`` (e.g. a matmul or row-reduction)."""
  return any(isinstance(t, IndexReduce) for t in ir.walk_stmts(s.body))


def compatible_maps(
  a: IndexMap, b: IndexMap, defs: dict[int, Expr] | IndexFacts, allocs: set[int] | BufferInfo
) -> bool:
  """True if two ``IndexMap`` statements over the same shape are independent
  and safe to merge into one loop."""
  if a.ordered != b.ordered or a.parallel != b.parallel or not same_shape(a.shape, b.shape, defs):
    return False
  # Leave matmul-shaped maps untouched so tiling / interchange still recognise them
  if contains_reduce(a) or contains_reduce(b):
    return False
  a_reads, a_writes = buffer_accesses(a.body)
  b_reads, b_writes = buffer_accesses(b.body)
  if buffers_may_alias(a_writes, b_reads | b_writes, allocs) or buffers_may_alias(b_writes, a_reads | a_writes, allocs):
    return False
  local_allocs = allocs.local_allocs if isinstance(allocs, BufferInfo) else allocs
  # Preserve exact state on exception: do not hoist a check past a write to an input array
  if any(isinstance(t, ir.Check) for t in ir.walk_stmts(b.body)) and any(u not in local_allocs for u in a_writes):
    return False
  return True


def absorb_map(a: IndexMap, b: IndexMap) -> None:
  """Merge ``b``'s body into ``a`` over ``a.index``."""
  a_body = [t for t in a.body if not (isinstance(t, ir.ReleaseScope) and t.scope is a.scope)]
  b_body = [t for t in b.body if not (isinstance(t, ir.ReleaseScope) and t.scope is b.scope)]
  cloner = Cloner({v.uid: v for v in defined([b]).values()})
  cloner.m.update({v.uid: i for v, i in zip(b.index, a.index)})
  a.body[:] = a_body + [cloner.stmt(t) for t in b_body] + [ir.ReleaseScope(a.scope)]
  a.scope.owned.extend(cloner.scope(b.scope).owned)


def fuse_block(block: list, defs: dict[int, Expr] | IndexFacts, allocs: set[int] | BufferInfo) -> list:
  """Walk ``block`` merging adjacent compatible reductions and maps."""
  facts = defs.fork() if isinstance(defs, IndexFacts) else IndexFacts(defs)
  for s in block:
    facts.record_stmt(s)
    sub_facts = facts.fork()
    if isinstance(s, (IndexMap, IndexReduce)):
      sub_facts.activate_nonempty(s.shape)
    for b in ir.sub_blocks(s):
      b[:] = fuse_block(b, sub_facts, allocs)
  out: list = []
  walk_facts = defs.fork() if isinstance(defs, IndexFacts) else IndexFacts(defs)
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    walk_facts.record_stmt(s)
    if isinstance(s, IndexReduce):
      # ``s`` may sink past independent scalar definitions (hoisted before ``s``)
      # and step past pure scalar post-processing of its own results (sunk after ``s``)
      # to fuse with an independent sibling reduction.
      sunk: list = []
      sunk_defs: set[int] = {v.uid for f in s.folds for v in (f.result, *(p.result for p in f.payloads))}
      while True:
        hoist: list = []
        sink_more: list = []
        sink_more_defs = set(sunk_defs)
        step_facts = walk_facts.fork()
        j = k
        while j < len(block):
          t = block[j]
          if isinstance(t, ir.Comment):
            hoist.append(t)
          elif isinstance(t, Assign) and not isinstance(t.var.type, ir.BufType) and pure(t.value):
            if uses([t], sink_more_defs):
              sink_more.append(t)
              sink_more_defs.add(t.var.uid)
            else:
              step_facts.record_stmt(t)
              hoist.append(t)
          else:
            break
          j += 1
        nxt = block[j] if j < len(block) else None
        if not isinstance(nxt, IndexReduce) or uses([nxt], sink_more_defs) or not compatible(s, nxt, step_facts):
          break
        out.extend(hoist)
        walk_facts = step_facts
        sunk.extend(sink_more)
        sunk_defs = sink_more_defs | {v.uid for f in nxt.folds for v in (f.result, *(p.result for p in f.payloads))}
        absorb(s, nxt)
        k = j + 1
      out.append(s)
      out.extend(sunk)
      continue
    if isinstance(s, IndexMap):
      sunk = []
      while True:
        _, a_writes = buffer_accesses(s.body)
        map_outputs = set(defined([s])) | a_writes
        hoist = []
        sink_more = []
        step_facts = walk_facts.fork()
        j = k
        while j < len(block):
          t = block[j]
          if isinstance(t, ir.Comment):
            hoist.append(t)
          elif isinstance(t, ir.Incref):
            sink_more.append(t)
          elif isinstance(t, ir.Alloc) and not uses([t], map_outputs):
            hoist.append(t)
          elif (
            isinstance(t, Assign)
            and not isinstance(t.var.type, ir.BufType)
            and pure(t.value)
            and not uses([t], map_outputs)
          ):
            step_facts.record_stmt(t)
            hoist.append(t)
          else:
            break
          j += 1
        nxt = block[j] if j < len(block) else None
        if not isinstance(nxt, IndexMap) or not compatible_maps(s, nxt, step_facts, allocs):
          break
        out.extend(hoist)
        walk_facts = step_facts
        sunk.extend(sink_more)
        absorb_map(s, nxt)
        k = j + 1
      out.append(s)
      out.extend(sunk)
      continue
    out.append(s)
  return out


def fuse_siblings(fn: ir.Function, options) -> None:
  """Sibling fusion of adjacent reductions and independent maps."""
  if not options.fusion:
    return
  binfo = BufferInfo.analyze(fn)
  facts = IndexFacts.from_function(fn)
  fn.body[:] = fuse_block(fn.body, facts, binfo)


__all__ = [
  "AffineInlineBufferLoad",
  "CollapseFilterChain",
  "InlineBufferLoad",
  "Survivors",
  "absorb",
  "absorb_map",
  "buffer_accesses",
  "buffers_may_alias",
  "combine_nested_map",
  "compatible",
  "compatible_maps",
  "dedup_filters",
  "filter_filter",
  "filter_reduce",
  "filter_signature",
  "fuse_filters",
  "fuse_siblings",
  "fuse_vertical",
  "fuse_vertical_pair",
  "record_check_equality",
  "split_invariant_map",
  "squeeze_unit_map",
]
