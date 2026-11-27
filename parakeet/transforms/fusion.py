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
from ..syntax.ir import Assign, Const, Expr, IndexFilter, IndexMap, IndexReduce, IndexScan, IndexScatter, Var
from .analysis import (
  _EXPENSIVE_PRIMS,
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
  if any(isinstance(t, (ir.Store, ir.IndexScatter)) for t in ir.walk_stmts(prefix)):
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
  if any(isinstance(t, (ir.Store, ir.IndexScatter)) for t in ir.walk_stmts(emit_prefix)):
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

      # Reshape-transparent epilogue fusion: producer writes row-major (e.g. matmul (B*T, N))
      # and consumer maps over a reshaped layout of the same total size (e.g. (B, T, N)).
      if (
        not allow_partial
        and not contains_reduce(consumer)
        and _is_row_major_store(store_index, producer, facts)
        and facts.same_expr(ops.iprod(producer.shape), ops.iprod(consumer.shape))
        and consumer.index
      ):
        elem = Var("v", buf.type.elt)
        consumer_body = [t for t in consumer.body if not (isinstance(t, ir.ReleaseScope) and t.scope is consumer.scope)]
        c_facts = facts.fork()
        for t in consumer_body:
          c_facts.record_stmt(t)
        c_row: Expr = consumer.index[0]
        for idx_v, dim in zip(consumer.index[1:], consumer.shape[1:]):
          c_row = ops.iadd(ops.imul(c_row, dim), idx_v)
        c_expected_key = c_facts.expr_key(c_row)
        rewriter = InlineBufferLoad(
          {v.uid: v for v in defined([consumer]).values()}, buf, c_expected_key, elem, c_facts
        )
        # First verify every Load of buf is at row-major consumer index and rewrite to elem
        replaced_body = rewriter.block(consumer_body)
        if rewriter.ok and rewriter.replaced == consumer_reads:
          # Map producer.index -> consumer.index, sharing any identical trailing dimensions
          p_rank, c_rank = len(producer.shape), len(consumer.shape)
          tail = 0
          while tail < min(p_rank, c_rank) and facts.same_expr(
            producer.shape[p_rank - 1 - tail], consumer.shape[c_rank - 1 - tail]
          ):
            tail += 1
          p_lead = p_rank - tail
          c_lead = c_rank - tail
          c_subst: dict[int, Expr] = {}
          for t_i in range(tail):
            c_subst[consumer.index[c_rank - 1 - t_i].uid] = producer.index[p_rank - 1 - t_i]
          unravel_stmts: list[ir.Stmt] = []
          if c_lead > 0:
            if p_lead == 0:
              flat_lead: Expr = Const(0, ops.I)
            else:
              flat_lead = producer.index[0]
              for idx_v, dim in zip(producer.index[1:p_lead], producer.shape[1:p_lead]):
                flat_lead = ops.iadd(ops.imul(flat_lead, dim), idx_v)
            rem = flat_lead
            if c_lead == 1:
              cv = Var(consumer.index[0].name, consumer.index[0].type)
              unravel_stmts.append(Assign(cv, rem))
              c_subst[consumer.index[0].uid] = cv
            else:
              c_strides: list[Expr] = [Const(1, ops.I)] * c_lead
              for k_i in range(c_lead - 2, -1, -1):
                c_strides[k_i] = ops.imul(c_strides[k_i + 1], consumer.shape[k_i + 1])
              for k_i in range(c_lead - 1):
                qv = Var(consumer.index[k_i].name, consumer.index[k_i].type)
                rv = Var("rem", ops.I)
                unravel_stmts.append(Assign(qv, ops.idiv(rem, c_strides[k_i])))
                unravel_stmts.append(Assign(rv, ops.isub(rem, ops.imul(qv, c_strides[k_i]))))
                c_subst[consumer.index[k_i].uid] = qv
                rem = rv
              last_v = Var(consumer.index[c_lead - 1].name, consumer.index[c_lead - 1].type)
              unravel_stmts.append(Assign(last_v, rem))
              c_subst[consumer.index[c_lead - 1].uid] = last_v
          cl = Cloner(c_subst)
          final_consumer_body = [cl.stmt(t) for t in replaced_body]
          producer.body[:] = (
            prefix + [Assign(elem, value)] + unravel_stmts + final_consumer_body + [ir.ReleaseScope(producer.scope)]
          )
          producer.scope.owned.extend(cl.scope(rewriter.scope(consumer.scope)).owned)
          producer.parallel = producer.parallel and consumer.parallel
          return producer, buf.uid

      # Generalized affine Map -> Map: transpose, shifted slice/stencil, broadcast, multi-load.
      # Never inline into a contraction reduction (which would multiply execution count and undo
      # layout copies like ``bt = b.T.copy()``), EXCEPT allow inlining a single-reader producer
      # into a non-contraction map-of-reduce whose combined index space matches ``producer.shape``
      # (each element is read exactly once, e.g. ``g * gain * x_hat`` into LayerNorm backward sum).
      nested_buf_read = any(
        count_reads_in([t], buf) > 0
        for t in consumer.body
        if isinstance(t, (IndexReduce, IndexMap, ir.ForRange, ir.While))
      )
      allow_noncontraction_reduce = False
      if nested_buf_read and not allow_partial and readers.get(buf.uid, 0) == consumer_reads == 1:
        sr = _split_single_reduce_map(consumer)
        if sr is not None:
          _, c_red, _ = sr
          if not _looks_like_contraction(consumer, c_red) and facts.same_expr(
            ops.iprod(producer.shape), ops.iprod((*consumer.shape, *c_red.shape))
          ):
            allow_noncontraction_reduce = True
      if (
        (not nested_buf_read or allow_noncontraction_reduce)
        and not any(isinstance(t, ir.Check) for t in ir.walk_stmts(prefix))
        and _is_row_major_store(store_index, producer, facts)
        and (
          allow_noncontraction_reduce
          or _affine_fusion_profitable(
            producer, consumer.shape, prefix, value, readers.get(buf.uid, consumer_reads), facts
          )
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

    if isinstance(consumer, IndexScatter) and facts.same_shape(producer.shape, consumer.shape):
      # Map -> Scatter: fuse producer elementwise map directly into IndexScatter
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
      new_body = [rewriter.stmt(t) for t in consumer.body]
      new_target = rewriter.expr(consumer.target)
      new_updates = [ir.ScatterUpdate(u.buf, rewriter.expr(u.value), u.op) for u in consumer.updates]
      new_where = None if consumer.where is None else rewriter.expr(consumer.where)
      if rewriter.ok and rewriter.replaced == consumer_reads:
        consumer.body[:] = inlined_prefix + [Assign(elem, inlined_value)] + new_body
        consumer.target = new_target
        consumer.updates = new_updates
        consumer.where = new_where
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
    if isinstance(t, (IndexMap, IndexReduce, IndexFilter, IndexScatter)) and count_reads_in([t], buf) > 0:
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
    if uses([t], {buf.uid} | producer_defs) or uses([producer], set(defined([t]))):
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
    producer_reads, producer_writes = buffer_accesses([s])
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
        and not uses([s], {t.var.uid})
      ):
        step_facts.record_stmt(t)
        between.append(t)
      elif (
        producer_cannot_raise
        and isinstance(t, ir.If)
        and not uses([t], producer_defs | producer_writes)
        and not uses([s], set(defined([t])))
        and not any(isinstance(x, (ir.Break, ir.Continue)) for x in ir.walk_stmts([t]))
        and not binfo.may_alias(producer_reads, buffer_accesses([t])[1])
      ):
        step_facts.record_stmt(t)
        between.append(t)
      else:
        break
      j += 1
    nxt = block[j] if j < len(block) else None
    if nxt is not None and isinstance(nxt, (IndexMap, IndexFilter, IndexReduce, IndexScatter)):
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
      # Bounded lookahead for single-consumer producers separated from their sole
      # consumer by independent operators (e.g. matmul backward emits dX, then dW,
      # then activation/residual consumer of dX). Sink ``s`` down to its sole consumer
      # and fuse them when nothing in between aliases ``s``'s reads or writes.
      m_s = match_single_output_map(s, pair_facts, binfo.local_allocs, binfo.store_counts)
      if m_s is not None:
        _, s_buf, _, _, _ = m_s
        if s_buf.uid not in binfo.escapes and binfo.readers.get(s_buf.uid, 0) >= 1 and producer_cannot_raise:
          total_r = binfo.readers[s_buf.uid]
          scan_facts = pair_facts.fork()
          scan_defs: set[int] = set()
          scan_writes: set[int] = set()
          limit = min(len(block), k + 40)
          found_sink = False
          for m in range(k, limit):
            t = block[m]
            c_reads = count_reads_in([t], s_buf)
            if c_reads > 0:
              if (
                c_reads == total_r
                and isinstance(t, (IndexMap, IndexFilter, IndexReduce, IndexScatter))
                and not uses([s], scan_defs)
                and not binfo.may_alias(producer_reads, scan_writes)
              ):
                res = fuse_vertical_pair(s, t, binfo.readers, scan_facts, binfo, binfo.store_counts, options)
                if res is not None:
                  fused, dead_buf = res
                  block[m] = fused
                  gone.add(dead_buf)
                  changed = True
                  found_sink = True
              break
            if uses([t], producer_defs | producer_writes) or uses([s], set(defined([t]))):
              break
            _, tw = buffer_accesses([t])
            if binfo.may_alias(producer_reads, tw):
              break
            scan_defs |= set(defined([t]))
            scan_writes |= tw
            scan_facts.record_stmt(t)
          if found_sink:
            continue
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


def _resolve_local_expr(e: Expr, local_assigns: dict[int, Expr], budget: int = 12) -> Expr:
  """Inline single-assignment local variables within a reduction body."""
  if isinstance(e, Var) and e.uid in local_assigns and budget > 0:
    return _resolve_local_expr(local_assigns[e.uid], local_assigns, budget - 1)
  if isinstance(e, ir.Prim):
    return ir.Prim(e.op, tuple(_resolve_local_expr(a, local_assigns, budget - 1) for a in e.args), e.type)
  if isinstance(e, ir.Cast):
    return ir.Cast(_resolve_local_expr(e.arg, local_assigns, budget - 1), e.type)
  if isinstance(e, ir.Load):
    return ir.Load(e.buf, _resolve_local_expr(e.index, local_assigns, budget - 1))
  return e


def _alpha_expr_key(e: Expr, idx_slots: dict[int, int], facts: IndexFacts) -> object:
  """Alpha-invariant key for comparing two reduction bodies' value expressions."""
  if isinstance(e, Var):
    if e.uid in idx_slots:
      return ("idx", idx_slots[e.uid], e.type)
    if not isinstance(e.type, ir.BufType) and e.type.kind in "iu":
      return ("iv", facts.expr_key(e))
    return ("v", e.uid)
  if isinstance(e, Const):
    return ("c", e.value, e.type)
  if isinstance(e, ir.Load):
    return ("ld", e.buf.uid, _alpha_expr_key(e.index, idx_slots, facts))
  if isinstance(e, ir.Prim):
    return ("p", e.op, e.type, tuple(_alpha_expr_key(a, idx_slots, facts) for a in e.args))
  if isinstance(e, ir.Cast):
    return ("t", e.type, _alpha_expr_key(e.arg, idx_slots, facts))
  if isinstance(e, ir.RtCall):
    return ("rt", e.name, e.pure, e.type, tuple(_alpha_expr_key(a, idx_slots, facts) for a in e.args))
  return ("?", id(e))


def fuse_online_logsumexp(a: IndexReduce, b: IndexReduce, facts: IndexFacts) -> IndexReduce | None:
  """Fuse adjacent ``m = max(x)`` followed by ``s = sum(exp(x - m))`` over the
  same index space into ONE single-pass associative monoidal :class:`IndexReduce`
  using :class:`~parakeet.syntax.ir.Carry`:

      (m_a, s_a) (+) (m_b, s_b) = (m_new, s_a * exp(m_a - m_new) + s_b * exp(m_b - m_new))
      where m_new = max(m_a, m_b).
  """
  if len(a.folds) != 1 or len(b.folds) != 1:
    return None
  if a.where is not None or b.where is not None or a.parallel != b.parallel:
    return None
  if len(a.shape) != len(b.shape) or not facts.same_shape(a.shape, b.shape):
    return None
  if has_effects(a.body) or has_effects(b.body) or a.scope.owned or b.scope.owned:
    return None
  f1, f2 = a.folds[0], b.folds[0]
  if f1.op != "max" or f1.init is not None or f1.combine or f1.carries or f1.a.type.kind != "f":
    return None
  if (
    f2.op != "add"
    or f2.combine
    or f2.carries
    or f2.a.type is not f1.a.type
    or not (isinstance(f2.init, Const) and f2.init.value == 0.0)
  ):
    return None

  a_locals = {t.var.uid: t.value for t in a.body if isinstance(t, Assign)}
  b_locals = {t.var.uid: t.value for t in b.body if isinstance(t, Assign)}
  v1 = _resolve_local_expr(f1.value, a_locals)
  v2 = _resolve_local_expr(f2.value, b_locals)
  # Match v2 == exp(x2 - f1.result)
  if not (isinstance(v2, ir.Prim) and v2.op == "exp" and len(v2.args) == 1):
    return None
  sub_expr = v2.args[0]
  if not (isinstance(sub_expr, ir.Prim) and sub_expr.op == "sub" and len(sub_expr.args) == 2):
    return None
  x2, sub_rhs = sub_expr.args
  if sub_rhs is not f1.result:
    return None

  idx_a = {v.uid: k for k, v in enumerate(a.index)}
  idx_b = {v.uid: k for k, v in enumerate(b.index)}
  if _alpha_expr_key(v1, idx_a, facts) != _alpha_expr_key(x2, idx_b, facts):
    return None

  t = f1.a.type
  ma, mb = Var("ma", t), Var("mb", t)
  sa, sb = Var("sa", t), Var("sb", t)
  mnew = Var("mnew", t)
  comb_stmts: list[ir.Stmt] = [Assign(mnew, ops.prim("max", (ma, mb), t))]
  scale_a = ops.prim("mul", (sa, ops.prim("exp", (ops.prim("sub", (ma, mnew), t),), t)), t)
  scale_b = ops.prim("mul", (sb, ops.prim("exp", (ops.prim("sub", (mb, mnew), t),), t)), t)
  snew = ops.prim("add", (scale_a, scale_b), t)
  carry = ir.Carry(f2.result, Const(1.0, t), sa, sb, snew)
  fused_fold = ir.Fold(
    result=f1.result,
    value=f1.value,
    a=ma,
    b=mb,
    combine=comb_stmts,
    combined=mnew,
    init=None,
    op="online_lse",
    empty=f1.empty,
    carries=[carry],
  )
  return IndexReduce(a.shape, a.index, list(a.body), [fused_fold], a.scope, a.home, a.parallel, a.loc)


def contains_reduce(s: IndexMap) -> bool:
  """True if ``s`` wraps an ``IndexReduce`` (e.g. a matmul or row-reduction)."""
  return any(isinstance(t, IndexReduce) for t in ir.walk_stmts(s.body))


def _split_single_reduce_map(s: IndexMap) -> tuple[list[ir.Stmt], IndexReduce, list[ir.Stmt]] | None:
  """Return ``(pre, red, post)`` if ``s`` has a single top-level ``IndexReduce`` and no other nested loops."""
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  red_pos = [idx for idx, t in enumerate(body) if isinstance(t, IndexReduce)]
  if len(red_pos) != 1:
    return None
  pos = red_pos[0]
  pre, red, post = body[:pos], body[pos], body[pos + 1 :]
  assert isinstance(red, IndexReduce)
  if any(
    isinstance(t, (IndexMap, IndexReduce, ir.ForRange, ir.While))
    for t in (*ir.walk_stmts(pre), *ir.walk_stmts(post), *ir.walk_stmts(red.body))
  ):
    return None
  return pre, red, post


def _looks_like_contraction(m: IndexMap, red: IndexReduce) -> bool:
  """True if a 2D+ ``IndexMap(IndexReduce)`` has separable row/col loads (like matmul/allpairs/einsum)."""
  if len(m.shape) < 2 or len(red.shape) != 1:
    return False
  row_uid = m.index[-2].uid
  col_uid = m.index[-1].uid
  has_row_only = False
  has_col_only = False
  for s in ir.walk_stmts([*m.body]):
    for e in ir.stmt_exprs(s):
      for x in ir.walk_expr(e):
        if isinstance(x, ir.Load):
          uids = {v.uid for v in ir.walk_expr(x.index) if isinstance(v, Var)}
          if row_uid in uids and col_uid not in uids:
            has_row_only = True
          elif col_uid in uids and row_uid not in uids:
            has_col_only = True
  return has_row_only and has_col_only


def fuse_reduce_maps(a: IndexMap, b: IndexMap, facts: IndexFacts, allocs: set[int] | BufferInfo) -> bool:
  """Sibling-fuse two independent non-contraction ``IndexMap(IndexReduce)`` statements
  over the same outer and inner shapes (e.g. LayerNorm backward ``d_bias`` + ``d_gain``
  or ``g_mean`` + ``gx_mean``)."""
  if a.ordered != b.ordered or a.parallel != b.parallel or not facts.same_shape(a.shape, b.shape):
    return False
  sa = _split_single_reduce_map(a)
  sb = _split_single_reduce_map(b)
  if sa is None or sb is None:
    return False
  a_pre, a_red, a_post = sa
  b_pre, b_red, b_post = sb
  # Leave 2D+ contractions (matmul/einsum/allpairs) untouched for tiling
  if _looks_like_contraction(a, a_red) or _looks_like_contraction(b, b_red):
    return False
  a_reads, a_writes = buffer_accesses(a.body)
  b_reads, b_writes = buffer_accesses(b.body)
  if buffers_may_alias(a_writes, b_reads | b_writes, allocs) or buffers_may_alias(b_writes, a_reads | a_writes, allocs):
    return False
  local_allocs = allocs.local_allocs if isinstance(allocs, BufferInfo) else allocs
  if any(isinstance(t, ir.Check) for t in ir.walk_stmts(b.body)) and any(u not in local_allocs for u in a_writes):
    return False
  if any(isinstance(t, (ir.Store, ir.IndexScatter)) for t in (*ir.walk_stmts(a_pre), *ir.walk_stmts(b_pre))):
    return False
  cloner = Cloner({v.uid: v for v in defined([b]).values()})
  cloner.m.update({v.uid: i for v, i in zip(b.index, a.index)})
  b_pre_c = [cloner.stmt(t) for t in b_pre]
  b_red_c = cloner.stmt(b_red)
  assert isinstance(b_red_c, IndexReduce)
  b_post_c = [cloner.stmt(t) for t in b_post]
  sub_facts = facts.fork()
  for t in (*a_pre, *b_pre_c):
    sub_facts.record_stmt(t)
  if not compatible(a_red, b_red_c, sub_facts):
    return False
  absorb(a_red, b_red_c)
  a.body[:] = a_pre + b_pre_c + [a_red] + a_post + b_post_c + [ir.ReleaseScope(a.scope)]
  a.scope.owned.extend(cloner.scope(b.scope).owned)
  return True


def compatible_maps(
  a: IndexMap, b: IndexMap, defs: dict[int, Expr] | IndexFacts, allocs: set[int] | BufferInfo
) -> bool:
  """True if two ``IndexMap`` statements over the same shape are independent
  and safe to merge into one loop."""
  facts = defs if isinstance(defs, IndexFacts) else IndexFacts(defs)
  if a.ordered != b.ordered or a.parallel != b.parallel or not facts.same_iteration_shape(a.shape, b.shape):
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


def fuse_write_through_map(a: IndexMap, b: IndexMap, facts: IndexFacts, allocs: set[int] | BufferInfo) -> bool:
  """Merge dependent same-shape ``IndexMap`` ``b`` into ``a`` when ``b`` reads
  local buffers written by ``a`` at the exact same index (e.g. Adam's
  ``mn, vn -> pn`` or saved forward activations).  Keeps ``a``'s stores
  (write-through) while passing the computed values directly in registers to ``b``."""
  if a.ordered != b.ordered or a.parallel != b.parallel or not facts.same_iteration_shape(a.shape, b.shape):
    return False
  if contains_reduce(a) or contains_reduce(b):
    return False
  a_reads, a_writes = buffer_accesses(a.body)
  b_reads, b_writes = buffer_accesses(b.body)
  # b must not overwrite anything a reads or writes
  if buffers_may_alias(b_writes, a_reads | a_writes, allocs):
    return False
  local_allocs = allocs.local_allocs if isinstance(allocs, BufferInfo) else allocs
  shared_uids = a_writes & b_reads
  if not shared_uids or not (shared_uids <= local_allocs):
    return False
  # Every buffer in a_writes that may alias b_reads must be in shared_uids (all local)
  if buffers_may_alias(a_writes - shared_uids, b_reads, allocs):
    return False
  if any(isinstance(t, ir.Check) for t in ir.walk_stmts(b.body)) and any(u not in local_allocs for u in a_writes):
    return False
  # Each shared buffer must have exactly one top-level Store in a.body (no nested writes)
  a_body = [t for t in a.body if not (isinstance(t, ir.ReleaseScope) and t.scope is a.scope)]
  store_pos: dict[int, int] = {}
  store_stmts: dict[int, ir.Store] = {}
  for idx, t in enumerate(a_body):
    if isinstance(t, ir.Store) and t.buf.uid in shared_uids:
      if t.buf.uid in store_pos:
        return False
      store_pos[t.buf.uid] = idx
      store_stmts[t.buf.uid] = t
    else:
      _, inner_w = buffer_accesses([t])
      if inner_w & shared_uids:
        return False
  if set(store_pos) != shared_uids:
    return False
  # Build facts for a.body and clone b.body onto a.index
  a_facts = facts.fork()
  for t in a_body:
    a_facts.record_stmt(t)
  b_body = [t for t in b.body if not (isinstance(t, ir.ReleaseScope) and t.scope is b.scope)]
  cloner = Cloner({v.uid: v for v in defined([b]).values()})
  cloner.m.update({v.uid: i for v, i in zip(b.index, a.index)})
  b_cloned = [cloner.stmt(t) for t in b_body]
  b_facts = a_facts.fork()
  for t in b_cloned:
    b_facts.record_stmt(t)
  # Replace every Load of each shared buffer in b_cloned with a fresh scalar Var holding a's stored value
  repl_vars: dict[int, Var] = {}
  cur_b = b_cloned
  for uid in sorted(shared_uids):
    st = store_stmts[uid]
    v = Var("wt", st.buf.type.elt)
    expected_key = a_facts.expr_key(st.index)
    n_reads = count_reads_in(cur_b, st.buf)
    rew = InlineBufferLoad({x.uid: x for x in defined(cur_b).values()}, st.buf, expected_key, v, b_facts)
    nxt_b = [rew.stmt(t) for t in cur_b]
    if not rew.ok or rew.replaced != n_reads:
      return False
    cur_b = nxt_b
    repl_vars[uid] = v
  # Commit: rewrite each shared Store(buf, idx, val) in a_body to Assign(v, val); Store(buf, idx, v)
  new_a: list[ir.Stmt] = []
  for t in a_body:
    if isinstance(t, ir.Store) and t.buf.uid in repl_vars:
      v = repl_vars[t.buf.uid]
      new_a.append(Assign(v, t.value))
      new_a.append(ir.Store(t.buf, t.index, v))
    else:
      new_a.append(t)
  a.body[:] = new_a + cur_b + [ir.ReleaseScope(a.scope)]
  a.scope.owned.extend(cloner.scope(b.scope).owned)
  return True


def _fuse_keyed_updates(a: IndexScatter, b: IndexScatter, facts: IndexFacts, allocs) -> bool:
  """Merge independent updates to the same key (and guard), sharing its computation and checks."""
  if a.scope.owned or b.scope.owned or a.parallel != b.parallel or (a.where is None) != (b.where is None):
    return False
  if not facts.same_shape(a.shape, b.shape) or not facts.same_expr(a.dest_size, b.dest_size):
    return False
  ra, wa = buffer_accesses([a])
  rb, wb = buffer_accesses([b])
  local = allocs.local_allocs if isinstance(allocs, BufferInfo) else allocs
  if not (wa | wb) <= local:
    return False
  if buffers_may_alias(wa, rb | wb, allocs) or buffers_may_alias(wb, ra, allocs):
    return False
  if any(
    not isinstance(t, (ir.Check, Assign)) or not all(pure(e) for e in ir.stmt_exprs(t)) for t in (*a.body, *b.body)
  ):
    return False
  if not all(pure(u.value) for u in (*a.updates, *b.updates)):
    return False

  def _same_expr(ex: Expr, ey: Expr) -> bool:
    if ir.same(ex, ey):
      return True
    return facts.same_expr(ex, ey)

  cl = Cloner({y.uid: x for x, y in zip(a.index, b.index)})
  common = 0
  for x, y in zip(a.body, b.body):
    if isinstance(x, Assign) and isinstance(y, Assign) and _same_expr(x.value, cl.expr(y.value)):
      cl.m[y.var.uid] = x.var
    elif isinstance(x, ir.Check) and isinstance(y, ir.Check) and (x.err.exc, x.err.fmt) == (y.err.exc, y.err.fmt):
      if (
        not _same_expr(x.cond, cl.expr(y.cond))
        or len(x.args) != len(y.args)
        or not all(_same_expr(ex, cl.expr(ey)) for ex, ey in zip(x.args, y.args))
      ):
        return False
    else:
      break
    common += 1
  if (
    not _same_expr(a.target, cl.expr(b.target))
    or (a.where is not None and not _same_expr(a.where, cl.expr(b.where)))
    or any(isinstance(t, ir.Check) for t in (*a.body[common:], *b.body[common:]))
  ):
    return False
  a.body.extend(cl.block(b.body[common:]))
  a.updates.extend(ir.ScatterUpdate(u.buf, cl.expr(u.value), u.op) for u in b.updates)
  return True


def _scatter_siblings(block, k, s, facts, allocs, out):
  """Cross only pure scalar defs, shape checks, and fresh destinations. On a shape error,
  reproduce earlier per-index checks before raising the later error."""
  step_facts = facts.fork()
  while k < len(block):
    j = k
    while j < len(block) and (
      isinstance(block[j], (ir.Check, ir.Alloc))
      or (isinstance(block[j], Assign) and not isinstance(block[j].var.type, ir.BufType) and pure(block[j].value))
    ):
      if any(isinstance(x, (ir.Load, ir.RtCall)) for e in ir.stmt_exprs(block[j]) for x in ir.walk_expr(e)):
        break
      j += 1
    if j == len(block) or not isinstance(block[j], IndexScatter):
      break
    nxt = block[j]
    fresh = {t.var.uid for t in block[k:j] if isinstance(t, ir.Alloc)}
    if fresh - {u.buf.uid for u in nxt.updates} or uses([s], fresh):
      break
    for t in block[k:j]:
      step_facts.record_stmt(t)
    # Keep only the prefix through the last bounds check for the cold error path.
    last = max((i + 1 for i, t in enumerate(s.body) if isinstance(t, ir.Check)), default=0)
    validation = s.body[:last]
    if not _fuse_keyed_updates(s, nxt, step_facts, allocs):
      break
    for t in block[k:j]:
      if validation and isinstance(t, ir.Check):
        cl = Cloner()
        indices = tuple(cl.var(v) for v in s.index)
        check_map = IndexMap(s.shape, indices, cl.block(validation), cl.scope(s.scope), parallel=False)
        out.append(ir.If(ops.lnot(t.cond), [check_map, t], []))
      else:
        out.append(t)
    k = j + 1
  return k


def fuse_block(
  block: list, defs: dict[int, Expr] | IndexFacts, allocs: set[int] | BufferInfo, max_bufs: int = 20
) -> list:
  """Walk ``block`` merging adjacent compatible reductions and maps."""
  facts = defs.fork() if isinstance(defs, IndexFacts) else IndexFacts(defs)
  for s in block:
    facts.record_stmt(s)
    sub_facts = facts.fork()
    if isinstance(s, (IndexMap, IndexReduce)):
      sub_facts.activate_nonempty(s.shape)
    for b in ir.sub_blocks(s):
      b[:] = fuse_block(b, sub_facts, allocs, max_bufs=max_bufs)
  out: list = []
  walk_facts = defs.fork() if isinstance(defs, IndexFacts) else IndexFacts(defs)
  local_allocs = allocs.local_allocs if isinstance(allocs, BufferInfo) else allocs
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    walk_facts.record_stmt(s)
    if isinstance(s, IndexScatter):
      k = _scatter_siblings(block, k, s, walk_facts, allocs, out)
    if isinstance(s, IndexReduce):
      # ``s`` may sink past independent scalar definitions (hoisted before ``s``)
      # and step past pure scalar post-processing of its own results (sunk after ``s``)
      # to fuse with an independent sibling reduction.
      sunk: list = []
      sunk_defs: set[int] = {v.uid for f in s.folds for v in (f.result, *(c.result for c in f.carries))}
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
        if isinstance(nxt, IndexReduce):
          if not uses([nxt], sink_more_defs) and compatible(s, nxt, step_facts):
            out.extend(hoist)
            walk_facts = step_facts
            sunk.extend(sink_more)
            sunk_defs = sink_more_defs | {v.uid for f in nxt.folds for v in (f.result, *(c.result for c in f.carries))}
            absorb(s, nxt)
            k = j + 1
            continue
          # Online monoidal fusion: m = max(x); s = sum(exp(x - m)) -> 1-pass (m, s) monoid
          if not sink_more and (online := fuse_online_logsumexp(s, nxt, step_facts)) is not None:
            out.extend(hoist)
            walk_facts = step_facts
            s = online
            sunk_defs = sunk_defs | {v.uid for f in s.folds for v in (f.result, *(c.result for c in f.carries))}
            k = j + 1
            continue
        break
      out.append(s)
      out.extend(sunk)
      continue
    if isinstance(s, IndexMap):
      sunk = []
      while True:
        a_reads, a_writes = buffer_accesses(s.body)
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
            isinstance(t, ir.Check)
            and not uses([t], map_outputs)
            and (a_writes <= local_allocs)
            and not any(isinstance(x, (ir.Load, ir.RtCall)) for e in ir.stmt_exprs(t) for x in ir.walk_expr(e))
          ):
            step_facts.record_stmt(t)
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
        if isinstance(nxt, IndexMap):
          nxt_r, nxt_w = buffer_accesses(nxt.body)
          if len(a_reads | a_writes | nxt_r | nxt_w) <= max_bufs:
            if compatible_maps(s, nxt, step_facts, allocs):
              out.extend(hoist)
              walk_facts = step_facts
              sunk.extend(sink_more)
              absorb_map(s, nxt)
              k = j + 1
              continue
            if fuse_write_through_map(s, nxt, step_facts, allocs):
              out.extend(hoist)
              walk_facts = step_facts
              sunk.extend(sink_more)
              k = j + 1
              continue
            if fuse_reduce_maps(s, nxt, step_facts, allocs):
              out.extend(hoist)
              walk_facts = step_facts
              sunk.extend(sink_more)
              k = j + 1
              continue
        # Bounded lookahead: pull later independent same-shape maps (e.g. Adam's pn/mn/vn
        # or Q/K/V backward transposes separated by matmuls) into s while keeping buffers <= max_bufs.
        if not contains_reduce(s) and not s.ordered:
          merged_ahead = False
          a_reads, a_writes = buffer_accesses(s.body)
          s_bufs = a_reads | a_writes
          if len(s_bufs) < max_bufs:
            def_defs: set[int] = set()
            def_reads: set[int] = set()
            def_writes: set[int] = set()
            hoist_assign_idx: dict[int, int] = {}
            alloc_idx_by_uid: dict[int, tuple[int, list[int]]] = {}
            hoist_checks: list[int] = []
            passed_effectful_map = False
            scan_facts = walk_facts.fork()
            limit = min(len(block), k + 2400)
            for m in range(k, limit):
              t = block[m]
              if isinstance(t, ir.Comment):
                continue
              if isinstance(t, ir.Alloc):
                need_assigns = [
                  hoist_assign_idx[x.uid]
                  for x in ir.walk_expr(t.count)
                  if isinstance(x, Var) and x.uid in hoist_assign_idx
                ]
                needed_uids = {block[idx].var.uid for idx in need_assigns}
                if not uses([t], (def_defs - needed_uids) | def_writes | map_outputs):
                  alloc_idx_by_uid[t.var.uid] = (m, need_assigns)
                def_defs.add(t.var.uid)
                continue
              if isinstance(t, Assign) and not isinstance(t.var.type, ir.BufType) and pure(t.value):
                scan_facts.record_stmt(t)
                if not uses([t], def_defs | def_writes | map_outputs):
                  hoist_assign_idx[t.var.uid] = m
                def_defs.add(t.var.uid)
                continue
              if isinstance(t, ir.Incref):
                continue
              if (
                isinstance(t, ir.Check)
                and not passed_effectful_map
                and not uses([t], def_defs | def_writes | map_outputs)
                and (a_writes <= local_allocs)
                and not any(isinstance(x, (ir.Load, ir.RtCall)) for e in ir.stmt_exprs(t) for x in ir.walk_expr(e))
              ):
                scan_facts.record_stmt(t)
                hoist_checks.append(m)
                continue
              if isinstance(t, IndexMap) and not t.ordered:
                tr, tw = buffer_accesses(t.body)
                if not contains_reduce(t):
                  hoist_indices: set[int] = set(hoist_checks)
                  hoisted_uids: set[int] = set()
                  for u in tw:
                    if u in alloc_idx_by_uid:
                      a_idx, dep_assigns = alloc_idx_by_uid[u]
                      hoist_indices.add(a_idx)
                      hoisted_uids.add(u)
                      for da in dep_assigns:
                        hoist_indices.add(da)
                        hoisted_uids.add(block[da].var.uid)
                  rem_defs = def_defs - hoisted_uids
                  # Do not pull a single-consumer temp map (e.g. LayerNorm inv_std) across
                  # a reduction away from its sole consumer row-pipeline.
                  is_single_temp = (
                    passed_effectful_map
                    and isinstance(allocs, BufferInfo)
                    and len(tw) == 1
                    and allocs.readers.get(next(iter(tw)), 0) == 1
                    and next(iter(tw)) not in allocs.escapes
                  )
                  if (
                    not is_single_temp
                    and len(s_bufs | tr | tw) <= max_bufs
                    and not uses([t], rem_defs)
                    and not buffers_may_alias(tr, def_writes, allocs)
                    and not buffers_may_alias(tw, def_reads | def_writes, allocs)
                  ):
                    is_compat = compatible_maps(s, t, scan_facts, allocs)
                    if is_compat or fuse_write_through_map(s, t, scan_facts, allocs):
                      if is_compat:
                        absorb_map(s, t)
                      for idx in sorted(hoist_indices):
                        out.append(block[idx])
                      for idx in sorted({m, *hoist_indices}, reverse=True):
                        block.pop(idx)
                      merged_ahead = True
                      break
                else:
                  if any(isinstance(x, ir.Check) for x in ir.walk_stmts(t.body)) or not (tw <= local_allocs):
                    break
                  passed_effectful_map = True
                def_defs |= set(defined([t]))
                def_reads |= tr
                def_writes |= tw
                continue
              break
          if merged_ahead:
            continue
        break
      out.append(s)
      out.extend(sunk)
      continue
    out.append(s)
  return out


# ---------------------------------------------------------------------------
# Disjoint-union horizontal batching of independent pure GPU maps (e.g. Adam)
# ---------------------------------------------------------------------------


def _flatten_contiguous_map(m: IndexMap, flat_var: Var, facts: IndexFacts) -> list[ir.Stmt] | None:
  """If ``m`` is a pure, loop-free, check-free ``IndexMap`` whose body only uses
  ``m.index`` through its row-major flat offset, return its body rewritten in
  terms of 1-D ``flat_var`` (zero integer divisions/modulos)."""
  if m.ordered or not m.parallel or m.scope.owned or not m.shape:
    return None
  body = [t for t in m.body if not (isinstance(t, ir.ReleaseScope) and t.scope is m.scope)]
  if not body:
    return None
  if any(
    isinstance(t, (ir.Check, ir.Alloc, ir.Release, ir.Incref, *ir.OPERATORS, ir.ForRange, ir.While))
    for t in ir.walk_stmts(body)
  ):
    return None
  if len(m.shape) == 1:
    cl = Cloner({m.index[0].uid: flat_var})
    for v in defined(body).values():
      cl.var(v)
    return [cl.stmt(t) for t in body]

  # ND contiguous map: match ``base_offset + row_major(m.index, m.shape)`` and rewrite
  # to ``base_offset + flat_var`` (handling array view offsets like ``params_off + i * N + j``).
  rm: Expr = m.index[0]
  for iv, d in zip(m.index[1:], m.shape[1:]):
    rm = ops.iadd(ops.imul(rm, d), iv)
  m_facts = facts.fork()
  for t in body:
    m_facts.record_stmt(t)
  rm_aff = m_facts.affine(rm)
  idx_uids = {v.uid for v in m.index}
  local_int_uids = {
    t.var.uid
    for t in body
    if isinstance(t, Assign) and not isinstance(t.var.type, ir.BufType) and t.var.type.kind in "iu"
  }
  forbidden_uids = idx_uids | local_int_uids

  def _as_flat_plus_inv(e: Expr) -> Expr | None:
    if isinstance(e.type, ir.BufType) or e.type.kind not in "iu":
      return None
    diff = m_facts.affine(e).add(rm_aff, -1)
    # Every factor in diff must be loop-invariant (no m.index or local int vars)
    for mon, _ in diff.terms:
      for fac in mon:
        if fac[0] == "v" and fac[1] in forbidden_uids:
          return None
        if fac[0] == "o":
          return None
    base_e = m_facts._rebuild_expr(diff)
    if base_e is None:
      return None
    if isinstance(base_e, Const) and base_e.value == 0:
      res: Expr = flat_var
    else:
      res = ops.iadd(base_e, flat_var)
    return ir.Cast(res, e.type) if e.type is not res.type else res

  class _FlatRewriter(Cloner):
    def expr(self, e: Expr) -> Expr:
      if not isinstance(e.type, ir.BufType) and e.type.kind in "iu":
        if any(isinstance(x, Var) and x.uid in idx_uids for x in ir.walk_expr(e)):
          rep = _as_flat_plus_inv(e)
          if rep is not None:
            return rep
      if isinstance(e, ir.Prim):
        return ir.Prim(e.op, tuple(self.expr(a) for a in e.args), e.type)
      if isinstance(e, ir.Cast):
        return ir.Cast(self.expr(e.arg), e.type)
      if isinstance(e, ir.Load):
        return ir.Load(e.buf, self.expr(e.index))
      if isinstance(e, ir.RtCall):
        return ir.RtCall(e.name, tuple(self.expr(a) for a in e.args), e.type, e.pure)
      return super().expr(e)

  rew = _FlatRewriter({})
  for v in defined(body).values():
    rew.var(v)
  new_body: list[ir.Stmt] = []
  dropped_uids: set[int] = set()
  for t in body:
    if isinstance(t, Assign) and not isinstance(t.var.type, ir.BufType) and t.var.type.kind in "iu":
      rep = _as_flat_plus_inv(t.value)
      if rep is not None:
        rew.m[t.var.uid] = rep
        continue
      if any(isinstance(x, Var) and x.uid in idx_uids for x in ir.walk_expr(t.value)):
        dropped_uids.add(rew.var(t.var).uid)
        dropped_uids.add(t.var.uid)
        continue
    new_body.append(rew.stmt(t))
  if uses(new_body, idx_uids | dropped_uids):
    return None
  return new_body


def _batch_maps_block(block: list, facts: IndexFacts, binfo: BufferInfo, max_maps: int = 8, max_bufs: int = 64) -> list:
  """Horizontally batch independent pure contiguous GPU ``IndexMap``s of
  arbitrary shapes into one flat disjoint-union ``IndexMap`` using a balanced
  binary dispatch tree on the flat thread index."""
  local_facts = facts.fork()
  for s in block:
    local_facts.record_stmt(s)
    sub_facts = local_facts.fork()
    for b in ir.sub_blocks(s):
      b[:] = _batch_maps_block(b, sub_facts, binfo, max_maps=max_maps, max_bufs=max_bufs)

  out: list = []
  walk_facts = facts.fork()
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    walk_facts.record_stmt(s)
    if not isinstance(s, IndexMap) or s.ordered or not s.parallel or s.scope.owned:
      out.append(s)
      continue
    probe_v = Var("tid", ops.I)
    if _flatten_contiguous_map(s, probe_v, walk_facts) is None:
      out.append(s)
      continue
    s_reads, s_writes = buffer_accesses(s.body)
    group: list[IndexMap] = [s]
    group_bufs: set[int] = s_reads | s_writes
    group_writes: set[int] = set(s_writes)
    group_defs: set[int] = set(defined([s]))
    hoisted_stmts: list[ir.Stmt] = []
    sunk_stmts: list[ir.Stmt] = []

    while len(group) < max_maps and len(group_bufs) < max_bufs and k < len(block):
      # Step over hoistable statements (allocs, pure scalar assigns, comments, increfs, checks)
      j = k
      step_hoist: list[ir.Stmt] = []
      step_sink: list[ir.Stmt] = []
      step_facts = walk_facts.fork()
      blocked = group_defs | group_writes
      while j < len(block):
        t = block[j]
        if isinstance(t, ir.Comment):
          step_hoist.append(t)
        elif isinstance(t, ir.Incref):
          step_sink.append(t)
        elif isinstance(t, ir.Alloc) and not uses([t], blocked):
          step_hoist.append(t)
        elif (
          isinstance(t, ir.Check)
          and not uses([t], blocked)
          and (group_writes <= binfo.local_allocs)
          and not any(isinstance(x, (ir.Load, ir.RtCall)) for e in ir.stmt_exprs(t) for x in ir.walk_expr(e))
        ):
          step_facts.record_stmt(t)
          step_hoist.append(t)
        elif (
          isinstance(t, Assign) and not isinstance(t.var.type, ir.BufType) and pure(t.value) and not uses([t], blocked)
        ):
          step_facts.record_stmt(t)
          step_hoist.append(t)
        else:
          break
        j += 1
      nxt = block[j] if j < len(block) else None
      if not isinstance(nxt, IndexMap) or nxt.ordered or not nxt.parallel or nxt.scope.owned:
        break
      nr, nw = buffer_accesses(nxt.body)
      if len(group_bufs | nr | nw) > max_bufs:
        break
      if buffers_may_alias(group_writes, nr | nw, binfo) or buffers_may_alias(nw, group_bufs, binfo):
        break
      if _flatten_contiguous_map(nxt, probe_v, step_facts) is None:
        break
      hoisted_stmts.extend(step_hoist)
      sunk_stmts.extend(step_sink)
      walk_facts = step_facts
      group.append(nxt)
      group_bufs |= nr | nw
      group_writes |= nw
      group_defs |= set(defined([nxt]))
      k = j + 1

    if len(group) == 1:
      out.append(s)
      continue

    # Emit hoisted allocations/scalars, prefix-sum offsets, and balanced binary dispatch map
    out.extend(hoisted_stmts)
    offs: list[Var] = []
    cur_off: Expr = Const(0, ops.I)
    for gi, gm in enumerate(group):
      sz = ops.iprod(gm.shape)
      nv = Var(f"boff_{gi}", ops.I)
      cur_off = sz if gi == 0 else ops.iadd(cur_off, sz)
      out.append(Assign(nv, cur_off))
      offs.append(nv)
    total_v = offs[-1]
    tid = Var("btid", ops.I)

    def _emit_tree(lo: int, hi: int) -> list[ir.Stmt]:
      if lo + 1 == hi:
        loc_v = Var(f"bloc_{lo}", ops.I)
        loc_expr: Expr = tid if lo == 0 else ops.isub(tid, offs[lo - 1])
        flat_b = _flatten_contiguous_map(group[lo], loc_v, walk_facts)
        assert flat_b is not None
        return [Assign(loc_v, loc_expr), *flat_b]
      mid = (lo + hi) // 2
      cond = ops.cmp("lt", tid, offs[mid - 1])
      return [ir.If(cond, _emit_tree(lo, mid), _emit_tree(mid, hi))]

    b_scope = ir.Scope("loop")
    b_body = _emit_tree(0, len(group)) + [ir.ReleaseScope(b_scope)]
    out.append(IndexMap((total_v,), (tid,), b_body, b_scope, False, True, group[0].loc))
    out.extend(sunk_stmts)
  return out


def fuse_siblings(fn: ir.Function, options) -> None:
  """Sibling fusion of adjacent reductions and independent maps."""
  if not options.fusion:
    return
  target = getattr(options, "target", None)
  max_bufs = 96 if (target is not None and target.kind == "cuda") else 20
  binfo = BufferInfo.analyze(fn)
  facts = IndexFacts.from_function(fn)
  fn.body[:] = fuse_block(fn.body, facts, binfo, max_bufs=max_bufs)
  if target is not None and target.kind == "cuda":
    binfo2 = BufferInfo.analyze(fn)
    facts2 = IndexFacts.from_function(fn)
    fn.body[:] = _batch_maps_block(fn.body, facts2, binfo2, max_maps=12, max_bufs=96)


# ---------------------------------------------------------------------------
# Multi-stage row-pipeline fusion (LayerNorm fwd/bwd, RMSNorm, Softmax)
# ---------------------------------------------------------------------------


def _row_major_expr(index: tuple[Var, ...], shape: tuple[Expr, ...]) -> Expr:
  acc: Expr = index[0]
  for iv, d in zip(index[1:], shape[1:]):
    acc = ops.iadd(ops.imul(acc, d), iv)
  return acc


def _match_row_pipeline(s: IndexMap, facts: IndexFacts) -> tuple[Expr, bool] | None:
  """If ``s`` is a non-contraction ``IndexMap(S)`` with 1+ sequential 1-D
  associative ``IndexReduce((K,))`` stages and an optional trailing 1-D
  ``IndexMap((K,))`` epilogue, return ``(K, has_epilogue)``."""
  if not isinstance(s, IndexMap) or not s.parallel or s.ordered or s.scope.owned:
    return None
  if getattr(s, "monoidal", None) is not None:
    return None
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  if not body:
    return None
  has_ep = False
  if isinstance(body[-1], IndexMap):
    ep = body[-1]
    if ep.ordered or ep.scope.owned or len(ep.shape) != 1:
      return None
    if any(isinstance(t, (IndexMap, IndexReduce, ir.Check, ir.ForRange, ir.While)) for t in ir.walk_stmts(ep.body)):
      return None
    has_ep = True
    ep_k = ep.shape[0]
    body = body[:-1]
  else:
    ep_k = None

  red_pos = [i for i, t in enumerate(body) if isinstance(t, IndexReduce)]
  if not red_pos:
    return None
  lfacts = facts.fork()
  for t in body:
    lfacts.record_stmt(t)
  k_dim: Expr | None = None
  prev = 0
  for idx_i, pos in enumerate(red_pos):
    red = body[pos]
    assert isinstance(red, IndexReduce)
    if red.scope.owned or red.where is not None or len(red.shape) != 1 or not red.folds:
      return None
    if not all(f.op in ("add", "mul", "max", "min") and not f.combine and not f.carries for f in red.folds):
      return None
    if not all(f.a.type.name in ("float32", "float64", "int32", "int64") for f in red.folds):
      return None
    if _looks_like_contraction(s, red):
      return None
    if k_dim is None:
      k_dim = red.shape[0]
    elif not lfacts.same_expr(k_dim, red.shape[0]):
      return None
    before = body[prev:pos] if idx_i == 0 else []
    nxt_p = red_pos[idx_i + 1] if idx_i + 1 < len(red_pos) else len(body)
    after = body[pos + 1 : nxt_p]
    prev = nxt_p
    if not all(isinstance(t, (Assign, ir.Comment)) for t in (*before, *red.body)):
      return None
    if any(isinstance(t, (IndexMap, IndexReduce, ir.Check, ir.ForRange, ir.While)) for t in ir.walk_stmts(after)):
      return None
    if any(isinstance(t, (ir.Store, ir.IndexScatter)) for t in ir.walk_stmts(before)):
      return None
  assert k_dim is not None
  if has_ep and ep_k is not None and not lfacts.same_expr(k_dim, ep_k):
    return None
  return k_dim, has_ep


def _row_local_stores(s: IndexMap, facts: IndexFacts, local_allocs: set[int]) -> dict[int, ir.Store] | None:
  """Return ``{buf.uid: store_stmt}`` for every top-level row-major local store in ``s.body``."""
  if not s.index:
    return None
  body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
  lfacts = facts.fork()
  for t in body:
    lfacts.record_stmt(t)
  rm_key = lfacts.expr_key(_row_major_expr(s.index, s.shape))
  stores: dict[int, ir.Store] = {}
  bad: set[int] = set()
  for t in body:
    if isinstance(t, ir.Store) and t.atomic is None and t.buf.uid in local_allocs:
      if lfacts.expr_key(t.index) == rm_key and t.buf.uid not in stores and t.buf.uid not in bad:
        stores[t.buf.uid] = t
      else:
        stores.pop(t.buf.uid, None)
        bad.add(t.buf.uid)
    else:
      _, iw = buffer_accesses([t])
      for u in iw:
        stores.pop(u, None)
        bad.add(u)
  return stores


def _rewrite_row_loads_in_stmts(
  stmts: list[ir.Stmt],
  shared_stores: dict[int, ir.Store],
  expected_idx_key: object,
  facts: IndexFacts,
) -> tuple[list[ir.Stmt], dict[int, Var], dict[int, int]] | None:
  """Replace every ``Load(buf, expected_idx)`` for ``buf`` in ``shared_stores``
  with a fresh scalar ``Var``.  Returns ``(new_stmts, scalar_vars, replaced_counts)``
  or ``None`` if any shared buffer is loaded at a non-matching index."""
  cur = stmts
  repl_vars: dict[int, Var] = {}
  repl_counts: dict[int, int] = {}
  for uid in sorted(shared_stores):
    st = shared_stores[uid]
    n_reads = count_reads_in(cur, st.buf)
    if n_reads == 0:
      continue
    v = Var("rp", st.buf.type.elt)
    c_facts = facts.fork()
    for t in cur:
      c_facts.record_stmt(t)
    rew = InlineBufferLoad({x.uid: x for x in defined(cur).values()}, st.buf, expected_idx_key, v, c_facts)
    nxt = rew.block(cur)
    if not rew.ok or rew.replaced != n_reads:
      return None
    cur = nxt
    repl_vars[uid] = v
    repl_counts[uid] = n_reads
  return cur, repl_vars, repl_counts


def _commit_row_stores(
  body: list[ir.Stmt],
  repl_vars: dict[int, Var],
  repl_counts: dict[int, int],
  binfo: BufferInfo,
  gone: set[int],
) -> list[ir.Stmt]:
  """Rewrite each shared ``Store(buf, idx, val)`` in ``body`` into
  ``Assign(v, val)`` plus a write-through ``Store`` only if external readers remain."""
  out: list[ir.Stmt] = []
  for t in body:
    if isinstance(t, ir.Store) and t.buf.uid in repl_vars:
      uid = t.buf.uid
      v = repl_vars[uid]
      out.append(Assign(v, t.value))
      remaining = binfo.readers.get(uid, 0) - repl_counts.get(uid, 0)
      if remaining > 0 or uid in binfo.escapes:
        out.append(ir.Store(t.buf, t.index, v))
      else:
        gone.add(uid)
    else:
      out.append(t)
  return out


def _row_pipeline_block(block: list, binfo: BufferInfo, facts: IndexFacts, gone: set[int]) -> tuple[list, bool]:
  """Fuse sequential row reductions + trailing row epilogues + single-consumer
  row prologues into single multi-stage row-pipeline ``IndexMap``s."""
  changed = False
  local_facts = facts.fork()
  for s in block:
    local_facts.record_stmt(s)
    sub_facts = local_facts.fork()
    if isinstance(s, (IndexMap, IndexReduce)):
      sub_facts.activate_nonempty(s.shape)
    for b in ir.sub_blocks(s):
      b[:], inner = _row_pipeline_block(b, binfo, sub_facts, gone)
      changed |= inner

  out: list = []
  walk_facts = facts.fork()
  k = 0
  while k < len(block):
    s = block[k]
    k += 1
    walk_facts.record_stmt(s)
    if not isinstance(s, IndexMap):
      out.append(s)
      continue

    rp_s = _match_row_pipeline(s, walk_facts)
    if rp_s is not None:
      k_dim, has_ep = rp_s
      if not has_ep:
        # Step over hoistable statements to inspect the immediately following operator
        s_reads, s_writes = buffer_accesses(s.body)
        s_defs = set(defined([s]))
        hoist: list = []
        step_facts = walk_facts.fork()
        j = k
        while j < len(block):
          t = block[j]
          if isinstance(t, ir.Comment):
            hoist.append(t)
          elif isinstance(t, ir.Alloc) and not uses([t], s_defs | s_writes):
            hoist.append(t)
          elif (
            isinstance(t, Assign)
            and not isinstance(t.var.type, ir.BufType)
            and pure(t.value)
            and not uses([t], s_defs | s_writes)
          ):
            step_facts.record_stmt(t)
            hoist.append(t)
          elif (
            isinstance(t, ir.Check)
            and (s_writes <= binfo.local_allocs)
            and not uses([t], s_defs | s_writes)
            and not any(isinstance(x, (ir.Load, ir.RtCall)) for e in ir.stmt_exprs(t) for x in ir.walk_expr(e))
          ):
            step_facts.record_stmt(t)
            hoist.append(t)
          else:
            break
          j += 1
        nxt = block[j] if j < len(block) else None
        if isinstance(nxt, IndexMap) and not nxt.ordered and nxt.parallel and not nxt.scope.owned:
          stores_s = _row_local_stores(s, step_facts, binfo.local_allocs) or {}
          nxt_r, nxt_w = buffer_accesses(nxt.body)
          shared_uids = set(stores_s) & nxt_r
          if (
            shared_uids
            and not buffers_may_alias(nxt_w, s_reads | s_writes, binfo)
            and not buffers_may_alias(s_writes - shared_uids, nxt_r, binfo)
            and not any(isinstance(t, ir.Check) for t in ir.walk_stmts(nxt.body))
          ):
            # Rule B: Row-Pipeline(S; K) + Row-Reduce(S; K) -> Multi-stage Row-Pipeline(S; K)
            rp_nxt = _match_row_pipeline(nxt, step_facts)
            if (
              rp_nxt is not None
              and not rp_nxt[1]
              and step_facts.same_shape(s.shape, nxt.shape)
              and step_facts.same_expr(k_dim, rp_nxt[0])
            ):
              nxt_body = [t for t in nxt.body if not (isinstance(t, ir.ReleaseScope) and t.scope is nxt.scope)]
              cl = Cloner({v.uid: u for v, u in zip(nxt.index, s.index)})
              for v in defined(nxt_body).values():
                cl.var(v)
              nxt_cloned = [cl.stmt(t) for t in nxt_body]
              rm_key = step_facts.expr_key(_row_major_expr(s.index, s.shape))
              rew_res = _rewrite_row_loads_in_stmts(
                nxt_cloned, {u: stores_s[u] for u in shared_uids}, rm_key, step_facts
              )
              if rew_res is not None:
                nxt_rew, repl_vars, repl_counts = rew_res
                s_body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
                new_s_body = _commit_row_stores(s_body, repl_vars, repl_counts, binfo, gone)
                s.body[:] = new_s_body + nxt_rew + [ir.ReleaseScope(s.scope)]
                out.extend(hoist)
                walk_facts = step_facts
                out.append(s)
                k = j + 1
                changed = True
                continue

            # Rule C: Row-Pipeline(S; K) + Pure Map(*S, K) -> Row-Pipeline(S; K) with trailing epilogue
            if (
              not contains_reduce(nxt)
              and len(nxt.shape) == len(s.shape) + 1
              and step_facts.same_shape(nxt.shape[:-1], s.shape)
              and step_facts.same_expr(nxt.shape[-1], k_dim)
              and not any(
                isinstance(t, (IndexMap, IndexReduce, ir.Check, ir.ForRange, ir.While)) for t in ir.walk_stmts(nxt.body)
              )
            ):
              nxt_body = [t for t in nxt.body if not (isinstance(t, ir.ReleaseScope) and t.scope is nxt.scope)]
              ep_k = Var(nxt.index[-1].name, nxt.index[-1].type)
              subst: dict[int, Expr] = {v.uid: u for v, u in zip(nxt.index[:-1], s.index)}
              subst[nxt.index[-1].uid] = ep_k
              cl = Cloner(subst)
              for v in defined(nxt_body).values():
                cl.var(v)
              ep_cloned = [cl.stmt(t) for t in nxt_body]
              rm_key = step_facts.expr_key(_row_major_expr(s.index, s.shape))
              rew_res = _rewrite_row_loads_in_stmts(
                ep_cloned, {u: stores_s[u] for u in shared_uids}, rm_key, step_facts
              )
              if rew_res is not None:
                ep_rew, repl_vars, repl_counts = rew_res
                s_body = [t for t in s.body if not (isinstance(t, ir.ReleaseScope) and t.scope is s.scope)]
                new_s_body = _commit_row_stores(s_body, repl_vars, repl_counts, binfo, gone)
                ep_scope = ir.Scope("loop")
                ep_map = IndexMap(
                  (k_dim,), (ep_k,), ep_rew + [ir.ReleaseScope(ep_scope)], ep_scope, False, False, nxt.loc
                )
                s.body[:] = new_s_body + [ep_map, ir.ReleaseScope(s.scope)]
                out.extend(hoist)
                walk_facts = step_facts
                out.append(s)
                k = j + 1
                changed = True
                continue

    # Case 2: Pure single-output Map -> downstream Row-Pipeline (Rule A: (S)->(S;K), Rule D: (*S,K)->(S;K))
    m_s = match_single_output_map(s, walk_facts, binfo.local_allocs, binfo.store_counts)
    if (
      m_s is not None
      and not contains_reduce(s)
      and not any(isinstance(t, (ir.Check, *ir.OPERATORS, ir.ForRange, ir.While)) for t in ir.walk_stmts(m_s[0]))
    ):
      prefix, s_buf, store_index, _, s_val = m_s
      total_r = binfo.readers.get(s_buf.uid, 0)
      if s_buf.uid not in binfo.escapes and total_r >= 1 and _is_row_major_store(store_index, s, walk_facts):
        s_reads, s_writes = buffer_accesses([s])
        s_defs = set(defined([s]))
        scan_facts = walk_facts.fork()
        scan_defs: set[int] = set()
        scan_writes: set[int] = set()
        limit = min(len(block), k + 64)
        fused_into_rp = False
        for m in range(k, limit):
          t = block[m]
          c_reads = count_reads_in([t], s_buf)
          if c_reads > 0:
            if (
              c_reads == total_r
              and isinstance(t, IndexMap)
              and not uses([s], scan_defs)
              and not binfo.may_alias(s_reads, scan_writes)
              and not binfo.may_alias(s_reads, buffer_accesses([t])[1])
            ):
              rp_t = _match_row_pipeline(t, scan_facts)
              if rp_t is not None:
                tk_dim, _ = rp_t
                # Rule A: Pure row map (S) -> Row-Pipeline (S; K) (e.g. inv_std -> LN bwd)
                if scan_facts.same_shape(s.shape, t.shape):
                  t_body = [st for st in t.body if not (isinstance(st, ir.ReleaseScope) and st.scope is t.scope)]
                  rm_key = scan_facts.expr_key(_row_major_expr(t.index, t.shape))
                  pv = Var("rpre", s_buf.type.elt)
                  rew = InlineBufferLoad({x.uid: x for x in defined(t_body).values()}, s_buf, rm_key, pv, scan_facts)
                  new_t_body = rew.block(t_body)
                  if rew.ok and rew.replaced == total_r:
                    cl = Cloner({u.uid: v for u, v in zip(s.index, t.index)})
                    for v in defined(prefix).values():
                      cl.var(v)
                    inlined_pre = [cl.stmt(st) for st in prefix] + [Assign(pv, cl.expr(s_val))]
                    t.body[:] = inlined_pre + new_t_body + [ir.ReleaseScope(t.scope)]
                    gone.add(s_buf.uid)
                    changed = True
                    fused_into_rp = True
                # Rule D: Cheap arithmetic (*S, K) map -> Row-Pipeline (S; K) (e.g. (dQ+dK+dV)*g1 -> LN1 bwd)
                elif (
                  len(s.shape) == len(t.shape) + 1
                  and scan_facts.same_shape(s.shape[:-1], t.shape)
                  and scan_facts.same_expr(s.shape[-1], tk_dim)
                  and op_cost(prefix) + expr_cost(s_val) <= 32
                  and not any(
                    isinstance(x, ir.Prim) and x.op in _EXPENSIVE_PRIMS
                    for st in (*prefix, Assign(s_buf, s_val))
                    for e in ir.stmt_exprs(st)
                    for x in ir.walk_expr(e)
                  )
                ):
                  t_facts = scan_facts.fork()
                  t_facts.activate_nonempty(t.shape)
                  for v, d in zip(t.index, t.shape):
                    t_facts.bind_index(v, d)
                  t_body = [st for st in t.body if not (isinstance(st, ir.ReleaseScope) and st.scope is t.scope)]
                  aff = AffineInlineBufferLoad(
                    s, prefix, s_val, s_buf, t_facts, {x.uid: x for x in defined(t_body).values()}
                  )
                  new_t_body = aff.block(t_body)
                  if aff.ok and aff.replaced == total_r:
                    t.body[:] = new_t_body + [ir.ReleaseScope(t.scope)]
                    gone.add(s_buf.uid)
                    changed = True
                    fused_into_rp = True
            break
          if uses([t], s_defs | s_writes) or uses([s], set(defined([t]))):
            break
          _, tw = buffer_accesses([t])
          if binfo.may_alias(s_reads, tw):
            break
          scan_defs |= set(defined([t]))
          scan_writes |= tw
          scan_facts.record_stmt(t)
        if fused_into_rp:
          continue

    out.append(s)
  return out, changed


def fuse_row_pipelines(fn: ir.Function, options) -> None:
  """Fuse sequential 1-D row reductions and their trailing row epilogues /
  leading row prologues (LayerNorm fwd/bwd, RMSNorm, Softmax) into single
  multi-stage row-pipeline ``IndexMap``s.  Runs after ``monoidal tiling`` so
  FlashAttention and fused Cross-Entropy are matched first."""
  if not options.fusion:
    return
  target = getattr(options, "target", None)
  if target is not None and target.kind == "metal":
    return
  any_changed = False
  for _ in range(6):
    gone: set[int] = set()
    binfo = BufferInfo.analyze(fn)
    facts = IndexFacts.from_function(fn)
    fn.body[:], changed = _row_pipeline_block(fn.body, binfo, facts, gone)
    if gone:
      drop_buffers(fn, gone, drop_allocs=True)
    if not changed:
      break
    any_changed = True
  if any_changed:
    # One sibling cleanup pass merges adjacent reductions newly unblocked (e.g. LN1 d_b1 + d_g1)
    max_bufs = 96 if (target is not None and target.kind == "cuda") else 20
    binfo = BufferInfo.analyze(fn)
    facts = IndexFacts.from_function(fn)
    fn.body[:] = fuse_block(fn.body, facts, binfo, max_bufs=max_bufs)


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
  "fuse_online_logsumexp",
  "fuse_row_pipelines",
  "fuse_siblings",
  "fuse_vertical",
  "fuse_vertical_pair",
  "record_check_equality",
  "split_invariant_map",
  "squeeze_unit_map",
]
