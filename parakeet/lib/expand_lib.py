"""1-to-K ragged compaction (``flatmap`` / :class:`~parakeet.syntax.ir.IndexExpand`) and ``filter_pairs``."""

from __future__ import annotations

from ..errors import TypeInferenceError, UnsupportedError
from ..ndtypes import BufType, Int64
from ..ragged import RaggedArray
from ..syntax import ir, ops
from ..syntax.ir import Assign, Const, Expr, ForRange, If, IndexExpand, Store, Var
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, isub
from ..syntax.records import record_type
from ..syntax.values import ArrayVal, RangeVal, Root, describe
from .adverbs import libfunction
from .ragged_lib import segment_ids


def _check_pure(el, stmts: list, what: str) -> None:
  for s in ir.walk_stmts(stmts):
    if isinstance(s, (ir.Store, ir.IndexScatter)):
      raise el.error(UnsupportedError, f"{what} must be pure (cannot mutate arrays)")
    if any(isinstance(e, ir.RtCall) and not e.pure for x in ir.stmt_exprs(s) for e in ir.walk_expr(x)):
      raise el.error(UnsupportedError, f"{what} must be pure")


def _flatten_scalars(el, v, what: str) -> tuple[list[Expr], object]:
  """Flatten a scalar, tuple, or frozen Record/NamedTuple of scalars into ``(leaves, rebuild)``."""
  if isinstance(v, Expr):
    return [v], lambda xs: xs[0]
  if isinstance(v, ArrayVal) and v.rank == 0:
    return [el.as_scalar(v, what)], lambda xs: xs[0]
  if rec := record_type(v):
    leaves, rb = _flatten_scalars(el, rec.values(v), what)
    return leaves, lambda xs, _r=rec, _rb=rb: _r.build(_rb(xs))
  if isinstance(v, tuple):
    parts = [_flatten_scalars(el, x, what) for x in v]
    flat: list[Expr] = []
    counts: list[int] = []
    for ls, _ in parts:
      flat.extend(ls)
      counts.append(len(ls))

    def rebuild(xs, _parts=parts, _counts=counts):
      out = []
      pos = 0
      for (_, rb), cnt in zip(_parts, _counts):
        out.append(rb(xs[pos : pos + cnt]))
        pos += cnt
      return tuple(out)

    return flat, rebuild
  raise el.error(
    TypeInferenceError,
    f"{what} must return a scalar or tuple/record of scalars, got {describe(v)}",
  )


def _source_domain(el, v, what: str):
  if isinstance(v, RaggedArray):
    if el.options.bounds_check:
      el.check(cmp("ge", v.offsets.shape[0], ONE), "ValueError", "RaggedArray offsets must have length >= 1")
    n = el.tmp(isub(v.offsets.shape[0], ONE), "nseg")
    return n, lambda i: el._ragged_getitem(v, i)
  if isinstance(v, RangeVal):
    v = el.range_array(v)
  if isinstance(v, ArrayVal):
    if v.rank == 0:
      raise el.error(TypeInferenceError, f"{what}() arguments must be at least 1-dimensional")
    if v.buf is None and v.cost > 24:
      v = el.as_concrete(v)
    el.consume(v)
    return v.shape[0], lambda i: el.axis_view(v, 0, i, consumed=True)
  raise el.error(TypeInferenceError, f"{what}() expects an array, RaggedArray, or range, got {describe(v)}")


@libfunction("flatmap")
def flatmap(el, count_fn, emit_fn, *xs):
  """Two-pass 1-to-K ragged expansion returning ``RaggedArray(values, offsets)``.

  For each index ``i`` in ``0 .. len(xs[0])``:
  - ``count_fn(x0[i], ...)`` returns ``c_i >= 0`` (how many items ``i`` emits),
  - ``emit_fn(x0[i], ..., k)`` returns the ``k``-th emitted scalar or SoA record
    for ``k in 0 .. c_i``.
  """
  if not xs:
    raise el.error(TypeInferenceError, "flatmap() requires at least one sequence argument")
  domains = [_source_domain(el, x, "flatmap") for x in xs]
  n = el.tmp(domains[0][0], "n")
  for d, _ in domains[1:]:
    if not el.dim_equal(n, d):
      el.check(cmp("eq", n, d), "ValueError", "flatmap() arguments have different lengths ({0} and {1})", n, d)

  tape = el.tape
  ad_active = tape is not None and not tape.muted
  if ad_active:
    for x in xs:
      if not isinstance(x, ArrayVal) and tape.is_active(el, x):
        raise el.error(
          UnsupportedError, "differentiating flatmap() with respect to non-array sequences is not supported"
        )

  # Elaborate count_fn and emit_fn with tape paused (discrete topology / IndexExpand)
  saved_tape = el.tape
  if ad_active:
    el.tape = None
  try:
    idx = Var("i", I)
    el.nonneg.add(idx.uid)
    el.upper[idx.uid] = n
    count_scope = el.new_scope("expand_count")
    count_blk: list = []
    with el.in_block(count_blk, count_scope, loop=True, par=el.can_parallel((n,))):
      elems = [get(idx) for _, get in domains]
      c_val = el.as_index(el.call(count_fn, elems), "flatmap count")
      if el.options.bounds_check:
        el.check(cmp("ge", c_val, ZERO), "ValueError", "flatmap() count must be non-negative, got {0}", c_val)
      count_expr = el.tmp(c_val, "cnt")
    _check_pure(el, count_blk, "flatmap() count_fn")

    emit_k = Var("k", I)
    el.nonneg.add(emit_k.uid)
    emit_scope = el.new_scope("expand_emit")
    emit_blk: list = []
    with el.in_block(emit_blk, emit_scope, loop=True, par=el.can_parallel((n,))):
      elems2 = [get(idx) for _, get in domains]
      emitted = el.call(emit_fn, [*elems2, emit_k])
      leaves, rebuild = _flatten_scalars(el, emitted, "flatmap() emit_fn")
    _check_pure(el, emit_blk, "flatmap() emit_fn")

    offsets_buf = Var("exp_off", BufType(Int64))
    total_var = Var("exp_tot", I)
    el.nonneg.add(total_var.uid)
    el.scope.owned.append(offsets_buf)

    out_bufs = []
    values_spec = []
    for m, leaf in enumerate(leaves):
      elt = leaf.type.strong
      ob = Var(f"exp_v{m}", BufType(elt))
      el.scope.owned.append(ob)
      out_bufs.append((ob, elt))
      values_spec.append((ob, cast(leaf, elt)))

    parallel = el.can_parallel((n,))
    el.emit(
      IndexExpand(
        n,
        idx,
        count_blk,
        count_expr,
        emit_k,
        emit_blk,
        values_spec,
        offsets_buf,
        total_var,
        count_scope,
        emit_scope,
        parallel,
        el.loc(),
      )
    )
    offsets_arr = ArrayVal(
      Int64,
      (el.tmp(iadd(n, ONE), "n1"),),
      buf=offsets_buf,
      offset=ZERO,
      strides=(ONE,),
      roots=frozenset({Root(el.scope, "exp_off")}),
      contig=True,
    )
    offsets_arr.inj = True
    out_arrays = []
    for ob, elt in out_bufs:
      arr = ArrayVal(
        elt,
        (total_var,),
        buf=ob,
        offset=ZERO,
        strides=(ONE,),
        roots=frozenset({Root(el.scope, ob.name)}),
        contig=True,
      )
      arr.inj = True
      out_arrays.append(arr)
  finally:
    el.tape = saved_tape

  if not ad_active:
    return RaggedArray(rebuild(out_arrays), offsets_arr)

  # Differentiable path: discrete CSR offsets are fixed; gather array inputs via src_ids
  # and replay emit_fn via differentiable map so both xs and closed-over params get exact grads!
  src_ids = segment_ids.impl(el, offsets_arr, total_var)
  from .adverbs import map as _pk_map

  edge_indices = el.range_array(RangeVal(ZERO, total_var, ONE))
  bases = el.getitem(offsets_arr, src_ids)
  local_ks = el.binop("sub", edge_indices, bases)
  gathered_xs = [el.getitem(x, src_ids) if isinstance(x, ArrayVal) else None for x in xs]

  if all(g is not None for g in gathered_xs):
    diff_vals = _pk_map.impl(el, emit_fn, *gathered_xs, local_ks, axis=0)
  else:

    def _map_one(*args):
      e_scalar = args[-1]
      si = el.tmp(cast(el.elem(src_ids, (cast(e_scalar, I),)), I), "si")
      base = el.tmp(cast(el.elem(offsets_arr, (si,)), I), "base")
      local_k = el.tmp(isub(cast(e_scalar, I), base), "lk")
      it = iter(args[:-1])
      call_args = [next(it) if g is not None else get(si) for g, (_, get) in zip(gathered_xs, domains)]
      return el.call(emit_fn, [*call_args, local_k])

    live_g = [g for g in gathered_xs if g is not None]
    diff_vals = _pk_map.impl(el, _map_one, *live_g, edge_indices, axis=0)
  return RaggedArray(diff_vals, offsets_arr)


@libfunction("filter_pairs")
def filter_pairs(el, pred_fn, x, y, value_fn=None):
  """Sparse pairwise neighbor expansion with zero ``O(N * M)`` materialization.

  For each ``i in range(len(x))``, finds all ``j in range(len(y))`` where
  ``pred_fn(x[i], y[j])`` holds.
  - When ``value_fn is None``, returns ``RaggedArray(dst_indices, offsets)``.
  - When ``value_fn`` is provided, returns ``RaggedArray((dst_indices, values), offsets)``
    where ``values[e] = value_fn(x[i], y[j])`` (differentiable w.r.t. ``x`` and ``y``!).
  """
  nx, get_x = _source_domain(el, x, "filter_pairs")
  ny, get_y = _source_domain(el, y, "filter_pairs")
  nx = el.tmp(nx, "nx")
  ny = el.tmp(ny, "ny")

  tape = el.tape
  ad_active = tape is not None and not tape.muted and value_fn is not None
  if ad_active:
    if (not isinstance(x, ArrayVal) and tape.is_active(el, x)) or (
      not isinstance(y, ArrayVal) and tape.is_active(el, y)
    ):
      raise el.error(
        UnsupportedError, "differentiating filter_pairs() with respect to non-array sequences is not supported"
      )

  # Step 1: Discrete two-pass CSR topology discovery (zero N*M allocation, tape paused)
  saved_tape = el.tape
  if tape is not None:
    el.tape = None
  try:
    # Pass 1: count matching j per i -> counts[i]
    counts = el.alloc_array(Int64, (nx,), label="fp_cnt")

    def count_i(idx):
      i = idx[0]
      xi = get_x(i)
      cnt = Var("cnt", Int64)
      el.mutables.add(cnt.uid)
      el.emit(Assign(cnt, Const(0, Int64)))
      j = Var("j", I)
      el.nonneg.add(j.uid)
      el.upper[j.uid] = ny
      jscope = el.new_scope("loop")
      jbody: list = []
      with el.in_block(jbody, jscope, loop=True):
        keep = el.truth(el.call(pred_fn, [xi, get_y(j)]))
        el.emit(Assign(cnt, ops.prim("add", (cnt, ops.select(keep, Const(1, Int64), Const(0, Int64))), Int64)))
      _check_pure(el, jbody, "filter_pairs() pred_fn")
      el.emit(ForRange(j, ZERO, ny, ONE, jbody, jscope))
      el.emit(Store(counts.buf, i, cnt))

    el.index_map((nx,), count_i, ordered=False)

    # Exclusive prefix sum of counts -> offsets[nx + 1]
    from .numpy_lib import cumsum

    n1 = el.tmp(iadd(nx, ONE), "n1")
    offsets = el.alloc_array(Int64, (n1,), zero=True, label="fp_off")
    if not (isinstance(nx, Const) and nx.value == 0):
      cs = cumsum(el, counts, None, None)
      el.store_array(el.getitem(offsets, slice(ONE, n1, None)), cs)
    total = el.tmp(cast(el.elem(offsets, (nx,)), I), "fp_tot")

    # Pass 2: scatter matching j (and optional values when not AD) into [offsets[i], offsets[i + 1])
    dst = el.alloc_array(Int64, (total,), label="fp_dst")

    if value_fn is None or ad_active:

      def fill_dst(idx):
        i = idx[0]
        xi = get_x(i)
        pos = Var("pos", I)
        el.mutables.add(pos.uid)
        el.emit(Assign(pos, cast(el.elem(offsets, (i,)), I)))
        j = Var("j", I)
        el.nonneg.add(j.uid)
        el.upper[j.uid] = ny
        jscope = el.new_scope("loop")
        jbody: list = []
        with el.in_block(jbody, jscope, loop=True):
          keep = el.truth(el.call(pred_fn, [xi, get_y(j)]))
          tblk = [Store(dst.buf, pos, cast(j, Int64)), Assign(pos, iadd(pos, ONE))]
          el.emit(If(keep, tblk, []))
        el.emit(ForRange(j, ZERO, ny, ONE, jbody, jscope))

      el.index_map((nx,), fill_dst, ordered=False)
      fused_vals = None
    else:
      # Probe value_fn output type/structure cleanly in a trial block
      with el.trial():
        probe_i, probe_j = Var("pi", I), Var("pj", I)
        v_sample = el.call(value_fn, [get_x(probe_i), get_y(probe_j)])
        sample_leaves, rebuild = _flatten_scalars(el, v_sample, "filter_pairs() value_fn")
        leaf_elts = [lf.type.strong for lf in sample_leaves]

      val_arrays = [el.alloc_array(elt, (total,), label=f"fp_v{m}") for m, elt in enumerate(leaf_elts)]

      def fill_dst_and_vals(idx):
        i = idx[0]
        xi = get_x(i)
        pos = Var("pos", I)
        el.mutables.add(pos.uid)
        el.emit(Assign(pos, cast(el.elem(offsets, (i,)), I)))
        j = Var("j", I)
        el.nonneg.add(j.uid)
        el.upper[j.uid] = ny
        jscope = el.new_scope("loop")
        jbody: list = []
        with el.in_block(jbody, jscope, loop=True):
          yj = get_y(j)
          keep = el.truth(el.call(pred_fn, [xi, yj]))
          tblk: list = []
          with el.in_block(tblk):
            val = el.call(value_fn, [xi, yj])
            vleaves, _ = _flatten_scalars(el, val, "filter_pairs() value_fn")
            el.emit(Store(dst.buf, pos, cast(j, Int64)))
            for va, lf, elt in zip(val_arrays, vleaves, leaf_elts):
              el.emit(Store(va.buf, pos, cast(lf, elt)))
            el.emit(Assign(pos, iadd(pos, ONE)))
          _check_pure(el, tblk[: -len(val_arrays) - 2], "filter_pairs() value_fn")
          el.emit(If(keep, tblk, []))
        el.emit(ForRange(j, ZERO, ny, ONE, jbody, jscope))

      el.index_map((nx,), fill_dst_and_vals, ordered=False)
      fused_vals = rebuild(val_arrays)
  finally:
    el.tape = saved_tape

  if value_fn is None:
    return RaggedArray(dst, offsets)
  if not ad_active:
    return RaggedArray((dst, fused_vals), offsets)

  # Differentiable value_fn: compute edge values via standard gather + map so pk.grad differentiates cleanly!
  src_ids = segment_ids.impl(el, offsets, total)
  x_edges = el.getitem(x, src_ids) if isinstance(x, ArrayVal) else None
  y_edges = el.getitem(y, dst) if isinstance(y, ArrayVal) else None
  from .adverbs import map as _pk_map

  if x_edges is not None and y_edges is not None:
    diff_vals = _pk_map.impl(el, value_fn, x_edges, y_edges, axis=0)
  else:
    edge_idx = el.range_array(RangeVal(ZERO, total, ONE))

    def _eval_edge(e):
      ei = cast(e, I)
      si = el.tmp(cast(el.elem(src_ids, (ei,)), I), "si")
      dj = el.tmp(cast(el.elem(dst, (ei,)), I), "dj")
      return el.call(value_fn, [get_x(si), get_y(dj)])

    diff_vals = _pk_map.impl(el, _eval_edge, edge_idx, axis=0)
  return RaggedArray((dst, diff_vals), offsets)
