"""Staged implementations of ``np.searchsorted``, ``np.sort``, ``np.argsort``, ``np.unique``,
and read-only lookup indices (:class:`~parakeet.SortedIndex`, :class:`~parakeet.FrozenHashTable`)."""

from __future__ import annotations

import numpy as np

from ..errors import StaticValueError, TypeInferenceError, UnsupportedError
from ..index import FrozenHashTable, SortedIndex
from ..ndtypes import Bool, Int64, UInt64, join
from ..syntax import ops
from ..syntax.ir import Assign, Break, Const, Expr, If, RtCall, Store, Var, While
from ..syntax.ops import ONE, ZERO, I, cast, cmp, iadd, isub
from ..syntax.values import ArrayVal, describe
from .core import method, register

_DTYPE_TAG = {
  "bool": 0,
  "int8": 1,
  "uint8": 2,
  "int16": 3,
  "uint16": 4,
  "int32": 5,
  "uint32": 6,
  "int64": 7,
  "uint64": 8,
  "float32": 9,
  "float64": 10,
}


# ---------------------------------------------------------------------------
# Fusible binary search: np.searchsorted
# ---------------------------------------------------------------------------


def _searchsorted_scalar(el, a: ArrayVal, x: Expr, right: bool, sorter: ArrayVal | None = None) -> Expr:
  n = a.shape[0]
  common = join(a.elt, x.type).strong
  xv = el.tmp(cast(x, common), "sv")
  lo = Var("lo", I)
  hi = Var("hi", I)
  el.mutables.update((lo.uid, hi.uid))
  el.emit(Assign(lo, ZERO))
  el.emit(Assign(hi, n))
  wscope = el.new_scope("loop")
  wbody: list = []
  with el.in_block(wbody, wscope, loop=True):
    el.emit(If(ops.lnot(cmp("lt", lo, hi)), [Break()], []))
    mid = el.tmp(iadd(lo, ops.prim("rshift", (isub(hi, lo), ONE), I)), "mid")
    at_idx = cast(el.elem(sorter, (mid,)), I) if sorter is not None else mid
    am = el.tmp(cast(el.elem(a, (at_idx,)), common), "am")
    if common.kind == "f":
      nan_am = ops.prim("isnan", (am,), Bool)
      nan_x = ops.prim("isnan", (xv,), Bool)
      if right:
        go_right = ops.lor(nan_x, ops.land(ops.lnot(nan_am), ops.prim("le", (am, xv), Bool)))
      else:
        go_right = ops.land(ops.lnot(nan_am), ops.lor(nan_x, ops.prim("lt", (am, xv), Bool)))
    else:
      go_right = ops.prim("le" if right else "lt", (am, xv), Bool)
    el.emit(If(go_right, [Assign(lo, iadd(mid, ONE))], [Assign(hi, mid)]))
  el.emit(While(wbody, wscope))
  return cast(lo, Int64)


@register(np.searchsorted)
def searchsorted(el, a, v, side="left", sorter=None):
  a_arr = el.as_array(a, "a")
  if a_arr.rank != 1:
    raise el.error(TypeInferenceError, f"searchsorted() requires a 1-D sorted array, got {a_arr.rank}-D")
  if a_arr.buf is None and a_arr.cost > 24:
    a_arr = el.as_concrete(a_arr)
  el.consume(a_arr)
  side_s = el.require_static(side, "side")
  if side_s not in ("left", "right"):
    raise el.error(StaticValueError, f"searchsorted() side must be 'left' or 'right', got {side_s!r}")
  right = side_s == "right"
  srt = None
  if sorter is not None:
    srt = el.as_concrete(el.as_array(sorter, "sorter"))
    if srt.rank != 1 or srt.elt.kind not in "iu":
      raise el.error(TypeInferenceError, "searchsorted() sorter must be a 1-D integer array")
    if el.options.bounds_check:
      el.check(cmp("eq", srt.shape[0], a_arr.shape[0]), "ValueError", "sorter must have same length as a")
  if isinstance(v, Expr):
    return _searchsorted_scalar(el, a_arr, v, right, srt)
  v_arr = el.as_array(v, "v")
  if v_arr.rank == 0:
    return _searchsorted_scalar(el, a_arr, el.as_scalar(v_arr), right, srt)
  if v_arr.buf is None and v_arr.cost > 24:
    v_arr = el.as_concrete(v_arr)
  el.consume(v_arr)
  return el.delayed(
    Int64,
    v_arr.shape,
    lambda idx: _searchsorted_scalar(el, a_arr, el.elem(v_arr, idx), right, srt),
    cost=16,
  )


# ---------------------------------------------------------------------------
# Parallel 8-bit LSD radix argsort, sort, unique
# ---------------------------------------------------------------------------


def _argsort_1d(el, lane: ArrayVal) -> ArrayVal:
  if lane.rank == 0:
    return el.alloc_array(Int64, (ONE,), zero=True, label="order")
  if lane.rank != 1:
    lane = el.flat_view(lane)
  lane = el.as_concrete(lane)
  n = lane.shape[0]
  tag = _DTYPE_TAG.get(lane.elt.name)
  if tag is None:
    raise el.error(UnsupportedError, f"argsort() does not support dtype {lane.elt.name}")
  out = el.alloc_array(Int64, (n,), label="order")
  el.emit(
    Assign(
      Var("sort_ok", I),
      RtCall(
        "argsort",
        (Const(tag, I), n, lane.buf, lane.offset, lane.strides[0], out.buf, ZERO),
        I,
        pure=False,
      ),
    )
  )
  el.record_store(out)
  return out


def _sort_1d(el, lane: ArrayVal) -> ArrayVal:
  if lane.rank == 0:
    return lane
  if lane.rank != 1:
    lane = el.flat_view(lane)
  c = el.as_concrete(lane)
  ord1 = _argsort_1d(el, c)
  return el.as_concrete(el.getitem(c, ord1))


@register(np.unique)
def unique(el, ar, return_index=False, return_inverse=False, return_counts=False, axis=None, equal_nan=True):
  if axis is not None:
    raise el.error(UnsupportedError, "unique() with axis != None is not supported")
  ret_idx = el.static_bool(return_index, "return_index") if not isinstance(return_index, bool) else return_index
  ret_inv = el.static_bool(return_inverse, "return_inverse") if not isinstance(return_inverse, bool) else return_inverse
  ret_cnt = el.static_bool(return_counts, "return_counts") if not isinstance(return_counts, bool) else return_counts
  eq_nan = el.static_bool(equal_nan, "equal_nan") if not isinstance(equal_nan, bool) else equal_nan
  if ret_inv:
    raise el.error(UnsupportedError, "unique(..., return_inverse=True) is not yet supported")

  flat = el.as_concrete(el.flat_view(el.as_array(ar, "ar")))
  n = flat.shape[0]
  order = _argsort_1d(el, flat)
  sorted_v = el.as_concrete(el.getitem(flat, order))

  def is_new(idx):
    i = idx[0]
    prev_i = ops.imax(isub(i, ONE), ZERO)
    cur = el.elem(sorted_v, (i,))
    prv = el.elem(sorted_v, (prev_i,))
    if sorted_v.elt.kind == "f" and eq_nan:
      both_nan = ops.land(ops.prim("isnan", (cur,), Bool), ops.prim("isnan", (prv,), Bool))
      diff = ops.land(ops.prim("ne", (cur, prv), Bool), ops.lnot(both_nan))
    else:
      diff = ops.prim("ne", (cur, prv), Bool)
    return ops.lor(cmp("eq", i, ZERO), diff)

  mask = el.delayed(Bool, (n,), is_new, cost=4)
  starts = el.nonzero1d(mask)
  uniq_vals = el.as_concrete(el.getitem(sorted_v, starts))

  outputs = [uniq_vals]
  if ret_idx:
    outputs.append(el.as_concrete(el.getitem(order, starts)))
  if ret_cnt:
    m = starts.shape[0]
    counts = el.alloc_array(Int64, (m,), label="uniq_cnt")

    def _cnt_at(idx):
      j = idx[0]
      s0 = cast(el.elem(starts, (j,)), I)
      next_j = ops.imin(iadd(j, ONE), isub(m, ONE))
      s1 = ops.select(cmp("lt", iadd(j, ONE), m), cast(el.elem(starts, (next_j,)), I), n)
      el.emit(Store(counts.buf, j, cast(isub(s1, s0), Int64)))

    el.index_map((m,), _cnt_at, ordered=False)
    outputs.append(counts)
  return outputs[0] if len(outputs) == 1 else tuple(outputs)


# ---------------------------------------------------------------------------
# SortedIndex & FrozenHashTable staged lookup methods
# ---------------------------------------------------------------------------


def _sorted_index_contains(el, idx_obj: SortedIndex, key) -> Expr:
  keys = idx_obj.keys
  k_expr = el.as_scalar(key, "key")
  pos = el.tmp(cast(_searchsorted_scalar(el, keys, k_expr, right=False), I), "pos")
  in_bounds = cmp("lt", pos, keys.shape[0])
  safe_pos = ops.select(in_bounds, pos, ZERO)
  common = join(keys.elt, k_expr.type).strong
  matched = ops.land(in_bounds, ops.prim("eq", (cast(el.elem(keys, (safe_pos,)), common), cast(k_expr, common)), Bool))
  return matched


def _sorted_index_get(el, idx_obj: SortedIndex, key, default=None):
  keys, values = idx_obj.keys, idx_obj.values
  k_expr = el.as_scalar(key, "key")
  if default is None:
    default = Const(0.0 if values.elt.kind == "f" else 0, values.elt)
  def_e = el.as_scalar(default, "default")
  out_t = join(values.elt, def_e.type).strong
  pos = el.tmp(cast(_searchsorted_scalar(el, keys, k_expr, right=False), I), "pos")
  in_bounds = cmp("lt", pos, keys.shape[0])
  safe_pos = ops.select(in_bounds, pos, ZERO)
  common = join(keys.elt, k_expr.type).strong
  hit = el.tmp(
    ops.land(in_bounds, ops.prim("eq", (cast(el.elem(keys, (safe_pos,)), common), cast(k_expr, common)), Bool)),
    "hit",
  )
  if values.rank != 1:
    raise el.error(UnsupportedError, "SortedIndex.get() inside @jit currently expects 1-D values")
  res = Var("si_val", out_t)
  el.mutables.add(res.uid)
  el.emit(If(hit, [Assign(res, cast(el.elem(values, (pos,)), out_t))], [Assign(res, cast(def_e, out_t))]))
  return res


def _sorted_index_range(el, idx_obj: SortedIndex, lo, hi):
  keys = idx_obj.keys
  s = _searchsorted_scalar(el, keys, el.as_scalar(lo, "lo"), right=False)
  e = _searchsorted_scalar(el, keys, el.as_scalar(hi, "hi"), right=False)
  return (s, e)


def _hashtable_probe(el, ht: FrozenHashTable, key) -> tuple[Expr, Expr]:
  """Linear probe returning ``(found: Bool, slot: I)``."""
  keys = ht.keys
  cap = keys.shape[0]
  mask = el.tmp(isub(cap, ONE), "ht_mask")
  k_expr = el.as_scalar(key, "key")
  k_t = keys.elt
  k_cast = el.tmp(cast(k_expr, k_t), "hk")
  ek = Const(ht.empty_key, k_t) if isinstance(ht.empty_key, int) else cast(ht.empty_key, k_t)
  h = el.tmp(RtCall("mix64", (cast(k_cast, UInt64),), UInt64, pure=True), "h")
  slot = Var("slot", I)
  found = Var("found", Bool)
  steps = Var("steps", I)
  el.mutables.update((slot.uid, found.uid, steps.uid))
  el.emit(Assign(slot, ops.prim("bitand", (cast(h, I), mask), I)))
  el.emit(Assign(found, ops.FALSE))
  el.emit(Assign(steps, ZERO))
  wscope = el.new_scope("loop")
  wbody: list = []
  with el.in_block(wbody, wscope, loop=True):
    el.emit(If(ops.lnot(cmp("lt", steps, cap)), [Break()], []))
    cur = el.tmp(el.elem(keys, (slot,)), "curk")
    el.emit(If(ops.prim("eq", (cur, k_cast), Bool), [Assign(found, ops.TRUE), Break()], []))
    el.emit(If(ops.prim("eq", (cur, ek), Bool), [Break()], []))
    el.emit(Assign(slot, ops.prim("bitand", (iadd(slot, ONE), mask), I)))
    el.emit(Assign(steps, iadd(steps, ONE)))
  el.emit(While(wbody, wscope))
  return found, slot


def _hashtable_contains(el, ht: FrozenHashTable, key) -> Expr:
  found, _ = _hashtable_probe(el, ht, key)
  return found


def _hashtable_get(el, ht: FrozenHashTable, key, default=None):
  values = ht.values
  if values.rank != 1:
    raise el.error(UnsupportedError, "FrozenHashTable.get() inside @jit currently expects 1-D values")
  if default is None:
    default = Const(0.0 if values.elt.kind == "f" else 0, values.elt)
  def_e = el.as_scalar(default, "default")
  out_t = join(values.elt, def_e.type).strong
  found, slot = _hashtable_probe(el, ht, key)
  res = Var("ht_val", out_t)
  el.mutables.add(res.uid)
  el.emit(If(found, [Assign(res, cast(el.elem(values, (slot,)), out_t))], [Assign(res, cast(def_e, out_t))]))
  return res


@method("get")
def _index_get(el, obj, key, default=None):
  if isinstance(obj, SortedIndex):
    return _sorted_index_get(el, obj, key, default)
  if isinstance(obj, FrozenHashTable):
    return _hashtable_get(el, obj, key, default)
  raise el.error(TypeInferenceError, f"{describe(obj)} has no method 'get'")


@method("range")
def _index_range(el, obj, lo, hi):
  if isinstance(obj, SortedIndex):
    return _sorted_index_range(el, obj, lo, hi)
  raise el.error(TypeInferenceError, f"{describe(obj)} has no method 'range'")
