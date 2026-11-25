"""Algebraic rewrite inspector and compatibility namespace for Parakeet operators.

All data-parallel operators (``map``, ``imap``, ``ireduce``, ``allpairs``,
``fold``, ``scan``, ``scatter_add``, ``grad``, ``value_and_grad``) and
:func:`show_rewrite` are exported directly on the top-level :mod:`parakeet`
module and work both inside ``@parakeet.jit`` and directly from Python::

    import parakeet as par

    @par.jit
    def tropical_min_plus(a, b):
        m, k = a.shape
        _, n = b.shape
        return par.imap(
            lambda i, j: par.ireduce(lambda r: a[i, r] + b[r, j], k, combine=par.minimum),
            (m, n),
        )

This module remains available as a thin compatibility alias (including compact
dtype shorthands like ``f32`` / ``i64`` and ``kernel = parakeet.jit``).
"""

from __future__ import annotations

import builtins as _builtins
from dataclasses import dataclass

import numpy as _np

from . import api, config, prims
from .index import FrozenHashTable, SortedIndex
from .lib.adverbs import (
  _DSL_MAX,
  _DSL_MIN,
  Gradient,
  LibFunction,
  allpairs,
  checkpoint,
  custom_jvp,
  custom_vjp,
  filter,  # noqa: A004
  filter_reduce,
  fold,
  grad,
  hessian,
  hvp,
  imap,
  ipa_attention,
  ireduce,
  jacfwd,
  jacrev,
  jvp,
  linear_transpose,
  linearize,
  map,  # noqa: A004
  outer_map,
  parfor,
  reduce,
  scan,
  scatter_add,
  stop_gradient,
  tanimoto_topk,
  topk,
  topk_fold,
  value_and_grad,
  vjp,
)
from .lib.expand_lib import filter_pairs, flatmap
from .lib.ragged_lib import (
  segment_ids,
  segmented_map,
  segmented_reduce,
  segmented_scan,
  segmented_softmax,
  varlen_flash_attention,
)
from .lib.spatial_lib import block_sparse_attention, morton_order, spatial_cutoff_sum
from .lib.wavefront_lib import batched_wavefront, soft_align, wavefront
from .ragged import RaggedArray
from .spatial import BlockMask
from .syntax import ir

# Explicit kernel decorator & operator aliases
kernel = api.jit
scatter = scatter_add
iscan = scan
iscatter = scatter_add
ifilter_reduce = filter_reduce
popcount = prims.popcount
clz = prims.clz
ctz = prims.ctz
rotl = prims.rotl
rotr = prims.rotr
bswap = prims.bswap
bf16_to_f32 = prims.bf16_to_f32
f32_to_bf16 = prims.f32_to_bf16

# Self-contained elementwise combiners and math (disjoint from numpy / top-level parakeet)
add = prims.add
sub = prims.subtract
mul = prims.multiply
div = prims.divide
min = _DSL_MIN  # noqa: A001
max = _DSL_MAX  # noqa: A001
minimum = prims.minimum
maximum = prims.maximum
mod = prims.remainder
pow = prims.power  # noqa: A001
and_ = prims.logical_and
or_ = prims.logical_or
bitand = prims.bitwise_and
bitor = prims.bitwise_or
bitxor = prims.bitwise_xor

abs = prims.UFUNCS["abs"]  # noqa: A001
neg = prims.negative
exp = prims.UFUNCS["exp"]
log = prims.UFUNCS["log"]
sqrt = prims.UFUNCS["sqrt"]
tanh = prims.UFUNCS["tanh"]
sin = prims.UFUNCS["sin"]
cos = prims.UFUNCS["cos"]
isnan = _np.isnan
isfinite = _np.isfinite
where = _np.where

# Array reductions & constructors (registered in Parakeet's compiler library)
sum = _np.sum  # noqa: A001
prod = _np.prod
mean = _np.mean
argmin = _np.argmin
argmax = _np.argmax
cumsum = _np.cumsum
any = _np.any  # noqa: A001
all = _np.all  # noqa: A001
zeros = _np.zeros
ones = _np.ones
full = _np.full
empty = _np.empty
arange = _np.arange

# Compact scalar dtype constructors (usable as casts and NamedTuple annotations)
f32 = float32 = _np.float32
f64 = float64 = _np.float64
i32 = int32 = _np.int32
i64 = int64 = _np.int64
u32 = uint32 = _np.uint32
u64 = uint64 = _np.uint64
bool_ = _np.bool_


@dataclass(frozen=True)
class RewriteReport:
  """Side-by-side summary of an algebraic rewrite performed by Parakeet."""

  fn_name: str
  laws_triggered: tuple[str, ...]
  monoidal_plans: tuple[str, ...]
  before_ir: str
  after_ir: str
  before_maps: int
  after_maps: int
  after_tiles: int
  after_scatters: int
  after_carries: int
  before_scatters: int = 0

  def summary(self) -> str:
    laws = ", ".join(self.laws_triggered) if self.laws_triggered else "(elementwise / standard tiling)"
    plans = ", ".join(self.monoidal_plans) if self.monoidal_plans else "none"
    return (
      f"Parakeet Algebraic Rewrite Report for `{self.fn_name}`\n"
      f"  Laws triggered : {laws}\n"
      f"  Monoidal plans : {plans}\n"
      f"  Operators      : IndexMap {self.before_maps} -> {self.after_maps} | "
      f"IndexTile {self.after_tiles} | IndexScatter {self.before_scatters} -> {self.after_scatters} | Carry folds {self.after_carries}"
    )


def _count_ops(ir_fn: ir.Function) -> tuple[int, int, int, int]:
  maps = tiles = scatters = carries = 0
  for s in ir.walk_stmts(ir_fn.body):
    if isinstance(s, ir.IndexMap):
      maps += 1
    elif isinstance(s, ir.IndexTile):
      tiles += 1
    elif isinstance(s, ir.IndexScatter):
      scatters += 1
    elif isinstance(s, ir.IndexReduce):
      for f in s.folds:
        if f.carries:
          carries += len(f.carries)
  return maps, tiles, scatters, carries


def _detect_laws(
  after_reassoc: ir.Function | None,
  after_monoid: ir.Function | None,
  after_scan: ir.Function | None,
  after_tiling: ir.Function | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
  laws: list[str] = []
  plans: list[str] = []

  def add_law(label: str) -> None:
    if label not in laws:
      laws.append(label)

  if after_reassoc is not None:
    guarded = _builtins.any(
      isinstance(s, ir.If)
      and _builtins.any(isinstance(t, ir.Alloc) and t.var.name.startswith("reassoc_m") for t in s.then)
      for s in ir.walk_stmts(after_reassoc.body)
    )
    if guarded:
      add_law("Law 3: Bilinear Contraction Reassociation / MLA Absorption (cost-guarded in IR)")
    elif _builtins.any(
      isinstance(s, ir.Alloc) and s.var.name.startswith("reassoc_m") for s in ir.walk_stmts(after_reassoc.body)
    ):
      add_law("Law 3: Bilinear Contraction Reassociation / MLA Absorption")

  if after_monoid is not None:
    for s in ir.walk_stmts(after_monoid.body):
      if isinstance(s, ir.IndexTile) and s.monoidal_tile is not None:
        p = s.monoidal_tile
        plans.append(p.kind)
        if p.kind in ("flash_monoid", "monoid_lse"):
          add_law("Law 1: Rescaled Semiring Monoid Tile Fusion")
        if p.kind in ("flash_monoid_bwd", "cce_bwd"):
          add_law("Law 2: Algebraic AD Rematerialization")
        if p.tile_false_guard is not None or p.tile_true_guard is not None:
          add_law("Law 5: Affine Tile-Interval Mask Pruning")
        elif p.mask_buf is not None:
          add_law("Law 5: Boolean-Mask Tile Ballot Skipping")
        if p.split_k and "unnormalized" in p.aux_exprs:
          add_law("Law 1: FlashDecoding Split-K Monoid Associativity (shape-guarded in IR)")
      if isinstance(s, ir.Assign) and s.var.name.startswith("gqa_meff"):
        add_law("Law 3: GQA Shared-KV Head-Group Tiling")

  if after_scan is not None:
    if _builtins.any(
      isinstance(s, ir.Alloc) and s.var.name.startswith("ts_chA") for s in ir.walk_stmts(after_scan.body)
    ):
      add_law("Law 4: Chunkwise Parallel Affine Prefix Scan (TileScan, length-guarded in IR)")
    if _builtins.any(
      isinstance(s, ir.Alloc) and s.var.name.startswith("ssd_Yi") for s in ir.walk_stmts(after_scan.body)
    ):
      add_law("Law 4: Chunkwise State-Space Duality (Mamba-2 SSD, length-guarded in IR)")

  if after_tiling is not None:
    for s in ir.walk_stmts(after_tiling.body):
      if isinstance(s, ir.IndexScatter):
        if len(s.updates) > 1:
          add_law(f"IndexScatter: {len(s.updates)} updates in one keyed traversal")
        else:
          add_law("IndexScatter: Privatized Associative Scatter-Add")
      elif isinstance(s, ir.IndexReduce) and _builtins.any(f.carries for f in s.folds):
        add_law("Carry: Single-Pass Multi-State Monoid Fold")
      elif isinstance(s, ir.Alloc) and (
        s.var.name in ("ckpt", "wckpt", "fckpt") or s.var.name.startswith(("ckpt_", "wckpt_", "fckpt_"))
      ):
        add_law("AD: Checkpointed Reverse Loop")
      elif isinstance(s, ir.Alloc) and s.var.name.startswith("fhvp_"):
        add_law("AD: Zero-O(T^2) Rematerialized Attention HVP")
      elif isinstance(s, ir.IndexMap) and _builtins.any(v.name.startswith("fji") for v in s.index):
        add_law("AD: Zero-O(T^2) Rematerialized Attention JVP")

  return tuple(laws), tuple(plans)


def show_rewrite(fn, *args, backend: str | None = None, blas: bool = False, **opts) -> RewriteReport:
  """Compile-only algebraic rewrite inspection for ``fn`` on the types of ``args``.

  Works reliably on cold or warm specializations (including repeated calls and
  calls after ordinary execution), supports plain functions,
  ``@parakeet.jit`` functions, and ``grad`` / ``value_and_grad`` wrappers without
  executing ``fn`` or mutating global compiler state.
  """
  snapshots: dict[str, str] = {}
  counts: dict[str, tuple[int, int, int, int]] = {}
  detected_laws: list[str] = []
  detected_plans: list[str] = []

  previous_tiles = 0

  def observer(name: str, f: ir.Function) -> None:
    nonlocal previous_tiles
    tiles = _builtins.sum(isinstance(stmt, ir.IndexTile) for stmt in ir.walk_stmts(f.body))
    if name == "pairwise fold tiling" and tiles > previous_tiles:
      detected_laws.append("Pairwise map and ordered tile-summary fold")
    previous_tiles = tiles
    if name == "before passes":
      snapshots["before"] = ir.fmt_function(f)
      counts["before"] = _count_ops(f)
    elif name == "contraction reassociation":
      lws, pls = _detect_laws(f, None, None, None)
      detected_laws.extend(lws)
      detected_plans.extend(pls)
    elif name == "monoidal tiling":
      lws, pls = _detect_laws(None, f, None, None)
      detected_laws.extend(lws)
      detected_plans.extend(pls)
    elif name == "chunkwise scan":
      lws, pls = _detect_laws(None, None, f, None)
      detected_laws.extend(lws)
      detected_plans.extend(pls)
    elif name == "tiling":
      lws, pls = _detect_laws(None, None, None, f)
      detected_laws.extend(lws)
      detected_plans.extend(pls)
      snapshots["after"] = ir.fmt_function(f)
      counts["after"] = _count_ops(f)

  run_opts = {"blas": blas, **opts}
  if backend is not None:
    run_opts["backend"] = backend

  with config.options(**run_opts):
    if isinstance(fn, Gradient):
      call_args = (fn.fn, *args)
      call_kwargs = {"argnum": fn.argnum}
      wrapper = api.staged_wrapper(fn.impl, call_args, call_kwargs)
      wrapper.trace_passes(*call_args, _backend=backend, _trace=observer, **call_kwargs)
    elif isinstance(fn, LibFunction):
      wrapper = api.staged_wrapper(fn, args, {})
      wrapper.trace_passes(*args, _backend=backend, _trace=observer)
    else:
      wrapper = fn if isinstance(fn, api.jit) else api.jit(fn)
      wrapper.trace_passes(*args, _backend=backend, _trace=observer)

  uniq_laws = tuple(dict.fromkeys(detected_laws))
  uniq_plans = tuple(detected_plans)
  bm, _, bsc, _ = counts.get("before", (0, 0, 0, 0))
  am, at, asc, ac = counts.get("after", (0, 0, 0, 0))
  return RewriteReport(
    fn_name=getattr(fn, "__name__", getattr(getattr(fn, "fn", fn), "__name__", "fn")),
    laws_triggered=uniq_laws,
    monoidal_plans=uniq_plans,
    before_ir=snapshots.get("before", ""),
    after_ir=snapshots.get("after", ""),
    before_maps=bm,
    after_maps=am,
    after_tiles=at,
    before_scatters=bsc,
    after_scatters=asc,
    after_carries=ac,
  )


__all__ = [
  "BlockMask",
  "FrozenHashTable",
  "RaggedArray",
  "RewriteReport",
  "SortedIndex",
  "abs",
  "add",
  "all",
  "allpairs",
  "and_",
  "any",
  "arange",
  "argmax",
  "argmin",
  "batched_wavefront",
  "bf16_to_f32",
  "bitand",
  "bitor",
  "bitxor",
  "block_sparse_attention",
  "bool_",
  "bswap",
  "checkpoint",
  "clz",
  "cos",
  "ctz",
  "cumsum",
  "custom_jvp",
  "custom_vjp",
  "div",
  "empty",
  "exp",
  "f32",
  "f32_to_bf16",
  "f64",
  "filter",
  "filter_pairs",
  "filter_reduce",
  "flatmap",
  "float32",
  "float64",
  "fold",
  "full",
  "grad",
  "hessian",
  "hvp",
  "i32",
  "i64",
  "ifilter_reduce",
  "imap",
  "int32",
  "int64",
  "ipa_attention",
  "ireduce",
  "iscan",
  "iscatter",
  "isfinite",
  "isnan",
  "jacfwd",
  "jacrev",
  "jvp",
  "kernel",
  "linear_transpose",
  "linearize",
  "log",
  "map",
  "max",
  "maximum",
  "mean",
  "min",
  "minimum",
  "mod",
  "morton_order",
  "mul",
  "neg",
  "ones",
  "or_",
  "outer_map",
  "parfor",
  "popcount",
  "pow",
  "prims",
  "prod",
  "reduce",
  "rotl",
  "rotr",
  "scan",
  "scatter",
  "scatter_add",
  "segment_ids",
  "segmented_map",
  "segmented_reduce",
  "segmented_scan",
  "segmented_softmax",
  "show_rewrite",
  "sin",
  "soft_align",
  "spatial_cutoff_sum",
  "sqrt",
  "stop_gradient",
  "sub",
  "sum",
  "tanh",
  "tanimoto_topk",
  "topk",
  "topk_fold",
  "u32",
  "u64",
  "uint32",
  "uint64",
  "value_and_grad",
  "varlen_flash_attention",
  "vjp",
  "wavefront",
  "where",
  "zeros",
]
