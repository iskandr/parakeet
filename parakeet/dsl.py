"""Explicit data-parallel DSL & algebraic rewrite inspector for Parakeet.

Unlike the NumPy frontend, :mod:`parakeet.dsl` does **not** rewrite or patch
NumPy functions.  Instead, it exposes Parakeet's core data-parallel operators
directly under explicit names alongside :func:`show_rewrite`, which inspects
how Parakeet's six algebraic laws transform a concise specification into
fused, SRAM-tiled, or chunkwise-dual native kernels::

    import numpy as np
    from parakeet import dsl

    @dsl.kernel
    def flash_attention(q, k, v):
        s = (q @ k.T) * np.float32(0.125)
        e = np.exp(s - np.max(s, axis=-1, keepdims=True))
        return (e / np.sum(e, axis=-1, keepdims=True)) @ v

    report = dsl.show_rewrite(
        flash_attention,
        np.zeros((64, 64), np.float32),
        np.zeros((64, 64), np.float32),
        np.zeros((64, 64), np.float32),
    )
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass

from . import api, config, prims
from .lib.adverbs import (
  Gradient,
  LibFunction,
  allpairs,
  filter_reduce,
  fold,
  grad,
  imap,
  ireduce,
  map,  # noqa: A004
  outer_map,
  parfor,
  reduce,
  scan,
  scatter_add,
  value_and_grad,
)
from .syntax import ir

# Explicit operator aliases mapping 1-to-1 to Parakeet's five core adverbs
kernel = api.jit
iscan = scan
iscatter = scatter_add
ifilter_reduce = filter_reduce


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
    guarded = any(
      isinstance(s, ir.If) and any(isinstance(t, ir.Alloc) and t.var.name.startswith("reassoc_m") for t in s.then)
      for s in ir.walk_stmts(after_reassoc.body)
    )
    if guarded:
      add_law("Law 3: Bilinear Contraction Reassociation / MLA Absorption (cost-guarded in IR)")
    elif any(isinstance(s, ir.Alloc) and s.var.name.startswith("reassoc_m") for s in ir.walk_stmts(after_reassoc.body)):
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
    if any(isinstance(s, ir.Alloc) and s.var.name.startswith("ts_chA") for s in ir.walk_stmts(after_scan.body)):
      add_law("Law 4: Chunkwise Parallel Affine Prefix Scan (TileScan, length-guarded in IR)")
    if any(isinstance(s, ir.Alloc) and s.var.name.startswith("ssd_Yi") for s in ir.walk_stmts(after_scan.body)):
      add_law("Law 4: Chunkwise State-Space Duality (Mamba-2 SSD, length-guarded in IR)")

  if after_tiling is not None:
    for s in ir.walk_stmts(after_tiling.body):
      if isinstance(s, ir.IndexScatter):
        if len(s.updates) > 1:
          add_law(f"IndexScatter: {len(s.updates)} updates in one keyed traversal")
        else:
          add_law("IndexScatter: Privatized Associative Scatter-Add")
      elif isinstance(s, ir.IndexReduce) and any(f.carries for f in s.folds):
        add_law("Carry: Single-Pass Multi-State Monoid Fold")

  return tuple(laws), tuple(plans)


def show_rewrite(fn, *args, backend: str | None = None, blas: bool = False, **opts) -> RewriteReport:
  """Compile-only algebraic rewrite inspection for ``fn`` on the types of ``args``.

  Works reliably on cold or warm specializations (including repeated calls and
  calls after ordinary execution), supports plain functions, ``@jit`` /
  ``@dsl.kernel`` functions, and ``grad`` / ``value_and_grad`` wrappers without
  executing ``fn`` or mutating global compiler state.
  """
  snapshots: dict[str, str] = {}
  counts: dict[str, tuple[int, int, int, int]] = {}
  detected_laws: list[str] = []
  detected_plans: list[str] = []

  previous_tiles = 0

  def observer(name: str, f: ir.Function) -> None:
    nonlocal previous_tiles
    tiles = sum(isinstance(stmt, ir.IndexTile) for stmt in ir.walk_stmts(f.body))
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
  "RewriteReport",
  "allpairs",
  "filter_reduce",
  "fold",
  "grad",
  "ifilter_reduce",
  "imap",
  "ireduce",
  "iscan",
  "iscatter",
  "kernel",
  "map",
  "outer_map",
  "parfor",
  "prims",
  "reduce",
  "scan",
  "scatter_add",
  "show_rewrite",
  "value_and_grad",
]
