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
  allpairs,
  filter_reduce,
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
from .transforms import pipeline

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

  def summary(self) -> str:
    laws = ", ".join(self.laws_triggered) if self.laws_triggered else "(elementwise / standard tiling)"
    plans = ", ".join(self.monoidal_plans) if self.monoidal_plans else "none"
    return (
      f"Parakeet Algebraic Rewrite Report for `{self.fn_name}`\n"
      f"  Laws triggered : {laws}\n"
      f"  Monoidal plans : {plans}\n"
      f"  Operators      : IndexMap {self.before_maps} -> {self.after_maps} | "
      f"IndexTile {self.after_tiles} | IndexScatter {self.after_scatters} | Carry folds {self.after_carries}"
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


def _detect_laws(after_reassoc: ir.Function | None, after_monoid: ir.Function | None, after_scan: ir.Function | None, after_tiling: ir.Function | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
  laws: list[str] = []
  plans: list[str] = []

  def add_law(label: str) -> None:
    if label not in laws:
      laws.append(label)

  if after_reassoc is not None:
    for s in ir.walk_stmts(after_reassoc.body):
      if isinstance(s, ir.Alloc) and s.var.name.startswith("reassoc_m"):
        add_law("Law 3: Bilinear Contraction Reassociation / MLA Absorption")
        break

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
          add_law("Law 1: FlashDecoding Split-K Monoid Associativity")
      if isinstance(s, ir.Assign) and s.var.name.startswith("gqa_meff"):
        add_law("Law 3: GQA Shared-KV Head-Group Tiling")

  if after_scan is not None:
    for s in ir.walk_stmts(after_scan.body):
      if isinstance(s, ir.Alloc) and s.var.name.startswith("ts_chA"):
        add_law("Law 4: Chunkwise Parallel Affine Prefix Scan (TileScan)")
      if isinstance(s, ir.Alloc) and s.var.name.startswith("ssd_Yi"):
        add_law("Law 4: Chunkwise State-Space Duality (Mamba-2 SSD)")

  if after_tiling is not None:
    for s in ir.walk_stmts(after_tiling.body):
      if isinstance(s, ir.IndexScatter):
        add_law("IndexScatter: Privatized Associative Scatter-Add")
      elif isinstance(s, ir.IndexReduce) and any(f.carries for f in s.folds):
        add_law("Carry: Single-Pass Multi-State Monoid Fold")

  return tuple(laws), tuple(plans)


def show_rewrite(fn, *args, backend: str | None = None, blas: bool = False, **opts) -> RewriteReport:
  """Compile ``fn`` for the types of ``args`` and report every algebraic rewrite applied.

  ``fn`` may be a plain Python function, a ``@parakeet.jit`` / ``@dsl.kernel``
  function, or a ``grad`` / ``value_and_grad`` wrapper.
  """
  # Always compile through a fresh jit wrapper so cached specializations never skip trace hooks
  raw_fn = getattr(fn, "fn", fn)
  target_fn = api.jit(raw_fn) if callable(raw_fn) and not hasattr(fn, "argnum") else fn

  snapshots: dict[str, str] = {}
  counts: dict[str, tuple[int, int, int, int]] = {}
  orig_run = pipeline.run

  # Snapshot lightweight summaries inside trace callback (before later passes mutate IR in-place)
  detected_laws: list[str] = []
  detected_plans: list[str] = []

  def hooked(ir_fn: ir.Function, options, trace=None):
    snapshots["before"] = ir.fmt_function(ir_fn)
    counts["before"] = _count_ops(ir_fn)

    def tr(name: str, f: ir.Function):
      if name == "contraction reassociation":
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
      if trace is not None:
        trace(name, f)

    orig_run(ir_fn, options, trace=tr)

  run_opts = {"blas": blas, **opts}
  if backend is not None:
    run_opts["backend"] = backend

  pipeline.run = hooked
  try:
    with config.options(**run_opts):
      if hasattr(target_fn, "specialize"):
        target_fn.specialize(*args)
      else:
        target_fn(*args)
  finally:
    pipeline.run = orig_run

  # Deduplicate preserving order
  uniq_laws = tuple(dict.fromkeys(detected_laws))
  uniq_plans = tuple(detected_plans)
  bm, _, _, _ = counts.get("before", (0, 0, 0, 0))
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
    after_scatters=asc,
    after_carries=ac,
  )


__all__ = [
  "RewriteReport",
  "allpairs",
  "filter_reduce",
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
