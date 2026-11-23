"""The middle end: passes over the operator statements, then lowering.

    elaboration  ->  IndexMap / IndexReduce / IndexScan / IndexFilter
                 ->  [ passes ]  ->  loops  ->  backends

Each pass takes a function and rewrites it in place.  ``PASSES`` is the
order they run in; ``parakeet.find_broken_transform`` switches them on one
at a time, and ``config.skip_transforms`` leaves some out.  Lowering always
runs.
"""

from __future__ import annotations

from ..config import config
from ..syntax import ir
from . import fusion, lowering, simplify, tiling
from .verify import verify

PASSES: list[tuple[str, object]] = [
  ("dead code", simplify.dead_code),
  ("vertical fusion", fusion.fuse_vertical),
  ("filter fusion", fusion.fuse_filters),
  ("sibling fusion", fusion.fuse_siblings),
  ("reduction interchange", tiling.interchange),
  ("tiling", tiling.tiling),
]


def run(fn: ir.Function, options, trace=None) -> None:
  """Optimize and lower ``fn`` in place; ``trace(name, fn)`` is told after every pass."""
  check = config.verify
  if check:
    verify(fn, "before passes")
  off = set(getattr(options, "skip", ()))
  for name, apply in PASSES:
    if name not in off:
      apply(fn, options)  # type: ignore[operator]
      if check:
        verify(fn, f"after {name}")
      if trace is not None:
        trace(name, fn)
  lowering.lower(fn)
  if check:
    verify(fn, "after lowering")


def names() -> list[str]:
  return [name for name, _ in PASSES]


__all__ = ["PASSES", "names", "run"]
