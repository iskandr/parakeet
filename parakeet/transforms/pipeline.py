"""The middle end: passes over the operator statements, then lowering.

    elaboration  ->  IndexMap / IndexReduce over loop IR
                 ->  [ passes ]  ->  loops  ->  backends

Each pass takes a function and rewrites it in place.  ``PASSES`` is the
order they run in; ``parakeet.find_broken_transform`` switches them on one
at a time, and ``config.skip_transforms`` leaves some out.  Lowering always
runs.
"""

from __future__ import annotations

from ..syntax import ir
from . import fusion, lowering, simplify, tiling

PASSES: list[tuple[str, object]] = [
  ("dead code", simplify.dead_code),
  ("filter-reduce fusion", fusion.fuse_filters),
  ("sibling fusion", fusion.fuse_siblings),
  ("tiling", tiling.tiling),
]


def run(fn: ir.Function, options, trace=None) -> None:
  """Optimize and lower ``fn`` in place; ``trace(name, fn)`` is told after every pass."""
  off = set(getattr(options, "skip", ()))
  for name, apply in PASSES:
    if name not in off:
      apply(fn, options)  # type: ignore[operator]
      if trace is not None:
        trace(name, fn)
  lowering.lower(fn)


def names() -> list[str]:
  return [name for name, _ in PASSES]


__all__ = ["PASSES", "names", "run"]
