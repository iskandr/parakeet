"""Tree utilities over nested tuples, dicts, dataclasses, and ``par.nn.Module`` objects.

Works identically in plain Python and inside ``@par.jit`` compiled functions so
parameter transforms and optimizers (``par.optim.adam_step``, ``par.optim.sgd_step``)
compile with zero boilerplate.
"""

from __future__ import annotations

import numpy as np

from .backends.device import DeviceArray
from .syntax.records import record_type


def _is_array_leaf(x) -> bool:
  return isinstance(x, (np.ndarray, DeviceArray))


def tree_map(fn, *trees):
  """Apply ``fn`` to matching leaves across one or more identically structured trees."""
  if not trees:
    raise TypeError("tree_map() expects at least one tree argument")
  first = trees[0]
  if isinstance(first, tuple) and not hasattr(first, "_fields"):
    return tuple(tree_map(fn, *parts) for parts in zip(*trees, strict=True))
  if (rec := record_type(first)) is not None:
    cols = [rec.values(t) for t in trees]
    mapped = tuple(tree_map(fn, *parts) for parts in zip(*cols, strict=True))
    return rec.build(mapped)
  return fn(*trees)


def tree_map_arrays(fn, *trees):
  """Like :func:`tree_map`, but only transforms array leaves and preserves static metadata."""
  if not trees:
    raise TypeError("tree_map_arrays() expects at least one tree argument")
  first = trees[0]
  if isinstance(first, tuple) and not hasattr(first, "_fields"):
    return tuple(tree_map_arrays(fn, *parts) for parts in zip(*trees, strict=True))
  if (rec := record_type(first)) is not None:
    cols = [rec.values(t) for t in trees]
    mapped = tuple(tree_map_arrays(fn, *parts) for parts in zip(*cols, strict=True))
    return rec.build(mapped)
  if _is_array_leaf(first):
    return fn(*trees)
  return first


def tree_leaves(tree) -> list:
  """Return all leaves of ``tree`` in deterministic depth-first order."""
  out: list = []

  def _walk(node):
    if isinstance(node, tuple) and not hasattr(node, "_fields"):
      for part in node:
        _walk(part)
    elif (rec := record_type(node)) is not None:
      for part in rec.values(node):
        _walk(part)
    else:
      out.append(node)

  _walk(tree)
  return out


def tree_array_leaves(tree) -> list:
  """Return only the array leaves of ``tree`` (skipping ints, floats, strings, etc.)."""
  return [x for x in tree_leaves(tree) if _is_array_leaf(x)]
