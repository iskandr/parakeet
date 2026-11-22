"""The library of functions available inside compiled code."""

from . import adverbs, builtins_lib, numpy_lib, pure, random_lib  # noqa: F401  (registration side effects)
from .core import (
  FUNCTIONS,
  METHODS,
  NAMES,
  PROPERTIES,
  array_literal,
  build_map,
  call_ufunc,
  convert,
  invoke,
  lookup,
  lookup_method,
  lookup_property,
  suggest_attribute,
  unsupported_message,
)
from .numpy_lib import matmul


def contains(el, container, item):
  """``item in container`` for tuples and arrays of scalars."""
  from ..syntax import ops
  from ..syntax.values import ArrayVal, RangeVal

  if isinstance(container, tuple):
    acc = ops.FALSE
    for x in container:
      acc = ops.lor(acc, el.truth(el.compare(_EQ, item, x)))
    return acc
  if isinstance(container, RangeVal):
    container = el.range_array(container)
  if isinstance(container, ArrayVal):
    from ..prims import UFUNCS

    return numpy_lib.any_(el, el.apply_ufunc(UFUNCS["eq"], [container, item]))
  from ..errors import TypeInferenceError
  from ..syntax.values import describe

  raise el.error(TypeInferenceError, f"argument of type {describe(container)} is not iterable")


import ast as _ast  # noqa: E402

_EQ = _ast.Eq()

__all__ = [
    "FUNCTIONS", "METHODS", "NAMES", "PROPERTIES", "array_literal", "build_map", "call_ufunc",
    "contains", "convert", "invoke", "lookup", "lookup_method", "lookup_property", "matmul",
    "suggest_attribute", "unsupported_message",
]  # fmt: skip
