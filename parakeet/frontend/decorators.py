"""The two decorators: ``jit`` for functions to compile, ``macro`` for
functions that run *inside* the compiler and build the code themselves."""

from __future__ import annotations

from ..api import jit
from ..lib.adverbs import LibFunction


def macro(fn=None, *, name: str | None = None):
  """Define a staged function: ``fn(el, *args, **kwargs)`` is called at compile
  time with the elaborator and the compile-time values of its arguments
  (scalar expressions, :class:`~parakeet.syntax.values.ArrayVal`, tuples,
  functions) and returns the value of the call.

  This is how ``parakeet.map``, ``reduce`` and the NumPy functions are
  implemented.  The result can be called from compiled code and, like the
  built-in adverbs, directly from Python."""
  if fn is None:
    return lambda f: macro(f, name=name)
  if not callable(fn):
    raise TypeError(f"@macro expects a function, got {type(fn).__name__}")
  return LibFunction(name or fn.__name__, fn)


__all__ = ["jit", "macro"]
