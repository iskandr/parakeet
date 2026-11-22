"""Find out which optimization breaks a function.

Parakeet has no pass pipeline to replay one transform at a time: fusion
happens while a function is elaborated, tiling while its loops are built, and
the rest is the C compiler and the parallel runtime.  What can be done is to
switch those on one after another and see where the answer changes.
"""

from __future__ import annotations

import numpy as np

from ..api import jit
from ..config import config, normalize_backend, options
from ..testing import eq

# (name, settings switched on at this step), applied cumulatively
STEPS = (
  ("fusion", {"fusion": True}),
  ("tiling", {"tiling": True}),
  ("C optimizer", {"opt_level": None}),
  ("parallel loops", {"backend": "auto"}),
)
_BASELINE = {"backend": "c", "fusion": False, "tiling": False, "opt_level": 0}


def _copy(x):
  if isinstance(x, np.ndarray):
    return x.copy()
  if isinstance(x, (tuple, list)):
    return type(x)(_copy(v) for v in x)
  return x


def _run(fn, inputs, settings: dict):
  with options(**settings):
    compiled = jit(fn)
    result = compiled(*[_copy(a) for a in inputs])
    return result, compiled.specialize(*[_copy(a) for a in inputs])


def find_broken_transform(fn, inputs, expected=None, print_transforms: bool = True, print_functions: bool = True):
  """Name of the first optimization that makes ``fn(*inputs)`` differ from
  ``expected`` (by default: what the function returns when run by Python).

  Returns ``None`` if the compiled function is right with everything on, and
  ``"frontend"`` if it is already wrong with every optimization off."""
  fn = getattr(fn, "__parakeet_fn__", fn)
  inputs = list(inputs)
  if expected is None:
    expected = fn(*[_copy(a) for a in inputs])
  final_backend = normalize_backend(config.backend)
  steps = list(STEPS)
  if final_backend in ("metal", "cuda"):
    steps.append((f"{final_backend} kernels", {"backend": final_backend}))
  elif final_backend not in ("auto", "multicore", "python"):
    steps[-1] = (steps[-1][0], {"backend": final_backend})
  settings = dict(_BASELINE)
  full = {"opt_level": config.opt_level}

  def say(*parts) -> None:
    if print_transforms:
      print("[Diagnose]", *parts)

  def attempt(name: str):
    say(f"Running with {name}...")
    try:
      result, spec = _run(fn, inputs, settings)
    except Exception as e:  # noqa: BLE001 - whatever went wrong is the finding
      say(f"{type(e).__name__}: {e}")
      return False
    if eq(result, expected):
      return True
    say(f"Expected {expected!r} but got {result!r}")
    if print_functions:
      print(spec.ir)
    return False

  if not attempt("every optimization off"):
    say("This function was busted before we optimized anything!")
    return "frontend"
  for name, switched_on in steps:
    for k, v in switched_on.items():
      settings[k] = full[k] if v is None else v
    if not attempt(name):
      say(f"The culprit is: {name}")
      return name
  say("Nothing broke")
  return None


__all__ = ["STEPS", "find_broken_transform"]
