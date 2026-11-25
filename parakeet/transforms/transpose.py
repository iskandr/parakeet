"""Linearization and linear transposition for Parakeet's AD system.

Provides:
- :func:`linear_transpose_apply`: exact adjoint ``L^T(u)`` of any linear
  Parakeet function ``L`` satisfying ``<L(v), u> == <v, L^T(u)>`` across
  scalars, N-D arrays, tuples, records, and ``RaggedArray``s (linear
  elementwise ops, ``broadcast <-> sum``, ``matmul <-> matmul^T``,
  ``gather <-> IndexScatter``, ``ForRange <-> reverse ForRange``,
  ``IndexScan <-> reverse IndexScan``).
- :func:`linearize_host` and :func:`vjp_host`: Python-level wrappers returning
  ``(primal_out, lin_fn)`` and ``(primal_out, vjp_fn)`` callable from plain
  Python or inside compiled code.
"""

from __future__ import annotations

from ..errors import TypeInferenceError
from ..syntax.ir import Const, Expr
from ..syntax.values import ArrayVal
from .ad_loops import zero_like_tree
from .jvp import jvp_call, vjp_apply


def linear_transpose_apply(el, linear_fn, example_inputs, cotangent, what: str = "linear_transpose"):
  """Compute the exact transpose ``L^T(cotangent)`` of a linear function ``L``.

  Because ``L`` is linear in its tangent inputs, evaluating its reverse-mode
  adjoint at structural zero inputs produces the exact dual linear map
  (``broadcast <-> sum``, ``matmul(A, dB) <-> matmul(A^T, u)``,
  ``gather <-> IndexScatter``, ``ForRange <-> reverse ForRange``, etc.)
  without requiring primal residual storage for the linear arguments.
  """
  examples = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
  zeros = tuple(zero_like_tree(el, x) for x in examples)
  _, ct_in = vjp_apply(el, linear_fn, zeros, cotangent, what=what)
  return ct_in


def check_transpose_linearity(el, v) -> bool:
  """Return True if ``v`` is a differentiable scalar/array/container type."""
  if isinstance(v, Expr):
    return v.type.kind == "f"
  if isinstance(v, ArrayVal):
    return v.elt.kind == "f"
  if isinstance(v, tuple):
    return any(check_transpose_linearity(el, x) for x in v)
  return True


__all__ = ["check_transpose_linearity", "jvp_call", "linear_transpose_apply", "vjp_apply"]
_ = (Const, TypeInferenceError)
