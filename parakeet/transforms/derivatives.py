"""Derivatives of the scalar primitives, and reverse accumulation over
expression trees.

A scalar computation in the IR is already its own record: an expression
tree whose leaves are constants, variables and loads.  Differentiating it
needs no tape, only a walk from the root that multiplies by one partial
derivative per edge.  :func:`backprop` does that walk; what the leaves mean
(a differentiated variable, an element of a differentiated array, a
constant) is the caller's business.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..errors import UnsupportedError
from ..ndtypes import Bool, ScalarType
from ..syntax import ops
from ..syntax.ir import Cast, Const, Expr, Load, Prim, Var
from ..syntax.ops import cast, prim


def _f(op: str, t: ScalarType, *args: Expr) -> Expr:
  """Build a float primitive ``op`` after casting all operands to ``t``."""
  return prim(op, tuple(cast(a, t) for a in args), t)


def _c(value: float, t: ScalarType) -> Const:
  """Return a typed floating-point constant of type ``t``."""
  return Const(value, t)


def _gt0(x: Expr) -> Expr:
  """Boolean predicate ``x > 0`` in ``x``'s scalar type."""
  return prim("gt", (x, Const(0.0, x.type)), Bool)


def _lt0(x: Expr) -> Expr:
  """Boolean predicate ``x < 0`` in ``x``'s scalar type."""
  return prim("lt", (x, Const(0.0, x.type)), Bool)


def _sign(x: Expr, t: ScalarType) -> Expr:
  """Branchless sign function returning ``-1``, ``0``, or ``+1`` in type ``t``."""
  return ops.select(_gt0(x), _c(1.0, t), ops.select(_lt0(x), _c(-1.0, t), _c(0.0, t)))


def _recip(x: Expr, t: ScalarType) -> Expr:
  """Reciprocal ``1 / x`` in type ``t``."""
  return _f("div", t, _c(1.0, t), x)


def _one_minus_square(x: Expr, t: ScalarType) -> Expr:
  """Compute ``1 - x*x`` (shared by ``arcsin``, ``arccos``, ``tanh``, ``arctanh``)."""
  return _f("sub", t, _c(1.0, t), _f("mul", t, x, x))


def _one_plus_square(x: Expr, t: ScalarType) -> Expr:
  """Compute ``1 + x*x`` (shared by ``tan``, ``arctan``, ``arcsinh``)."""
  return _f("add", t, _c(1.0, t), _f("mul", t, x, x))


def _pow(args, r, t):
  """Partials of ``a ** b`` with zero-exponent and non-positive-base guards."""
  a, b = args
  if isinstance(b, Const) and b.value == 0:
    return (None, None)
  da = _f("mul", t, b, _f("pow", t, a, _f("sub", t, b, _c(1.0, t))))
  if isinstance(b, Const):
    return (da, None)
  da = ops.select(prim("eq", (b, _c(0.0, t)), Bool), _c(0.0, t), da)
  # d/db a**b = a**b * log(a), which only exists for a positive base
  db = ops.select(_gt0(a), _f("mul", t, r, _f("log", t, a)), _c(0.0, t))
  return (da, db)


def _first_wins(op: str):
  """Subgradient rule for binary ``min`` / ``max`` breaking ties in favor of the first operand."""

  def partials(args, r, t):
    a, b = args
    first = prim("le" if op == "min" else "ge", (a, b), Bool)
    return (ops.select(first, _c(1.0, t), _c(0.0, t)), ops.select(first, _c(0.0, t), _c(1.0, t)))

  return partials


def _atan2(args, r, t):
  """Partials of ``arctan2(y, x)``: ``(x / (x^2 + y^2), -y / (x^2 + y^2))``."""
  y, x = args
  d = _f("add", t, _f("mul", t, x, x), _f("mul", t, y, y))
  return (_f("div", t, x, d), _f("neg", t, _f("div", t, y, d)))


def _gauss(sign: float):
  """Partial derivative of ``erf`` (``sign=+1``) or ``erfc`` (``sign=-1``)."""

  def partials(args, r, t):
    (a,) = args
    e = _f("exp", t, _f("neg", t, _f("mul", t, a, a)))
    return (_f("mul", t, _c(sign * 2.0 / math.sqrt(math.pi), t), e),)

  return partials


def _flat(args, r, t):
  """Zero-partial rule for piecewise-constant rounding and floor-division ops."""
  return tuple(None for _ in args)


# op -> (args, result, type) -> one partial derivative per argument (None: zero)
PARTIALS: dict[str, Callable] = {
  "add": lambda args, r, t: (_c(1.0, t), _c(1.0, t)),
  "sub": lambda args, r, t: (_c(1.0, t), _c(-1.0, t)),
  "mul": lambda args, r, t: (args[1], args[0]),
  "div": lambda args, r, t: (_recip(args[1], t), _f("neg", t, _f("div", t, r, args[1]))),
  "neg": lambda args, r, t: (_c(-1.0, t),),
  "abs": lambda args, r, t: (_sign(args[0], t),),
  "pow": _pow,
  "min": _first_wins("min"),
  "max": _first_wins("max"),
  "sqrt": lambda args, r, t: (_f("div", t, _c(0.5, t), r),),
  "cbrt": lambda args, r, t: (_recip(_f("mul", t, _c(3.0, t), _f("mul", t, r, r)), t),),
  "exp": lambda args, r, t: (r,),
  "exp2": lambda args, r, t: (_f("mul", t, r, _c(math.log(2.0), t)),),
  "expm1": lambda args, r, t: (_f("add", t, r, _c(1.0, t)),),
  "log": lambda args, r, t: (_recip(args[0], t),),
  "log2": lambda args, r, t: (_recip(_f("mul", t, args[0], _c(math.log(2.0), t)), t),),
  "log10": lambda args, r, t: (_recip(_f("mul", t, args[0], _c(math.log(10.0), t)), t),),
  "log1p": lambda args, r, t: (_recip(_f("add", t, _c(1.0, t), args[0]), t),),
  "sin": lambda args, r, t: (_f("cos", t, args[0]),),
  "cos": lambda args, r, t: (_f("neg", t, _f("sin", t, args[0])),),
  "tan": lambda args, r, t: (_one_plus_square(r, t),),
  "arcsin": lambda args, r, t: (_recip(_f("sqrt", t, _one_minus_square(args[0], t)), t),),
  "arccos": lambda args, r, t: (_f("neg", t, _recip(_f("sqrt", t, _one_minus_square(args[0], t)), t)),),
  "arctan": lambda args, r, t: (_recip(_one_plus_square(args[0], t), t),),
  "sinh": lambda args, r, t: (_f("cosh", t, args[0]),),
  "cosh": lambda args, r, t: (_f("sinh", t, args[0]),),
  "tanh": lambda args, r, t: (_one_minus_square(r, t),),
  "arcsinh": lambda args, r, t: (_recip(_f("sqrt", t, _one_plus_square(args[0], t)), t),),
  "arccosh": lambda args, r, t: (_recip(_f("sqrt", t, _f("sub", t, _f("mul", t, args[0], args[0]), _c(1.0, t))), t),),
  "arctanh": lambda args, r, t: (_recip(_one_minus_square(args[0], t), t),),
  "arctan2": _atan2,
  "hypot": lambda args, r, t: (_f("div", t, args[0], r), _f("div", t, args[1], r)),
  "copysign": lambda args, r, t: (_f("mul", t, _sign(args[0], t), _f("copysign", t, _c(1.0, t), args[1])), None),
  "mod": lambda args, r, t: (_c(1.0, t), _f("neg", t, _f("floor", t, _f("div", t, args[0], args[1])))),
  "fmod": lambda args, r, t: (_c(1.0, t), _f("neg", t, _f("trunc", t, _f("div", t, args[0], args[1])))),
  "erf": _gauss(1.0),
  "erfc": _gauss(-1.0),
  "floor": _flat,
  "ceil": _flat,
  "trunc": _flat,
  "rint": _flat,
  "floordiv": _flat,
}

# what to call the IR operations when one of them cannot be differentiated
_SPELLING = {"lgamma": "math.lgamma", "tgamma": "math.gamma"}


def is_float(e: Expr) -> bool:
  """Return ``True`` when scalar expression ``e`` has a floating-point type."""
  return isinstance(e.type, ScalarType) and e.type.kind == "f"


# ops whose partial-derivative rule reads the result expression ``r``
_USES_RESULT = frozenset({"div", "pow", "sqrt", "cbrt", "exp", "exp2", "expm1", "tan", "tanh", "hypot"})


def partials(el, e: Prim, r: Expr | None = None):
  """Look up and evaluate the partial derivatives of primitive ``e`` (reusing ``r`` when given)."""
  rule = PARTIALS.get(e.op)
  if rule is None:
    raise el.error(UnsupportedError, f"the derivative of {_SPELLING.get(e.op, e.op)} is not implemented")
  if r is None:
    r = el.tmp(e, "r") if e.op in _USES_RESULT else e
  return rule(e.args, r, e.type)


def backprop(el, root: Expr, seed: Expr, on_var: Callable, on_load: Callable, live: Callable | None = None) -> None:
  """Push ``seed`` (the adjoint of ``root``) down to the leaves of ``root``.

  ``on_var(v, adjoint)`` is called once per variable reached, with its total
  adjoint, and returns the expression that defines the variable when the
  walk should continue through it.  ``on_load(load, adjoint)`` is called for
  every load.  ``live(e)`` may prune subtrees that cannot reach a leaf of
  interest.  Statements needed to compute adjoints go to the current block.
  """
  pending: dict[int, list[Expr]] = {}
  variables: dict[int, Var] = {}
  memo: dict[int, Expr] = {}

  def eval_once(e: Expr) -> Expr:
    """Bind ``e`` (and any inner ops that need their result) once bottom-up."""
    hit = memo.get(id(e))
    if hit is not None:
      return hit
    if isinstance(e, Prim):
      args = tuple(eval_once(a) if isinstance(a, Prim) and a.op in _USES_RESULT else a for a in e.args)
      rebuilt = e if all(x is y for x, y in zip(args, e.args)) else Prim(e.op, args, e.type)
      v = el.tmp(rebuilt, "r")
      memo[id(e)] = v
      return v
    return e

  def visit(e: Expr, adj: Expr, bound: Var | None = None) -> None:
    if bound is not None:
      memo[id(e)] = bound
    stack = [(e, adj)]
    while stack:
      e, adj = stack.pop()
      if isinstance(e, Const) or not is_float(e) or (live is not None and not live(e)):
        continue
      if isinstance(e, Var):
        pending.setdefault(e.uid, []).append(adj)
        variables[e.uid] = e
      elif isinstance(e, Load):
        on_load(e, adj)
      elif isinstance(e, Cast):
        if is_float(e.arg):
          stack.append((e.arg, cast(adj, e.arg.type)))
      elif isinstance(e, Prim):
        t = e.type
        if e.op == "select":
          c, a, b = e.args
          zero = Const(0.0, t)
          adj = el.tmp(adj, "adj")
          stack.append((a, ops.select(c, adj, zero)))
          stack.append((b, ops.select(c, zero, adj)))
          continue
        r = eval_once(e) if e.op in _USES_RESULT else e
        parts = partials(el, e, r)
        if sum(1 for p in parts if p is not None) > 1:
          adj = el.tmp(adj, "adj")
        for arg, p in zip(e.args, parts):
          if p is None or not is_float(arg) or isinstance(arg, Const):
            continue
          if isinstance(p, Const) and p.value == 1.0:
            stack.append((arg, cast(adj, arg.type)))
          else:
            stack.append((arg, cast(_f("mul", t, adj, p), arg.type)))
      # runtime calls (the random number generator) are constants

  visit(root, seed)
  while pending:
    uid = max(pending)  # definitions only mention older variables
    parts = pending.pop(uid)
    v = variables[uid]
    total = parts[0]
    for p in parts[1:]:
      total = _f("add", v.type, total, p)
    total = el.tmp(total, "adj")
    definition = on_var(v, total)
    if definition is not None:
      visit(definition, total, bound=v)


def pushforward(
  el,
  root: Expr,
  tangent_var: Callable[[Var], Expr | None],
  tangent_load: Callable[[Load], Expr | None],
  bound: Var | None = None,
) -> Expr | None:
  """Forward-mode pushforward of scalar expression tree ``root``.

  Reuses :data:`PARTIALS` so forward and reverse scalar rules stay in one place.
  Returns ``None`` when ``root`` has an exact structural zero tangent.
  """
  memo: dict[int, Expr | None] = {}
  val_memo: dict[int, Expr] = {}
  if bound is not None:
    val_memo[id(root)] = bound

  def eval_once(e: Expr) -> Expr:
    hit = val_memo.get(id(e))
    if hit is not None:
      return hit
    if isinstance(e, Prim):
      args = tuple(eval_once(a) if isinstance(a, Prim) and a.op in _USES_RESULT else a for a in e.args)
      rebuilt = e if all(x is y for x, y in zip(args, e.args)) else Prim(e.op, args, e.type)
      v = el.tmp(rebuilt, "r")
      val_memo[id(e)] = v
      return v
    return e

  def rec(e: Expr) -> Expr | None:
    if isinstance(e, Const) or not is_float(e):
      return None
    key = id(e)
    if key in memo:
      return memo[key]
    res: Expr | None = None
    if isinstance(e, Var):
      res = tangent_var(e)
    elif isinstance(e, Load):
      res = tangent_load(e)
    elif isinstance(e, Cast):
      if is_float(e.arg):
        da = rec(e.arg)
        if da is not None:
          res = cast(da, e.type)
    elif isinstance(e, Prim):
      t = e.type
      if e.op == "select":
        c, a, b = e.args
        da = rec(a) if is_float(a) else None
        db = rec(b) if is_float(b) else None
        if da is not None or db is not None:
          zero = _c(0.0, t)
          res = ops.select(c, cast(da, t) if da is not None else zero, cast(db, t) if db is not None else zero)
      else:
        tangs = [rec(a) if is_float(a) else None for a in e.args]
        if any(dt is not None for dt in tangs):
          r = eval_once(e) if e.op in _USES_RESULT else e
          parts = partials(el, e, r)
          terms: list[Expr] = []
          for p, dt in zip(parts, tangs):
            if p is None or dt is None:
              continue
            dtc = cast(dt, t)
            if isinstance(p, Const) and p.value == 1.0:
              terms.append(dtc)
            elif isinstance(p, Const) and p.value == -1.0:
              terms.append(_f("neg", t, dtc))
            else:
              terms.append(_f("mul", t, p, dtc))
          if terms:
            acc = terms[0]
            for tm in terms[1:]:
              acc = _f("add", t, acc, tm)
            res = el.tmp(acc, "dt") if len(terms) > 1 else acc
    memo[key] = res
    return res

  return rec(root)


__all__ = ["PARTIALS", "backprop", "is_float", "partials", "pushforward"]
