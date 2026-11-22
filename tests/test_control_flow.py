"""Control flow, variable typing and function-call semantics."""

import numpy as np
import pytest

import parakeet
from parakeet import jit
from parakeet.testing import check


def test_loop_widens_int_to_float():
  def f(n):
    x = 0
    for i in range(n):
      x = x + 0.5
    return x

  check(f, 4)
  # the variable has one static type: a loop that never runs still yields a float
  assert jit(f)(0) == 0.0 and type(jit(f)(0)) is float


def test_loop_keeps_int():
  def f(n):
    x = 0
    for i in range(n):
      x += i
    return x

  assert type(check(f, 10)) is int


def test_fibonacci_tuple_swap():
  def fib(n):
    a, b = 0, 1
    for _ in range(n):
      a, b = b, a + b
    return a

  check(fib, 30)


def test_while_break_continue():
  def f(n):
    total = 0
    i = 0
    while True:
      i += 1
      if i > n:
        break
      if i % 3 == 0:
        continue
      total += i
    return total, i

  check(f, 20)


def test_early_return_in_loop():
  def find(x, target):
    for i in range(len(x)):
      if x[i] == target:
        return i
    return -1

  x = np.array([5, 3, 8, 1])
  check(find, x, 8)
  check(find, x, 7)


def test_return_in_nested_loops():
  def find2d(x, target):
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        if x[i, j] == target:
          return i * 10 + j
    return -1

  x = np.arange(12).reshape(3, 4)
  check(find2d, x, 7)
  check(find2d, x, 99)


def test_return_types_join():
  def f(x):
    if x > 0:
      return 1
    return 2.5

  # a function has one return type: the union of all its return statements
  for x in (3, -3):
    r = jit(f)(x)
    assert type(r) is float and r == f(x)


def test_return_none_and_tuple():
  def nothing(x):
    x[0] = 1

  def pair(x):
    return x.shape[0], x.sum(), (1, 2.0)

  check(nothing, np.zeros(3))
  check(pair, np.arange(4.0))


def test_if_merges_arrays():
  def f(x, flag):
    if flag:
      y = x * 2
    else:
      y = x[::-1]
    return y + 1

  x = np.arange(5.0)
  check(f, x, True)
  check(f, x, False)


def test_if_merges_dtypes():
  def f(x, flag):
    if flag:
      y = x
    else:
      y = x * 0.5
    return y

  x = np.arange(5)
  for flag in (True, False):
    r = jit(f)(x, flag)
    assert r.dtype == np.float64 and np.array_equal(r, f(x, flag))


def test_conditional_expression_and_short_circuit():
  def f(x, i):
    ok = i < len(x) and x[i] > 0
    y = x[i] if ok else np.float64(-1.0)
    return bool(ok), y, (i or 7), (i and 9)

  x = np.array([1.0, -2.0, 3.0])
  for i in (0, 1, 2, 3, 10):
    check(f, x, i)


def test_chained_comparison():
  def f(a, b, c):
    return a < b < c, a < b > c, a == b == c

  check(f, 1, 2, 3)
  check(f, 1, 5, 3)
  check(f, 2, 2, 2)


def test_nested_functions_and_closures():
  def f(x, k):
    def scale(v, factor=2):
      return v * factor + k

    g = lambda v: scale(v, 3) - 1  # noqa: E731
    return scale(x), g(x)

  check(f, np.arange(4.0), 10)


def test_closure_sees_later_assignment():
  def f(x):
    def g():
      return y + 1

    y = x * 2
    return g()

  check(f, 3)


def test_keyword_and_default_arguments():
  def inner(a, b=2, *rest, scale=1):
    return (a + b + len(rest)) * scale

  def f(x):
    return inner(x), inner(x, 5), inner(x, 5, 6, 7), inner(b=1, a=x, scale=3)

  check(f, 2)


def test_python_defaults_at_top_level():
  @jit
  def f(x, alpha=0.5, *, beta=2):
    return x * alpha + beta

  assert f(2.0) == 3.0
  assert f(2.0, 1.0) == 4.0
  assert f(2.0, beta=0) == 1.0
  assert f(x=4.0, alpha=0.25, beta=1) == 2.0


def test_varargs_at_top_level():
  @jit
  def f(first, *rest):
    return first + len(rest), rest

  assert f(1, 2.0, 3) == (3, (2.0, 3))


def test_loop_over_array_zip_enumerate():
  def f(x, y):
    total = 0.0
    for i, (a, b) in enumerate(zip(x, y)):
      total += i * a * b
    for row in x.reshape(2, -1):
      total += row[0]
    return total

  check(f, np.arange(6.0), np.arange(6.0) + 1)


def test_loop_over_tuple_is_unrolled():
  def f(x):
    total = 0.0
    for item in (1, 2.5, x):
      total = total + np.sum(item)
    return total

  check(f, np.arange(4.0))


def test_loop_variable_after_loop():
  def f(n):
    last = -1
    for i in range(n):
      last = i * 2
    return i, last

  check(f, 5)


def test_range_variants():
  def f(a, b, c):
    out = 0
    for i in range(a, b, c):
      out = out * 3 + i
    return out, len(range(a, b, c))

  check(f, 0, 10, 3)
  check(f, 10, 0, -3)
  check(f, 5, 5, 1)
  check(f, 5, 1, 2)


def test_array_carried_through_loop():
  def f(x, n):
    y = x
    for _ in range(n):
      y = y * 0.5 + 1
    return y

  check(f, np.arange(5.0), 6)
  check(f, np.arange(5.0), 0)


def test_shrinking_view_in_while():
  def f(x):
    total = 0.0
    while len(x) > 0:
      total += x[0]
      x = x[2:]
    return total

  check(f, np.arange(9.0))


def test_array_built_in_loop_then_used():
  def f(n):
    acc = np.zeros(3)
    for i in range(n):
      tmp = np.ones(3) * i
      acc = acc + tmp
    return acc

  check(f, 5)


def test_static_argument_requested_on_demand():
  @jit
  def total(x, axis):
    return np.sum(x, axis=axis)

  x = np.arange(12.0).reshape(3, 4)
  assert np.array_equal(total(x, 0), x.sum(0))
  assert np.array_equal(total(x, 1), x.sum(1))
  assert total(x, None) == x.sum()


def test_tuple_indexing_with_runtime_index():
  def f(t, i):
    return t[i]

  check(f, (1.5, 2.5, 3.5), 1)
  check(f, (1.5, 2.5, 3.5), -1)


def test_globals_are_read_at_call_time():
  global _SCALE, _TABLE
  _SCALE = 2.0
  _TABLE = np.arange(3.0)

  @jit
  def f(x):
    return x * _SCALE + _TABLE

  x = np.ones(3)
  assert np.array_equal(f(x), x * 2 + np.arange(3.0))
  _SCALE = 3.0
  _TABLE = np.arange(3.0) * 10
  assert np.array_equal(f(x), x * 3 + np.arange(3.0) * 10)
  _SCALE = 4  # type change triggers a new specialization
  assert np.array_equal(f(x), x * 4 + np.arange(3.0) * 10)


_SCALE = 1.0
_TABLE = np.zeros(3)


def test_closure_cells_are_read_at_call_time():
  def make(k):
    @jit
    def f(x):
      return x + k

    return f

  assert make(1)(1) == 2
  assert make(10)(1) == 11
  assert make(0.5)(1) == 1.5


def test_function_arguments():
  def apply(f, x):
    return f(x)

  check(apply, np.sqrt, np.arange(4.0))
  check(apply, lambda v: v * 3, 2)
  check(apply, len, np.arange(4.0))


def test_calling_other_jit_functions():
  @jit
  def helper(x, p=2):
    return x**p

  @jit
  def f(x):
    return helper(x) + helper(x, 3)

  assert f(2) == 12


def test_recursion_is_reported():
  @jit
  def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)

  with pytest.raises(parakeet.UnsupportedError, match="recursive"):
    fact(5)


def test_assert_and_raise():
  @jit
  def f(x):
    assert x >= 0, "x must be non-negative"
    if x > 100:
      raise ValueError("too big")
    return x

  assert f(3) == 3
  with pytest.raises(AssertionError, match="non-negative"):
    f(-1)
  with pytest.raises(ValueError, match="too big"):
    f(101)


def test_python_division_by_zero():
  @jit
  def f(a, b):
    return a // b, a % b, a / b

  assert f(7, 2) == (3, 1, 3.5)
  with pytest.raises(ZeroDivisionError):
    f(1, 0)
  with pytest.raises(ZeroDivisionError):
    f(1.0, 0.0)


def test_integer_semantics_match_python():
  def f(a, b):
    return a // b, a % b, a**2, -a, abs(a), a << 2, a >> 1, a & b, a | b, a ^ b, ~a, divmod(a, b)

  for a in (-7, 7, 0, 13):
    for b in (-3, 3, 5):
      check(f, a, b)


def test_float_semantics_match_python():
  def f(a, b):
    return a // b, a % b, a**b, -a, abs(a), round(a), int(a), a > b, min(a, b), max(a, b, 0.5)

  for a in (7.5, 7.25, 0.5, 2.5):
    for b in (-3.0, 3.5):
      check(f, a, b)


def test_walrus_and_delete():
  def f(x):
    if (n := len(x)) > 2:
      return n
    return -n

  check(f, np.arange(5))
  check(f, np.arange(1))
