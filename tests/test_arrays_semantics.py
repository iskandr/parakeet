"""Array semantics that must match NumPy exactly: views, aliasing, mutation,
indexing, broadcasting and dtype promotion."""

import numpy as np
import pytest

import parakeet
from parakeet import jit
from parakeet.testing import check

DTYPES = ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"]


def test_alias_sees_mutation():
  def f(x):
    y = x
    y[0] = 99
    return x

  check(f, np.arange(4.0))


def test_view_sees_mutation():
  def f(x):
    v = x[1:3]
    v[0] = -1
    w = x.T
    w[0] = 7
    return x.sum()

  check(f, np.arange(6.0).reshape(2, 3))


def test_expression_evaluated_before_later_mutation():
  def f(x):
    y = x * 2
    x[0] = 100
    return y

  check(f, np.arange(4.0))


def test_expression_evaluated_before_mutation_in_loop():
  def f(x):
    y = x + 1
    for i in range(len(x)):
      x[i] = 0
    return y, x

  check(f, np.arange(4.0))


def test_overlapping_assignment():
  def shift_right(x):
    x[1:] = x[:-1]
    return x

  def shift_left(x):
    x[:-1] = x[1:]
    return x

  def reverse_in_place(x):
    x[:] = x[::-1]
    return x

  def smooth(x):
    x[1:-1] = (x[:-2] + x[2:]) / 2
    return x

  for f in (shift_right, shift_left, reverse_in_place, smooth):
    check(f, np.arange(8.0))


def test_inplace_operators_mutate_the_argument():
  def f(x, y):
    x += 1
    x *= y
    x[1:] -= 0.5
    x[0] /= 4
    return x

  check(f, np.arange(4.0), np.arange(4.0))


def test_rebinding_does_not_mutate_the_argument():
  def f(x):
    x = x + 1
    x[0] = 50
    return x

  check(f, np.arange(4.0))


def test_inplace_cast_error():
  @jit
  def f(x):
    x += 0.5
    return x

  with pytest.raises(TypeError, match="cannot cast"):
    f(np.arange(3))


def test_lazy_array_mutated_then_returned():
  def f(x):
    y = x * 2
    y[1] = -1
    z = y
    z[2] = -2
    return y, z

  check(f, np.arange(4.0))


def test_lazy_array_mutated_in_branch():
  def f(x, flag):
    y = x * 2
    if flag:
      y[0] = -1
    return y

  check(f, np.arange(4.0), True)
  check(f, np.arange(4.0), False)


def test_expensive_expression_used_twice():
  def f(x):
    y = np.cumsum(x) * 2
    return y.sum() + y.max() + y[0]

  check(f, np.arange(5.0))


def test_basic_indexing_forms():
  def f(x):
    return (
            x[1], x[-1], x[1, 2], x[-1, -2], x[1:], x[:-1], x[::2], x[::-1], x[1:3, ::-2],
            x[..., 0], x[None, 1], x[:, None, 2], x[1][2], x[1:][0], x[:, 1:][:, ::2], x[-3:-1, -2:],
            x[5:], x[2:1], x[10:20, 1],
        )  # fmt: skip

  check(f, np.arange(20.0).reshape(4, 5))


def test_slices_with_runtime_bounds():
  def f(x, a, b, c):
    return x[a:b:c], x[a:], x[:b], x[::c]

  x = np.arange(10)
  for a, b, c in [(1, 8, 2), (-3, 20, 1), (8, 1, -2), (0, 0, 1), (-1, -8, -3), (20, -20, -1)]:
    check(f, x, a, b, c)


def test_zero_step_slice():
  @jit
  def f(x, s):
    return x[::s]

  with pytest.raises(ValueError, match="step"):
    f(np.arange(4), 0)


def test_fancy_indexing_get():
  def f(x, idx, mask):
    return x[idx], x[mask], x[idx, 1], x[:, idx[:2]], x[x[:, 0] > 4], x[[0, -1]], x[idx][:, ::2]

  x = np.arange(20.0).reshape(5, 4)
  check(f, x, np.array([3, 0, -1, 3]), np.array([True, False, True, False, True]))


def test_fancy_indexing_pairs_and_full_mask():
  def f(x, r, c):
    return x[r, c], x[x > 7], x[x > 100]

  check(f, np.arange(12.0).reshape(3, 4), np.array([0, 2, 1]), np.array([3, 0, -1]))


def test_fancy_indexing_set():
  def f(x, idx, mask):
    x[idx] = -1
    x[mask, 0] = 5
    x[x > 15] = 0
    x[[1, 2], 3] = np.array([7.0, 8.0])
    return x

  x = np.arange(20.0).reshape(5, 4)
  check(f, x, np.array([0, 3]), np.array([False, True, False, False, True]))


def test_out_of_bounds_errors():
  @jit
  def get(x, i):
    return x[i]

  @jit
  def get2(x, i, j):
    return x[i, j]

  @jit
  def put(x, i):
    x[i] = 1

  @jit
  def take(x, idx):
    return x[idx]

  x = np.arange(6.0).reshape(2, 3)
  with pytest.raises(IndexError, match="index 2 is out of bounds for axis 0 with size 2"):
    get(x, 2)
  with pytest.raises(IndexError, match="index -4 is out of bounds for axis 1 with size 3"):
    get2(x, 0, -4)
  with pytest.raises(IndexError):
    put(np.zeros(3), 3)
  with pytest.raises(IndexError):
    take(np.zeros(3), np.array([0, 5]))
  assert get(x, -1)[0] == 3.0


def test_bounds_checks_can_be_disabled():
  @jit(bounds_check=False)
  def get(x, i):
    return x[i]

  assert get(np.arange(4.0), 2) == 2.0
  assert "check" not in get.specialize(np.arange(4.0), 2).ir


def test_too_many_indices():
  @jit
  def f(x):
    return x[0, 0, 0]

  with pytest.raises(parakeet.CompileError, match="too many indices"):
    f(np.zeros((2, 2)))


def test_broadcasting():
  def f(a, b, c):
    return a + b, a * c, b - c, (a + b) * c, a[:, None, :] + a[None, :, :]

  a = np.arange(6.0).reshape(2, 3)
  check(f, a, np.arange(3.0), np.arange(2.0).reshape(2, 1))


def test_broadcasting_runtime_length_one():
  def f(a, n):
    b = np.ones(n)
    return a + b

  check(f, np.arange(4.0), 1)
  check(f, np.arange(4.0), 4)


def test_broadcast_mismatch():
  @jit
  def f(a, b):
    return a + b

  with pytest.raises(ValueError, match="broadcast"):
    f(np.zeros(3), np.zeros(4))
  with pytest.raises(ValueError, match="broadcast"):
    f(np.zeros((2, 3)), np.zeros((3, 2)))


def test_assignment_broadcast_mismatch():
  @jit
  def f(a, b):
    a[:] = b
    return a

  assert np.array_equal(f(np.zeros((2, 3)), np.arange(3.0)), np.tile(np.arange(3.0), (2, 1)))
  with pytest.raises(ValueError, match="broadcast"):
    f(np.zeros(3), np.zeros(4))


@pytest.mark.parametrize("dt", DTYPES)
def test_dtype_roundtrip_and_arithmetic(dt):
  def f(x):
    return x, x[::-1].copy(), x + x, x * 2, x[0], x.sum(), x.max(), x == x[0]

  x = np.array([1, 0, 3, 2]).astype(dt)
  check(f, x)


@pytest.mark.parametrize("a", ["bool", "int8", "uint8", "int32", "uint32", "int64", "uint64", "float32", "float64"])
@pytest.mark.parametrize("b", ["bool", "int8", "uint16", "int64", "uint64", "float32", "float64"])
def test_binary_promotion_matches_numpy(a, b):
  def f(x, y):
    return x + y, x * y, x > y, x / (y + 1), np.maximum(x, y)

  x = np.array([1, 0, 3]).astype(a)
  y = np.array([2, 1, 0]).astype(b)
  check(f, x, y)
  check(f, x, y[0])
  check(f, x[0], y[0])


@pytest.mark.parametrize("dt", ["int8", "uint8", "int32", "int64", "float32", "float64"])
def test_python_scalars_are_weak(dt):
  def f(x):
    return x + 1, x * 2.5, x - True, 2 ** x[:2], x / 2, x // 2, x % 2, x > 1.5

  check(f, np.array([1, 2, 3]).astype(dt))


def test_scalar_argument_types():
  def f(a, b):
    return a + b, a * 2, a / b

  for a in (3, 2.5, True, np.int8(3), np.uint16(7), np.float32(1.5), np.float64(2.5), np.bool_(True), np.int64(-4)):
    for b in (2, 0.5, np.float32(2), np.int32(5)):
      check(f, a, b)


def test_integer_division_numpy_semantics():
  def f(x, y):
    return x // y, x % y, np.fmod(x, y), x**2

  x = np.array([-7, 7, 0, 13, -1], dtype=np.int32)
  y = np.array([3, -3, 5, 0, 2], dtype=np.int32)
  check(f, x, y)


def test_integer_overflow_wraps():
  def f(x):
    return x + x, x * x, -x

  check(f, np.array([127, -128, 100], dtype=np.int8))
  check(f, np.array([255, 0, 200], dtype=np.uint8))


def test_casts():
  def f(x):
    return x.astype(np.int32), x.astype(np.uint8), x.astype(bool), x.astype(np.float32), np.int64(x[1]), float(x[2])

  check(f, np.array([0.0, 1.9, -2.5, 300.7]))


def test_strided_and_reversed_inputs():
  def f(x):
    return x * 2, x.sum(), x[1:]

  base = np.arange(24.0).reshape(4, 6)
  for view in (base[::2], base[:, ::-1], base.T, base[::-1, ::2], base[1], base[:, 2], np.asfortranarray(base)):
    check(f, view)


def test_mutating_a_strided_input():
  def f(x):
    x[0] = -1
    x += 1
    return None

  base = np.arange(24.0).reshape(4, 6)
  for make in (lambda b: b[::2], lambda b: b[:, ::-1], lambda b: b.T, lambda b: b[1:, 2]):
    mine, ref = base.copy(), base.copy()
    jit(f)(make(mine))
    f(make(ref))
    assert np.array_equal(mine, ref)


def test_returned_input_is_the_same_object():
  @jit
  def ident(x):
    return x

  @jit
  def tail(x):
    return x[1:]

  x = np.arange(5.0)
  assert ident(x) is x
  t = tail(x)
  t[0] = 42  # a view of the caller's memory, as in NumPy
  assert x[1] == 42
  del x
  assert t.sum() == 42 + 2 + 3 + 4  # the view keeps the memory alive


def test_read_only_input():
  @jit
  def read(x):
    return x.sum()

  @jit
  def write(x):
    x[0] = 1

  x = np.arange(4.0)
  x.flags.writeable = False
  assert read(x) == 6.0
  with pytest.raises(ValueError, match="read-only"):
    write(x)


def test_zero_dimensional_and_empty_arrays():
  def f(x, e):
    return x + 1, x.sum(), e + 1, e.sum(), e.shape, len(e), e[1:], np.concatenate((e, e))

  check(f, np.array(3.0), np.zeros(0))
  check(f, np.array(2), np.zeros((0, 3)))


def test_empty_reduction_without_identity():
  @jit
  def f(x):
    return x.max()

  with pytest.raises(ValueError, match="zero-size"):
    f(np.zeros(0))


def test_lists_tuples_and_other_arraylikes_as_arguments():
  @jit
  def f(x):
    return x * 2

  assert np.array_equal(f([1, 2, 3]), np.array([2, 4, 6]))
  assert np.array_equal(f([[1.0, 2.0], [3.0, 4.0]]), np.array([[2.0, 4.0], [6.0, 8.0]]))
  assert f((1, 2.5)) == (1, 2.5, 1, 2.5)
  assert np.array_equal(f(memoryview(b"")) if False else f(np.float16([1, 2])), np.float32([2, 4]))

  class Wrapper:
    def __array__(self, dtype=None, copy=None):
      return np.arange(3.0)

  assert np.array_equal(f(Wrapper()), np.arange(3.0) * 2)


def test_unsupported_argument_types():
  @jit
  def f(x):
    return x

  assert f({"a": 1}) == {"a": 1}
  assert np.array_equal(f(np.array([1 + 2j])), np.array([1 + 2j]))
  for bad in ({0: 1}, {"a": object()}, np.array(["a", "b"]), object(), 2**70):
    with pytest.raises(TypeError):
      f(bad)


def test_large_arrays_use_the_parallel_path():
  def f(x, y):
    z = np.sqrt(x * x + y * y)
    return z, z.sum(), z.max(), np.argmax(z), (z > 0.5).sum(), z.mean()

  rng = np.random.default_rng(0)
  x, y = rng.random(200_000), rng.random(200_000)
  check(f, x, y, rtol=1e-9)
  check(f, x.reshape(400, 500), y.reshape(400, 500), rtol=1e-9)
  check(f, x.astype(np.float32), y.astype(np.float32))


def test_mixed_scalar_and_array_indices_follow_numpy_dimension_order():
  def f(x, idx, i):
    return x[i, idx], x[idx, i], x[i, :, idx], x[:, i, idx], x[idx, :, i], x[:, idx, i], x[i, idx, :], x[:, idx]

  x = np.arange(60.0).reshape(3, 4, 5)
  check(f, x, np.array([2, 0, 2, 1]), 1)


def test_index_arrays_must_be_integers():
  @jit
  def f(x, idx):
    return x[idx]

  with pytest.raises(TypeError, match="integer or boolean"):
    f(np.arange(4.0), np.array([0.0, 1.0]))
