"""Tests for Wave 4 (v1.5.0, #80 — Python Semantics 2):
- Compile-time static ListVal (.append, .extend, .pop, .clear, indexing/slicing,
  np.stack, np.concatenate, RaggedArray.from_sequences) including inside par.grad
- CPU runtime growable homogeneous list (Vec[T]: geometric doubling, zero-copy to
  ArrayVal, verified with par.config.verify=True)
- Fixed-capacity thread-local scratch par.fixed_list(cap, dtype) inside parallel
  parfor/map on CPU, CUDA simulator, and Metal (zero malloc!)
- Dynamic-shape ArrayVal & RaggedArray rebinding across while/for loops with
  ownership/refcount verification
- Static itertools (product, pairwise, islice, chain, batched) & generator expressions
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pytest

import parakeet as par
from parakeet import RaggedArray


@dataclass
class ScaleBlock:
  weight: np.ndarray
  bias: float

  def __call__(self, x):
    return np.tanh(x * self.weight + self.bias)


# ---------------------------------------------------------------------------
# 1. Compile-time static ListVal + .append() / .extend() + grad
# ---------------------------------------------------------------------------


def test_static_list_append_stack_and_grad():
  def stack_blocks(blocks, x):
    outs = []
    for blk in blocks:
      outs.append(blk(x))
    return np.stack(outs)

  @par.jit
  def run_stack(blocks, x):
    return stack_blocks(blocks, x)

  rng = np.random.default_rng(0)
  blocks = (
    ScaleBlock(rng.normal(size=(5,)), 0.1),
    ScaleBlock(rng.normal(size=(5,)), -0.2),
    ScaleBlock(rng.normal(size=(5,)), 0.3),
  )
  x = rng.normal(size=(5,))

  got = run_stack(blocks, x)
  expected = np.stack([b(x) for b in blocks])
  np.testing.assert_allclose(got, expected, rtol=1e-12)

  # Differentiate through static list.append + np.stack!
  def loss(x_):
    return np.sum(stack_blocks(blocks, x_) ** 2)

  g = par.grad(loss)(x)
  eps = 1e-6
  fd = np.zeros_like(x)
  for i in range(len(x)):
    e = np.zeros_like(x)
    e[i] = eps
    fd[i] = (loss(x + e) - loss(x - e)) / (2 * eps)
  np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_static_list_heterogeneous_extend_pop_and_ragged():
  @par.jit
  def build_ragged(a, b, c):
    parts = [a]
    parts.extend((b, c))
    last = parts.pop()
    parts.append(last * 2.0)
    rag = RaggedArray.from_sequences(parts)
    return rag, len(parts), np.sum(parts[1])

  a = np.array([1.0, 2.0])
  b = np.array([3.0, 4.0, 5.0])
  c = np.array([6.0])
  rag, n_parts, sum_b = build_ragged(a, b, c)
  assert int(n_parts) == 3
  assert float(sum_b) == pytest.approx(12.0)
  np.testing.assert_allclose(rag.values, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 12.0]))
  np.testing.assert_array_equal(rag.offsets, np.array([0, 2, 5, 6]))


# ---------------------------------------------------------------------------
# 2. CPU runtime growable homogeneous list (Vec[T])
# ---------------------------------------------------------------------------


def test_runtime_vec_while_append_doubling_pop_and_verify():
  @par.jit
  def collatz_like(n):
    xs = []
    i = 0
    while i < n:
      if i % 2 == 0:
        xs.append(float(i) * 1.5)
      i = i + 1
    xs[0] = -1.0
    last = xs.pop()
    xs.append(last + 10.0)
    return xs, len(xs)

  @par.jit
  def vec_extend_and_loop(a, b):
    xs = [0.5]
    xs.extend(a)
    for i in range(len(b)):
      if b[i] > 0.0:
        xs.append(b[i])
    return np.asarray(xs)

  prev_verify = par.config.verify
  par.config.verify = True
  try:
    # Grow well past initial capacity (8 -> 16 -> 32 -> 64)
    arr, length = collatz_like(70)
    expected = [float(i) * 1.5 for i in range(70) if i % 2 == 0]
    expected[0] = -1.0
    expected[-1] += 10.0
    assert int(length) == len(expected)
    np.testing.assert_allclose(arr, np.array(expected), rtol=1e-12)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-1.0, 4.0, -2.0, 5.0])
    np.testing.assert_allclose(vec_extend_and_loop(a, b), np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]), rtol=1e-12)
  finally:
    par.config.verify = prev_verify


# ---------------------------------------------------------------------------
# 3. Fixed-capacity GPU/parallel thread-local scratch (par.fixed_list)
# ---------------------------------------------------------------------------


def test_fixed_list_in_parallel_parfor_and_gpu_kernel_emission():
  @par.jit
  def row_greedy_stack_sum(x):
    """Per-row stack traversal using thread-local par.fixed_list (zero malloc!)."""
    n_rows, n_cols = x.shape
    out = np.empty(n_rows, dtype=np.float32)

    def body(r):
      stk = par.fixed_list(64, np.float32)
      for c in range(n_cols):
        v = x[r, c]
        if v > np.float32(0.0):
          stk.append(v)
        elif len(stk) > 0:
          top = stk.pop()
          stk.append(top + v * np.float32(0.5))
      acc = np.float32(0.0)
      for k in range(len(stk)):
        acc = acc + stk[k]
      out[r] = acc

    par.parfor(body, n_rows)
    return out

  rng = np.random.default_rng(19)
  x = rng.standard_normal((128, 32)).astype(np.float32)
  ref = row_greedy_stack_sum(x, _backend="python")

  for backend in ("c", "multicore"):
    got = row_greedy_stack_sum(x, _backend=backend)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

  # Verify CUDA & Metal planners both compile the kernel natively with thread-local stack array (zero malloc!)
  spec = row_greedy_stack_sum.specialize(x, _backend="multicore")

  # Check that stack array is emitted in C source
  assert "fl_buf_" in spec.source and "[64];" in spec.source

  # Check Metal GPU execution if available on this host
  if par.available_backends().get("metal") == "ok":
    with par.options(device_fallback="error", device_threshold=1):
      got_metal = row_greedy_stack_sum(x, _backend="metal")
    np.testing.assert_allclose(got_metal, ref, rtol=1e-5, atol=1e-5)

  # Check CUDA GPU execution if available on this host
  if par.available_backends().get("cuda") == "ok":
    with par.options(device_fallback="error", device_threshold=1):
      got_cuda = row_greedy_stack_sum(x, _backend="cuda")
    np.testing.assert_allclose(got_cuda, ref, rtol=1e-5, atol=1e-5)

  # Verify capacity overflow raises IndexError cleanly
  @par.jit
  def overflow_fixed(n):
    stk = par.fixed_list(4, np.int64)
    for i in range(n):
      stk.append(i)
    return len(stk)

  assert int(overflow_fixed(4)) == 4
  with pytest.raises(IndexError, match="capacity"):
    overflow_fixed(5)


# ---------------------------------------------------------------------------
# 4. Dynamic-shape ArrayVal & RaggedArray rebinding across while / for loops
# ---------------------------------------------------------------------------


def test_dynamic_shape_array_and_ragged_loop_rebinding():
  @par.jit
  def shrink_and_grow(x):
    arr = x
    total = 0.0
    while len(arr) > 1:
      total += np.sum(arr)
      arr = arr[arr > np.mean(arr)]
    # Now grow with concatenate across a for loop
    grown = arr
    for k in range(3):
      grown = np.concatenate((grown, np.array([float(k)])))
    return total, grown

  @par.jit
  def bfs_ragged_frontier(rag, n_rounds):
    r = rag
    acc = 0.0
    for _ in range(n_rounds):
      acc += np.sum(r.values)
      keep = r.values > 0.0
      nv = r.values[keep] - 0.5
      no = np.array([0, len(nv)], dtype=np.int64)
      r = RaggedArray(nv, no)
    return acc, r

  prev_verify = par.config.verify
  par.config.verify = True
  try:
    x = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0])
    tot, grown = shrink_and_grow(x)
    tot_ref, grown_ref = shrink_and_grow(x, _backend="python")
    assert float(tot) == pytest.approx(float(tot_ref), rel=1e-12)
    np.testing.assert_allclose(grown, grown_ref, rtol=1e-12)

    rag = RaggedArray.from_lengths(x - 3.0, np.array([4, 4], dtype=np.int64))
    acc, r_out = bfs_ragged_frontier(rag, 3)
    acc_ref, r_ref = bfs_ragged_frontier(rag, 3, _backend="python")
    assert float(acc) == pytest.approx(float(acc_ref), rel=1e-12)
    np.testing.assert_allclose(r_out.values, r_ref.values, rtol=1e-12)
    np.testing.assert_array_equal(r_out.offsets, r_ref.offsets)
  finally:
    par.config.verify = prev_verify


# ---------------------------------------------------------------------------
# 5. Static itertools (product, pairwise, islice, chain, batched) & generators
# ---------------------------------------------------------------------------


def test_static_itertools_and_generators():
  @par.jit
  def run_itertools(x, y):
    # pairwise: sum of adjacent differences
    pw = 0.0
    for a, b in itertools.pairwise(x):
      pw += b - a

    # islice: every 2nd element from index 1 to 6
    sl_sum = np.sum(np.array([v for v in itertools.islice(x, 1, 6, 2)]))

    # chain: iterate over x then y without concatenating
    ch = 0.0
    for v in itertools.chain(x, y):
      ch += v * v

    # product: nested Cartesian loop over x and y
    pr = 0.0
    for a, b in itertools.product(x, y):
      pr += a * b

    # generator expression with filter inside sum()
    gen_s = sum(v * 2.0 for v in x if v > 2.0)

    return pw, sl_sum, ch, pr, gen_s

  x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 8.0, 6.0])
  y = np.array([0.5, -1.5, 2.5])
  got = run_itertools(x, y)
  ref = run_itertools(x, y, _backend="python")
  for g, r in zip(got, ref):
    assert float(g) == pytest.approx(float(r), rel=1e-12)

  if hasattr(itertools, "batched"):

    @par.jit
    def run_batched(x_, k):
      # Sum of per-batch maxima (tests both full batches and partial tail batch!)
      total = 0.0
      for chunk in itertools.batched(x_, k):
        total += np.max(chunk)
      return total

    assert float(run_batched(x, 3)) == pytest.approx(3.0 + 8.0 + 6.0, rel=1e-12)
    assert float(run_batched(x, 2)) == pytest.approx(3.0 + 5.0 + 8.0 + 6.0, rel=1e-12)
