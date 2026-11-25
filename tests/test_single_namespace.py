"""Tests for unified top-level ``parakeet`` operator exports and ``parakeet.dsl`` compatibility."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

import parakeet as pk
from parakeet import dsl


class _Pair(NamedTuple):
  val: np.float32
  idx: np.int64


def _pick_min(a, b):
  return b if b.val < a.val else a


def test_top_level_operators_standalone_and_jitted():
  a = np.array([[1.0, 2.0, 3.0], [4.0, 0.5, 2.0]], dtype=np.float32)
  b = np.array([[2.0, 1.0], [3.0, 0.25], [0.5, 4.0]], dtype=np.float32)

  # Standalone auto-JIT calls from Python
  im = pk.imap(lambda i, j: i * 10 + j, (3, 4))
  np.testing.assert_array_equal(im, np.arange(12).reshape(3, 4) // 4 * 10 + np.arange(12).reshape(3, 4) % 4)

  folded = pk.fold(_Pair, _pick_min, a, axis=1, indexed=True, emit=lambda p: p.idx)
  np.testing.assert_array_equal(folded, np.argmin(a, axis=1))

  # Inside @pk.jit using only top-level parakeet + numpy
  @pk.jit
  def tropical(x, y):
    m, k = x.shape
    _, n = y.shape
    return pk.imap(
      lambda i, j: pk.ireduce(lambda r: x[i, r] + y[r, j], k, combine=pk.minimum),
      (m, n),
    )

  got = tropical(a, b)
  ref = np.min(a[:, :, None] + b[None, :, :], axis=1)
  np.testing.assert_allclose(got, ref, rtol=1e-6)

  # Top-level show_rewrite works on plain and @pk.jit functions
  rep = pk.show_rewrite(tropical, a, b)
  assert isinstance(rep, pk.RewriteReport)
  assert rep.fn_name == "tropical"


def test_dsl_compatibility_aliases():
  assert dsl.kernel is pk.jit
  assert dsl.fold is pk.fold
  assert dsl.show_rewrite is pk.show_rewrite
  assert dsl.RewriteReport is pk.RewriteReport
  assert dsl.topk is pk.topk
  assert dsl.tanimoto_topk is pk.tanimoto_topk
  assert dsl.ipa_attention is pk.ipa_attention
  assert dsl.scatter is pk.scatter
  assert dsl.iscan is pk.iscan
  assert dsl.iscatter is pk.iscatter
  assert dsl.ifilter_reduce is pk.ifilter_reduce
  assert pytest is not None
