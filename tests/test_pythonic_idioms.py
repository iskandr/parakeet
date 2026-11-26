"""Tests for Pythonic string sequences, IntEnum/char-dict LUTs, and multi-output/ragged par.map."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pytest

import parakeet as par


class Base(IntEnum):
  A = 0
  C = 1
  G = 2
  T = 3


BASE_CODE = {"A": Base.A, "C": Base.C, "G": Base.G, "T": Base.T}
BASE_WEIGHT = {"A": 1.5, "C": 2.5, "G": 2.5, "T": 1.5}


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_list_str_equal_length_and_ragged_with_char_lut(backend):
  @par.jit
  def analyze_equal(reads):
    def one(seq):
      codes = BASE_CODE[seq]
      gc = np.mean((seq == "C") | (seq == "G")).astype(np.float32)
      gc_in = np.sum(np.array([1 if ch in "CG" else 0 for ch in seq]))
      unknown = np.sum(BASE_CODE.get(seq, 4) == 4)
      wt = np.sum(BASE_WEIGHT.get(seq, 0.0))
      return codes, gc, gc_in, unknown, wt

    return par.map(one, reads)

  reads = ["ACGTACGT", "GGGGCCCC", "AAAANTTT"]
  codes, gc, gc_in, unknown, wt = analyze_equal(reads, _backend=backend)
  assert codes.shape == (3, 8)
  np.testing.assert_array_equal(codes[0], [0, 1, 2, 3, 0, 1, 2, 3])
  np.testing.assert_allclose(gc, [0.5, 1.0, 0.0], rtol=1e-6)
  np.testing.assert_array_equal(gc_in, [4, 8, 0])
  np.testing.assert_array_equal(unknown, [0, 0, 1])
  np.testing.assert_allclose(wt, [16.0, 20.0, 10.5], rtol=1e-6)

  @par.jit
  def analyze_ragged(reads):
    def one(seq):
      gc = np.mean((seq == "C") | (seq == "G"))
      n_valid = np.sum(BASE_CODE.get(seq, 4) < 4)
      return gc, n_valid

    return par.map(one, reads)

  gc_r, valid_r = analyze_ragged(["ACGT", "GG", "TTTTTTN"], _backend=backend)
  np.testing.assert_allclose(gc_r, [0.5, 1.0, 0.0], rtol=1e-6)
  np.testing.assert_array_equal(valid_r, [4, 2, 6])


@pytest.mark.parametrize("backend", ["c", "multicore"])
def test_multi_output_map_over_dense_and_ragged(backend):
  @par.jit
  def row_stats(x):
    return par.map(lambda row: (np.sum(row), np.mean(row), row * np.float32(2.0)), x)

  rng = np.random.default_rng(7)
  x = rng.standard_normal((16, 8)).astype(np.float32)
  sums, means, doubled = row_stats(x, _backend=backend)
  np.testing.assert_allclose(sums, x.sum(axis=1), rtol=1e-5)
  np.testing.assert_allclose(means, x.mean(axis=1), rtol=1e-5)
  np.testing.assert_allclose(doubled, x * 2.0, rtol=1e-5)

  @par.jit
  def ragged_stats(rag, scale):
    return par.map(lambda seg, s: (np.sum(seg) * s, len(seg)), rag, scale)

  rag = par.RaggedArray.from_sequences([np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), np.array([10.0])])
  scales = np.array([2.0, 0.5, 3.0])
  weighted, lens = ragged_stats(rag, scales, _backend=backend)
  np.testing.assert_allclose(weighted, [6.0, 6.0, 30.0])
  np.testing.assert_array_equal(lens, [2, 3, 1])


def test_char_comparison_and_non_ascii_error():
  @par.jit
  def count_upper_dna(buf):
    return np.sum((buf >= "A") & (buf <= "Z"))

  assert int(count_upper_dna(b"AcGt")) == 2

  @par.jit
  def accept_strings(seqs):
    return len(seqs)

  with pytest.raises(TypeError, match="ASCII"):
    accept_strings(["ACGT", "αβγδ"])
