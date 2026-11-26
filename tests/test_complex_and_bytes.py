"""Tests for Wave 5 (v1.6.0, #81):
- Unboxed complex64 / complex128 arithmetic, math (abs, conj, angle, exp, log,
  sqrt, sin, cos), matmul / vdot, reductions, indexing, and NEP 50 promotion
- Native Apple Metal & multicore execution of complex64 (zero IR/backend changes)
- Wirtinger reverse-mode AD (par.grad) and forward-mode AD (par.jvp) over
  complex64 / complex128 losses and iterative Cooley-Tukey radix-2 FFT
- Byte / ASCII parsing helpers: bytes literals, ord, LUT remapping,
  par.split_bytes -> RaggedArray[uint8], par.parse_int64, par.parse_float32,
  par.parse_float64 (both inside @par.jit and directly from Python)
"""

from __future__ import annotations

import numpy as np
import pytest

import parakeet as par
from parakeet import RaggedArray


# ---------------------------------------------------------------------------
# 1. Unboxed complex64 / complex128 arithmetic, math, matmul, & Metal GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_arithmetic_math_and_matmul(dtype):
  @par.jit
  def complex_ops(a, b, M):
    s = a + b * (2.0 - 1.5j)
    q = s / (b + 3.0 + 0.5j)
    e = np.exp(a * 0.25) + np.sqrt(b + 2.0 + 1.0j)
    m = M @ a
    p = np.vdot(a, b)
    return (
      s,
      q,
      np.abs(e),
      np.angle(e),
      np.conj(m),
      p,
      np.sum(a),
      np.mean(b),
      a.real + b.imag,
    )

  rng = np.random.default_rng(11)
  n = 32
  a = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
  b = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
  M = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(dtype)

  ref = complex_ops(a, b, M, _backend="python")
  tol = 1e-4 if dtype == np.complex64 else 1e-10
  for backend in ("c", "multicore"):
    got = complex_ops(a, b, M, _backend=backend)
    for g, r in zip(got, ref):
      assert np.asarray(g).dtype == np.asarray(r).dtype
      np.testing.assert_allclose(g, r, rtol=tol, atol=tol)

  if dtype == np.complex64 and par.available_backends().get("metal") == "ok":
    with par.options(device_fallback="error", device_threshold=1):
      got_metal = complex_ops(a, b, M, _backend="metal")
    for g, r in zip(got_metal, ref):
      np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-4)


def test_complex_cooley_tukey_fft_and_inplace_updates():
  @par.jit
  def fft_radix2(x):
    """Iterative Cooley-Tukey radix-2 FFT in compiled complex64/complex128."""
    n = len(x)
    # Bit-reversal permutation
    bits = 0
    tmp = n
    while tmp > 1:
      tmp //= 2
      bits += 1
    out = np.empty_like(x)
    for i in range(n):
      rev = 0
      v = i
      for _ in range(bits):
        rev = (rev << 1) | (v & 1)
        v >>= 1
      out[rev] = x[i]

    # Butterfly stages
    size = 2
    while size <= n:
      half = size // 2
      angle_step = -2.0 * np.pi / size
      for start in range(0, n, size):
        for k in range(half):
          ang = angle_step * k
          w = complex(np.cos(ang), np.sin(ang))
          u = out[start + k]
          t = w * out[start + k + half]
          out[start + k] = u + t
          out[start + k + half] = u - t
      size *= 2
    return out

  rng = np.random.default_rng(7)
  for dtype, tol in ((np.complex64, 1e-4), (np.complex128, 1e-11)):
    x = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(dtype)
    got = fft_radix2(x)
    ref = np.fft.fft(x).astype(dtype)
    assert got.dtype == dtype
    np.testing.assert_allclose(got, ref, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# 2. Wirtinger AD (par.grad & par.jvp over complex64 / complex128)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_wirtinger_grad_and_jvp(dtype):
  def spectral_loss(z, target, w):
    # Real-valued loss over complex array: 0.5 * sum(|w * z - target|^2) + sum(real(exp(0.2 * z)))
    diff = w * z - target
    power = np.float64(0.5) * np.sum(diff.real * diff.real + diff.imag * diff.imag)
    reg = np.sum(np.exp(z * np.float32(0.2)).real)
    return power + reg

  rng = np.random.default_rng(42)
  n = 16
  z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
  target = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
  w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)

  val, g = par.value_and_grad(spectral_loss, argnum=0)(z, target, w)
  assert g.dtype == dtype
  assert g.shape == z.shape

  # Check Wirtinger directional derivative identity: Re(conj(g) . dz) == (L(z + eps dz) - L(z - eps dz)) / (2 eps)
  dz = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
  eps = 1e-3 if dtype == np.complex64 else 1e-6
  f_plus = float(par.jit(spectral_loss)(z + dtype(eps) * dz, target, w))
  f_minus = float(par.jit(spectral_loss)(z - dtype(eps) * dz, target, w))
  fd = (f_plus - f_minus) / (2.0 * eps)
  dir_grad = float(np.sum(np.conj(g) * dz).real)
  tol = 2e-2 if dtype == np.complex64 else 1e-6
  np.testing.assert_allclose(dir_grad, fd, rtol=tol, atol=tol)

  # Check par.jvp agrees with Wirtinger gradient dot product
  out_val, out_tan = par.jvp(lambda z_: spectral_loss(z_, target, w), (z,), (dz,))
  np.testing.assert_allclose(float(out_val), float(val), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(float(out_tan), dir_grad, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Byte / ASCII parsing helpers (split_bytes, parse_int64, parse_float32/64)
# ---------------------------------------------------------------------------


def test_bytes_literals_ord_lut_and_split_bytes():
  @par.jit
  def fasta_gc_counts(buf):
    lines = par.split_bytes(buf, ord("\n"), skip_empty=True)
    n_lines = len(lines)
    is_header = np.zeros(n_lines, dtype=np.int64)
    gc = np.zeros(n_lines, dtype=np.int64)
    bases = np.zeros(n_lines, dtype=np.int64)
    for k in range(n_lines):
      line = lines[k]
      if len(line) > 0 and line[0] == ord(">"):
        is_header[k] = 1
      else:
        bases[k] = len(line)
        c = 0
        for j in range(len(line)):
          b = line[j]
          if b == ord("G") or b == ord("C") or b == ord("g") or b == ord("c"):
            c += 1
        gc[k] = c
    return np.sum(is_header), np.sum(gc), np.sum(bases)

  fasta = b">seq1\nACGTGC\nTTAA\n>seq2\nGGCCggcc\n\n>seq3\nAAAA\n"
  n_hdr, n_gc, n_bases = fasta_gc_counts(fasta)
  assert int(n_hdr) == 3
  assert int(n_gc) == 4 + 0 + 8 + 0
  assert int(n_bases) == 6 + 4 + 8 + 4

  # Direct Python call to split_bytes matches compiled version
  rag_py = par.split_bytes(fasta, ord("\n"), skip_empty=True)
  rag_jit = par.jit(lambda b: par.split_bytes(b, ord("\n"), skip_empty=True))(fasta)
  np.testing.assert_array_equal(rag_jit.values, rag_py.values)
  np.testing.assert_array_equal(rag_jit.offsets, rag_py.offsets)
  assert bytes(rag_jit[0]) == b">seq1"
  assert bytes(rag_jit[1]) == b"ACGTGC"

  # Byte literal indexing & LUT remapping inside @par.jit
  @par.jit
  def complement_dna(seq):
    lut = np.arange(256, dtype=np.uint8)
    src = b"ACGT"
    dst = b"TGCA"
    for i in range(4):
      lut[src[i]] = dst[i]
    return lut[seq]

  comp = complement_dna(b"ACGTAC")
  assert bytes(comp) == b"TGCATG"


def test_ascii_parse_int64_and_float32_float64():
  @par.jit
  def parse_csv_cols(int_buf, flt_buf):
    i_toks = par.split_bytes(int_buf, ord(","), skip_empty=True)
    f_toks = par.split_bytes(flt_buf, ord(","), skip_empty=True)
    ints = par.parse_int64(i_toks)
    f32s = par.parse_float32(f_toks)
    f64s = par.parse_float64(f_toks)
    single_i = par.parse_int64(b" -4219 ")
    single_f = par.parse_float64(b" +3.14159e-2 ")
    return ints, f32s, f64s, single_i, single_f

  int_buf = b"0, 42, -17, +9999999999, 123456\n"
  flt_buf = b"0.0, -3.5, +1.25e2, 4.5E-3, 100, -0.000125"

  ref_ints = np.array([0, 42, -17, 9999999999, 123456], dtype=np.int64)
  ref_flts = np.array([0.0, -3.5, 125.0, 0.0045, 100.0, -0.000125], dtype=np.float64)

  for backend in ("c", "multicore", "python"):
    ints, f32s, f64s, si, sf = parse_csv_cols(int_buf, flt_buf, _backend=backend)
    np.testing.assert_array_equal(ints, ref_ints)
    np.testing.assert_allclose(f32s, ref_flts.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(f64s, ref_flts, rtol=1e-12, atol=1e-12)
    assert int(si) == -4219
    np.testing.assert_allclose(float(sf), 0.0314159, rtol=1e-12, atol=1e-12)
