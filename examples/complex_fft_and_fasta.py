"""Unboxed complex64/complex128 FFT with Wirtinger AD + compiled FASTA/CSV byte parsing.

Demonstrates two Wave 5 (v1.6.0) capabilities:
1. Iterative Cooley-Tukey radix-2 FFT and spectral loss differentiation using
   unboxed complex64/complex128 arrays (lowering cleanly onto real float32/float64
   buffers so CPU, multicore, Metal, and Wirtinger par.grad work natively).
2. Zero-PyObject FASTA/CSV byte parsing via par.split_bytes -> RaggedArray[uint8],
   byte-level LUT remapping, and par.parse_float32.

Run:
    python examples/complex_fft_and_fasta.py
"""

from __future__ import annotations

import time

import numpy as np

import parakeet as par


@par.jit
def fft_radix2(x):
  """Iterative Cooley-Tukey radix-2 FFT (power-of-two length)."""
  n = len(x)
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


def spectral_filter_loss(weights, spectrum, target_mag):
  """Real-valued spectral fitting loss over complex filter weights."""
  filtered = weights * spectrum
  mag = np.abs(filtered)
  diff = mag - target_mag
  return np.float64(0.5) * np.sum(diff * diff)


@par.jit
def fasta_gc_and_scores(fasta_buf, score_csv_buf):
  """Parse a FASTA byte buffer + CSV float scores with zero PyObjects."""
  lines = par.split_bytes(fasta_buf, ord("\n"), skip_empty=True)
  gc_lut = np.zeros(256, dtype=np.int64)
  for b in b"GCgc":
    gc_lut[b] = 1
  n_lines = len(lines)
  gc_total = 0
  base_total = 0
  for k in range(n_lines):
    line = lines[k]
    if len(line) > 0 and line[0] != ord(">"):
      base_total += len(line)
      gc_total += np.sum(gc_lut[line])
  scores = par.parse_float32(par.split_bytes(score_csv_buf, ord(","), skip_empty=True))
  return gc_total, base_total, np.mean(scores)


def main():
  rng = np.random.default_rng(0)
  n = 1024
  sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
  got_fft = fft_radix2(sig)
  ref_fft = np.fft.fft(sig).astype(np.complex64)
  max_err = float(np.max(np.abs(got_fft - ref_fft)))
  print(f"radix-2 FFT (N={n}, complex64) max error vs np.fft.fft: {max_err:.2e}")
  assert max_err < 1e-3

  weights = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
  target_mag = np.abs(ref_fft) * np.float32(0.5)
  loss, grad_w = par.value_and_grad(spectral_filter_loss, argnum=0)(weights, got_fft, target_mag)
  print(
    f"spectral filter loss: {float(loss):.4f}, grad dtype: {grad_w.dtype}, ||grad||: {float(np.linalg.norm(grad_w)):.4f}"
  )

  # Multi-MB FASTA + CSV benchmark
  seq_line = b"ACGTACGTGCGCATTAGCTA" * 4 + b"\n"
  fasta_buf = (b">read\n" + seq_line) * 2000
  csv_buf = b",".join(f"{0.01 * (i % 100):.2f}".encode() for i in range(50000))
  # Warmup
  gc, bases, mean_s = fasta_gc_and_scores(fasta_buf, csv_buf)
  t0 = time.perf_counter()
  for _ in range(10):
    gc, bases, mean_s = fasta_gc_and_scores(fasta_buf, csv_buf)
  dt = (time.perf_counter() - t0) / 10
  total_mb = (len(fasta_buf) + len(csv_buf)) / 1e6
  print(
    f"FASTA+CSV parse ({total_mb:.2f} MB): gc={int(gc)}/{int(bases)}, "
    f"mean_score={float(mean_s):.4f}, {dt * 1e3:.2f} ms ({total_mb / dt:.1f} MB/s)"
  )


if __name__ == "__main__":
  main()
