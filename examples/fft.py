"""Unboxed complex64/complex128 radix-2 FFT and Wirtinger spectral-filter AD.

Complex arrays lower onto unboxed real/imag float buffers so CPU, multicore,
Metal, CUDA, and Wirtinger ``par.value_and_grad`` compose without Python
overhead.

Run:
    python -m examples.fft
"""

from __future__ import annotations

import numpy as np

import parakeet as par


@par.jit
def fft_radix2(x):
  """Iterative Cooley-Tukey radix-2 FFT (power-of-two length)."""
  n = len(x)
  assert n > 0 and (n & (n - 1)) == 0, "fft_radix2 requires a power-of-two length"
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
    for k in range(half):
      ang = angle_step * k
      w = complex(np.cos(ang), np.sin(ang))
      for start in range(0, n, size):
        u = out[start + k]
        t = w * out[start + k + half]
        out[start + k] = u + t
        out[start + k + half] = u - t
    size *= 2
  return out


def spectral_filter_loss(weights, spectrum, target_mag):
  """Real-valued spectral fitting loss over complex filter weights."""
  diff = np.abs(weights * spectrum) - target_mag
  return np.float64(0.5) * np.sum(diff * diff)


def main(n: int = 1024):
  rng = np.random.default_rng(0)
  sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
  got_fft = fft_radix2(sig)
  ref_fft = np.fft.fft(sig).astype(np.complex64)
  max_err = float(np.max(np.abs(got_fft - ref_fft)))
  assert max_err < 1e-3, max_err

  weights = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
  target_mag = np.abs(ref_fft) * np.float32(0.5)
  loss, grad_w = par.value_and_grad(spectral_filter_loss, argnum=0)(weights, got_fft, target_mag)
  assert grad_w.dtype == np.complex64 and np.isfinite(float(loss))
  print(f"radix-2 FFT (N={n}, complex64) max error: {max_err:.2e} | loss: {float(loss):.4f}")
  return got_fft, loss, grad_w


if __name__ == "__main__":
  main()
