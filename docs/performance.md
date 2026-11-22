# Performance

All numbers on this page were measured with `python benchmarks/run.py` on an
Apple M1 Pro (10 cores, 32 GB), macOS 26.5, Python 3.12.6, NumPy 2.5.5; best of
five runs, compilation excluded. "NumPy" is the same computation written with
NumPy alone; "×" is the speed-up over it. Run the script on your own machine —
results vary a lot between CPUs, compilers and array sizes.

## CPU (float64)

| benchmark | NumPy | `backend="c"` | `backend="multicore"` |
|---|---:|---:|---:|
| `np.tanh(x * a) + b` (4M) | 14.5 ms | 24.8 ms (0.6×) | 3.5 ms (4.2×) |
| Euclidean norm (4M) | 3.4 ms | 1.1 ms (3.1×) | 0.65 ms (5.3×) |
| arc distance (4M) | 174 ms | 174 ms (1.0×) | 23.3 ms (7.5×) |
| Rosenbrock derivative (4M) | 11.4 ms | 8.7 ms (1.3×) | 0.72 ms (16×) |
| Harris corner response (2000²) | 19.4 ms | 4.0 ms (4.9×) | 1.1 ms (17×) |
| diffusion stencil, 20 steps (1000²) | 54.5 ms | 24.5 ms (2.2×) | 6.3 ms (8.6×) |
| all-pairs distances (40k × 64 × 32) | 103 ms | 11.1 ms (9.2×) | 1.6 ms (66×) |
| 3×3 convolution (2000²) | 21.2 ms | 9.1 ms (2.3×) | 1.4 ms (15×) |
| Julia set (1000²) | 402 ms | 47.2 ms (8.5×) | 6.1 ms (66×) |
| matrix product (300²) | 0.16 ms | 4.1 ms | 0.72 ms |
| explicit summation loop (4M) | — | 3.8 ms | 3.8 ms |
| GrowCut step (64², radius 5) | — | 1.9 ms | 0.47 ms |

## GPU (float32, Metal on the same machine)

| benchmark | NumPy | `"multicore"` | `"metal"` |
|---|---:|---:|---:|
| `np.tanh(x * a) + b` (4M) | 5.5 ms | 2.4 ms (2.2×) | 2.9 ms (1.9×) |
| Euclidean norm (4M) | 1.8 ms | 0.40 ms (4.5×) | 6.2 ms (0.3×) |
| arc distance (4M) | 50.5 ms | 13.5 ms (3.7×) | 8.7 ms (5.8×) |
| Rosenbrock derivative (4M) | 6.4 ms | 0.66 ms (9.8×) | 3.4 ms (1.9×) |
| Harris corner response (2000²) | 8.3 ms | 1.1 ms (7.8×) | 5.2 ms (1.6×) |
| diffusion stencil, 20 steps (1000²) | 24.8 ms | 5.5 ms (4.5×) | 57.0 ms (0.4×) |
| all-pairs distances (40k × 64 × 32) | 68.9 ms | 1.1 ms (63×) | 4.0 ms (17×) |
| 3×3 convolution (2000²) | 10.5 ms | 1.2 ms (8.9×) | 1.2 ms (8.6×) |
| Julia set (1000²) | 401 ms | 6.0 ms (67×) | 6.1 ms (66×) |
| matrix product (300²) | 0.06 ms | 0.42 ms | 3.9 ms |

## GPU (float32, CUDA on a Tesla T4)

A Google Colab T4 instance: the host has only 2 CPU threads, so `"multicore"`
is nothing like the M1 Pro above, and this NumPy has no fast `float32` `tanh`.

| benchmark | NumPy | `"multicore"` (2 threads) | `"cuda"` |
|---|---:|---:|---:|
| `np.tanh(x * a) + b` (4M) | 7.4 ms | 60.5 ms (0.1×) | 14.6 ms (0.5×) |
| Euclidean norm (4M) | 6.9 ms | 1.9 ms (3.6×) | 8.3 ms (0.8×) |
| arc distance (4M) | 53.0 ms | 179 ms (0.3×) | 42.8 ms (1.2×) |
| Rosenbrock derivative (4M) | 35.5 ms | 12.1 ms (2.9×) | 16.8 ms (2.1×) |
| Harris corner response (2000²) | 46.0 ms | 15.1 ms (3.0×) | 15.2 ms (3.0×) |
| diffusion stencil, 20 steps (1000²) | 79.1 ms | 38.0 ms (2.1×) | 98.8 ms (0.8×) |
| all-pairs distances (40k × 64 × 32) | 313 ms | 17.1 ms (18×) | 13.3 ms (24×) |
| 3×3 convolution (2000²) | 64.4 ms | 48.7 ms (1.3×) | 44.1 ms (1.5×) |
| Julia set (1000²) | 1064 ms | 43.4 ms (25×) | 10.6 ms (101×) |
| matrix product (300²) | 7.6 ms | 4.8 ms (1.6×) | 2.7 ms (2.8×) |

Arrays are copied to the GPU and back on every call, which is most of the time
in the rows where CUDA loses: nothing stays on the device between calls.

## Reading the numbers

**Where Parakeet wins.** Expressions with several operations (no temporaries,
one pass over memory), code that needs a Python-level loop in NumPy (Julia set,
GrowCut, pairwise kernels), and anything that can use all cores.

**Where it does not.**

* *Dense linear algebra.* `X @ Y` is a straightforward triple loop here; NumPy
  calls a tuned BLAS that is one to two orders of magnitude faster. Compute
  matrix products with NumPy and pass the result in.
* *Single-threaded transcendental functions.* NumPy ships SIMD implementations
  of `tanh`, `sin`, `exp`, ...; Parakeet's sequential backend calls the C math
  library one element at a time (the `tanh` row). With threads it is ahead again.
* *Memory-bound loops on a GPU.* On this machine the 10 CPU cores usually beat
  the integrated GPU unless each element needs a lot of arithmetic (arc distance,
  convolution). Loops that mostly move memory (norm, stencils) lose, and
  repeatedly launching cheap kernels over a NumPy-owned array (diffusion) loses
  badly because the array is copied to and from the device on every launch.
* *Tiny inputs.* A call into compiled code costs a few microseconds of argument
  marshalling; a NumPy call on a 10-element array is cheaper.

## Tips

* Prefer array expressions or comprehensions over index loops when they say the
  same thing: they parallelise, explicit `for` loops do not.
* Keep work inside one `@jit` function so expressions can fuse across helper
  calls (helpers are inlined automatically).
* `@jit(bounds_check=False)` removes index checks from explicit indexing in
  loops once the code is known to be correct.
* Compiled code is specialised on whether an input is C-contiguous; contiguous
  arrays give the C compiler unit strides to vectorise.
* `parakeet.config.fast_math = True` allows the C compiler to reorder
  floating-point arithmetic everywhere (results may change in the last digits).
