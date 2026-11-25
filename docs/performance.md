# Performance

!!! tip "Looking for cross-library benchmarks?"
    For head-to-head 1.0 comparisons against PyTorch, JAX, CuPy, Faiss, and
    PyKeOps on CPU and an NVIDIA A10, see [Release benchmarks](release-benchmarks.md).
    This page explains *when* and *why* Parakeet wins or loses on CPUs and GPUs,
    how cache tiling behaves, and practical tuning tips.

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

These numbers predate arrays staying on the device between the loops of a
call; the copy to the GPU and back on every *call* is still there, and is most
of the time in the rows where CUDA loses.

### A big GPU on the work it is for

`python benchmarks/gpu_showcase.py` on an NVIDIA A100 (Google Colab, 12 CPU
threads, float32). The same Python source runs in every column.

| kernel | size | NumPy | one core | 12 threads | A100 | A100 vs 12 threads |
|---|---|---:|---:|---:|---:|---:|
| n-body forces | 16,384 bodies | – | 1393 ms | 238 ms | 7.5 ms | 32× |
| k-means assignment | 2M points × 256 centers × 16 | 2247 ms | 9551 ms | 1913 ms | 75.8 ms | 25× |
| heat equation, 200 steps | 2048² | 7821 ms | 4532 ms | 370 ms | 118 ms | 3.1× |
| Black-Scholes | 20M options | 625 ms | 1746 ms | 227 ms | 111 ms | 2.0× |
| Julia set | 4096² | – | 1425 ms | 155 ms | 95.5 ms | 1.6× |

The first two are what a GPU is for: every element needs thousands of
operations and the result is small. The n-body kernel is 186× faster than one
core. The last three are bounded by moving their arrays across the bus once
each way (Black-Scholes reads 240 MB and writes 80 MB); the heat equation is
600 loops over a grid that never leaves the device between them, which before
arrays stayed on the device took 1.4 s here.

A float matrix product is *not* in this table: on the CPU it goes to the BLAS
(24 ms for 2048² on this machine), on the GPU it is Parakeet's own kernel
(73 ms). Dense linear algebra on a GPU wants cuBLAS, which Parakeet does not
call.

### Where the GPU starts to win

`python benchmarks/crossover.py` times the same kernel across sizes. On the same
kind of machine (Tesla T4, 2 CPU threads, float32), after tiling and with
gradients:

| kernel | size | NumPy | one core | 2 threads | CUDA |
|---|---|---:|---:|---:|---:|
| `np.tanh(x * a) + b` | 100k | 0.18 ms | 3.3 ms | 2.6 ms | 0.77 ms |
| | 1M | 1.7 ms | 29.9 ms | 30.6 ms | 2.8 ms |
| | 10M | 39.0 ms | 277 ms | 164 ms | 75.5 ms |
| all-pairs distances | 1k × 64 | 9.3 ms | 0.42 ms | 0.41 ms | 0.77 ms |
| | 100k × 64 | 760 ms | 34.5 ms | 27.3 ms | 42.5 ms |
| matrix product | 512² | 2.5 ms | 10.1 ms | 8.0 ms | 14.3 ms |
| | 1024² | 24.8 ms | 62.3 ms | 63.1 ms | 49.6 ms |
| | 2048² | 145 ms | 510 ms | 654 ms | 375 ms |
| Julia set | 256² | – | 5.2 ms | 3.0 ms | 0.81 ms |
| | 1024² | – | 82.2 ms | 47.0 ms | 6.4 ms |
| | 2048² | – | 338 ms | 197 ms | 78.2 ms |
| Harris corners | 1024² | 9.7 ms | 2.9 ms | 4.0 ms | 3.0 ms |
| | 4096² | 292 ms | 77.0 ms | 88.8 ms | 133 ms |
| MLP gradient (784-256-10) | batch 64 | – | 3.7 ms | 4.1 ms | 5.5 ms |
| | batch 4096 | – | 231 ms | 189 ms | 139 ms |
| decision tree fit (32 features) | 100k rows | – | 4.2 s | 3.0 s | 2.9 s (CPU) |

The GPU wins once each element costs real arithmetic (the Julia set from the
smallest size, `tanh` from about 100k elements) and the result is not much
bigger than the work that produced it. It ties or loses when a loop mostly
moves memory, because every launch pays for the copy both ways: Harris at
4096² and all-pairs at 100k rows are slower on the device than on two CPU
threads. Loops below `config.device_threshold` (32,768 iterations) never leave
the CPU, which is why the decision tree, whose parallel loops run over 32
features, is a CPU program on every backend.

The matrix product is the tiled one: on this machine it is within 2.5–3.5× of
NumPy's OpenBLAS on one core.

## Reading the numbers

**Where Parakeet wins.** Expressions with several operations (no temporaries,
one pass over memory), code that needs a Python-level loop in NumPy (Julia set,
GrowCut, pairwise kernels), and anything that can use all cores.

**Where it does not.**

* *Dense linear algebra, when there is no BLAS to call.* A product of two
  float matrices is handed to the BLAS NumPy itself uses, so it costs what it
  costs in NumPy. Parakeet's own [tiled](#tiling) loops run everything else
  that has that shape (integer products, `allpairs` distances, products inside
  a parallel map or on a GPU) and are 3–10× behind a hand-tuned BLAS.
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

## Tiling

A *matmul-shaped* operation is a map over pairs whose element is a reduction
over data picked by each member of the pair: a matrix product, `allpairs`
distances, a tropical product. Evaluated row by row, every row of the result
sweeps over all of the second operand, which stops fitting in cache long before
the arrays are large. Parakeet recognises the shape — any two-dimensional map
whose element function is one reduction plus straight-line code, whoever wrote
it — and does two things:

* **tiles** the inner dimension, so a cache-sized block of the second operand is
  reused for every row before moving on (`config.tile_bytes`, by default half of
  a core's L2 cache);
* **jams** the reductions of a 2×4 block of elements into one loop, so each
  loaded value is used several times.

Neither changes what is computed, only the order. `a @ b` on 1024×1024 float64
(Apple M1 Max):

| | one core | all cores |
|---|---:|---:|
| untiled | 190 ms | 43 ms |
| tiled | 92 ms | 13 ms |
| NumPy (Accelerate) | | 4.5 ms |

`@jit(tiling=False)` or `config.tiling = False` turns it off. The timings
above are with `config.blas = False`; by default a float matrix product on a
CPU backend calls the BLAS and takes the NumPy time.

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
