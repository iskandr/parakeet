# Performance and tuning

!!! tip "Cross-library benchmarks"
    For head-to-head comparisons against PyTorch, `torch.compile`, JAX, CuPy,
    Faiss, PyKeOps, RDKit, Biopython, and edlib on NVIDIA L4 and Apple M1 Pro,
    see [Benchmarks](release-benchmarks.md). This page explains *when* Parakeet
    wins or loses and how to tune your code.

## Where Parakeet wins vs. where it doesn't

| Parakeet excels at | Specialist libraries still lead on |
|---|---|
| Multi-step array expressions (eliminates temporaries, single pass over RAM) | Large standalone GPU GEMMs (use cuBLAS; on CPU, Parakeet already calls NumPy's BLAS) |
| Pairwise kernels, attention, cross-entropy, 1-NN (`O(N²)` buffers tiled away) | Single-threaded transcendentals (`np.tanh`, `np.exp` — NumPy uses SIMD; Parakeet wins once multi-core kicks in) |
| Algorithms that would otherwise need a Python loop (stencils, Julia set, DP, bit-parallel bio kernels) | Tiny inputs (~10–100 elements), where native call marshalling outweighs work |
| Multi-core CPU scaling & GIL-free concurrency | Memory-bound GPU loops that copy host arrays back and forth on every call (use `par.to_device`) |

## CPU micro-benchmarks (Apple M1 Pro, 10 cores, `float64`)

Measured via `python benchmarks/run.py` (best of 5, compilation excluded; `×` is
speedup over NumPy):

| Benchmark | NumPy | `backend="c"` (1T) | `backend="multicore"` (10T) |
|---|---:|---:|---:|
| `np.tanh(x * a) + b` (4M) | 14.5 ms | 24.8 ms (0.6×) | **3.5 ms (4.2×)** |
| Euclidean norm (4M) | 3.4 ms | 1.1 ms (3.1×) | **0.65 ms (5.3×)** |
| Arc distance (4M) | 174 ms | 174 ms (1.0×) | **23.3 ms (7.5×)** |
| Rosenbrock derivative (4M) | 11.4 ms | 8.7 ms (1.3×) | **0.72 ms (16×)** |
| Harris corner response (2000²) | 19.4 ms | 4.0 ms (4.9×) | **1.1 ms (17×)** |
| Diffusion stencil, 20 steps (1000²) | 54.5 ms | 24.5 ms (2.2×) | **6.3 ms (8.6×)** |
| All-pairs distances (40k × 64 × 32) | 103 ms | 11.1 ms (9.2×) | **1.6 ms (66×)** |
| 3×3 convolution (2000²) | 21.2 ms | 9.1 ms (2.3×) | **1.4 ms (15×)** |
| Julia set (1000²) | 402 ms | 47.2 ms (8.5×) | **6.1 ms (66×)** |
| Matrix product (300², `blas=False`) | **0.16 ms** | 4.1 ms | 0.72 ms |

## Compute-bound vs. transfer-bound on GPUs

GPUs win decisively when arithmetic intensity is high (many FLOPs per byte
loaded) or when arrays stay resident across iterations via `par.to_device`.
Measured with `python benchmarks/gpu_showcase.py` on an NVIDIA A100 (12 CPU
threads, `float32`, identical Python source):

| Kernel | Size | NumPy | 1 core | 12 threads | A100 | A100 vs 12T |
|---|---|---:|---:|---:|---:|---:|
| N-body forces | 16,384 bodies | – | 1,393 ms | 238 ms | **7.5 ms** | **32×** |
| K-means assignment | 2M × 256 × 16 | 2,247 ms | 9,551 ms | 1,913 ms | **75.8 ms** | **25×** |
| Heat equation, 200 steps | 2048² | 7,821 ms | 4,532 ms | 370 ms | **118 ms** | **3.1×** |
| Black-Scholes | 20M options | 625 ms | 1,746 ms | 227 ms | **111 ms** | **2.0×** |
| Julia set | 4096² | – | 1,425 ms | 155 ms | **95.5 ms** | **1.6×** |

N-body and k-means do thousands of operations per output element (186× faster
than 1 core on n-body). Heat equation runs 600 kernels on device-resident grids
with zero intermediate host copies (down from 1.4 s before residency).
Black-Scholes is mostly PCIe transfer (240 MB in, 80 MB out)—passing
`par.to_device` inputs removes that bus bottleneck across repeated calls.

## Cache and register tiling {#tiling}

Any 2D+ map wrapping a reduction (`a @ b`, `allpairs`, `einsum`, tropical
products) sweeps through the second operand on every row if run naively,
thrashing L1/L2 cache. Parakeet automatically tiles the inner dimension to fit
`config.tile_bytes` (default half of per-core L2) and jams `2×4` register
micro-tiles on CPU (and shared-memory + matrix-unit tiles on GPU).

`a @ b` on `1024 × 1024` `float64` (Apple M1 Max, `config.blas = False`):

| Schedule | 1 core | All cores |
|---|---:|---:|
| Untiled (`tiling=False`) | 190 ms | 43 ms |
| Tiled (default) | **92 ms** | **13 ms** |
| NumPy (Accelerate BLAS, `blas=True`) | — | 4.5 ms |

Disable tiling with `@par.jit(tiling=False)` or `par.config.tiling = False`. On
CPU, standalone float `a @ b` calls NumPy's BLAS directly by default
(`config.blas = True`).

## Practical tuning checklist

* **Keep arrays contiguous:** specializations track C-contiguity; unit-stride
  arrays vectorize cleanly on CPU and coalesce on GPU.
* **Keep GPU arrays resident:** wrap repeated GPU inputs with `par.to_device`
  so you do not pay host-to-device transfers on every call.
* **Use `float32` on Apple Metal:** Metal has no hardware `float64` and falls
  back to CPU for double-precision loops.
* **Enable Tensor Cores on NVIDIA `sm_80+`:** set `par.config.tensor_cores = True`
  (or `PARAKEET_TENSOR_CORES=1`) for TF32 WMMA matmuls and training steps.
* **Prefer expressions / operators over raw `for` loops:** NumPy expressions,
  comprehensions, `par.map`, `par.fold`, and `par.scatter_add` parallelize
  automatically; arbitrary sequential `for` loops do not (except recognized
  Law 4 affine/bilinear recurrences and canonical `out[idx[i]] += v` scatters).
* **Inline across helpers:** put the outer workflow inside one `@par.jit`
  function so fusion crosses helper boundaries automatically.
* **Drop checks when verified:** `@par.jit(bounds_check=False)` eliminates
  bounds checks in tight indexed loops; `par.config.fast_math = True` enables
  `-ffast-math` when exact IEEE rounding is not required.
