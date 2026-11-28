# Backends and GPUs

Every backend runs the same compiled host C program; backends differ only in
how parallel loops execute.

| `backend` | Parallel loops run on | Notes |
|---|---|---|
| `"auto"` | Best available multi-core CPU strategy | Default (same as `"multicore"`) |
| `"c"` | Calling thread (sequential) | Zero threading overhead |
| `"threads"` | Parakeet's persistent thread pool | POSIX; fork-safe; no extra runtime library |
| `"gcd"` | Grand Central Dispatch | macOS |
| `"openmp"` | Compiler's OpenMP runtime | Requires `libgomp` / `libomp` (`brew install libomp` on macOS) |
| `"metal"` / `"mps"` | Apple GPU | `pip install "parakeet[metal]"` |
| `"cuda"` | NVIDIA GPU | `pip install "parakeet[cuda12]"` (CuPy) |
| `"gpu"` | Auto-selects `"metal"` on macOS, `"cuda"` elsewhere | Portable GPU shorthand |
| `"python"` | Uncompiled Python | Runs the original function for debugging |

Check what works on your machine at any time:

```python
parakeet.available_backends()
# {'python': 'ok', 'c': 'ok', 'threads': 'ok', 'gcd': 'ok', 'openmp': 'ok', 'metal': 'ok',
#  'cuda': 'unavailable: the CUDA backend needs CuPy, which is not installed. ...'}
```

Unavailable backends fail immediately when selected with install instructions—never
later with a linker error.

## Multi-core CPU

* **Outermost parallelism:** `map`, `allpairs`, `imap`, `parfor`, array
  expressions, comprehensions, and associative reductions split across
  `config.num_threads` threads (default: all cores). Nested parallel ops run
  sequentially inside each worker.
* **Grain threshold:** Cheap loops run on the calling thread unless they have at
  least `2 * config.parallel_grain` iterations (default `8,192` per chunk).
  Loops containing inner loops always split.
* **Threads & processes:** Compiled functions release the GIL and are safe to
  call concurrently from Python threads. With `multiprocessing`, use `"threads"`
  (fork-safe) or set start method `"spawn"` when using `"openmp"` or `"gcd"`.

## Apple GPUs (Metal)

```python
@par.jit(backend="metal")
def blur_energy(img):
  return (img[1:-1, 1:-1] * 4 - img[:-2, 1:-1] - img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:]) ** 2
```

Eligible parallel loops compile to Metal Shading Language (MSL) compute kernels,
using Apple Silicon's `simdgroup_matrix` FP32 hardware units automatically on
tiled contractions.

!!! warning "Apple GPUs have no hardware `float64`"
    Loops touching `float64` data fall back to the CPU thread pool with a
    `FallbackWarning`. Pass `float32` arrays and `np.float32` scalars to stay on
    Metal (Python float literals like `x * 2.0` preserve `float32` via NEP 50).

Loops also stay on CPU if they allocate dynamic arrays per iteration or call
functions Metal lacks (`lgamma`, `erfc`). Control fallback behavior with
`config.device_fallback`: `"warn"` (default), `"silent"`, or `"error"`. Runtime
errors inside GPU kernels (`IndexError`, `AssertionError`) still raise normal
Python exceptions.

## NVIDIA GPUs (CUDA)

```python
@par.jit(backend="cuda")
def distances(X, C):
  return par.allpairs(lambda a, b: np.sum((a - b) ** 2), X, C)
```

Kernels are zero-header CUDA C++ (TF32 Tensor Cores use inline PTX
`mma.sync.aligned.m16n8k8` with register-direct epilogues, avoiding `#include
<mma.h>`) and compiled CUBIN binaries are cached persistently on disk under
`config.cache_dir`. Select the CUDA kernel compiler via `config.cuda_compiler`
(`PARAKEET_CUDA_COMPILER`):

* **`"nvrtc"` (default):** compiles via NVRTC with LLVM IR optimizations and
  caches CUBINs on disk. Requires only an NVIDIA driver and matching CuPy wheel.
* **`"ptx"`:** emits PTX 7.8 directly from Parakeet's flat `ParLoop` IR for
  elementwise, loop, atomic, and warp-reduce kernels (loaded through
  `cuModuleLoadDataEx`), falling back to NVRTC for tiled GEMMs and libm ops.
* **`"nvcc"`:** compiles via standalone `nvcc --cubin -O3` when installed on
  `PATH`.

All dtypes including `float64` are supported. Enable TF32 Tensor Cores on
`sm_80+` GPUs with `par.config.tensor_cores = True` (or
`PARAKEET_TENSOR_CORES=1`).

*(TPUs are not supported: TPUs require XLA whole-program static-shape graphs,
whereas Parakeet generates scalar C/MSL/CUDA kernels.)*

??? note "CI simulator and GPU test validation"
    In CI (no NVIDIA hardware), `PARAKEET_CUDA_SIMULATE=1` compiles generated
    CUDA kernels as host C++ and executes their thread grids on CPU.

    Full physical-GPU runs (L4, A10, A100, T4) validate the test suite,
    examples, and [benchmarks](release-benchmarks.md) with strict device
    dispatch. To run tests on your own GPU:

    ```bash
    PARAKEET_TEST_BACKENDS=cuda PARAKEET_DEVICE_THRESHOLD=1 pytest
    ```

    Or launch on Colab via `tools/colab_job.py`:

    ```bash
    pip install runplz google-colab-cli
    runplz colab tools/colab_job.py   # T4; set PK_COLAB_GPU=A100 for A100
    ```

## Device residency and data movement

Within a single `@par.jit` call, **arrays stay on the GPU across consecutive
kernels**:

1. The first GPU kernel touching an input uploads it; outputs allocated by the
   compiled function are created directly on the device (and on Apple Silicon
   unified memory, Metal shares buffers with zero copies).
2. Later GPU kernels reuse resident buffers without copying back to the host.
3. Temporary device buffers freed before the host touches them are never
   downloaded.
4. Only host-side loops, CPU fallbacks, BLAS calls, or returning to Python pull
   modified buffers back to host memory.

**Dispatch threshold:** A parallel loop goes to the GPU when it has at least
`1,024` elements and `elements × estimated_inner_iters >= config.device_threshold`
(default `32,768`). Tiny loops or low-parallelism outer loops stay on CPU where
launch overhead would dominate.

**Reusing inputs across calls:** Wrap arrays with `par.to_device` to avoid
re-uploading inputs on every call, and call `par.to_host` when Python needs the
result:

```python
x_dev = par.to_device(x, backend="cuda")
y_dev = my_kernel(x_dev)
y_np = par.to_host(y_dev)
```
