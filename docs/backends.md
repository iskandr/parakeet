# Backends and GPUs

Every backend runs the same compiled *host program* (C). They differ in how the
parallel loops inside it are executed.

| `backend` | Parallel loops run on | Notes |
|---|---|---|
| `"auto"` | the best multi-core strategy found | default; same as `"multicore"` |
| `"c"` | the calling thread | no threads at all; also compiles without parallel loop overhead |
| `"threads"` | Parakeet's persistent thread pool | POSIX; fork-safe; no extra runtime library |
| `"gcd"` | Grand Central Dispatch | macOS |
| `"openmp"` | the compiler's OpenMP runtime | needs `libgomp`/`libomp` (macOS: `brew install libomp`) |
| `"metal"` / `"mps"` | the Apple GPU | `pip install "parakeet[metal]"` |
| `"cuda"` | an NVIDIA GPU | `pip install "parakeet[cuda12]"` (CuPy) |
| `"python"` | — | runs the original function uncompiled |

```python
parakeet.available_backends()
# {'python': 'ok', 'c': 'ok', 'threads': 'ok', 'gcd': 'ok', 'openmp': 'ok', 'metal': 'ok',
#  'cuda': 'unavailable: the CUDA backend needs CuPy, which is not installed. Install ...'}
```

A backend that cannot work fails **when you select it**, with a message saying
what is missing and how to install it — not later with a linker error.

## Multi-core

Array expressions, `map`/`allpairs`/`imap`, comprehensions, `parfor` and
reductions over recognised associative operations are split into chunks that
run on `parakeet.config.num_threads` threads (default: all cores). Only the
outermost data-parallel operation is parallelised; anything nested inside it
runs sequentially within each thread.

Small loops are not worth a thread hand-off: loops with a cheap body run on the
calling thread unless they have at least `2 * config.parallel_grain` iterations
(default 8192 per chunk). Loops whose body contains loops of its own are always
split.

Compiled functions release the GIL and are safe to call from several Python
threads at once. With `multiprocessing`, prefer the `spawn` start method when
using `"openmp"` or `"gcd"` (those runtimes do not survive `fork`); the default
thread pool does.

## Apple GPUs (Metal)

```python
@jit(backend="metal")
def blur_energy(img):
  return (img[1:-1, 1:-1] * 4 - img[:-2, 1:-1] - img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:]) ** 2
```

Each eligible parallel loop is also compiled to a Metal compute kernel; the host
program hands loops of at least `config.device_threshold` iterations (default
32768) to the GPU and runs smaller ones on the CPU.

!!! warning "Apple GPUs have no float64"
    Metal does not support double precision. A loop that touches `float64` data
    runs on the CPU thread pool instead, and Parakeet emits a `FallbackWarning`
    saying so. Pass `float32` arrays and `np.float32` scalars to stay on the GPU
    (a Python `float` argument is a double; `x * 2.0` with a *literal* is fine).

Other reasons a loop stays on the CPU: it allocates temporary arrays per
iteration, or it uses a function Metal lacks (`lgamma`, `erfc`). The warning
always names the reason. Control this with `config.device_fallback`:
`"warn"` (default), `"silent"`, or `"error"` to refuse instead.

Runtime errors inside kernels (out-of-bounds indices, failed assertions) are
reported as normal Python exceptions.

## NVIDIA GPUs (CUDA)

```python
@jit(backend="cuda")
def distances(X, C):
  return parakeet.allpairs(lambda a, b: np.sum((a - b) ** 2), X, C)
```

Kernels are compiled at run time by CuPy's `RawKernel` (NVRTC), so only the
NVIDIA driver and a CuPy wheel matching your CUDA version are needed. All dtypes
including `float64` are supported. The same per-loop fallback rules apply.

!!! note "Validation status"
    CI has no NVIDIA hardware, so there the CUDA code generator runs against a
    *simulator* (`PARAKEET_CUDA_SIMULATE=1`) that compiles the generated kernels
    as C++ and executes their thread grid on the CPU.

    On real hardware the backend was last checked on a Tesla T4 (Google Colab,
    driver 580, CuPy with CUDA 12, Python 3.13): the whole test suite with every
    parallel loop forced onto the GPU (595 tests, about 3,200 kernel launches),
    the examples and the benchmarks all passed. That check is run on demand,
    not by CI:

    ```bash
    pip install runplz google-colab-cli
    runplz colab tools/colab_job.py          # T4; PK_COLAB_GPU=A100 for others
    ```

    To push your own test suite through a GPU set `PARAKEET_TEST_BACKENDS=cuda`
    (the backends used by `parakeet.testing.expect` and `check`) and
    `PARAKEET_DEVICE_THRESHOLD=1` (so small arrays go to the device too).

### What about TPUs?

No. A TPU only runs programs compiled by XLA, which wants whole-array operations
with static shapes; Parakeet generates scalar loops in C-like kernel languages.
Parakeet's CPU backends do work on a Colab TPU host (24 x86 cores), which is all
`PK_COLAB_TPU=v5e1 runplz colab tools/colab_job.py` checks.

## How data moves

Arrays live in host memory. For each GPU loop Parakeet makes the arrays it uses
available to the device and brings back the ones it wrote:

* Metal: arrays allocated by compiled code are page-aligned and shared with the
  GPU without copying (Apple Silicon has unified memory); arrays that came from
  NumPy are copied in, and copied back if written.
* CUDA: arrays are copied to the device and written arrays copied back.

This keeps semantics simple and exact, but it means a GPU only pays off when the
loop does substantially more arithmetic than memory traffic. A program that runs
many cheap loops over the same large array (an explicit stencil iterated a
hundred times, say) is usually faster on the multi-core backend; see
[Performance](performance.md).
