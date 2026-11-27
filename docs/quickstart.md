# Getting started

## Requirements and install

* Python 3.11+ and NumPy 2
* A system C compiler: Xcode command line tools on macOS
  (`xcode-select --install`), `gcc` or `clang` on Linux, MinGW-w64 or LLVM on
  Windows

```bash
pip install parakeet
python -c "import parakeet; print(parakeet.available_backends())"
```

`available_backends()` reports `"ok"` for each usable backend or explains what
is missing and how to install it. For GPUs, install `parakeet[metal]` on Apple
Silicon or `parakeet[cuda12]` with an NVIDIA GPU.

## Your first compiled function

Decorate any numerical function with `@par.jit`:

```python
import numpy as np
import parakeet as par


@par.jit
def rms(x):
  return np.sqrt(np.mean(x * x))


x = np.random.rand(1_000_000)
rms(x)  # first call: compiles for (float64, 1-d), then runs
rms(x)  # later calls: straight into cached native code
rms(x.astype(np.float32))  # compiles a separate float32 specialization
```

Compilation takes a few hundred milliseconds per specialization (mostly in the
C compiler) and is cached on disk across runs. `par.jit` also wraps lambdas or
plain functions directly, and `par.run(fn, *args)` compiles and calls in one
step:

```python
fast_rms = par.jit(lambda x: np.sqrt(np.mean(x * x)))
par.run(rms.fn, x)
```

## Loops, comprehensions, and helpers

Anything in the [compiled language](language.md) mixes freely. Helper functions
are inlined automatically—they do not need `@par.jit`:

```python
def sqr_dist(a, b):
  return np.sum((a - b) ** 2)


@par.jit
def nearest(X, centroids):
  return np.array([np.argmin([sqr_dist(x, c) for c in centroids]) for x in X])
```

When you want explicit data-parallel control over slices, pairs, scans, or
histograms, use [`par.map`](operators.md#map), [`par.reduce`](operators.md#reduce),
[`par.fold`](operators.md#fold), [`par.scan`](operators.md#scan),
[`par.allpairs`](operators.md#allpairs), or
[`par.scatter_add`](operators.md#scatter_add).

## Gradients in one line

Wrap any scalar-valued function with `par.grad` or `par.value_and_grad`. The
forward and backward passes compile together into one native function:

```python
def mse_loss(w, x, y):
  return np.mean((np.tanh(x @ w) - y) ** 2)


loss_val, dw = par.value_and_grad(mse_loss)(w, x, y)
```

See [Automatic differentiation](autodiff.md) for nested tuples, `@dataclass`
parameters, `par.jvp`, `par.hvp`, and custom rules.

## Picking a backend

By default (`backend="auto"`), Parakeet parallelizes across all CPU cores.
Switch backends per function, per call, globally, or for a block:

```python
@par.jit(backend="metal")  # per function ("cuda", "gpu", "c", "threads", ...)
def f(x): ...


f(x, _backend="c")  # single call
par.config.backend = "threads"  # process-wide
with par.options(backend="c", num_threads=1):  # scoped block
  f(x)
```

See [Backends and GPUs](backends.md) for Metal, CUDA, and keeping arrays
resident on device with `par.to_device`.

## Errors and debugging

Anything outside the supported subset fails at compile time with your exact
source line:

```text
parakeet.errors.UnsupportedError: numpy.fft.fft is not supported in compiled code
  File "demo.py", line 6, in spectrum
    return np.fft.fft(x)
           ^^^^^^^^^^^^^
```

Run any function uncompiled with `f(x, _backend="python")` to check whether an
issue is in your algorithm or the compiler. To inspect the generated C or GPU
kernels, call `rms.specialize(x).source` (see
[Errors and debugging](debugging.md)).

## Where to go next

* [The compiled language](language.md) — supported types, control flow, and
  differences from CPython.
* [Data-parallel operators](operators.md) — `map`, `fold`, `scan`, `scatter_add`,
  and `wavefront`.
* [Kernel gallery](gallery.md) — runnable examples from nearest-neighbor search
  to attention and nanoGPT.
* [Operators & compilation walkthrough](primitives.md) — how Parakeet fuses and
  tiles your code under the hood.
