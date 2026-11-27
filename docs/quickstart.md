# Getting started

## Requirements

* Python 3.11 or newer and NumPy 2.
* A C compiler: the Xcode command line tools on macOS (`xcode-select --install`),
  `gcc` or `clang` on Linux, MinGW-w64 `gcc` or LLVM `clang` on Windows.

Install Parakeet in your environment (or `python -m pip install .` in a local checkout):

```bash
pip install parakeet
python -c "import parakeet; print(parakeet.available_backends())"
```

`available_backends()` reports, for every backend, either `"ok"` or the reason
it cannot be used on this machine and how to fix it.

## Your first compiled function

```python
import numpy as np
import parakeet as par


@par.jit
def rms(x):
  return np.sqrt(np.mean(x * x))


x = np.random.rand(1_000_000)
rms(x)  # first call: compiles for (float64 array, 1-d), then runs
rms(x)  # later calls: straight into native code
rms(x.astype(np.float32))  # a new specialization, compiled on demand
```

Compilation takes a few hundred milliseconds per specialization (most of it in
the C compiler) and is cached on disk, so a second run of your program starts
fast.

`par.jit` also works without decorator syntax, including on lambdas and NumPy
functions:

```python
fast_rms = par.jit(lambda x: np.sqrt(np.mean(x * x)))
par.run(rms.fn, x)  # compile-and-call in one step
```

## Loops, comprehensions, helpers

Anything in the [supported subset](language.md) can be mixed freely. Helper
functions are inlined automatically; they do not need their own `@par.jit`.

```python
def sqr_dist(a, b):
  return np.sum((a - b) ** 2)


@par.jit
def nearest(X, centroids):
  return np.array([np.argmin([sqr_dist(x, c) for c in centroids]) for x in X])
```

You can also call Parakeet's data-parallel operators directly (`par.map`,
`par.reduce`, `par.scan`, `par.allpairs`, `par.scatter_add`, `par.fold`) when
you want to apply a Python function over slices or pairs in parallel. See
[Data-parallel operators](operators.md).

## Gradients in one line

Wrap any scalar-valued function with `par.grad` or `par.value_and_grad`.
The forward and backward passes compile together into one native function:

```python
def mse_loss(w, x, y):
  return np.mean((np.tanh(x @ w) - y) ** 2)


loss_val, dw = par.value_and_grad(mse_loss)(w, x, y)
```

See [Gradients](autodiff.md) for nested parameter tuples, `NamedTuple`s,
dataclasses, and training loops.

## Picking a backend

```python
@par.jit(backend="metal")  # per function
def f(x): ...


f(x, _backend="c")  # per call
par.config.backend = "threads"  # process-wide
with par.options(backend="c", num_threads=1):  # temporarily
  f(x)
```

See [Backends and GPUs](backends.md) for multi-core tuning, Metal, CUDA, and
keeping arrays resident on a GPU with `parakeet.to_device`.

## Looking inside

Inspect what Parakeet fused and what native code it generated at any time:

```python
print(par.show_rewrite(rms, x).summary())  # algebraic laws and operator counts

spec = rms.specialize(x)  # compile without running
print(spec.source)  # generated C
```

For `rms(x)`, `x * x` fuses directly into the chunked parallel reduction with no
temporary array:

```python
# Pseudocode of the fused parallel reduction for rms(x):
partials = parallel_map_chunks(x, lambda chunk: sum(xi * xi for xi in chunk))
return sqrt(sum(partials) / len(x))
```

To inspect the raw typed compiler IR (`spec.operators` and lowered `spec.ir`),
see [Compiler architecture: Lowered IR](internals.md#lowered-ir).

## When something is not supported

You get a `CompileError` pointing at your code, not a crash:

```text
parakeet.errors.UnsupportedError: numpy.fft.fft is not supported in compiled code
  File "demo.py", line 6, in spectrum
    return np.fft.fft(x)
           ^^^^^^^^^^^^^
```

Run the function uncompiled at any time with `backend="python"` to separate a
bug in your code from a limitation of the compiler. More in
[Errors and debugging](debugging.md).

## Where to go next

* [Kernel gallery](gallery.md): runnable examples ranging from nearest-neighbor
  search and tree splits to attention, state-space recurrences, and classic
  numerical kernels.
* [The compiled language](language.md): exact types, statements, and where
  Parakeet differs from CPython.
* [Ideas, history, and boundaries](philosophy.md): design principles, project
  history, and current limits.
* [Performance](performance.md) and [Release benchmarks](release-benchmarks.md):
  when Parakeet wins, when it doesn't, and measured numbers on CPU and GPU.
