# Getting started

## Requirements

* Python 3.11 or newer and NumPy 2.
* A C compiler: the Xcode command line tools on macOS (`xcode-select --install`),
  `gcc` or `clang` on Linux, MinGW-w64 `gcc` or LLVM `clang` on Windows.

Install the release in your Python environment. For development, run `python -m pip install .` in a checkout.

```bash
pip install "parakeet==1.0.0"
python -c "import parakeet; print(parakeet.available_backends())"
```

`available_backends()` reports, for every backend, either `"ok"` or the reason it
cannot be used on this machine and how to fix that.

## Your first compiled function

```python
import numpy as np
from parakeet import jit


@jit
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

`jit` also works without the decorator syntax, including on lambdas and on NumPy
functions:

```python
fast_rms = jit(lambda x: np.sqrt(np.mean(x * x)))
parakeet.run(rms.fn, x)  # compile-and-call in one step
```

## Loops, comprehensions, helpers

Anything in the [supported subset](language.md) can be mixed freely. Helper
functions are inlined automatically — they do not need their own `@jit`.

```python
def sqr_dist(a, b):
  return np.sum((a - b) ** 2)


@jit
def nearest(X, centroids):
  return np.array([np.argmin([sqr_dist(x, c) for c in centroids]) for x in X])
```

## Picking a backend

```python
@jit(backend="metal")  # per function
def f(x): ...


f(x, _backend="c")  # per call
parakeet.config.backend = "threads"  # process-wide
with parakeet.options(backend="c", num_threads=1):  # temporarily
  f(x)
```

See [Backends and GPUs](backends.md).

## Looking inside

```python
spec = rms.specialize(x)  # compile without running
print(spec.ir)  # the typed, fused loop IR
print(spec.source)  # the generated C
```

```text
def rms(x_1: array[float64, 1d]):
    nchunks_9: int64? = @num_chunks(x_n0_3, True)
    partials_10 = alloc(float64, nchunks_9)
    parallel[reduce] for chunk_11, (start_12, stop_13) in chunks(x_n0_3, n=nchunks_9):
        x_15: float64 = (x_1[(x_off_2 + start_12)] * x_1[(x_off_2 + start_12)])
        acc_14: float64 = x_15
        for i_16 in range((start_12 + 1), stop_13, 1):  # simd
            x_17: float64 = (x_1[(x_off_2 + i_16)] * x_1[(x_off_2 + i_16)])
            acc_14: float64 = (acc_14 + x_17)
        partials_10[chunk_11] = acc_14
    acc_8: float64 = 0.0
    for i_18 in range(0, nchunks_9, 1):
        acc_8: float64 = (acc_8 + partials_10[i_18])
    red_19: float64 = acc_8
    res_20: float64 = sqrt((red_19 / float64(x_n0_3)))
    return res_20
```

No temporary for `x * x` exists: the multiplication was fused into the parallel
reduction.

## When something is not supported

You get a `CompileError` pointing at your code, not a crash:

```text
parakeet.errors.UnsupportedError: numpy.fft.fft is not supported in compiled code
  File "demo.py", line 6, in spectrum
    return np.fft.fft(x)
           ^^^^^^^^^^^^^
```

Run the function uncompiled at any time with `backend="python"` to tell a bug in
your code from a limitation of the compiler. More in
[Errors and debugging](debugging.md).
