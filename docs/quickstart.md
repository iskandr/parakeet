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

## Four styles, one compiler

Parakeet compiles NumPy broadcasting (`_numpy`), comprehensions (`_comp`),
explicit loops (`_loop`), and data-parallel operators (`_par`) through the same
optimizer. Helper functions inline automatically—they do not need `@par.jit`:

=== "NumPy (`_numpy`)"

    ```python
    --8 < --"examples/algebraic_gallery.py:nearest_numpy"
    ```

=== "Comprehension (`_comp`)"

    ```python
    --8 < --"examples/algebraic_gallery.py:squared_distance"

    --8 < --"examples/algebraic_gallery.py:nearest_comp"
    ```

=== "Loop (`_loop`)"

    ```python
    --8 < --"examples/algebraic_gallery.py:squared_distance"

    --8 < --"examples/algebraic_gallery.py:nearest_loop"
    ```

=== "Operators (`_par`)"

    ```python
    --8 < --"examples/algebraic_gallery.py:squared_distance"

    --8 < --"examples/algebraic_gallery.py:nearest_par"
    ```

See the [Kernel gallery](gallery.md) for side-by-side comparisons across
smoothing, histograms, tropical products, scans, attention, and cross-entropy.

## Gradients, modules, and optimizers

Wrap any scalar-valued function with `par.grad` or `par.value_and_grad`. You
can differentiate plain NumPy functions, dicts/tuples, `@dataclass` records, or
ordinary Python classes inheriting from `par.nn.Module`:

```python
class TwoLayerMLP(par.nn.Module):
  def __init__(self, width, hidden, rng):
    self.fc1 = par.nn.Linear(width, hidden, rng=rng)
    self.fc2 = par.nn.Linear(hidden, width, rng=rng)

  def __call__(self, x):
    return self.fc2(np.tanh(self.fc1(x)))


def mse_loss(model, x, y):
  return np.mean((model(x) - y) ** 2)


loss_and_grad = par.value_and_grad(mse_loss)
opt = par.optim.Adam(model, lr=1e-3)
loss_val, grads = loss_and_grad(model, x, y)
model = opt.update(model, grads)
```

See [Automatic differentiation](autodiff.md) and [Train nanoGPT](gallery/nanogpt.md)
for both helper-based and from-scratch functional training loops.

## Sequences, alphabets, and tokenization

To pass character sequences (DNA reads, protein chains, text corpora) into
compiled code without `IntEnum` or dict boilerplate, use `par.Alphabet` (fixed
categorical alphabets) or `par.Vocab` (corpus-derived tokenizers). Both compile
down to a 256-entry lookup table:

```python
DNA = par.Alphabet("ACGT")
GC = par.Alphabet.mask("GC")
codes = DNA.encode("ACGTGCGT")  # array([0, 1, 2, 3, 2, 1, 2, 3], dtype=uint8)

vocab = par.Vocab.from_text("parakeet compiles numerical python.")
tokens = vocab.encode("parakeet")
```

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

* [The compiled language](language.md) — supported types, `par.nn.Module`,
  `par.Alphabet`/`par.Vocab`, and differences from CPython.
* [Data-parallel operators](operators.md) — `map`, `fold`, `scan`, `scatter_add`,
  and `wavefront`.
* [Kernel gallery](gallery.md) — runnable four-flavor examples from nearest
  neighbor to attention and nanoGPT.
* [Operators & compilation walkthrough](primitives.md) — how Parakeet fuses and
  tiles your code under the hood.
