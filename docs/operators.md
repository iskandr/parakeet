# Data-parallel operators

Parakeet exposes one unified operator surface on the top-level `parakeet`
module. Every operator below works **both** inside `@parakeet.jit` (where it
fuses into surrounding NumPy code) **and** directly from ordinary Python
(automatically JIT-compiling any callable arguments):

```python
import numpy as np
import parakeet as pk
```

| Operator | Meaning |
|---|---|
| `pk.map(f, x, y, ..., axis=0)` | `f` applied to corresponding slices along `axis`; `axis=None` applies it to every scalar element |
| `pk.allpairs(f, x, y, axis=0)` | `result[i, j] = f(x[i], y[j])` |
| `pk.outer_map(f, x, y, z, ...)` | the n-ary generalisation of `allpairs` |
| `pk.reduce(f, x, ..., init=None, axis=None, map=None)` | fold `f` over all elements, or over slices along `axis`; with `map=g` and several arrays, over `g(x[i], y[i], ...)` |
| `pk.fold(lift, merge, x, init=None, *, axis=None, indexed=False, emit=None)` | associative structured fold over tuple / `NamedTuple` states (tiles pairwise folds automatically) |
| `pk.scan(f, x, ..., init=None, axis=0, map=None, emit=None)` | running fold (a generalised `cumsum`); `emit=h` gives `h` of each running value |
| `pk.imap(f, shape)` | `result[i, j, ...] = f(i, j, ...)` for every index of `shape` (unpacks multi-axis indices automatically; 1-arg tuple `lambda ij:` also works) |
| `pk.ireduce(f, shape, init=None, combine=pk.add)` | reduce `f(i, ...)` over every index of `shape` |
| `pk.parfor(f, shape)` | call `f(i, ...)` for its side effects, in parallel |
| `pk.filter(pred, x, ..., map=None)` | the elements of `x` for which `pred` is true; with several arrays, `map(x[i], y[i], ...)` where `pred(x[i], y[i], ...)` holds |
| `pk.filter_reduce(f, pred, x, ..., map=None, init=None, axis=None)` | reduce with `f` over elements where `pred` holds (fuses into one zero-allocation guarded reduction loop when `init` is given) |
| `pk.flatmap(count_fn, emit_fn, n)` / `pk.filter_pairs(...)` | two-pass 1-to-K ragged compaction (`IndexExpand`) |
| `pk.scatter_add(n, idx, vals)` (alias `pk.scatter`) | index-directed associative scatter-add (`IndexScatter`) with thread-privatized parallel lowering and exact gather adjoint |
| `pk.wavefront(...)` / `pk.batched_wavefront(...)` / `pk.soft_align(...)` | 2D and banded semiring dynamic programming (`IndexWavefront`) with checkpointed VJP |
| `pk.grad(f, argnum=0)`, `pk.value_and_grad(f, argnum=0)` | reverse-mode gradient of a scalar-valued `f` — see [Gradients](autodiff.md) |
| `pk.show_rewrite(f, *args)` | inspect which of Parakeet's [six algebraic laws](algebra.md) rewrote `f` and view before/after IR |

```python
pk.map(lambda row: row / np.sum(row), X)  # normalise rows
pk.allpairs(lambda a, b: np.sum((a - b) ** np.float32(2)), X, C)  # distance matrix
pk.reduce(pk.maximum, X, axis=0)  # column maxima
pk.imap(lambda i, j: i * 10 + j, (3, 4))  # an index-defined 2D array
```

```python
pk.reduce(pk.add, x, y, map=pk.multiply)  # a dot product; no array of products is built
pk.scan(pk.add, x, y, map=lambda a, b: a - b)  # running sum of differences
pk.filter(lambda a, b: a > b, x, y, map=pk.add)  # x + y where x > y
```

For multi-dimensional shapes, `imap`, `ireduce`, and `parfor` unpack coordinates
directly when the function takes `len(shape)` parameters (`lambda i, j: ...`),
and still accept a single index tuple (`lambda ij: ...`) for backward
compatibility. `parakeet.dsl` remains available as a compatibility alias
re-exporting these same operators alongside compact dtype shorthands
(`dsl.f32`, `dsl.i64`).

## Functions that return arrays

The function given to `map` may return an array; the results are stacked.

```python
pk.map(lambda row: np.cumsum(row), X)  # shape (n, m)
```

All results must have the same shape — this is checked at run time
(`ValueError`).

## Parallelism and associativity

`map`-like operators evaluate elements in parallel, in no particular order.
`reduce` runs in parallel when the combining function is recognisably one of
`+`, `*`, `min`, `max`, `and`, `or`, `&`, `|`, `^` (whether written as
`pk.add`, `np.add` or `lambda a, b: a + b`), and sequentially otherwise — so a
non-associative function such as `lambda acc, x: acc * 0.5 + x` still gives the
same answer as `functools.reduce`. Floating-point sums are reassociated, like
NumPy's own pairwise summation: expect differences in the last digits.

## Primitive functions

`pk.add`, `pk.subtract`, `pk.multiply`, `pk.divide`, `pk.minimum`, `pk.maximum`,
the comparison and logical/bitwise operators are first-class functions for use
with the operators above. Outside compiled code they behave like the NumPy
ufuncs of the same name.

### Structured reductions

`pk.fold(lift, merge, x, init)` lifts each axis-0 slice of `x` to a
nonempty tuple or named tuple of scalar fields. The pure `merge(left, right)`
must be associative and return the same field types. Chunks may regroup adjacent
values while preserving input order. The initial state is applied once, and is
the result for empty input. Read any field of the result in compiled code.

For example, affine updates `s -> a*s + b` compose in order with
`merge(left, right) = (right[0]*left[0], right[0]*left[1] + right[1])`.
Custom folds support forward evaluation. The initial state is optional. With no
seed, the first lifted element starts the fold and evaluating an empty lane
raises `ValueError`.

Use `axis=k` to fold the scalar values along each lane of a multidimensional
array. Pass `emit=project` to return a scalar per lane. With `indexed=True`,
`lift(value, index)` receives the zero-based position on that axis. The
projection runs after the final merge, including on the seed for an empty lane.
The [gallery](gallery.md#exact-nearest-neighbor) composes this operation with
`allpairs` and uses the same score with two different merge rules.

### Tiled pairwise folds

On CPU, a map over queries can consume an associative structured fold over
references whose inputs come from a feature reduction. For example, both the
gallery's nested-map 1-NN and the following NumPy expression compile to the
same tiled computation:

```python
@pk.jit
def nearest(x, y):
  return np.argmin(np.sum((x[:, None, :] - y[None, :, :]) ** np.float32(2), axis=2), axis=1)
```

The compiler evaluates a bounded rectangle of scores, folds each row, and
merges consecutive tile summaries in reference order. Each summary carries the
global reference index, so ties and NaNs retain their first-index semantics
across tiles. The same rewrite handles other pure score reductions, including
a sum of absolute differences followed by argmax. The rewrite inspector reports
`Pairwise map and ordered tile-summary fold` when it applies.
