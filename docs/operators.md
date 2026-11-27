# Data-parallel operators

Parakeet exposes one unified operator surface on the top-level `parakeet`
module. Every operator below works **both** inside `@parakeet.jit` (where it
fuses into surrounding NumPy code) **and** directly from ordinary Python
(automatically JIT-compiling any callable arguments):

```python
import numpy as np
import parakeet as par
```

| Operator | Meaning |
|---|---|
| [`par.map(f, x, y, ..., axis=0)`](#map) | `f` applied to corresponding slices along `axis`; `axis=None` applies it to every scalar element |
| [`par.allpairs(f, x, y, axis=0)`](#allpairs) | `result[i, j] = f(x[i], y[j])` |
| [`par.outer_map(f, x, y, z, ...)`](#outer_map) | the n-ary generalisation of `allpairs` |
| [`par.reduce(f, x, ..., init=None, axis=None, map=None)`](#reduce) | fold `f` over all elements, or over slices along `axis`; with `map=g` and several arrays, over `g(x[i], y[i], ...)` |
| [`par.fold(lift, merge, x, init=None, *, axis=None, indexed=False, emit=None)`](#fold) | associative structured fold over tuple / `NamedTuple` states (tiles pairwise folds automatically) |
| [`par.scan(f, x, ..., init=None, axis=0, map=None, emit=None)`](#scan) | running fold (a generalised `cumsum`); `emit=h` gives `h` of each running value |
| [`par.imap(f, shape)`](#imap) | `result[i, j, ...] = f(i, j, ...)` for every index of `shape` (unpacks multi-axis indices automatically; 1-arg tuple `lambda ij:` also works) |
| [`par.ireduce(f, shape, init=None, combine=np.add)`](#ireduce) | reduce `f(i, ...)` over every index of `shape` |
| [`par.parfor(f, shape)`](#parfor) | call `f(i, ...)` for its side effects, in parallel |
| [`par.filter(pred, x, ..., map=None)`](#filter) | the elements of `x` for which `pred` is true; with several arrays, `map(x[i], y[i], ...)` where `pred(x[i], y[i], ...)` holds |
| [`par.filter_reduce(f, pred, x, ..., map=None, init=None, axis=None)`](#filter_reduce) | reduce with `f` over elements where `pred` holds (fuses into one zero-allocation guarded reduction loop when `init` is given) |
| [`par.flatmap(count_fn, emit_fn, *xs)`](#flatmap) / [`par.filter_pairs(...)`](#filter_pairs) | two-pass 1-to-K ragged compaction ([`IndexExpand`](primitives.md#filter-and-expand)) |
| [`par.scatter_add(n, idx, vals, where=None)`](#scatter_add) | index-directed associative scatter-add ([`IndexScatter`](primitives.md#scatter)) with thread-privatized parallel lowering and exact gather adjoint |
| [`par.wavefront(...)`](#wavefront) / [`par.batched_wavefront(...)`](#batched_wavefront) / [`par.soft_align(...)`](#soft_align) | 2D and banded semiring dynamic programming ([`IndexWavefront`](primitives.md#wavefront)) with checkpointed VJP |
| `par.partition_by_size(offsets, ...)` / `par.segmented_parfor(body, offsets, ...)` / `par.segmented_histogram2d(...)` / `par.ragged_partition(...)` | adaptive 3-tier ragged dispatch (heavy/medium/light tiers for skewed workloads), single-pass fused 2D `(grad, hess, count)` histograms, and parallel CSR sample-index partitioning |
| `par.split_bytes(buf, sep=10, skip_empty=False)` / `par.parse_int64` / `par.parse_float32` / `par.parse_float64` | zero-`PyObject` delimiter splitting into `RaggedArray[uint8]` and compiled ASCII integer/float parsing over `uint8` slices or `RaggedArray[uint8]` |
| `par.fixed_list(capacity, dtype=np.float32)` | fixed-capacity, zero-`malloc` thread-local scratch list (`.append`, `.pop`, `.clear`, indexing, `len`) for stacks and per-thread buffers inside `par.parfor` / `par.map` on CPU, CUDA, and Metal |
| `par.grad`, `par.value_and_grad`, `par.jvp`, `par.vjp`, `par.hvp`, `par.linearize`, `par.linear_transpose`, `par.jacfwd`, `par.jacrev`, `par.hessian` | composable reverse-mode, forward-mode, and higher-order AD; see [Automatic differentiation](autodiff.md) |
| `par.show_rewrite(f, *args)` | inspect which algebraic optimizations rewrote `f` and view before/after IR; see [Algebraic laws](algebra.md) |

---

## Operator reference

### `par.map(f, *args, axis=0)` {#map}

Apply `f` in parallel to corresponding slices along `axis` (or to every scalar
element with broadcasting when `axis=None`). Lowers to
[`IndexMap`](primitives.md#map). Non-array arguments are passed to every call
unchanged, and `f` may return a scalar, tuple, or array (results are stacked).

```python
par.map(lambda row: row / np.sum(row), X)  # normalise each row of X
par.map(lambda a, b: a * a + b, x, y, axis=None)  # element-wise with broadcasting
```

### `par.allpairs(f, x, y, axis=0)` {#allpairs}

Evaluate `f(x[i], y[j])` for every pair of slices `(i, j)` along `axis`. Lowers
to [`IndexMap`](primitives.md#map) and fuses directly into downstream reductions
such as [`par.fold`](#fold) or `np.argmin` without materializing the full
`N × M` score matrix.

```python
par.allpairs(lambda a, b: np.sum((a - b) ** np.float32(2)), X, C)
```

### `par.outer_map(f, *args)` {#outer_map}

The n-ary generalisation of [`par.allpairs`](#allpairs), evaluating `f` on the
Cartesian product of the leading axes of two or more arrays. Lowers to
[`IndexMap`](primitives.md#map).

### `par.reduce(f, *args, init=None, axis=None, map=None)` {#reduce}

Reduce elements with binary combiner `f` (such as `np.add`, `np.maximum`, or a
custom function), over all elements (`axis=None`) or along a specific axis.
Passing `map=g` fuses a map step into the reduction with zero intermediate
allocation. Lowers to [`IndexReduce`](primitives.md#reduce-and-fold).

```python
par.reduce(np.maximum, X, axis=0)  # column maxima
par.reduce(np.add, x, y, map=np.multiply)  # dot product, no temporary array
```

### `par.fold(lift, merge, x, init=None, *, axis=None, indexed=False, emit=None)` {#fold}

Associative structured reduction over multi-field tuple or `NamedTuple` states.
`lift` maps each element (or `(value, index)` when `indexed=True`) to a state,
`merge(left, right)` combines two adjacent states in input order (preserving
exact left-to-right tie and NaN semantics), and `emit` optionally projects the
final state. Lowers to [`IndexReduce`](primitives.md#reduce-and-fold) with
`Carry` fields, and triggers tiled pairwise folds on both CPU and GPU when
composed with [`par.allpairs`](#allpairs).

```python
# Exact argmin carrying (best_dist, best_idx) with first-index tie breaking
par.fold(
  lambda v, i: (v, i),
  lambda a, b: a if a[0] <= b[0] else b,
  scores,
  axis=1,
  indexed=True,
  emit=lambda state: state[1],
)
```

### `par.scan(f, *args, init=None, axis=0, map=None, emit=None)` {#scan}

Running prefix fold along `axis` (a generalisation of `np.cumsum` and
`np.cumprod`). Associative combiners lower to a 3-stage parallel prefix scan,
and affine/bilinear recurrences trigger
[Law 4 chunkwise duality](algebra.md#law-4-chunkwise-state-space-duality).
Lowers to [`IndexScan`](primitives.md#scan).

```python
par.scan(np.add, x, y, map=lambda a, b: a - b)  # running sum of differences
par.scan(lambda acc, v: 0.9 * acc + 0.1 * v, x)  # exponential moving average
```

### `par.imap(f, shape)` {#imap}

Construct an array of `shape` where entry `(i, j, ...)` equals `f(i, j, ...)`.
Multi-axis coordinates are unpacked automatically when `f` accepts `len(shape)`
arguments (a single tuple parameter `lambda ij:` also works). Lowers to
[`IndexMap`](primitives.md#map).

```python
par.imap(lambda i, j: i >= j, (T, T))  # causal mask (fuses without allocating)
```

### `par.ireduce(f, shape, init=None, combine=np.add)` {#ireduce}

Reduce `f(i, ...)` over every coordinate of `shape` using `combine`, without
constructing an intermediate array first. Lowers to
[`IndexReduce`](primitives.md#reduce-and-fold).

```python
par.ireduce(lambda r: a[i, r] + b[r, j], k, combine=np.minimum)
```

### `par.parfor(f, shape)` {#parfor}

Execute `f(i, ...)` in parallel across every coordinate of `shape` for its side
effects (such as writing disjoint output slices or updating thread-local
`par.fixed_list` scratch buffers). For pure array outputs, prefer NumPy
expressions, comprehensions, [`par.map`](#map), or [`par.imap`](#imap).

### `par.filter(pred, *args, map=None)` {#filter}

Return the elements where `pred` holds (optionally transforming kept elements
with `map`). Equivalent to NumPy boolean masking `x[pred(x)]`. Lowers to
[`IndexFilter`](primitives.md#filter-and-expand).

```python
par.filter(lambda a, b: a > b, x, y, map=np.add)  # x + y where x > y
```

### `par.filter_reduce(f, pred, *args, map=None, init=None, axis=None)` {#filter_reduce}

Reduce with `f` over elements where `pred` holds. When `init` is provided,
fuses into a single zero-allocation guarded reduction loop (also triggered
automatically by `np.sum(x[x > 0])`). Lowers to
[`IndexFilter`](primitives.md#filter-and-expand) +
[`IndexReduce`](primitives.md#reduce-and-fold).

### `par.flatmap(count_fn, emit_fn, *xs)` {#flatmap}

Two-pass 1-to-K ragged expansion returning a `RaggedArray(values, offsets)`.
For each input index `i`, `count_fn(x0[i], ...)` returns the count `c_i >= 0`
of items to emit, and `emit_fn(x0[i], ..., k)` emits the `k`-th item
(`0 <= k < c_i`) in parallel after a prefix sum of counts. Lowers to
[`IndexExpand`](primitives.md#filter-and-expand).

```python
# Expand run-length-encoded CIGAR operations into a flat RaggedArray
par.flatmap(lambda op, n: n, lambda op, n, k: op, ops, lengths)
```

### `par.filter_pairs(pred, x, y, emit=None)` {#filter_pairs}

Compact all pairs `(i, j)` where `pred(x[i], y[j])` holds into a
`RaggedArray` of matching pairs (or `emit(x[i], y[j])` payloads) using two-pass
[`IndexExpand`](primitives.md#filter-and-expand) without allocating a dense
`N × M` mask.

### `par.scatter_add(n, idx, vals, where=None)` {#scatter_add}

Accumulate `vals[i]` into output bin `idx[i]` of length `n` (optionally guarded
by boolean mask `where`, or passed as a generator `((k, v) for ...)`). Lowers
to [`IndexScatter`](primitives.md#scatter) with thread-privatized histograms on
CPU, parallel atomics on GPU, sibling histogram fusion, and an exact gather
adjoint under `par.grad`. Also targeted automatically by `np.bincount`,
`np.add.at`, and canonical `out[idx[i]] += v` loops.

```python
grad_hist = par.scatter_add(256, bin_ids, grads)
```

### `par.wavefront(step_fn, shape, init_top, init_left, *, init_corner=None, band=None, track_max=False)` {#wavefront}

Execute a 2D or diagonally banded dynamic programming recurrence along parallel
anti-diagonals in `O(M)` rolling memory (never allocating the full `M × N`
table). `step_fn(top, left, diag, i, j)` returns the next scalar or tuple /
`NamedTuple` state (e.g., 3-state Gotoh affine-gap alignment). Lowers to
[`IndexWavefront`](primitives.md#wavefront).

### `par.batched_wavefront(...)` {#batched_wavefront}

Run [`par.wavefront`](#wavefront) independently across a batch of sequence or
grid pairs in parallel, with one thread/lane per batch item on CPU or GPU.

### `par.soft_align(scores, gamma=1.0, gap=-1.0, band=None)` {#soft_align}

Differentiable soft-maximum global/banded sequence alignment over a `scores`
matrix, lowered to [`IndexWavefront`](primitives.md#wavefront) with a
checkpointed reverse-mode VJP under `par.grad` / `par.value_and_grad`.

---

## Functions that return arrays

The function given to [`par.map`](#map) may return an array; the results are
stacked.

```python
par.map(lambda row: np.cumsum(row), X)  # shape (n, m)
```

All results must have the same shape; this is checked at run time
(`ValueError`).

## Parallelism and associativity

`map`-like operators evaluate elements in parallel, in no particular order.
[`par.reduce`](#reduce) runs in parallel when the combining function is
recognisably one of `+`, `*`, `min`, `max`, `and`, `or`, `&`, `|`, `^`
(whether written as `np.add`, `lambda a, b: a + b`, or `par.add`), and
sequentially otherwise; so a non-associative function such as
`lambda acc, x: acc * 0.5 + x` still gives the same answer as
`functools.reduce`. Floating-point sums are reassociated, like NumPy's own
pairwise summation: expect differences in the last digits.

## Primitive combiner functions

Standard NumPy ufuncs (`np.add`, `np.subtract`, `np.multiply`, `np.divide`,
`np.minimum`, `np.maximum`) are the idiomatic combiners to pass to
[`par.reduce`](#reduce), [`par.ireduce`](#ireduce), [`par.scan`](#scan), and
[`par.filter`](#filter). Parakeet also exports matching aliases (`par.add`,
`par.minimum`, etc.) for backward compatibility, and `parakeet.dsl` remains
available as a compatibility alias alongside compact dtype shorthands
(`dsl.f32`, `dsl.i64`).

## Structured reductions and tiled pairwise folds

[`par.fold`](#fold) lifts each axis-0 slice of `x` to a nonempty tuple or
`NamedTuple` of scalar fields. The pure `merge(left, right)` must be
associative and return the same field types. Chunks may regroup adjacent values
while preserving input order. The initial state is applied once and is the
result for empty input; with no seed, evaluating an empty lane raises
`ValueError`.

On both CPU and GPU, a map over queries consuming an associative structured
fold over references whose inputs come from a feature reduction compiles to a
tiled pairwise fold. For example, both the gallery's nested-map 1-NN and the
following NumPy expression compile to the same tiled computation:

```python
@par.jit
def nearest(x, y):
  return np.argmin(np.sum((x[:, None, :] - y[None, :, :]) ** np.float32(2), axis=2), axis=1)
```

The compiler evaluates a bounded rectangle of scores, folds each row, and
merges consecutive tile summaries in reference order. Each summary carries the
global reference index, so ties and NaNs retain their first-index semantics
across tiles. The rewrite inspector (`par.show_rewrite`) reports
`Pairwise map and ordered tile-summary fold` when it applies.
