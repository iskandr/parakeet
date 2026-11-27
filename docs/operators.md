# Data-parallel operators

Parakeet exposes one unified set of data-parallel operators on the top-level
`parakeet` module. Every operator works **both** inside `@par.jit` (fusing with
surrounding NumPy code) **and** directly from ordinary Python (automatically
compiling callable arguments):

```python
import numpy as np
import parakeet as par
```

## Choosing an operator

| Goal | Operator(s) |
|---|---|
| Transform slices or elements in parallel | [`par.map`](#map), [`par.imap`](#imap) |
| Pairwise or Cartesian-product operations | [`par.allpairs`](#allpairs), [`par.outer_map`](#outer_map) |
| Reduce an array or index grid | [`par.reduce`](#reduce), [`par.ireduce`](#ireduce), [`par.fold`](#fold) |
| Running prefix sum or recurrence | [`par.scan`](#scan) |
| Boolean compaction or 1-to-K ragged expansion | [`par.filter`](#filter), [`par.filter_reduce`](#filter_reduce), [`par.flatmap`](#flatmap), [`par.filter_pairs`](#filter_pairs) |
| Histograms and indexed accumulation | [`par.scatter_add`](#scatter_add) |
| 2D / banded dynamic programming | [`par.wavefront`](#wavefront), [`par.batched_wavefront`](#batched_wavefront), [`par.soft_align`](#soft_align) |
| Side-effecting parallel loops | [`par.parfor`](#parfor) |

For differentiation (`par.grad`, `par.jvp`, `par.hvp`), see
[Automatic differentiation](autodiff.md). To see how all of these operators
indexify onto Parakeet's internal operator core (and how fusion and tiling fire
on them), see [Operators & compilation walkthrough](primitives.md#what-the-index-prefix-means).

---

## Mapping and indexing {#operator-reference}

### `par.map(f, *args, axis=0)` {#map}

Apply `f` in parallel to corresponding slices along `axis` (or to every scalar
element with broadcasting when `axis=None`). Non-array arguments are passed
unchanged, and `f` may return a scalar, tuple, or array (results are stacked).

```python
par.map(lambda row: row / np.sum(row), X)  # normalize each row
par.map(lambda a, b: a * a + b, x, y, axis=None)  # element-wise with broadcast
```

### `par.imap(f, shape)` {#imap}

Build an array of `shape` where entry `(i, j, ...)` is `f(i, j, ...)`.
Multi-axis indices unpack automatically when `f` takes `len(shape)` arguments
(a 1-arg tuple `lambda ij:` also works).

```python
par.imap(lambda i, j: i >= j, (T, T))  # causal mask (fuses with zero allocation)
```

### `par.allpairs(f, x, y, axis=0)` {#allpairs}

Evaluate `f(x[i], y[j])` for every pair of slices `(i, j)` along `axis`. Fuses
directly into downstream reductions like [`par.fold`](#fold) or `np.argmin`
without materializing the `N × M` matrix.

```python
par.allpairs(lambda a, b: np.sum((a - b) ** np.float32(2)), X, C)
```

### `par.outer_map(f, *args)` {#outer_map}

N-ary generalization of [`par.allpairs`](#allpairs) over the Cartesian product
of the leading axes of two or more arrays.

### `par.parfor(f, shape)` {#parfor}

Run `f(i, ...)` in parallel across `shape` for its side effects (such as writing
disjoint slices or using thread-local `par.fixed_list` scratch). For pure array
outputs, prefer NumPy expressions, comprehensions, [`par.map`](#map), or
[`par.imap`](#imap).

---

## Reductions and structured folds

### `par.reduce(f, *args, init=None, axis=None, map=None)` {#reduce}

Reduce elements with binary combiner `f` (such as `np.add`, `np.maximum`, or a
custom function), globally (`axis=None`) or along `axis`. Passing `map=g` fuses
a map step directly into the reduction.

```python
par.reduce(np.maximum, X, axis=0)  # column maxima
par.reduce(np.add, x, y, map=np.multiply)  # dot product, zero temporaries
```

### `par.ireduce(f, shape, init=None, combine=np.add)` {#ireduce}

Reduce `f(i, ...)` directly over every coordinate of `shape` using `combine`,
without constructing an input array first.

```python
par.ireduce(lambda r: a[i, r] + b[r, j], k, combine=np.minimum)
```

### `par.fold(lift, merge, x, init=None, *, axis=None, indexed=False, emit=None)` {#fold}

Associative structured reduction over multi-field tuple or `NamedTuple` states:

* `lift` maps each element (or `(value, index)` when `indexed=True`) to a state
  of scalar fields.
* `merge(left, right)` combines two adjacent states **in input order**
  (preserving exact left-to-right tie and NaN semantics).
* `emit` optionally projects the final state.
* `init` seeds the fold and is returned for empty input (without `init`, an
  empty lane raises `ValueError`).

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

When composed with [`par.allpairs`](#allpairs) (or broadcasting NumPy
`np.argmin(np.sum((x[:, None] - y[None, :]) ** 2, axis=2), axis=1)`), Parakeet
automatically compiles a tiled pairwise fold on both CPU and GPU without
allocating the `Q × R` score matrix.

---

## Prefix scans

### `par.scan(f, *args, init=None, axis=0, map=None, emit=None)` {#scan}

Running prefix fold along `axis` (a generalization of `np.cumsum` and
`np.cumprod`). Associative combiners compile to a 3-stage parallel prefix scan,
and affine/bilinear recurrences trigger
[Law 4 chunkwise duality](algebra.md#law-4-chunkwise-state-space-duality).

```python
par.scan(np.add, x, y, map=lambda a, b: a - b)  # running sum of differences
par.scan(lambda acc, v: 0.9 * acc + 0.1 * v, x)  # exponential moving average
```

---

## Filtering and ragged expansion

### `par.filter(pred, *args, map=None)` {#filter}

Return elements where `pred` holds (optionally transforming kept elements with
`map`). Equivalent to NumPy boolean masking `x[pred(x)]`.

```python
par.filter(lambda a, b: a > b, x, y, map=np.add)  # x + y where x > y
```

### `par.filter_reduce(f, pred, *args, map=None, init=None, axis=None)` {#filter_reduce}

Reduce with `f` over elements satisfying `pred`. When `init` is provided, fuses
into a single zero-allocation guarded reduction loop (also triggered
automatically by `np.sum(x[x > 0])`).

### `par.flatmap(count_fn, emit_fn, *xs)` {#flatmap}

Two-pass 1-to-K ragged expansion returning a `RaggedArray(values, offsets)`. For
each index `i`, `count_fn(x0[i], ...)` returns the output count `c_i >= 0`, and
`emit_fn(x0[i], ..., k)` emits item `k` (`0 <= k < c_i`) in parallel after a
prefix sum of counts.

```python
# Expand run-length-encoded CIGAR operations into a RaggedArray
par.flatmap(lambda op, n: n, lambda op, n, k: op, ops, lengths)
```

### `par.filter_pairs(pred, x, y, emit=None)` {#filter_pairs}

Compact all pairs `(i, j)` where `pred(x[i], y[j])` holds into a `RaggedArray`
of matching pairs (or `emit(x[i], y[j])` payloads) without allocating a dense
`N × M` mask.

---

## Histograms and scatter

### `par.scatter_add(n, idx, vals, where=None)` {#scatter_add}

Accumulate `vals[i]` into output bin `idx[i]` of length `n` (optionally guarded
by `where`, or passed as a keyed generator `((k, v) for ...)`). Compiles with
thread-privatized histograms on CPU, parallel atomics on GPU, sibling fusion for
shared `idx`, and an exact gather adjoint under `par.grad`. Also triggered
automatically by `np.bincount`, `np.add.at`, and canonical `out[idx[i]] += v`
loops.

```python
grad_hist = par.scatter_add(256, bin_ids, grads)
```

---

## 2D dynamic programming

### `par.wavefront(step_fn, shape, init_top, init_left, *, init_corner=None, band=None, track_max=False)` {#wavefront}

Execute a 2D or diagonally banded dynamic programming recurrence along parallel
anti-diagonals in `O(M)` rolling memory (never allocating the full `M × N`
table). `step_fn(top, left, diag, i, j)` returns the next scalar or
tuple/`NamedTuple` state (e.g. 3-state affine-gap alignment).

### `par.batched_wavefront(...)` {#batched_wavefront}

Run [`par.wavefront`](#wavefront) independently across a batch of sequences or
grids in parallel (one thread/lane per batch item on CPU or GPU).

### `par.soft_align(scores, gamma=1.0, gap=-1.0, band=None)` {#soft_align}

Differentiable soft-maximum global or banded sequence alignment with a
checkpointed reverse-mode VJP under `par.grad` / `par.value_and_grad`.

---

## Specialized data & scratch helpers

For ragged workloads, byte parsing, and per-thread scratch inside parallel loops:

| Helper | Purpose |
|---|---|
| `par.fixed_list(capacity, dtype=np.float32)` | Stack-allocated, zero-`malloc` scratch list (`.append`, `.pop`, `.clear`, indexing, `len`) for CPU, CUDA, and Metal parallel bodies |
| `par.split_bytes(buf, sep=10, skip_empty=False)` / `par.parse_int64` / `par.parse_float32` / `par.parse_float64` | Zero-`PyObject` delimiter splitting into `RaggedArray[uint8]` and compiled ASCII numeric parsing |
| `par.partition_by_size` / `par.segmented_parfor` / `par.segmented_histogram2d` / `par.ragged_partition` | Adaptive 3-tier ragged dispatch (heavy/medium/light tiers), fused 2D `(grad, hess, count)` histograms, and parallel CSR partitioning |

---

## Semantics notes

* **Array-returning functions:** `par.map` may return arrays (stacked along the
  mapped axis). All outputs must share the same shape (`ValueError` at runtime
  if ragged).
* **Associativity & parallelism:** `par.reduce` runs in parallel only when its
  combiner is recognizably one of `+`, `*`, `min`, `max`, `and`, `or`, `&`, `|`,
  `^` (written as `np.add`, `lambda a, b: a + b`, or `par.add`). Non-associative
  combiners run sequentially left-to-right like `functools.reduce`. Floating-point
  parallel reductions may differ in the last ULP, like NumPy's pairwise summation.
* **Combiners:** Standard NumPy ufuncs (`np.add`, `np.multiply`, `np.minimum`,
  `np.maximum`) are idiomatic; `par.add` / `parakeet.dsl` aliases remain
  available for compatibility.
