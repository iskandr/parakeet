# Rewrite rules and optimizations

Everything Parakeet does to a program between your source and the loops it
runs, in one place, and how that compares with the two theses that describe
the original system:

* **[R]** Alex Rubinsteyn, *Runtime Compilation of Array-Oriented Python
  Programs* (NYU, 2014) — the operators and their fusion rules.
* **[H]** Eric Hielscher, *Locality Optimization for Data Parallel Programs*
  (NYU, 2013) — tiling.

The short version: the rules are the same algebra, applied in two places.

* **While elaborating.** An array expression is a *delayed* value — a function
  from index to element — and the vertical fusion rules fire at the moment an
  operator is applied to one. `F ∘ G` below means "the element function of
  the result is `F` applied to the element function of the argument"; no pass
  looks for the pattern afterwards.
* **In the middle end.** Elaboration produces the original compiler's
  *indexified* level: operator statements (`IndexMap`, `IndexReduce`,
  `IndexScan`, `IndexFilter`, and `IndexScatter`) over index spaces with
  ordinary scalar bodies (`Specialization.operators` prints it). Middle-end
  passes rewrite those statements in order:
  dead code elimination, normalization, vertical fusion (exact and affine
  map/scan/filter/reduce/scatter), filter fusion, sibling and online monoidal
  fusion, bilinear contraction reassociation, monoidal tiling and
  tile-interval pruning, chunkwise affine/bilinear scans,
  pairwise fold tiling, reduction interchange, and cache/register/GPU tiling.
  A final pass (`LowerAdverbs` in spirit) lowers operators into loops.
  `parakeet.transforms.pipeline.PASSES` is the authoritative list, and
  `find_broken_transform` toggles its passes on one at a time. See
  [Algebraic laws](algebra.md) for the mathematical identities behind these
  passes and the [Kernel gallery](gallery.md) for runnable examples.

```
map i_27, i_28 over (x_n0_3, y_n0_8):               # x @ y.T, as the passes see it
    off_29: int64 = (x_off_2 + (i_27 * x_n1_4))
    off_30: int64 = (y_off_7 + (i_28 * y_n1_9))
    reduce i_35 over (x_n1_4):
        x_36: float64 = (x_1[(off_29 + i_35)] * y_6[(off_30 + i_35)])
        red_39: float64 = fold(add, x_36, init=0.0)
    arr_24[((i_27 * y_n0_8) + i_28)] = red_39
```

## Operators

| [R] §4.6 | here |
|---|---|
| `Map(f, X1..Xn, axis)` | `map(f, x1, ..., xn, axis=0)`, `each`; `axis=None` for every scalar |
| `OuterMap(f, X1..Xn, axis)` | `outer_map(f, x1, ..., xn)`, `allpairs(f, x, y)` |
| `IndexMap(f, shape)` | `imap(f, shape)` |
| `Reduce(f, ⊕, X1..Xn, init, axis)` | `reduce(⊕, x1, ..., xn, map=f, init=, axis=)` |
| `IndexReduce(f, ⊕, shape)` | `ireduce(f, shape, init=, combine=⊕)` |
| `Scan(f, ⊕, emit, X1..Xn, init, axis)` | `scan(⊕, x1, ..., xn, map=f, emit=, init=, axis=)` |
| `Filter(f, pred, X1..Xn)` | `filter(pred, x1, ..., xn, map=f)` |
| `ParFor(f, bounds)` | `parfor(f, bounds)` |
| `IndexFilter(pred, shape)` | what `x[mask]`, `np.nonzero` and `filter` are built on: an `IndexFilter` statement |
| `FilterReduce`, `IndexFilterReduce` | `filter_reduce(⊕, pred, x1, ..., xn, map=, init=, axis=)` — and any reduction over a filter also *becomes* one (see fusion) |
| `IndexScan(f, ⊕, emit, shape)` | what `scan`, `np.cumsum` and `np.cumprod` are built on: an `IndexScan` statement, whose `emit` is a block (it stores the running value) |
| — | `IndexScatter` (`par.scatter_add`, `np.bincount`, `np.add.at`, canonical `for i: out[idx[i]] += val` loops): index-directed associative scatter-add with thread-privatized parallel lowering and exact gather adjoint |
| — | `grad`, `value_and_grad`, `jvp`, `vjp`, `linearize`, `linear_transpose`, `hvp`, `jacfwd`, `jacrev`, `hessian` |

Every NumPy function is defined in terms of these: `np.sum` is a `reduce`,
`np.cumsum` a `scan`, `a @ b` and `np.einsum` a map of reductions, a
comprehension an `imap`.

## Fusion

### Vertical: a producer into its consumer

| rule ([R] Fig. 6.1) | status |
|---|---|
| `Map(g, Map(f, X))` ⇒ `Map(g∘f, X)` | yes — including affine coordinate inversion (`AffineInlineBufferLoad`) for transpose, shifted-slice, and stencil fusion, plus cost-bounded multi-consumer fusion |
| `Map(g, IndexMap(f, σ))` ⇒ `IndexMap(g∘f, σ)` | yes |
| `OuterMap(g, Map(f, X1), X2..)` ⇒ `OuterMap(λ. g(f(x1), x2..), X1..)` | yes |
| `Reduce(g, ⊕, Map(f, X))` ⇒ `Reduce(g∘f, ⊕, X)` | yes |
| `Reduce(g, ⊕, IndexMap(f, σ))` ⇒ `IndexReduce(g∘f, ⊕, σ)` | yes |
| `Scatter(⊕, idx, Map(f, X))` ⇒ `IndexScatter(⊕, idx, f(X))` | **yes** (`transforms/fusion.py`): elementwise weight maps fuse directly into `np.bincount` / `parakeet.scatter_add` |
| `Scan(g, ⊕, ε, Map(f, X))` ⇒ `Scan(g∘f, ⊕, ε, X)` | yes |
| `Filter(g, pred, Map(f, X))` ⇒ `Filter(g∘f, pred∘f, X)` | yes: the predicate and the kept values both read the unfused map |
| `Map(g, Filter(f, pred, X))` ⇒ `Filter(g∘f, pred, X)` | yes: the selected elements are a delayed gather that the map fuses into |
| `Map(g, Scan(f, ⊕, ε, X))` ⇒ `Scan(f, ⊕, g∘ε, X)` | **yes** (`transforms/fusion.py`): a consumer `IndexMap` whose single input is an `IndexScan`'s output fuses into the scan's `emit` block (`np.cumsum(x) * 2.0 + 1.0` runs in one scan pass) |
| `Reduce(g, ⊕, Filter(f, p, X))` ⇒ `FilterReduce(g∘f, ⊕, p, X)` | **yes** (`transforms/fusion.py`): becomes a guarded `IndexReduce` with no index array (see below) |
| `Filter(g, p2, Filter(f, p1, X))` ⇒ `Filter(g∘f, p1 ∧ (p2∘f), X)` | **yes** (`transforms/fusion.py`): short-circuiting single compaction, which also feeds into `FilterReduce` |

**Side conditions.** [R] states one: nothing may modify the intermediate array
between producer and consumer, decided by alias analysis. Here every delayed
array records how many stores had happened when it was created; if a store
that may alias what it reads has happened by the time it is consumed,
elaboration restarts with that expression evaluated eagerly where it was
written.

[R] states no condition on how often the intermediate is used, so fusing
could duplicate work. Here a delayed array that costs more than a few
operations per element may be consumed once; a second consumer, or a consumer
inside a loop, triggers the same restart. That is the main correction to the
thesis's rules.

### Nested: a map of maps

| rule ([R] Fig. 6.2) | status |
|---|---|
| `Map(λxi. Map(f, xi), x)` ⇒ `Map(f, x)` | yes, when the inner map is cheap and its shape does not depend on `xi`: the result is one delayed array over the whole index space |
| `Map(λxi. Map(f, y), x)` ⇒ `OuterMap(f, x, y)` | yes, same condition |
| `IndexMap(λi. IndexMap(f, σ2), σ1)` ⇒ `IndexMap(f, σ1 ++ σ2)` | yes, same condition |

When the inner map is expensive, elaboration emits a nested `IndexMap`
writing directly into its row of one output array (no per-row allocation), and
the middle-end `vertical fusion` pass (`CombineNestedMaps`) flattens it into a
single multi-dimensional `IndexMap` whenever the inner shape does not depend on
the outer index — exposing the full 2D/ND index space to multi-core CPUs and
GPUs.

### Horizontal: two consumers of one array

| rule ([R] Fig. 6.3) | status |
|---|---|
| `a = Reduce(f, ⊕1, X); b = Reduce(g, ⊕2, X)` ⇒ `a, b = Reduce(f×g, ⊕1×⊕2, X)` | **yes**, as *sibling fusion* (`transforms/fusion.py`): two `IndexReduce` statements over the same index space, the second not using the first's result, become one statement with two folds. `x.min(), x.max()` reads `x` once |
| `a = Map(f, X); b = Map(g, X)` ⇒ `a, b = Map(f×g, X)` | **yes** (`transforms/fusion.py`): two pure `IndexMap` statements over the same index space where the second does not read what the first writes share one loop (`np.sin(x), np.cos(x)` reads `x` once) |

Conditions the thesis leaves out: the index spaces must be provably equal
(the same expression once shape variables are resolved), both seeded or both
unseeded, and both must stay vectorizable — a float sum is not fused with a
`max`, which would cost the sum its SIMD loop. Scalar definitions between the
two statements are stepped over when they do not use the first result.

Tiling (below) uses the same rule on copies of one reduction.

### Filter fusion

| rule ([R] Fig. 6.1) | status |
|---|---|
| `Reduce(g, ⊕, Filter(f, p, X))` ⇒ `FilterReduce(g ∘ f, ⊕, p, X)` | **yes** (`transforms/fusion.py`): an `IndexFilter` followed by an `IndexReduce` over what it kept becomes one guarded `IndexReduce` over the filter's own index space. `np.sum(x[x > 0])` is one parallel loop and allocates nothing (about 3× faster on 5M elements) |
| `Filter(q, Filter(p, X))` ⇒ `Filter(p ∧ q, X)` | **yes** (`transforms/fusion.py`): adjacent `IndexFilter` statements where the second filters the first's survivors collapse into one short-circuiting `IndexFilter` (and if reduced afterwards, into one `FilterReduce`) |
| `Map(f, Filter(p, X))` ⇒ one loop | half: the map is a delayed array over the kept indices, so it fuses with its consumer (or directly into `FilterReduce`) but a returned filter still compacts first |

Conditions: nothing else reads the kept indices (returning `x[mask]` keeps
the filter); the reduction has an `init`, because the first element may not
survive (`np.min(x[mask])` stays two steps so that an empty selection still
raises); it runs in parallel only when every fold has an identity to start a
chunk from. If the number of survivors is used (`np.mean`), counting it is
one more fold. Inside the fused loop the bounds check on the gathered index
is dropped: the index is the loop's own.

## Reductions and scans

* **Associativity.** [R] calls the combiner "commutative" and never requires
  associativity; [H] requires both. Here a fold is regrouped (run in chunks,
  vectorized) only when its combiner is recognisably one of `+ * min max and
  or & | ^`, however it was written (`np.add`, `lambda a, b: a + b`). Any
  other function is folded strictly left to right.
* **Several axes at once.** `Reduce` over axes `A` of `X` ⇒ a map over the
  other axes of one reduction over a block: the reduced axes are moved last (a
  view) and each block is folded in a single pass.
* **A reduction with carried state (`Carry`).** `argmin`/`argmax` fold `(value, index)`
  pairs per chunk and combine the pairs in index order, so the first of equal
  elements still wins. The same `Fold.carries` monoid machinery also fuses
  dependent `m = max(x); s = sum(exp(x - m))` pairs into a single parallel
  associative `(m, s)` pass (`fuse_online_logsumexp`) for 1-pass `logsumexp`
  and zero-temp `softmax`.
* **Parallel scan.** `Scan(⊕, X)` with an associative primitive ⇒ fold every
  chunk; prefix the chunk totals; scan every chunk from its prefix.
* **Parallel filter.** `X[mask]` ⇒ count per chunk; prefix the counts; every
  chunk writes its survivors from its own offset.

## Linear algebra

* `a @ b` ⇒ `Map((i, j) ↦ Reduce(+, a[i, :] * b[:, j]))`: a map of
  reductions like any other, so it fuses with what feeds it.
* **Layout.** The inner loop should walk both operands with stride one: `b`
  is replaced by its transpose copied once, unless it is already the transpose
  of a contiguous array, and `transpose(transpose(x))` ⇒ `x`.
* **Stacks.** `A @ B` with `B` a matrix ⇒ reshape the stack to one tall matrix,
  one product, reshape back.
* **Einstein summation.** `einsum(spec, X1..Xn)` ⇒ view every operand in one
  index space (output labels, then summed labels), multiply element-wise,
  reduce the trailing block. `'ij,jk->ik'` exactly ⇒ `@`.
* **BLAS.** A product of two float matrices outside a parallel loop ⇒
  `cblas_?gemm` from the library NumPy itself uses, with the transpose flags
  set for transposed views. (`config.blas = False` keeps the generated loops.)

## Tiling

| [H] | here |
|---|---|
| `PermuteReductions`: interchange a strided inner reduction with a stride-1 outer map | **yes** (`transforms/tiling.py`, *target-aware reduction interchange*): on CPU, when an `IndexMap` of any rank wraps a commutative+associative `IndexReduce` whose inner axis is strided while the map's last index has unit stride (`np.sum(X, axis=0)`), the reduction is hoisted outside an L1-sized strip of contiguous output columns; on GPU (`TargetProfile.is_gpu`), interchange is skipped so warp/SIMD threads coalesce across columns directly |
| `Map(f, X)` ⇒ `TiledMap(f↑, X)`, applied recursively through a nest of adverbs, stopping at a `Reduce` | **generalized `IndexTile` + `TileStage` operators** (`syntax/ir.py`, `transforms/tiling.py`, `transforms/lowering.py`): any `IndexMap` of rank ≥ 2 wrapping one `IndexReduce` (any number of folds and `Carry` states) plus straight-line prologues/epilogues (`a @ b`, batched 3D+ matmul, `allpairs`, `einsum`, `np.tanh(a @ b + c)`) is rewritten into nested `IndexTile(level="cache")` → `IndexTile(level="register")` operators on CPU, and extracts maximal row/col/k `TileStage`s for GPU `__shared__` (CUDA) and `threadgroup` (Metal) block tiling (`backends/tiled_gpu.py`) |
| register tiling: a second, innermost tiling with compile-time tile sizes, then unroll and scalar-replace | **general N-D register jamming** (`lowering.jam_register_tile`) on CPU (exact hyper-rectangle partition jamming `2×4` interior blocks and partial remainder faces) + **`4×4` / `2×2` per-thread register micro-tiles** on CUDA and Metal |
| `TiledReduce`, `TiledScan` | `PermuteReductions` (L1 column-strip tiled reduction), invariant row-reduction hoisting (`split_invariant_map` / `squeeze_unit_map` for `softmax` and `layer_norm`), and chunkwise `TileScan` for affine/bilinear recurrences; Hielscher's general cache-tiled `TiledScan` over arbitrary scans is not implemented |
| cache tile sizes found by an online search between a pessimistic and an optimistic estimate | analytical compile-time sizing from queried `TargetProfile` (`parakeet/target.py`: CPU L1/L2/SIMD, CUDA SMs/SMEM/compute capability, Metal threadgroup memory/Apple7 `simdgroup_matrix`), with hardware matrix-unit lowering (`simdgroup_matrix` on Metal, `nvcuda::wmma` TF32 on CUDA `sm_80+` with `config.tensor_cores=True`) |
| aborts unless each function body is a single adverb (simple algorithm) | aborts unless the element is exactly one reduction; anything else is a plain parallel map |

[H] reports that tiling slows down a stencil (an 11×11 blur). Stencils are not
a map of reductions over two different operands, so they are never tiled here.

## Scalars and shapes

Done while expressions are built; there is no simplification pass:

* constant folding with NumPy's semantics;
* `x + 0`, `x - 0`, `x * 1`, `x // 1` ⇒ `x` and `x * 0` ⇒ `0` for integers
  (not floats: `-0.0` and `nan`); `select(c, a, b)` with constant `c`;
  `not not x`; `and`/`or` with a constant side;
* `x ** 2` ⇒ `x * x`, `x ** 0.5` ⇒ `sqrt(x)`, `x ** 1` ⇒ `x`;
* a slice of a slice, a transpose, a `reshape` of a contiguous array, an
  `expand_dims` are arithmetic on offset and strides: no copy, no loop;
* dimensions proven equal (by a broadcast check that already ran) are not
  checked again; an index known to be within its loop's bound gets no bounds
  check.

**Specialization on values** ([R] ch. 6.6 partially evaluates on inputs equal
to 0 or 1). Here: a dimension of length 1 is always a constant (which is what
makes broadcasting static — [R] lists `200x1 + 1x300` as giving a wrong
answer); a scalar argument or an array dimension becomes a constant when the
code needs one (`axis=`, the length of a tuple, a loop that has to be
unrolled).

Left to the C compiler, which does them well on the loops it is given: common
subexpressions, dead code, loop-invariant code motion, scalar replacement,
unrolling, vectorization. [R]'s pipeline did these itself, on its own IR.

## Parallelism and placement

* Every map, reduction, scan, filter and `parfor` is a parallel loop over
  chunks; a parallel loop inside another is a sequential one.
* A loop goes to the GPU when *elements × estimated inner iterations per
  element* reaches `config.device_threshold`: a short loop over long rows goes,
  a long loop of cheap elements below the threshold does not.
* Between the loops of one call, arrays stay on the device. Host code that
  reads or writes array memory itself asks for the device's copies first.

## Differentiation

`grad(f)` adds one rule per operator, applied in reverse order of the
operators `f` applied ([details](autodiff.md)):

| operator | adjoint of its argument, given the adjoint `ȳ` of its result |
|---|---|
| `Map(f, X)` element-wise | `Map(∂f/∂x · ȳ, X)`, summed over broadcast dimensions |
| `Reduce(+, X)` | `ȳ` broadcast to `X` |
| `Reduce(max, X)` | `ȳ` where `X` equals the result, shared between ties |
| `Reduce(*, X)` | `ȳ · result / X`, with a zero-safe branch (exact when one or more elements are zero) |
| `Reduce(f, X)`, any `f` | the fold replayed backwards over its stored running values |
| `Scan(+, X)` | `Scan(+)` of `ȳ` from the other end |
| `A @ B` | `ȳ @ Bᵀ` and `Aᵀ @ ȳ` |
| view (slice, transpose, reshape) | the same view of the adjoint |
| gather `X[i]` | scatter-add of `ȳ` at `i` |
| `sort(X)` | gather by the sorting permutation, so: scatter by it |
| `Map(f, X)` with any `f` | `f` differentiated again per element, in parallel, each chunk adding to its own copy of the adjoints |
| `if c: A else: B` | both run; values and adjoints are selected by `c` |

## What the theses have that this does not

* Differentiation as an IR-to-IR pass over the operator statements (here `grad`
  still records operators on a tape during elaboration, while element-wise
  expressions fuse both by construction during elaboration and in the
  middle-end vertical fusion pass).
* Recursive tiling of deeper adverb nests, general cache-tiled `TiledReduce`/`TiledScan` (outside affine/bilinear `TileScan`), and the
  online search for tile sizes.
* Its own low-level optimizer (it relies on the C compiler instead).

## What this has that the theses do not

* Differentiation; `einsum`; products of stacks.
* Parallel scans, filters and index-carrying reductions.
* The cost and single-use conditions on fusion, and checked aliasing.
* Static unit-dimension broadcasting; `break`/`continue`/early `return`;
  bounds and shape checks that raise.
* Metal and CUDA kernels from the same loops, launch by work, arrays resident
  on the device.
