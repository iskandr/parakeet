# Core ideas, history, and boundaries

Parakeet began at New York University as a research compiler by **Alex
Rubinsteyn** (`@iskandr`), with major contributions from **Eric Hielscher**
(`@hielscher`) and **Nathaniel Weinman**, guidance from **Dennis Shasha**, and
advice from **Russell Power** — originally hosted at `parakeetpython.com` and
described in:

* **Alex Rubinsteyn**, [*Runtime Compilation of Array-Oriented Python Programs*](https://cs.nyu.edu/media/visual/files/Rubinsteyn_Alex.pdf)
  (PhD thesis, NYU, 2014) — type/value specialization, data-parallel adverbs,
  and algebraic fusion.
* **Eric Hielscher**, *Locality Optimization for Data Parallel Programs*
  (PhD thesis, NYU, 2013) — tiling, register blocking, and loop interchange
  over adverb trees.
* Earlier papers and talks: [HotPar '12](https://www.usenix.org/conference/hotpar12/parakeet-just-time-parallel-accelerator-python)
  (*Parakeet: A Just-In-Time Parallel Accelerator for Python*),
  [PyData Boston '13](https://vimeo.com/73895275), and
  [PyData NYC '13](https://vimeo.com/79556629) (*Python on the GPU with Parakeet*).

Parakeet 1.0 is a ground-up rewrite for Python 3.11+, NumPy 2, Apple Metal,
NVIDIA CUDA, and reverse-mode automatic differentiation (`grad`), built to
preserve the algebraic spirit of those theses while generalizing their
operators and fixing the places where the original rules were unsound or
incomplete.

---

## Core thesis ideas that define Parakeet

### 1. Specialization replaces dynamic dispatch ([R] Ch. 3 & §6.6)

Python has no static types, yet numerical kernels are almost always monomorphic
at each call site. Parakeet uses an untyped Python function as a template and
generates a native *specialization* for each distinct combination of argument
dtypes, ranks, memory layouts (contiguous vs. strided), and tuple structure.

When an expression's type or rank depends on a runtime value (such as `axis=`,
a tuple length, or a unit dimension `1` that governs broadcasting), Parakeet
specializes on that value automatically—resolving the famous broadcasting bug
noted in [R] (`200×1 + 1×300`) statically.

### 2. A small closed algebra of data-parallel adverbs ([R] Ch. 4)

Rather than implementing hundreds of bespoke C/CUDA kernels for every NumPy
function, Parakeet defines NumPy operations, comprehensions, matrix products,
and `einsum` in terms of a tiny set of data-parallel operators (*adverbs*):

* `IndexMap` — independent element-wise / slice-wise computation over an index space
* `IndexReduce` — associative or sequential folds (including multi-payload monoids such as `argmin`/`argmax` and 1-pass online `logsumexp`/`softmax`, plus guarded `FilterReduce`)
* `IndexScan` — prefix folds with an `emit` stage
* `IndexFilter` — stream compaction of indices matching a predicate
* `IndexScatter` — index-directed associative scatter-reductions (`np.bincount`, `parakeet.scatter_add`), lowered via thread-privatized parallel bins on CPU, CUDA, and Metal

Every higher-level construct—whether written as `np.tanh(x * a) + b`, a list
comprehension, `a @ b`, `np.cumsum`, `x[x > 0]`, `np.bincount`, or
`parakeet.allpairs`—lowers to these five operator statements
(`f.specialize(*args).operators`).

### 3. Algebraic fusion *before* loop lowering ([R] Ch. 6)

Once a program is expressed in adverbs, optimizations become concise algebraic
identities instead of fragile loop-nest analyses, backed by a single shared
`IndexFacts` affine analysis and `BufferInfo` alias/cost table:

* **Vertical fusion** ([R] Fig. 6.1): `Map ∘ Map` (including matmul epilogues like `np.tanh(a @ b + c)`, affine transpose/slice/stencil fusion, and cost-bounded multi-consumer fusion), `Map ∘ Scan` into `IndexScan.emit`, `Map → Filter`, `Map → Reduce`, `Map → Scatter`, `Filter ∘ Filter` (short-circuiting), and `Reduce ∘ Filter ⇒ FilterReduce` (single zero-allocation pass).
* **Nested map flattening** ([R] Fig. 6.2, `CombineNestedMaps`): `IndexMap` inside `IndexMap` collapses into one multi-dimensional `IndexMap`, exposing the full 2D/ND index space to multi-core CPUs and GPUs.
* **Horizontal / sibling & online monoidal fusion** ([R] Fig. 6.3): adjacent `IndexReduce`s (`x.min(), x.max()`, `x.argmin(), x.argmax()`) and independent `IndexMap`s (`np.sin(x), np.cos(x)`) share one loop, and dependent `m = max(x); s = sum(exp(x - m))` pairs collapse into a single parallel associative `(m, s)` monoid pass via `Fold.payloads` (`logsumexp`, 1-loop `softmax`).
* **Dead-operator elimination**: pure maps, reductions, scans, filters, scatters, and unused allocations nobody reads are removed before lowering.

Parakeet 1.0 also adds the side conditions the original thesis omitted:
single-use / cost guards (so fusion never silently duplicates expensive work),
alias tracking, explicit associativity checks (non-associative folds stay
sequential), and keeping SIMD-vectorizable float sums separate from non-SIMD
folds.

### 4. Generalized tiling & locality from the same algebra ([H] 2013)

Cache, register, and GPU SRAM locality optimizations all operate on the same
operator IR via two composable operators—`IndexTile` and `TileStage`—rather
than recovering structure from raw `for` loops:

* **Reduction interchange (`PermuteReductions`)**: when an `IndexMap` wraps a
  strided commutative+associative `IndexReduce` whose outer map index has unit
  stride (`np.sum(X, axis=0)`), the reduction is hoisted outside an L1-sized
  strip of contiguous output columns.
* **Invariant map splitting & unit-axis squeezing**: `keepdims=True` unit axes
  are squeezed and inner reductions independent of trailing map axes are
  hoisted (`softmax`, `layer_norm`), eliminating redundant $O(M \cdot N^2)$
  recomputation and collapsing multi-pass row normalizations into single
  L1-resident loops.
* **Nested `IndexTile` + `TileStage` (CPU cache/register & GPU SRAM)**: any
  N-D `IndexMap` wrapping an `IndexReduce` plus straight-line code (`a @ b`,
  batched 3D+ matmul, `allpairs`, `einsum`, fused prologues & epilogues) is
  rewritten into nested `IndexTile(level="cache")` → `IndexTile(level="register")`
  operators on CPU (arbitrary N-D register jamming with exact remainder-face
  partitioning) and automatically extracts maximal row/col/k prologue
  `TileStage`s into a `SharedTilePlan` for GPU backends.
* **GPU `__shared__` (CUDA) & `threadgroup` (Metal) block tiling**: analytically
  sized `(BM, BN, BK, TM, TN)` tiles (up to `64×64×16` with `4×4` per-thread
  register micro-tiles and bank-conflict-free `+1` padding) stage arbitrary
  fused prologues into SRAM once per tile and keep fused epilogues
  (`np.tanh(a @ b + c)`) in registers with zero extra HBM round-trips.

### 5. Verifiable, inspectable, fail-loud compilation

* `parakeet.verify` (`config.verify` / legacy `opt_verify`) checks scoping,
  typing, and buffer-ownership invariants before passes, after every pass, and
  after lowering.
* `f.specialize(*args).operators`, `.ir`, `.source`, and `.kernels` expose
  every stage of compilation.
* `parakeet.find_broken_transform` bisects the pipeline one pass at a time.
* Unsupported Python/NumPy constructs fail at compile time with the exact
  source caret and inlined call chain instead of silently falling back or
  returning wrong numbers.

---

## The boundary: what does *not* work yet

Parakeet is an array-oriented numerical compiler, not a drop-in replacement for
all of CPython or SciPy. Knowing the boundary up front avoids surprises:

### Python language & types

| Works | Not supported yet |
|---|---|
| `bool`, `int8`–`int64`, `uint8`–`uint64`, `float32`, `float64` scalars & arrays (any rank/strides) | Complex numbers (`complex64`/`complex128`), native `float16` compute (`float16` is widened to `float32`), structured/record dtypes, object/string arrays |
| Fixed-structure tuples (nested arbitrarily), `slice`, `range`, `zip`, `enumerate` | `dict`, `set`, runtime string manipulation, user-defined classes / methods (`self`) |
| List literals and comprehensions as fixed-shape arrays | Mutable/dynamically resized Python lists (`.append`, `.extend`, ragged lists) |
| `if`/`elif`/`else`, `for`, `while`, `break`, `continue`, early `return`, nested `def`/`lambda`, closures, `assert`, `raise` | `try`/`except`, `with`, generators (`yield`), `global`/`nonlocal`, `import` inside `@jit`, `**kwargs`, recursion |
| Static array rank; compile-time specialization on `axis`, `keepdims`, `dtype`, tuple lengths | Value-dependent output rank (e.g. unguided `np.squeeze(x)` without `axis=`), dynamically typed variables that cannot unify |

### NumPy & SciPy library coverage

Roughly 200 core NumPy functions and ufuncs are built in (see
[Supported functions](supported.md)), covering arithmetic, reductions, scans,
sorting, indexing/masks, `@`/`dot`/`einsum`, `np.linalg.norm`, and parallel
`np.random`. Not yet built in:

* Matrix decompositions and solvers (`np.linalg.svd`, `solve`, `inv`, `eig`,
  `cholesky`, `qr`, `det`, `lstsq`)
* Signal/spectral transforms (`np.fft.*`)
* Variable-cardinality set/grouping ops (`np.unique`, `np.lexsort`, `np.isin`)
* SciPy modules (`scipy.special`, `scipy.optimize`, `scipy.ndimage`, ...)

### Automatic differentiation (`grad` / `value_and_grad`)

`grad` differentiates element-wise ops, reductions (`sum`, `prod` zero-safe,
`min`, `max`, general folds, active `initial`), `cumsum`, `sort`, `@` (and
stacks), `einsum`, views/concatenation, gather/scatter, `map`/`allpairs` with
user functions, and pure runtime `if` branches—enough to train small
transformers ([`examples/nanogpt.py`](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt.py))
and convnets ([`examples/convnet.py`](https://github.com/iskandr/parakeet/blob/master/examples/convnet.py)).
Boundaries:

* **Still an elaboration-time tape, not yet an IR-to-IR pass:** differentiation
  records operators during elaboration rather than transforming `IndexMap` /
  `IndexReduce` statements directly.
* **No dynamic-length loops over differentiated values:** `for` loops with a
  compile-time-known length (≤ 4096, such as transformer layers or kernel
  offsets) unroll and differentiate; `while` loops or loops over runtime array
  lengths are refused (rewrite with `reduce`/`map`/`scan`).
* **Pure runtime `if` branches only:** both branches execute and are selected
  by the condition, so branches touching differentiated values cannot contain
  array stores, `return`, `break`, or `raise`.
* **Missing derivative rules (refused loudly):** `cumprod`, `convolve`,
  user-defined `scan` combiners, in-place indexed assignment (`out[idx] = v`;
  use `parakeet.scatter_add` / `np.bincount(..., weights=w)` for differentiable
  scatter-add), and overwriting an array still needed by an earlier
  operation's backward pass.
* **Second derivatives:** work for element-wise ops, reductions, `@`, and
  views; not yet supported through `map`, gather indexing, or `cumsum`.

### GPU & locality boundaries

* **Apple Metal has no hardware `float64`:** `float64` loops fall back to CPU
  cores with a warning; use `float32` for Metal acceleration.
* **No per-thread heap allocation on GPU:** a `ParLoop` that allocates
  temporary arrays or selects between buffer aliases inside the loop body runs
  on the CPU instead of CUDA/Metal.
* **GPU matrix multiplication & cuBLAS:** CPU backends call BLAS for standalone
  float `A @ B`; GPU backends run Parakeet's cooperative `__shared__` (CUDA) /
  `threadgroup` (Metal) block-tiled + register-micro-tiled kernels (handling
  arbitrary contractions, fused prologues like `sin(a) @ cos(b)`, `allpairs`,
  and fused epilogues like `np.tanh(a @ b + c)` in a single kernel), rather
  than calling cuBLAS or Tensor Cores (`mma`). Very large standalone GEMMs are
  therefore still faster in Tensor-Core cuBLAS, while fused contractions and
  custom `allpairs` workloads benefit directly from single-kernel SRAM reuse.
* **Online autotuning & tiled scans:** tile sizes are chosen analytically from
  cache/SRAM/register budgets (`config.tile_bytes`, GPU shared-memory budget)
  rather than Hielscher's runtime online search, and chunked GPU `TiledScan`
  is not yet implemented.
