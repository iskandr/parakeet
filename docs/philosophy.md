# Ideas, history, and boundaries

## Five design principles

### 1. Specialize on types and shapes at call time

Python functions act as templates. On each call, Parakeet specializes the
function on its argument dtypes, array ranks, memory layouts (contiguous vs.
strided), and container structure, caching the compiled binary per type key.

When a type or rank depends on a runtime value (`axis=`, tuple length, or a
size-1 broadcast axis), Parakeet specializes on that value automatically. That
fixes the broadcasting corner case (`200×1 + 1×300`) noted in
[Rubinsteyn's thesis](https://cs.nyu.edu/media/publications/rubinsteyn_alex.pdf)
statically at compile time.

### 2. Lower everything to seven algebraic primitives

Instead of shipping hundreds of hand-written C or GPU kernels for individual
NumPy functions, Parakeet lowers NumPy expressions, comprehensions, matrix
products, `einsum`, and `par.*` operators onto seven core index-space operators
(`Map`, `Reduce`/`Fold`, `Scan`, `Filter`/`Expand`, `Scatter`, `Wavefront`, and
`Tile`). See [Algebraic primitives & walkthrough](primitives.md) for the full
catalog and step-by-step examples.

### 3. Fuse algebraically before lowering to loops

Keeping programs in operator form lets fusion run as algebraic rewrites rather
than loop-nest dependence analysis:

* **Vertical fusion:** collapses producer-consumer chains (`Map -> Map`,
  `Map -> Reduce/Scan/Filter/Scatter`, `Scan -> Map`, `Filter -> Filter`, and
  `Filter -> Reduce` so `np.sum(x[x > 0])` allocates nothing).
* **Nested map flattening:** merges nested maps into one multi-dimensional map
  for CPU threads and GPU grids.
* **Horizontal and online fusion:** merges sibling maps, reductions, and
  same-key histograms, and combines dependent `max` + `sum(exp(x - m))` pairs
  into one pass.
* **Safety guards:** unlike the 2014 thesis, fusion checks operation cost,
  single-use counts, aliasing, and associativity so it never silently
  duplicates expensive work or reorders non-associative user folds.

See [Rewrite rules and optimizations](rewrites.md) for the full rule catalog.

### 4. Tile for cache and GPU shared memory from the same IR

Following [Hielscher's 2013 thesis](https://cs.nyu.edu/media/publications/eric_h.pdf),
locality passes rewrite operators directly before loop lowering:

* **Reduction interchange:** hoists strided reductions (`np.sum(X, axis=0)`)
  outside L1-sized output strips on CPU, while leaving columns coalesced on GPU.
* **Cache, register, and GPU SRAM tiling:** turns maps wrapping reductions
  (`a @ b`, `allpairs`, `einsum`, fused prologues/epilogues) into nested cache
  and register tiles on CPU and shared-memory tiles on CUDA and Metal (with
  Metal `simdgroup_matrix` and optional CUDA `nvcuda::wmma` TF32 Tensor Cores).
* **Modern algebraic laws:** extends the same tiling machinery to online
  softmax/attention, cut cross-entropy, Gaussian kernel regression, contraction
  reassociation, chunkwise affine/bilinear scans, and tile-interval mask
  pruning. See [Algebraic laws](algebra.md).
* **Device residency:** `par.to_device` and `par.to_host` keep arrays on CUDA
  or Metal across calls with zero host copies.

### 5. Inspectable, fail-loud compilation

* `par.show_rewrite(f, *args)` summarizes which algebraic laws fired.
* `f.specialize(*args)` exposes `.operators`, `.ir`, `.source`, and `.kernels`.
* `par.verify` checks types, scoping, and buffer ownership after every pass;
  `par.find_broken_transform` bisects passes automatically.
* Unsupported Python or NumPy constructs fail at compile time with the exact
  source caret rather than falling back silently.

---

## Current boundaries

Parakeet compiles a static-type numerical subset of Python, not all of CPython
or SciPy.

### Python language and types

| Supported | Not supported yet |
|---|---|
| `bool`, `int8`–`int64`, `uint8`–`uint64`, `float32`, `float64`, `complex64`, `complex128` scalars and arrays | Native `float16` compute (widened to `float32`), object or string arrays |
| Tuples, `NamedTuple`, fixed string-key dicts, `@dataclass` (pure methods, `@property`, `__call__`), `slice`, `range`, `zip`, `enumerate`, `itertools` | Dynamic dict keys, `set`, runtime string manipulation, arbitrary mutable classes |
| Compile-time lists, CPU growable homogeneous lists, `par.fixed_list` scratch, dynamic-shape array/`RaggedArray` loop rebinding | Heterogeneously typed lists grown inside runtime loops |
| `if`/`elif`/`else`, `for`, `while`, `break`, `continue`, early `return`, `match`/`case`, closures, `*args`/`**kwargs`, `assert`, `raise` | `try`/`except`, `with`, `yield`, `global`/`nonlocal`, `import` inside `@jit`, recursion |
| Static array rank; value specialization on `axis`, `keepdims`, `dtype`, tuple length | Value-dependent output rank (e.g. `np.squeeze(x)` without `axis=`) |

### NumPy and SciPy

About 215 NumPy functions and ufuncs are built in ([full list](supported.md)),
including arithmetic, reductions, scans, sorting, masks/fancy indexing,
`@`/`einsum`, 2D and batched 3D `np.linalg` (`norm`, `cholesky`, `det`,
`solve`, `inv`, `qr`, `eigh`, `eigvalsh`, `svd`, `pinv`, `lstsq`), and parallel
`np.random`. Not built in yet:

* Non-symmetric eigensolvers (`np.linalg.eig`, `eigvals`) and 4D+ batched
  `np.linalg.*` (wrap with `par.map` for extra batch axes)
* `np.fft.*` (see `examples/complex_fft_and_fasta.py` for a compiled radix-2 FFT)
* Variable-cardinality set ops (`np.unique`, `np.lexsort`, `np.isin`)
* SciPy modules (`scipy.special`, `scipy.optimize`, `scipy.ndimage`, ...)

### Automatic differentiation

Reverse mode (`par.grad`, `par.value_and_grad`, `par.vjp`), forward mode
(`par.jvp`, `par.linearize`), `par.linear_transpose`, and higher-order AD
(`par.hvp`, `par.jacfwd`, `par.jacrev`, `par.hessian`) cover element-wise ops,
reductions, scans, `sort`/`topk`, `@`/`einsum`, `np.linalg`, gather/scatter-add,
`par.map`/`allpairs`/`wavefront`, pure runtime `if`, and checkpointed
`for`/`while` loops. Current limits:

* **Tape-based reverse mode:** records array operators during elaboration (with
  algebraic rematerialization for attention, cross-entropy, softmax, and norms)
  rather than running as a separate IR-to-IR pass over lowered loops.
* **Control flow:** loops with `break`/`continue` over differentiated state are
  refused; runtime `if` branches over differentiated values run both sides and
  cannot contain array stores, `return`, `break`, or `raise`.
* **Missing rules (refused loudly):** `cumprod`, `convolve`, arbitrary indexed
  overwrites (`out[idx] = v`; use scatter-add instead), overwriting a buffer
  still needed by an earlier backward rule, and reverse-over-reverse through
  non-scalar `par.map` bodies.

### GPU and locality

* **Apple Metal `float64`:** Metal lacks hardware `float64`; those loops run on
  CPU with a warning. Use `float32` on Metal.
* **Per-thread heap allocation:** parallel loops that allocate dynamic arrays
  inside their body run on CPU rather than CUDA or Metal.
* **Standalone GPU GEMM:** CPU calls NumPy's BLAS for standalone float `A @ B`.
  CUDA and Metal run Parakeet's own tiled kernels (Metal `simdgroup_matrix`,
  optional CUDA TF32 WMMA), which fuse prologues/epilogues in one kernel but
  trail cuBLAS on large standalone matmuls.
* **Tile sizing:** chosen analytically from `TargetProfile` cache and shared
  memory budgets (`config.tile_bytes`) rather than
  [Hielscher's](https://cs.nyu.edu/media/publications/eric_h.pdf) online
  runtime search. Chunkwise `TileScan` covers affine and bilinear recurrences,
  not arbitrary general scans.

---

## History

Parakeet began in OCaml as an array compiler modeling the array language Q,
shifted to modeling Python, and was reimplemented in Python at New York
University (Courant Institute) by co-creators
[Alex Rubinsteyn](https://rubinsteyn.com/)
([`@iskandr`](https://github.com/iskandr)) and
[Eric Hielscher](https://github.com/hielscher)
([`@hielscher`](https://github.com/hielscher)), with
[Nathaniel Weinman](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-257.html),
advised by [Dennis Shasha](https://cs.nyu.edu/~shasha/), and with early advice
from [Russell Power](https://rjp.io/) ([`@rjpower`](https://github.com/rjpower)).
That earlier system was hosted at `parakeetpython.com` and documented in:

* [Eric Hielscher](https://github.com/hielscher),
  [*Locality Optimization for Data Parallel Programs*](https://cs.nyu.edu/media/publications/eric_h.pdf)
  (PhD thesis, NYU, 2013; also [NYU TR2013-955](https://cs.nyu.edu/media/publications/TR2013-955.pdf) /
  [arXiv:1304.1835](https://arxiv.org/abs/1304.1835) with
  [Alex Rubinsteyn](https://rubinsteyn.com/) and
  [Dennis Shasha](https://cs.nyu.edu/~shasha/)):
  `TiledMap`, `TiledReduce`, and `TiledScan` for cache tiling, register
  blocking, and loop interchange over adverb trees.
* [Alex Rubinsteyn](https://rubinsteyn.com/),
  [*Runtime Compilation of Array-Oriented Python Programs*](https://cs.nyu.edu/media/publications/rubinsteyn_alex.pdf)
  (PhD thesis, NYU, 2014): type and value specialization, data-parallel adverbs,
  and algebraic fusion.
* Papers and talks: [HotPar '12](https://www.usenix.org/conference/hotpar12/workshop-program/presentation/rubinsteyn)
  (*Parakeet: A Just-In-Time Parallel Accelerator for Python*, with
  [Alex Rubinsteyn](https://rubinsteyn.com/),
  [Eric Hielscher](https://github.com/hielscher),
  [Nathaniel Weinman](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-257.html),
  and [Dennis Shasha](https://cs.nyu.edu/~shasha/)),
  [PyData Boston '13](https://vimeo.com/73895275), and
  [PyData NYC '13](https://vimeo.com/79556629) (*Python on the GPU with Parakeet*).

Parakeet 1.x is a rewrite for Python 3.11+ and NumPy 2 that keeps `@jit` NumPy
and explicit data-parallel operators together, closes the soundness gaps from
the earlier design, and adds Apple Metal, NVIDIA CUDA, composable AD, and tiled
monoidal reductions.
