# Design, boundaries, and history

## Why Parakeet is built this way

Numerical Python sits between two worlds. **NumPy** gives concise, readable
code and fast C/Fortran kernels for individual operations, but composing
expressions allocates temporary arrays and sweeps DRAM on every step—and any
algorithm that does not fit a built-in ufunc falls back to slow CPython loops.
**C/CUDA compilers** generate fast scalar loops, but once array code is
flattened into imperative `for`-loops, high-level structure (independence,
associativity, data reuse) is lost and must be painfully recovered by
dependence analysis.

Parakeet—originating in the NYU theses of
[Eric Hielscher (2013)](https://cs.nyu.edu/media/publications/eric_h.pdf) and
[Alex Rubinsteyn (2014)](https://cs.nyu.edu/media/publications/rubinsteyn_alex.pdf)
and [HotPar '12](https://www.usenix.org/conference/hotpar12/workshop-program/presentation/rubinsteyn)—bridges
that gap with one guiding idea: **keep array programs in algebraic operator
form (data-parallel adverbs over explicit index spaces) all the way through
optimization.** That single choice unifies five design principles:

1. **Specialize on types, values, and shape buckets at call time.** Python
   functions act as templates specialized on argument dtypes, array ranks,
   layouts, and container structure. Compile-time parameters (`axis=`, tuple
   length, size-1 broadcast dimensions) specialize automatically, and tiled GPU
   schedules propagate power-of-2 shape buckets and divisibility so aligned
   sizes share one branchless binary.
2. **Normalize everything into coordinate-driven primitives.** NumPy
   expressions, comprehensions, matrix products, `einsum`, and `par.*` calls
   all indexify into six source primitives (`IndexMap`, `IndexReduce`,
   `IndexScan`, `IndexFilter`/`IndexExpand`, `IndexScatter`, `IndexWavefront`),
   turning broadcasts, transposes, and slices into plain index arithmetic
   ([Operators & compilation walkthrough](primitives.md)).
3. **Fuse algebraically before lowering to loops.** Vertical, nested, filter,
   sibling, and online fusion operate directly on the operator IR with explicit
   alias, single-use cost, and associativity guards
   ([Rewrite rules catalog](rewrites.md)).
4. **Tile for cache and GPU shared memory from one IR.** Following Hielscher's
   locality framework, target-aware reduction interchange, `IndexTile`
   cache/register blocking, GPU SRAM staging, hardware matrix units (Metal
   `simdgroup_matrix`, CUDA TF32 WMMA), and [Laws 1–6](algebra.md) all run on
   the same operator IR before codegen.
5. **Inspectable, fail-loud compilation.** Unsupported constructs raise a
   source-located `CompileError` instead of silently falling back; every pass
   is verified by `par.verify` and inspectable via `par.show_rewrite`,
   `f.specialize()`, and `par.find_broken_transform`.

---

## Current boundaries

Parakeet compiles a static-type numerical subset of Python, not all of CPython
or SciPy.

### Python language and types

| Supported | Not supported |
|---|---|
| `bool`, `int8`–`int64`, `uint8`–`uint64`, `float32`, `float64`, `complex64`, `complex128` scalars & arrays | Native `float16` compute (widened to `float32`), object or string arrays |
| Tuples, `NamedTuple`, fixed string-key dicts, `@dataclass` (pure methods, `@property`, `__call__`), `slice`, `range`, `zip`, `enumerate`, `itertools` | Dynamic dict keys, `set`, runtime string manipulation, arbitrary non-dataclass classes |
| Compile-time lists, CPU growable homogeneous lists, `par.fixed_list`, dynamic-shape array / `RaggedArray` loop rebinding | Heterogeneously typed lists mutated inside runtime loops |
| `if`/`elif`/`else`, `for`, `while`, `break`, `continue`, early `return`, scalar `match`/`case`, closures, `*args`/`**kwargs`, `assert`, `raise` | `try`/`except`, `with`, `yield`, `global`/`nonlocal`, `import` inside `@jit`, recursion, `print` |
| Static array rank; automatic value specialization on `axis`, `keepdims`, `dtype`, tuple length | Value-dependent output rank (e.g. `np.squeeze(x)` without `axis=`) |

### NumPy, SciPy, AD, and GPUs

* **NumPy & SciPy:** ~215 NumPy functions and ufuncs are built in
  ([full list](supported.md)), including `einsum`, `np.unique`, `np.searchsorted`,
  2D & batched 3D `np.linalg` (`cholesky`, `det`, `solve`, `inv`, `qr`, `eigh`,
  `eigvalsh`, `svd`, `pinv`, `lstsq`, `norm`), and parallel `np.random`. Not
  built in: non-symmetric `eig`/`eigvals`, 4D+ `np.linalg` (wrap with
  `par.map`), `np.fft.*` (see `examples/fft.py` for a compiled
  radix-2 FFT), `np.lexsort`/`np.isin`, and `scipy.*`.
* **Automatic differentiation:** covers reverse mode (`grad`, `vjp`, `jacrev`),
  forward mode (`jvp`, `linearize`, `jacfwd`), `linear_transpose`, higher-order
  AD (`hvp`, `hessian`), checkpointed `for`/`while` loops, and Law 2
  rematerialization. Refused loudly: `break`/`continue` over differentiated
  state, runtime `if` branches with array stores or control exits, `cumprod`,
  `convolve`, arbitrary indexed overwrites (`out[idx] = v`), overwriting a
  buffer needed by an earlier VJP, and reverse-over-reverse through non-scalar
  `par.map` bodies.
* **GPUs & locality:** Apple Metal lacks hardware `float64` (falls back to CPU
  with a warning); parallel bodies that dynamically allocate arrays per thread
  run on CPU; standalone GPU GEMMs use Parakeet's tiled kernels (fusing
  prologues/epilogues, though trailing cuBLAS on large standalone matmuls);
  tile sizes are chosen analytically from `TargetProfile` rather than online
  search; chunkwise `TileScan` covers affine and bilinear recurrences rather
  than arbitrary scans.

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

Parakeet 1.x is a ground-up rewrite for Python 3.11+ and NumPy 2 that keeps
`@jit` NumPy and explicit data-parallel operators together, closes the soundness
gaps from the earlier design, and adds Apple Metal, NVIDIA CUDA, composable AD,
and tiled monoidal reductions.
