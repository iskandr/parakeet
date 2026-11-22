# Migrating from Parakeet 0.x

Parakeet 1.0 is a rewrite for Python 3.11+. The decorator and the operators keep
their names, and most 0.x programs run after the usual Python 2 → 3 changes
(`xrange` → `range`, `/` → `//` for integer division, `print(...)`).

| 0.x | 1.0 |
|---|---|
| Python 2.7, `dsltools`, `appdirs` | Python 3.11+, NumPy 2, nothing else |
| `config.backend = "openmp"` | still works; default is `"auto"`, which needs no OpenMP |
| `"llvm"` backend | removed |
| `"interp"` backend | `"python"` (runs the original function) |
| `"cuda"` via PyCUDA + `nvcc` | `"cuda"` via CuPy, no toolkit install; new `"metal"` backend |
| `f(x, _backend="c")` | unchanged; also `@jit(backend="c")` |
| `@macro`, `@typed_macro`, `@staged_macro` | removed — arguments such as `axis` are specialised automatically |
| `parakeet.testing_helpers` | `parakeet.testing` (the old name still imports) |
| `config.opt_fusion`, `config.print_specialized_function`, `config.print_generated_code` | still accepted, as aliases of `fusion`, `print_ir` and `print_code`; the other `opt_*` flags are gone (see [Configuration](configuration.md)) |
| `parakeet.prims.add`, `parakeet.ndtypes.Int64`, `Float64`, ... | unchanged |
| tuple parameters `def f((a, b))` | not valid Python 3; unpack in the body |
| `typed_repr(f, args)`, `specialize(f, args)`, `run_python_fn(f, args)` | unchanged (also `f.specialize(*args).ir`) |
| `find_broken_transform` | removed: there is no transformation pipeline to bisect |
| `parakeet.rank`, `size`, `shape`, `erf`, `norm`, ... | unchanged |

## Behaviour changes

* **Result types follow NumPy 2.** In 0.x `bool + bool` gave an integer and
  integer `power` gave a float; now both give what NumPy gives. Python scalars
  no longer upcast `float32` arrays.
* **Bounds are checked** by default. Programs that silently read out of bounds
  under 0.x now raise `IndexError`.
* **`np.array(x)` copies**, `x += 1` updates in place, views alias — as in NumPy.
* **Globals are read at call time** rather than captured at first compilation.
* `parakeet.reduce` is parallel only for combiners known to be associative;
  0.x assumed associativity for any function.
* `imap(f, shape)` passes `f` a tuple whenever `shape` is a tuple, including
  one-element tuples.
