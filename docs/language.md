# The compiled language

Parakeet compiles a statically typed numerical subset of Python: every value has
a type and array rank known at compile time. This page summarizes what is
supported and where compiled code intentionally differs from CPython.

## Values and types

| Python value | Inside compiled code |
|---|---|
| `int`, `float`, `bool` | 64-bit integer, `float64`, `bool` (weakly typed like NumPy scalars) |
| NumPy scalars (`np.float32(1)`, `np.complex64(1+2j)`) | Exact scalar dtype (`complex` lowers to unboxed `(re, im)` pairs) |
| NumPy arrays | `bool`, `int8`–`int64`, `uint8`–`uint64`, `float32`, `float64`, `complex64`, `complex128`; any rank and strides |
| Tuples (arbitrarily nested) | Compile-time structures (zero runtime overhead) |
| `NamedTuple` and `@dataclass` (frozen or mutable) | Compile-time records with pure methods, `@property`, `__call__`, and functional field rebinding (`obj.x = v`) |
| Fixed string-key `dict` | Compile-time record keyed by literal strings (key order is part of the type) |
| Lists & comprehensions | Numeric literals/comprehensions build arrays; local `[]` lists unbox at compile time or promote to growable CPU buffers in dynamic loops |
| `slice`, `range`, `zip`, `enumerate`, `itertools` | Supported as values and zero-allocation iterators (`pairwise`, `islice`, `chain`, `product`, `batched`) |
| `None`, strings, dtypes, functions, modules | Compile-time constants |

Type promotion follows **NumPy 2 (NEP 50)** exactly: `float32_arr * 2.0` stays
`float32`, `int8_arr + 1` stays `int8`, and `int32_arr / 2` is `float64`.

* **`float16`** inputs are widened to `float32`.
* **`complex64` / `complex128`** lower to unboxed `(re, im)` float views across
  CPU, Metal (`complex64`), and CUDA, and compose with Wirtinger `par.grad` and
  `par.jvp`.
* **Unsupported objects** (object arrays, `set`, arbitrary non-dataclass class
  instances) raise a `TypeError` naming the unsupported type.

## Statements and control flow

| Supported | Not supported (`UnsupportedError`) |
|---|---|
| Assignments (`=`, tuple unpack, `+=`, `:=`, annotations) | `try` / `except` / `finally` |
| `if` / `elif` / `else`, scalar `match` / `case` | `with` context managers |
| `for` (ranges, arrays, tuples, `zip`, `enumerate`), `while`, loop `else` | `class` definitions inside `@jit` |
| `break`, `continue`, early `return` anywhere | Generators (`yield`) |
| Nested `def` and `lambda`, closures | `global` / `nonlocal` statements |
| `assert`, `raise` built-in exceptions (literal message), `pass`, `del` | `import` inside `@jit`, f-strings, `print` |

Key loop and comprehension semantics:

* **Loop `else`** runs on normal exhaustion (including empty inputs) and is
  skipped by `break`. `zip(..., strict=True)` checks equal lengths.
* **Chained comparisons** (`0 <= i < n`) evaluate each reached operand once and
  short-circuit.
* **Cartesian comprehensions** (`[f(x, y) for x in xs for y in ys]`) over
  independent ranges or arrays compile to parallel maps (rightmost iterator
  varies fastest) and fuse directly into scalar consumers like `sum(...)`.
  Dependent inner bounds (`for j in range(i)`) run sequentially.

## Functions and closures

* **All helper calls are inlined** automatically—helpers do not need `@par.jit`.
  Because calls inline at compile time, recursion is not supported.
* Positional, keyword, default, keyword-only, `*args`, `**kwargs`, `**d`
  unpacking, `functools.partial`, and `operator.*` callables all work.
* Functions are first-class at compile time (pass them, store them, close over
  locals).
* **Globals and closures are read on every call**, just like Python. If a global
  changes type, a new specialization is compiled.

## Arrays, views, and indexing

Compiled code matches NumPy's view and mutation semantics:

```python
@par.jit
def f(x):
  v = x[1:]  # view of caller's array
  v[0] = 0  # mutates x in place
  x += 1  # visible to caller
  y = x * 2  # newly allocated array
  x[1:] = x[:-1]  # overlapping slice assignment is safe
  return y
```

* Basic slicing (`x[i, 1:-1, ::2, None, ...]`), integer-array indexing, and
  boolean masks work for both reads and writes.
* Returning an input returns the same object; returning a view returns a NumPy
  view over caller memory. Writing to a read-only array raises `ValueError`.
* Indices are bounds-checked (`IndexError`) and broadcast shapes are checked
  (`ValueError`). Pass `@par.jit(bounds_check=False)` once code is verified.

## Lists and dynamic loops

* **Numeric list literals** (`[1.0, 2.0, 3.0]`) and comprehensions build NumPy
  arrays directly.
* **Local scratch lists** (`outs = []; outs.append(...)`) stay unboxed at
  compile time in straight-line or unrolled code (so `np.stack(outs)` and
  `RaggedArray.from_sequences(outs)` allocate zero Python objects and compose
  with `par.grad`), and automatically promote to a growable CPU buffer when
  mutated inside a runtime loop or dynamic branch.
* **Parallel & GPU scratch:** inside `par.parfor` / `par.map` bodies and GPU
  kernels, use `par.fixed_list(capacity, dtype)` for a stack-allocated,
  zero-`malloc` buffer.
* **Shape rebinding:** variables holding arrays or `RaggedArray`s can be rebound
  to new shapes across `for` / `while` iterations (for example `active = active[mask]`).

## Dataclasses and fixed records

`NamedTuple`s, `@dataclass`es (frozen or non-frozen), and fixed string-key
`dict`s can cross the `@par.jit` boundary and hold arrays, scalars, static
metadata, or nested records.

* Pure methods, `@property`, `@staticmethod`, `@classmethod`, and `__call__`
  inline at compile time with zero `PyObject` overhead.
* Local field updates (`obj.x = v`, `obj.x += v`) on non-frozen dataclasses
  functionally rebind the compile-time record (value semantics, no heap
  mutation).
* Returning a record reconstructs its Python type while preserving NumPy view
  semantics for array fields.

## Differences from CPython / NumPy

| Topic | Behavior in Parakeet |
|---|---|
| **One static type per variable** | Merging `int` and `float` across branches promotes both paths to `float`. Incompatible types (e.g. array and `None`) fail at compile time if used afterward. |
| **Static array rank** | Every array's `ndim` is fixed at compile time. Operations with value-dependent output rank (such as `np.squeeze(x)` without `axis=`) require an explicit `axis=`. |
| **Compile-time constants** | Arguments like `axis`, `keepdims`, `dtype`, and `shape` tuple length must be static. When passed as function arguments, Parakeet specializes on their values automatically. |
| **64-bit integer overflow** | Python `int`s compile to 64-bit signed integers that wrap on overflow (like NumPy `int64`), not arbitrary-precision bigints. |
| **Unentered loop variables** | A variable first assigned inside a loop keeps its final value after the loop, or `0` if the loop ran zero times (CPython raises `NameError`). |
| **Random number streams** | `np.random` uses Parakeet's parallel RNG ([`par.seed`](api.md)); values are statistically valid but differ from NumPy's stream. |
| **Parallel evaluation order** | Array expressions, comprehensions, and `par.map` run elements in parallel in unspecified order. User reductions run in parallel only when associativity is proved (`+`, `*`, `min`, `max`, `&`, `|`, `^`); otherwise left-to-right. |
