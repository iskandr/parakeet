# The compiled language

Parakeet compiles a subset of Python chosen so that every value has a type known
at compile time. This page describes that subset and the places where compiled
code deliberately differs from running the same function in CPython.

## Values and types

| Python value | Inside compiled code |
|---|---|
| `int`, `float`, `bool` | 64-bit integer / double / bool, *weakly typed* like NumPy treats Python scalars |
| NumPy scalars (`np.float32(1)`, ...) | scalars of exactly that dtype |
| NumPy arrays | bool, int8–int64, uint8–uint64, float32, float64; any number of dimensions, any strides |
| tuples (also nested) | compile-time structures of the above |
| lists, list comprehensions | **arrays** (there are no Python lists in compiled code) |
| `None`, strings, dtypes, functions, modules | compile-time constants |
| `slice`, `range`, `zip`, `enumerate` | supported as values and in loops |

`float16` data is widened to `float32`. Complex numbers, object arrays, structured
dtypes, dictionaries, sets and user-defined classes are not supported; passing
one raises a `TypeError` naming the type.

Type promotion follows NumPy 2 (NEP 50) exactly: `float32_array * 2.0` stays
`float32`, `int8_array + 1` stays `int8`, `int32_array / 2` is `float64`.

## Statements

Supported: assignment (including tuple unpacking, augmented and annotated
assignment, `:=`), `if`/`elif`/`else`, `for` over ranges, arrays, tuples, `zip`
and `enumerate`, `while`, `break`, `continue`, `return` anywhere, nested `def`
and `lambda`, `assert`, `raise` of built-in exceptions with a literal message,
`pass`, `del`.

Not supported: `try`/`except`, `with`, classes, generators (`yield`), `global`
/`nonlocal`, `import` inside a compiled function, `for ... else`, f-strings and
`print`. Each of these produces an `UnsupportedError` at the offending line.

## Functions

* Any Python function called from compiled code is **inlined** — helpers do not
  need `@jit`. Recursion is therefore not possible and is reported as an error.
* Positional, keyword, default, keyword-only and `*args` arguments work;
  `**kwargs` does not.
* Functions are first-class at compile time: pass them as arguments, store them
  in variables, close over local variables.
* **Globals and closure variables are read on every call**, like in Python. If a
  global changes type, a new specialization is compiled.

## Arrays

Compiled code follows NumPy semantics for views and mutation:

```python
@jit
def f(x):
    v = x[1:]  # a view of the caller's array
    v[0] = 0  # ... so this modifies x
    x += 1  # in place, visible to the caller
    y = x * 2  # a new array
    x[1:] = x[:-1]  # overlapping assignment behaves as if a temporary were made
    return y
```

* Basic indexing (`x[i, 1:-1, ::2, None, ...]`), integer-array indexing and
  boolean masks work for reading and writing.
* Indices are bounds-checked (`IndexError`); shapes are checked when
  broadcasting (`ValueError`). Use `@jit(bounds_check=False)` to drop index checks.
* Returning an input returns *the same object*; returning a view of an input
  returns a NumPy view of the caller's memory.
* Read-only arrays may be read; writing to one raises `ValueError`.

## Differences from CPython / NumPy

One static type per variable
:   A variable assigned an `int` on one path and a `float` on another is a
    `float` everywhere (the usual numeric promotion). A function returning `1`
    in one branch and `2.5` in another returns a `float` from both. Types that
    cannot be unified — say an array and `None` — are an error if the variable
    is used afterwards.

Rank is static
:   The number of dimensions of every array is fixed at compile time. Functions
    whose result rank depends on *values* (for example `np.squeeze(x)` when it
    cannot be known which dimensions have length 1) ask you to pass `axis=`.

Compile-time constants
:   Some arguments must be known at compile time: `axis`, `keepdims`, the length
    of a `shape` tuple, a `dtype`. If such a value comes straight from an
    argument of the compiled function, Parakeet specialises on its *value*
    automatically; otherwise it reports which expression must be constant.

Lists are arrays
:   `[a, b, c]` and `[f(x) for x in xs]` build arrays. `sorted(x)` returns an
    array. There is no `append`.

Integer arithmetic
:   Python `int`s are 64-bit and wrap on overflow (like NumPy), they are not
    arbitrary precision.

`float16`
:   Operations NumPy would compute in half precision are computed in `float32`.

Loop variables after a loop
:   A variable first assigned inside a loop body keeps its last value after the
    loop, but if the loop never ran it is zero (CPython raises `NameError`).

Random numbers
:   `np.random` functions use Parakeet's own generator (see
    [`parakeet.seed`](api.md)); the streams differ from NumPy's.

Parallel side effects
:   `parakeet.map`, comprehensions and array expressions may evaluate their
    elements in parallel and in any order. Reductions with a user-defined
    function assume it is associative only when Parakeet can prove it (for
    `+`, `*`, `min`, `max`, `&`, `|`, `^`); otherwise they run sequentially.
