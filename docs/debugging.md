# Errors and debugging

## Compile-time errors

Everything Parakeet cannot compile is reported when the function is first called
with a given set of argument types, as a subclass of `parakeet.CompileError`:

| Exception | Meaning |
|---|---|
| `UnsupportedError` | a Python or NumPy feature outside the supported subset |
| `TypeInferenceError` (also a `TypeError`) | types that cannot be reconciled, wrong arguments |
| `StaticValueError` (also a `ValueError`) | something that must be a compile-time constant is not, or is invalid |

The message carries the source line, a caret under the offending expression,
and — because calls are inlined — the chain of calls that led there:

```text
parakeet.errors.TypeInferenceError: indices must be a scalar, got str 'a'
  File "model.py", line 15, in helper
    return v[k] * 2
           ^^^^
called from
  File "model.py", line 21, in outer
    total += helper(x, "a")
             ^^^^^^^^^^^^^^
```

`err.location` and `err.call_chain` give the same information programmatically,
and a note records the argument types being compiled.

## Run-time errors

Generated code raises ordinary exceptions, with a note pointing at the source
line that failed:

```python
@jit
def third(x):
  return x[2]


third(np.zeros(2))
# IndexError: index 2 is out of bounds for axis 0 with size 2
#   raised by compiled code at demo.py:5 (in third)
```

| Situation | Exception |
|---|---|
| index out of range | `IndexError` |
| shapes that do not broadcast, bad `reshape`, ragged `map` results, empty reduction without identity | `ValueError` |
| `//`, `%` or `/` by zero on Python numbers | `ZeroDivisionError` |
| `assert` | `AssertionError` |
| allocation failure | `MemoryError` |
| writing to a read-only array | `ValueError` |

Checks are also performed inside parallel loops and GPU kernels. When an error
occurs, temporaries allocated so far are released.

NumPy *array* arithmetic follows NumPy: integer division by zero yields 0 and
floating-point operations yield `inf`/`nan` rather than raising.

## Tools

`backend="python"`
:   Run the original function. If the answer is still wrong the bug is in your
    code, not in the compiler: `f(x, _backend="python")`.

`f.specialize(*args)`
:   Compile for example arguments without running and return a
    [`Specialization`](api.md): `.operators` (maps and reductions over index
    spaces, as elaborated), `.ir` (typed loop IR after the passes), `.source` (C),
    `.kernels` (GPU kernel sources), `.signature`.

`f.inspect(*args)`
:   Print the IR.

`parakeet.config.print_ir`, `parakeet.config.print_code`
:   Print the IR / C of everything as it is compiled (also `PARAKEET_PRINT_IR=1`).

`@jit(fusion=False)`
:   Evaluate every array expression into its own temporary. Useful to check
    whether a problem is related to loop fusion.

`parakeet.find_broken_transform(fn, inputs)`
:   Switch the optimizations on one at a time — fusion, each pass of the
    middle end (dead code, filter-reduce fusion, sibling fusion, tiling), the C optimizer, parallel loops, the
    GPU — and name the first one that makes the
    result differ from plain Python (or from `expected`, if given).

`parakeet.clear_cache()`
:   Delete cached compiled modules (they are keyed by the generated source and
    compiler, so this is never *needed* for correctness).

If you hit a `parakeet.errors.InternalError` or a `CCompilerError`, that is a bug
in Parakeet: the message includes where the generated source was kept. Please
open an issue with it.
