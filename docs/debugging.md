# Errors and debugging

## Compile-time errors

Anything outside Parakeet's supported subset fails at specialization time as a
subclass of `parakeet.CompileError`:

| Exception | Meaning |
|---|---|
| `UnsupportedError` | Python or NumPy feature outside the compiled subset |
| `TypeInferenceError` (also `TypeError`) | Incompatible types or invalid argument signature |
| `StaticValueError` (also `ValueError`) | A compile-time constant (`axis`, `dtype`, `shape` length) is dynamic or invalid |

Errors report the exact source line, caret, and inlined call chain
(programmatically accessible via `err.location` and `err.call_chain`):

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

## Run-time errors

Compiled code raises standard Python exceptions with the failing source line,
even from inside parallel CPU loops and GPU kernels (freeing any temporary
allocations first):

```python
@par.jit
def third(x):
  return x[2]


third(np.zeros(2))
# IndexError: index 2 is out of bounds for axis 0 with size 2
#   raised by compiled code at demo.py:5 (in third)
```

| Condition | Exception |
|---|---|
| Out-of-bounds index | `IndexError` |
| Broadcast mismatch, bad `reshape`, ragged `map` output, empty unseeded reduction, writing to a read-only array | `ValueError` |
| Scalar `//`, `%`, or `/` by zero on Python numbers | `ZeroDivisionError` |
| Failed `assert` | `AssertionError` |
| Out of memory | `MemoryError` |

NumPy *array* division matches NumPy: integer division by zero yields `0`, and
float division by zero yields `inf` / `nan`.

## Debugging toolkit

| Tool | What it does |
|---|---|
| `f(x, _backend="python")` | Run the original Python function uncompiled to separate an algorithm bug from a compiler issue |
| `par.show_rewrite(f, *args)` | Report which algebraic laws fired and operator counts before/after optimization |
| `f.specialize(*args)` | Compile without running; returns a [`Specialization`](api.md) with `.operators`, `.ir`, `.source`, `.kernels`, `.signature` |
| `f.inspect(*args)` | Print the lowered loop IR |
| `@par.jit(fusion=False)` | Disable fusion to isolate whether an issue is fusion-related |
| `par.config.print_ir = True` / `print_code = True` | Print lowered IR or generated C for every specialization (also `PARAKEET_PRINT_IR=1`) |
| `par.find_broken_transform(fn, inputs)` | Enable fusion, each middle-end pass, C optimization, threads, and GPU one at a time and report the first pass that diverges from Python (or `expected`) |
| `par.clear_cache()` | Delete disk-cached compiled modules (keyed by generated source + compiler, so never required for correctness) |

If you ever hit an `InternalError` or `CCompilerError`, the exception message
includes the saved path to the generated source—please attach it to a GitHub
issue.
