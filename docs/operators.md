# Data-parallel operators

Most code needs none of these — NumPy expressions and comprehensions compile to
the same loops. They are useful when you want to apply *your own function* over
an array in parallel, and they can be called directly from Python without
decorating anything.

```python
import parakeet as pk
```

| Operator | Meaning |
|---|---|
| `pk.map(f, x, y, ..., axis=0)` (alias `each`) | `f` applied to corresponding slices along `axis`; `axis=None` applies it to every scalar element |
| `pk.allpairs(f, x, y, axis=0)` | `result[i, j] = f(x[i], y[j])` |
| `pk.outer_map(f, x, y, z, ...)` | the n-ary generalisation of `allpairs` |
| `pk.reduce(f, x, ..., init=None, axis=None, map=None)` | fold `f` over all elements, or over slices along `axis`; with `map=g` and several arrays, over `g(x[i], y[i], ...)` |
| `pk.scan(f, x, ..., init=None, axis=0, map=None, emit=None)` | running fold (a generalised `cumsum`); `emit=h` gives `h` of each running value |
| `pk.imap(f, shape)` | `result[idx] = f(idx)` for every index of `shape` |
| `pk.ireduce(f, shape, init=None, combine=add)` | reduce `f(idx)` over every index |
| `pk.parfor(f, shape)` | call `f(idx)` for its side effects, in parallel |
| `pk.filter(pred, x, ..., map=None)` | the elements of `x` for which `pred` is true; with several arrays, `map(x[i], y[i], ...)` where `pred(x[i], y[i], ...)` holds |
| `pk.filter_reduce(f, pred, x, ..., map=None, init=None, axis=None)` | reduce with `f` over elements where `pred` holds (fuses into one zero-allocation guarded reduction loop when `init` is given) |
| `pk.scatter_add(n, idx, vals)` | index-directed associative scatter-add (`IndexScatter`) with thread-privatized parallel lowering and exact gather adjoint |
| `pk.pmap1(f, x, w)`, `pk.pmap2(f, x, (wx, wy))` | `f` applied to a sliding window around each element |
| `pk.grad(f, argnum=0)`, `pk.value_and_grad(f, argnum=0)` | the gradient of a scalar-valued `f` — see [Gradients](autodiff.md) |
| `pk.dsl.show_rewrite(f, *args)` | inspect which of Parakeet's [six algebraic laws](algebra.md) rewrote `f` and view before/after IR |

```python
pk.map(lambda row: row / row.sum(), X)  # normalise rows
pk.allpairs(lambda a, b: np.sum((a - b) ** 2), X, C)  # distance matrix, no temporaries
pk.reduce(pk.maximum, X, axis=0)  # column maxima
pk.imap(lambda ij: ij[0] * 10 + ij[1], (3, 4))  # an index-defined array
```

```python
pk.reduce(pk.add, x, y, map=pk.multiply)  # a dot product; no array of products is built
pk.scan(pk.add, x, y, map=lambda a, b: a - b)  # running sum of differences
pk.filter(lambda a, b: a > b, x, y, map=pk.add)  # x + y where x > y
```

`imap`, `ireduce` and `parfor` pass an integer when `shape` is an integer and a
tuple of integers when it is a tuple.

## Functions that return arrays

The function given to `map` may return an array; the results are stacked.

```python
pk.map(lambda row: np.cumsum(row), X)  # shape (n, m)
```

All results must have the same shape — this is checked at run time
(`ValueError`).

## Parallelism and associativity

`map`-like operators evaluate elements in parallel, in no particular order.
`reduce` runs in parallel when the combining function is recognisably one of
`+`, `*`, `min`, `max`, `and`, `or`, `&`, `|`, `^` (whether written as
`pk.add`, `np.add` or `lambda a, b: a + b`), and sequentially otherwise — so a
non-associative function such as `lambda acc, x: acc * 0.5 + x` still gives the
same answer as `functools.reduce`. Floating-point sums are reassociated, like
NumPy's own pairwise summation: expect differences in the last digits.

## Primitive functions

`pk.add`, `pk.subtract`, `pk.multiply`, `pk.divide`, `pk.minimum`, `pk.maximum`,
the comparison and logical/bitwise operators are first-class functions for use
with the operators above. Outside compiled code they behave like the NumPy
ufuncs of the same name.
