# Gradients

`grad` turns a function into its gradient. It is an adverb like `map` and
`reduce`: it is applied while your code is compiled, and what comes out is
ordinary loops that are fused, tiled and run in parallel like everything else.

```python
import numpy as np
import parakeet as par


def loss(w, x, y):
  return np.mean((np.tanh(x @ w) - y) ** 2)


g = par.grad(loss)(w, x, y)  # d loss / d w, shaped like w
value, g = par.value_and_grad(loss)(w, x, y)
gw, gx = par.grad(loss, (0, 1))(w, x, y)  # several arguments at once
```

The function has to return one floating point number. The arguments you
differentiate with respect to can be floats, float arrays, or tuples of those
nested however you like — the gradient comes back with the same structure:

```python
def mlp(params, x, y):
  (w1, b1), (w2, b2) = params
  h = np.tanh(x @ w1 + b1)
  return np.mean((h @ w2 + b2 - y) ** 2)


(gw1, gb1), (gw2, gb2) = par.grad(mlp)(params, x, y)
```

`par.grad(f)` can be called from plain Python, where it compiles on first use,
or inside a `@par.jit` function, where the whole training step becomes one
compiled function:

```python
@par.jit
def descend(w, x, y, rate, steps):
  for _ in range(steps):
    w = w - rate * par.grad(loss)(w, x, y)
  return w
```

[`examples/nanogpt.py`](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt.py)
trains a small transformer this way, and
[`examples/convnet.py`](https://github.com/iskandr/parakeet/blob/master/examples/convnet.py)
a convolutional network.

## What can be differentiated

| | |
|---|---|
| arithmetic and math functions | `+ - * / **`, `np.exp`, `np.tanh`, `np.maximum`, `np.where`, `np.abs`, `np.clip`, … with broadcasting |
| reductions | `np.sum`, `np.mean`, `np.var`, `np.std`, `np.max`, `np.min`, `np.prod`, `np.linalg.norm`, with `axis` and `keepdims`; `parakeet.reduce` with `add`, `multiply`, `minimum`, `maximum` |
| linear algebra | `@`, `np.dot`, `np.matmul` for vectors, matrices and stacks of matrices; `np.einsum` |
| running and ordered | `np.cumsum`, `parakeet.scan(add, x)`, `np.sort` |
| folds | `parakeet.reduce(f, x)` with any function `f` |
| views | slicing, `x.T`, `reshape`, `ravel`, `np.transpose`, `x[:, None]` |
| indexing with arrays | `x[idx]`, `x[rows, cols]` |
| building arrays | `np.concatenate`, `np.stack`, `np.array([a, b, c])`, comprehensions |
| adverbs | `map`/`each`, `allpairs`, `imap` with your own functions, including functions that use arrays from outside |
| scalar code | anything written with scalars, conditional expressions included |

Derivatives of `min`, `max`, `abs` and friends at a tie follow the usual
conventions: ties in a reduction share the gradient equally.

## Loops and branches

Parakeet differentiates **both** small static loops and arbitrary runtime-bound
`for` and `while` loops:

* **Small static `for` loops** (compile-time trip count `<= 32` or iterating
  over a heterogeneous tuple of weight matrices) are **unrolled** onto the tape
  for zero checkpoint overhead and cross-iteration fusion.
* **Runtime-bound and long `for` loops** (`for t in range(n):`, `for x in arr:`,
  or static ranges longer than 32 iterations — including 10,000+ iterations)
  compile to an **IR-level `O(sqrt(T))` checkpointed reverse loop** (`ForRange`)
  with constant IR size and bounded memory.
* **Data-dependent `while` loops** (`while err > tol:`) compile to a fixed-slot
  stride-doubling **checkpointed reverse `While` loop** (`O(log T)` slots, zero
  dynamic reallocation inside the loop).
* **Custom `par.reduce` folds** use `O(sqrt(N))` checkpointed reverse mode, and
  1-D **`par.scan`** with any user combiner is differentiable directly.

```python
def deep(weights, x):  # a tuple of matrices (unrolled)
  for w in weights:
    x = np.tanh(x @ w)
  return np.sum(x)


def iir_loss(a, b, s0):  # runtime-length recurrence (O(sqrt(T)) checkpointed reverse loop)
  s = s0
  for t in range(a.shape[0]):
    s = a[t] * s + b[t]
  return s * s
```

Loops that do not touch differentiated values are left alone, so a training
loop around `par.grad` is an ordinary loop. Loops containing `break`,
`continue`, or early `return` over differentiated state are refused loudly.

## Custom rules, `stop_gradient`, and `checkpoint`

* **`par.stop_gradient(x)`** detaches a scalar, array, tuple, record, or
  `RaggedArray` from the tape with zero copy (useful for straight-through
  estimators and detached targets):
  ```python
  y = x + par.stop_gradient(np.round(x) - x)
  ```
* **`@par.custom_vjp`** lets you attach custom `(fwd, bwd)` rules via
  `.defvjp(fwd, bwd)` — supporting `NamedTuple` / frozen `@dataclass` /
  `RaggedArray` residuals, higher-order AD, and Implicit Function Theorem
  differentiation through `while`-loop equilibrium solvers without storing
  iteration history.
* **`@par.checkpoint`** (or `par.checkpoint(fn)(*args)`) drops a block's
  forward intermediates and rematerializes them during the backward pass,
  composing cleanly with Law 2 algebraic VJPs.

An `if` statement whose condition is only known at run time is differentiated
by running **both** branches and choosing between their results, which is what
`np.where` does. That is only possible when running the branch not taken has
no effect, so the branches may only assign variables: no `return`, `raise`,
`break`, no stores into arrays, no in-place updates. Remember that both sides
are computed: `if x > 0: y = np.log(x)` evaluates the logarithm either way.

```python
if np.sum(x) > 10.0:  # fine
  h, scale = np.tanh(h), 2.0
else:
  h, scale = h * h, 3.0
```

A guard that only raises (`if n == 0: raise ValueError(...)`) is fine too, and
is not run for both sides.

## Writing into arrays

Stores are differentiated: assigning a slice, an element, or updating in place
all work, in straight-line code and in loops that can be unrolled.

```python
out = np.zeros((n, 3))
out[:3] = x[:3] @ w
out[3:] = np.tanh(x[3:] @ w)
out += bias
```

Going backwards, whatever adjoint arrived for the stored region belongs to the
value that was stored, and the region starts again from zero.

One thing cannot work: overwriting a value that an earlier operation's
derivative still has to read. `y = np.tanh(h); h[0] = 5.0` is refused, because
the derivative of `tanh` needs the old `h`; `y = np.sum(h * 3.0); h[0] = 5.0`
is fine, because that derivative does not. Stores through an array of indices
(`out[idx] = v`) are not differentiated.

## The gradient is right or there is none

Parakeet never returns a gradient that silently ignores part of your function.
If something on the way from the arguments to the result has no derivative
rule — `np.cumprod`, `np.convolve`, a `scan` with your own function — you get
an `UnsupportedError` naming it:

```text
UnsupportedError: cannot differentiate through cumprod() (line 12): its derivative is not implemented
```

## Second derivatives

`grad` of a function that calls `grad` works for element-wise operators,
reductions, matrix products and views — enough for gradient penalties and
Hessian-vector products:

```python
def penalty(w, x):
  g = grad(loss)(w, x)  # the gradient, as part of the function
  return np.sum(g * g)


grad(penalty)(w, x)  # 2 H g, reverse over reverse
```

It is refused, by name, through `map` with your own function, indexing with
arrays, `cumsum`, `sort`, folds and `np.stack`/`np.concatenate`.

## Cost

A gradient costs about two to three times the function. Element-wise chains
are fused in both directions; a matrix product's derivative is two more
matrix products.

The derivative of `map`/`allpairs` with your own function runs in parallel
like the map itself. Every element may add to the gradient of an array the
function captured (the weights of a network mapped over a batch), so each
chunk of the loop adds to a copy of its own and the copies are summed at the
end: a reduction whose accumulator is the whole gradient. That costs memory:
one copy of the captured parameters per thread, up to 256 MB, beyond which
fewer chunks are used.

Loops are unrolled, so a long unrolled loop makes for a long compile.

## How it works

While the function is elaborated, a tape records each array operator applied
to a value that depends on the differentiated arguments. The record is made
where the operator still has its algebraic meaning, so each rule is one line:
an element-wise operator becomes a map of the scalar function's partial
derivative, a sum broadcasts its adjoint back, a matrix product becomes two
more, a `map` differentiates its function again per element.

Scalars are not recorded. An expression in Parakeet's IR is already a record
of how it was computed, so the adjoint of a scalar is pushed straight down its
expression tree. Views are not recorded either: a slice of an array is the
same slice of its adjoint.

See [`parakeet/transforms/autodiff.py`](https://github.com/iskandr/parakeet/blob/master/parakeet/transforms/autodiff.py).
