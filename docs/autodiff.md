# Automatic differentiation

Parakeet provides composable **reverse-mode**, **forward-mode**, and
**higher-order** automatic differentiation directly over its operator IR. Every
differentiated function compiles into ordinary fused, tiled, parallel CPU or
GPU loops alongside the rest of your program:

| API | What it computes |
|---|---|
| `par.grad(f, argnum=0)` | Reverse-mode gradient of a scalar-valued function |
| `par.value_and_grad(f, argnum=0)` | Forward value and reverse-mode gradient in one compiled call |
| `par.jvp(f, primals, tangents)` | Forward-mode Jacobian-vector product `(f(*primals), J_f(*primals) · tangents)` |
| `par.vjp(f, *primals)` | General non-scalar reverse-mode `(y, pullback)` for array, tuple, record, or `RaggedArray` outputs |
| `par.linearize(f, *primals)` | Forward value `y` plus reusable linear JVP function `t -> J_f(x) · t` |
| `par.linear_transpose(f, *primals)` | Exact linear transpose `u -> Lᵀ(u)` of a linear function `L` |
| `par.hvp(f, primals, tangents)` | Forward-over-reverse Hessian-vector product `∇²f(x) · v` |
| `par.jacfwd(f, argnum=0)` / `par.jacrev(f, argnum=0)` | Full Jacobian matrix via forward-mode or reverse-mode AD |
| `par.hessian(f, argnum=0)` | Full Hessian matrix via forward-over-reverse AD |
| `par.stop_gradient(x)` | Zero-copy gradient & tangent barrier |
| `@par.checkpoint` | Rematerialize a block's intermediates during backpropagation |
| `@par.custom_vjp` / `@par.custom_jvp` | Attach user-defined reverse-mode or forward-mode derivative rules |

## Reverse-mode gradients (`grad` and `value_and_grad`)

```python
import numpy as np
import parakeet as par


def loss(w, x, y):
  return np.mean((np.tanh(x @ w) - y) ** 2)


g = par.grad(loss)(w, x, y)  # d loss / d w, shaped like w
value, g = par.value_and_grad(loss)(w, x, y)
gw, gx = par.grad(loss, (0, 1))(w, x, y)  # several arguments at once
```

For `par.grad` and `par.value_and_grad`, the function returns a single
floating-point scalar. Differentiated arguments may be floats, float arrays,
`RaggedArray`s, or arbitrarily nested tuples, `NamedTuple`s, dicts, and frozen
`@dataclass` records (discrete metadata fields such as `int`, `bool`, or `str`
pass through cleanly as zero-tangent static leaves):

```python
def mlp(params, x, y):
  (w1, b1), (w2, b2) = params
  h = np.tanh(x @ w1 + b1)
  return np.mean((h @ w2 + b2 - y) ** 2)


(gw1, gb1), (gw2, gb2) = par.grad(mlp)(params, x, y)
```

`par.grad(f)` can be called from plain Python, where it compiles on first use,
or inside a `@par.jit` function, where the entire training step becomes one
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
trains a convolutional network.

## Forward-mode AD, general VJPs, and linear transposes

Use `par.jvp` for directional derivatives in a single forward pass (no backward
tape, O(1) extra memory even through long `for`/`while` loops), and `par.vjp`
when differentiating functions with vector, matrix, tuple, record, or
`RaggedArray` outputs:

```python
def f(x, w):
  z = np.tanh(x @ w)
  return z, np.sum(z * z, axis=1)


# Forward-mode directional derivative (JVP)
(y1, y2), (dy1, dy2) = par.jvp(f, (x, w), (vx, vw))

# General non-scalar reverse-mode pullback (VJP)
(y1, y2), pullback = par.vjp(f, x, w)
gx, gw = pullback((u1, u2))
```

For any differentiable function `f`, the transpose identity
`⟨jvp(f, x, v), u⟩ == ⟨v, vjp(f, x)(u)⟩` holds to machine precision.
`par.linearize(f, *primals)` returns `(y, lin_fn)` for repeated JVP evaluation
at a point, and `par.linear_transpose(L, *example_inputs)` constructs the exact
adjoint `Lᵀ` of a linear operator.

## Higher-order derivatives, HVPs, and Jacobians

`par.hvp` computes Hessian-vector products via **forward-over-reverse AD**
(`jvp(grad(f))`) in one fused pass; including through checkpointed loops,
LayerNorm, and **zero-$O(T^2)$ rematerialized attention**:

```python
def objective(w, x):
  h = np.tanh(x @ w)
  return np.sum(h * h * h)


hv = par.hvp(lambda w_: objective(w_, x), w, v)  # ∇²f(w) · v (forward-over-reverse)
H = par.hessian(lambda w_: objective(w_, x))(w)  # full Hessian matrix
J_fwd = par.jacfwd(lambda w_: np.tanh(x @ w_))(w)  # forward-mode Jacobian
J_rev = par.jacrev(lambda w_: np.tanh(x @ w_))(w)  # reverse-mode Jacobian
```

Nested reverse-over-reverse `par.grad(par.grad(...))` also works through
element-wise operations, reductions, matrix products, views, `np.stack` /
`np.concatenate`, `np.cumsum`, gather indexing (`x[idx]`), scatter-add
(`np.add.at`, `par.scatter_add`), and scalar 1-to-1 `par.map`:

```python
def penalty(w, x):
  g = par.grad(loss)(w, x)
  return np.sum(g * g)


par.grad(penalty)(w, x)  # 2 H g, reverse over reverse
```

## What can be differentiated

| Category | Supported operations |
|---|---|
| Arithmetic and math | `+ - * / **`, `np.exp`, `np.tanh`, `np.maximum`, `np.where`, `np.abs`, `np.clip`, … with broadcasting |
| Reductions | `np.sum`, `np.mean`, `np.var`, `np.std`, `np.max`, `np.min`, `np.prod` (zero-safe), `np.linalg.norm` with `axis`/`keepdims`; `par.reduce` with `np.add`, `np.multiply`, `np.minimum`, `np.maximum`, or arbitrary user folds |
| Linear algebra | `@`, `np.dot`, `np.matmul` for vectors, matrices, and batches; `np.einsum` |
| Scans, sorting, and selection | `np.cumsum`, `par.scan`, `np.sort`, `par.topk`, `par.topk_fold` |
| Indexing, lookup, and scatter | Views (slice, `T`, `reshape`, `ravel`, `np.transpose`), gather `x[idx]`, `FrozenHashTable.get`, `SortedIndex.get`, `np.add.at`, canonical `for`-loop `out[idx[i]] += val[i]`, `par.scatter_add` |
| Building arrays | `np.concatenate`, `np.stack`, `np.array([a, b, c])`, comprehensions |
| Parallel operators | `par.map` / `par.each`, `par.allpairs`, `par.imap`, `par.fold`, `par.wavefront` / `par.soft_align` |
| Structured types | Nested tuples, `NamedTuple`, fixed string-key dicts, frozen `@dataclass`, `RaggedArray` |

Derivatives of `min`, `max`, `abs`, and `topk` at a tie follow standard
conventions: ties share the derivative equally.

## Loops and branches

Parakeet differentiates **both** small static loops and arbitrary runtime-bound
`for` and `while` loops in forward and reverse mode:

* **Small static `for` loops** (compile-time trip count `<= 32` or iterating
  over a heterogeneous tuple of weight matrices) are **unrolled** onto the tape
  for zero checkpoint overhead and cross-iteration fusion.
* **Runtime-bound and long `for` loops** (`for t in range(n):`, `for x in arr:`,
  or static ranges longer than 32 iterations; including 10,000+ iterations)
  compile to an **IR-level `O(sqrt(T))` checkpointed reverse loop** (`ForRange`)
  with constant IR size and bounded memory.
* **Data-dependent `while` loops** (`while err > tol:`) compile to a fixed-slot
  stride-doubling **checkpointed reverse `While` loop** (`O(log T)` slots, zero
  dynamic reallocation inside the loop).
* **Custom `par.reduce` folds** use `O(sqrt(N))` checkpointed reverse mode, and
  1-D **`par.scan`** with any user combiner is differentiable directly.
* **Forward-mode `par.jvp`** propagates tangents alongside the loop in a single
  forward pass with O(1) extra memory.

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

An `if` statement whose condition is only known at run time is differentiated
in reverse mode by running **both** branches and selecting between their
results (like `np.where`). Branches touching differentiated values may only
assign variables: no `return`, `raise`, `break`, or array stores. A guard that
only raises (`if n == 0: raise ValueError(...)`) is allowed without running
both sides.

## Custom rules, `stop_gradient`, and `checkpoint`

* **`par.stop_gradient(x)`** detaches a scalar, array, tuple, record, or
  `RaggedArray` from both reverse-mode `grad`/`vjp` and forward-mode `jvp` with
  zero copy:
  ```python
  y = x + par.stop_gradient(np.round(x) - x)
  ```
* **`@par.custom_vjp`** attaches custom `(fwd, bwd)` rules via `.defvjp(fwd, bwd)`: supporting `NamedTuple`, frozen `@dataclass`, and `RaggedArray` residuals,
  higher-order AD, and Implicit Function Theorem differentiation through
  `while`-loop equilibrium solvers without storing iteration history.
* **`@par.custom_jvp`** attaches custom forward-mode rules via `.defjvp(jvp_rule)`:
  ```python
  @par.custom_jvp
  def smooth_relu(x):
    return np.maximum(x, 0.0)


  @smooth_relu.defjvp
  def _smooth_relu_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    return np.maximum(x, 0.0), dx / (1.0 + np.exp(-x))
  ```
* **`@par.checkpoint`** (or `par.checkpoint(fn)(*args)`) drops a block's
  forward intermediates and rematerializes them during backpropagation,
  composing cleanly with Parakeet's
  [algebraic rematerialization](algebra.md#law-2-algebraic-rematerialization-in-reverse-mode-ad)
  rules for attention, linear cross-entropy, softmax, and LayerNorm / RMSNorm.

## Writing into arrays

Stores are differentiated: assigning a slice, an element, or updating in place
(`out[i] += v`, `np.add.at(out, idx, vals)`) all work cleanly:

```python
out = np.zeros((n, 3))
out[:3] = x[:3] @ w
out[3:] = np.tanh(x[3:] @ w)
out += bias
```

Going backwards, whatever adjoint arrived for the stored region belongs to the
value that was stored, and the region starts again from zero. Overwriting a
value that an earlier operation's derivative still has to read
(`y = np.tanh(h); h[0] = 5.0`) is refused loudly, because the derivative of
`tanh` needs the original `h`.

## The gradient is right or there is none

Parakeet never returns a derivative that silently drops part of your function.
If something on the path from inputs to output lacks a derivative rule; such
as `np.cumprod`, `np.convolve`, or an arbitrary indexed overwrite
`out[idx] = v`; you get an `UnsupportedError` naming it:

```text
UnsupportedError: cannot differentiate through cumprod() (line 12): its derivative is not implemented
```

## How it works

While a function is elaborated, reverse-mode AD records each array operator on
an algebraic tape (and forward-mode AD propagates symbolic tangents alongside
primals). Because rules operate at the index-operator level before lowering:

* Element-wise operators differentiate symbolically via partial derivatives.
* `parakeet/transforms/algebraic_vjp.py` runs a composable registry of
  algebraic rematerialization rules (`ALGEBRAIC_AD_RULES`) for FlashAttention
  (VJP, JVP, and HVP with zero $O(T^2)$ allocations), linear cross-entropy,
  softmax, and LayerNorm / RMSNorm; both at top level and inside checkpointed
  loops.
* Scalars and views are not recorded on the tape: scalar adjoints push straight
  down the expression tree, and a view of an array is the same view of its
  adjoint.

See [`parakeet/transforms/autodiff.py`](https://github.com/iskandr/parakeet/blob/master/parakeet/transforms/autodiff.py),
[`parakeet/transforms/jvp.py`](https://github.com/iskandr/parakeet/blob/master/parakeet/transforms/jvp.py),
and [`parakeet/transforms/algebraic_vjp.py`](https://github.com/iskandr/parakeet/blob/master/parakeet/transforms/algebraic_vjp.py).
