# Train nanoGPT in Parakeet

This character-level GPT learns to predict the next character in a short passage.
For the fragment `para`, one training sequence has inputs `p a r` and targets
`a r a`. A lower-triangular causal mask ensures position `t` only attends to
positions `0..t` while computing every position's logits in parallel.

Below we show **two complete implementations** of the exact same model:

1. **Pythonic objects with `par.nn` and `par.optim`**
   ([`examples/nanogpt.py`](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt.py)):
   a normal Python class inheriting from `par.nn.Module` with a standard
   `__init__`, `par.nn.TransformerBlock`, `par.nn.cross_entropy`,
   `par.optim.Adam`, and `par.Vocab.from_text`.
2. **Minimalist functional from scratch**
   ([`examples/nanogpt_functional.py`](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt_functional.py)):
   plain dicts and tuples of NumPy arrays with every layer (layer norm, softmax,
   multi-head attention, cross-entropy, and Adam) written inline in raw NumPy—no
   `par.nn` or `par.optim` helpers at all.

Run either version directly from a checkout:

```bash
python -m examples.nanogpt
python -m examples.nanogpt_functional
```

---

## How `par.nn.Module` compiles Python objects

Parakeet does not require external `init()` factories or manual dataclass
unpacking to compile neural networks. Any class inheriting from `par.nn.Module`
(or setting `__parakeet_record__ = True`) is lowered at `@par.jit` and
`par.value_and_grad` boundaries into a flat compile-time record of its
`self.__dict__` attributes:

* **Array attributes** (`weight`, `bias`, `gain`, `wpe`) become native typed
  array parameters that receive gradients under `par.value_and_grad`, transfer
  cleanly with `par.to_device`, and update via `par.tree_map_arrays` /
  `par.optim`.
* **Static attributes** (`heads`, `eps`, `activation`) become compile-time
  constants—zero runtime memory and untouched by optimizers.
* **Methods** (`__call__`, helper methods) inline completely during elaboration
  with zero `PyObject` overhead.

Because every layer in `par.nn` (`Linear`, `LayerNorm`, `Embedding`, `MLP`,
`CausalSelfAttention`, `TransformerBlock`) is just plain NumPy under the hood,
Parakeet's monoidal FlashAttention tiling, warp-row LayerNorm reduction, and
rematerialized AD trigger automatically on both versions.

---

## Model definition

=== "With `par.nn` and `par.optim`"

    ```python
    --8<-- "examples/nanogpt.py:gpt_model"
    ```

=== "Minimalist functional from scratch"

    ```python
    --8<-- "examples/nanogpt_functional.py:gpt_functional_primitives"
    ```

    ```python
    --8<-- "examples/nanogpt_functional.py:gpt_functional_loss"
    ```

---

## Tokenization and training loop

`par.Vocab.from_text(TEXT)` builds a compact character vocabulary backed by a
256-entry lookup table—the exact same categorical encoding pattern as
`par.Alphabet("ACGT")` in the [bioinformatics gallery](../bioinformatics.md).
Calling `vocab.encode(TEXT)` returns a 1D `int64` token array ready for
windowed batching.

```python
--8<-- "examples/nanogpt.py:gpt_train"
```

### Fusing the optimizer step on GPU

In the walkthrough above, `optimizer.update(model, grads)` runs in Python so
each step is easy to inspect. When training on a GPU, wrap both
`par.value_and_grad(loss)` and `par.optim.adam_step` inside one `@par.jit`
function and keep `(model, m, v)` resident on the device with `par.to_device`:

```python
@par.jit(backend="cuda", tensor_cores=True)
def train_step(model, m, v, step, tokens, targets, mask):
  value, grads = par.value_and_grad(loss)(model, tokens, targets, mask)
  model, m, v = par.optim.adam_step(model, grads, m, v, step, lr=3e-3)
  return value, model, m, v
```

That compiles forward, backward, and Adam into a single zero-round-trip device
step. See the [nanoGPT GPU benchmarks](../release-benchmarks.md#end-to-end-nanogpt-training-nvidia-l4)
for full timings comparing Parakeet FP32/TF32, PyTorch (`eager`, `SDPA`,
`torch.compile`), and JAX.
