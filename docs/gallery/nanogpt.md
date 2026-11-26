# Train nanoGPT from scratch

This character-level GPT learns to predict the next character in a short passage.
For the fragment `para`, one training sequence has inputs `p a r` and targets
`a r a`. At the second position, the model can see `p a` and must predict `r`.
A causal mask enforces that rule even though the model computes predictions for
every position at once.

The six Python blocks below make one complete file. You can copy them in order
into `train_gpt.py` and run `python train_gpt.py`, or run the [same file from
the repository](https://github.com/iskandr/parakeet/blob/master/examples/nanogpt.py).
To run the repository version, use Python 3.11 or newer and a C compiler.

```bash
git clone https://github.com/iskandr/parakeet.git
cd parakeet
python -m pip install -e .
python -m examples.nanogpt
```

The text is embedded in the script. The default run trains on batches of 16
sequences, each 32 characters long, for 60 steps.

## Normalize one position at a time

One input sequence becomes a matrix of shape `(context, width)`. Each row holds
one character's embedding. `LayerNorm` is a plain `@dataclass` with a `__call__`
method that normalizes across the features of each row. Parakeet lowers
`@dataclass` modules and their pure methods directly during compilation with no
`PyObject` overhead. Later, each row passed to `softmax` holds attention scores
from one position to every position in the sequence; subtracting the row
maximum keeps the exponential stable.

```python
--8<-- "examples/nanogpt.py:gpt_setup"
```

## Mix information across visible characters

`Block` groups one transformer layer's norms, attention projections (`wq`, `wk`,
`wv`, `wo`), and feed-forward weights (`w1`, `b1`, `w2`, `b2`). Each attention
head uses part of the model width, computes a `(context, context)` score matrix,
masks future positions before softmax, and projects back through `wo`. The
block adds the attention output to its input, runs a two-layer `tanh` MLP on
each position, and adds that residual too.

```python
--8<-- "examples/nanogpt.py:gpt_attention"
```

## Score one sequence, then a batch

`NanoGPT.sequence_loss` embeds the token IDs (`self.wte[tokens] + self.wpe`),
runs each block in `self.blocks`, projects to vocabulary logits with
`self.ln_f(x) @ self.head`, and computes the mean negative log-probability of
the next character at each position.

In `loss`, both `tokens` and `targets` have shape `(batch, context)`.
`par.map` applies `model.sequence_loss` across every `(inp, tgt)` pair in the
batch, and `np.mean` averages the per-sequence losses.
`par.value_and_grad(loss)` compiles one native function returning both the
scalar loss and a `NanoGPT` dataclass holding the gradient of every parameter
array.

```python
--8<-- "examples/nanogpt.py:gpt_loss"
```

## Build the model

`init` constructs a `NanoGPT` instance containing the token and position
embedding tables, a tuple of `Block` modules, the final `LayerNorm`, and the
output projection `head`. With the defaults, `wte` has shape `(vocab, 32)` and
`head` has shape `(32, vocab)`.

```python
--8<-- "examples/nanogpt.py:gpt_params"
```

## Update every parameter with Adam

`loss_and_grad` returns the batch loss and a `NanoGPT` gradient tree matching
the structure of `model`. `tree_map` walks matching tuples and dataclass fields
together so Adam can update embeddings, norms, biases, and projections with the
same few lines. `m` tracks the moving average of gradients and `v` tracks the
moving average of squared gradients.

```python
--8<-- "examples/nanogpt.py:gpt_update"
```

## Train on shifted character windows

`main` assigns one integer ID to each distinct character in `TEXT`. For a
sample starting at offset `i`, `tokens` starts at `i` and `targets` starts at
`i + 1`. Stacking those windows gives the two arrays consumed by `par.map`.
The lower triangle of `causal` includes the diagonal, so each position can read
its own input when predicting the following character.

```python
--8<-- "examples/nanogpt.py:gpt_train"
```

The first training step compiles the forward and gradient kernels; subsequent
steps reuse the compiled native code. With `--verbose` (the default when run as
a script), `main` prints the cross-entropy loss at steps 1, 20, 40, and 60 and
asserts that training makes progress (final loss below 80% of step 1). Using
the fixed `seed=0` above, the printed loss drops from **3.173** at step 1 to
**2.307** (step 20), **1.796** (step 40), and **1.250** (step 60) on GPU (with
nearly identical values on CPU).

For larger four-head training runs where the Adam update is also compiled into
the training step, see the [nanoGPT training benchmark](../release-benchmarks.md#end-to-end-nanogpt-training-nvidia-l4).
The compact two-head walkthrough above keeps Adam in plain Python so every line
is easy to follow.
