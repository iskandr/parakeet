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
one character's embedding. `layer_norm` normalizes across the features of that
row. Later, each row passed to `softmax` contains the attention scores
from one position to every position in the sequence. Subtracting
the row maximum keeps the exponential stable.

```python
--8<-- "examples/nanogpt.py:gpt_setup"
```

## Mix information across visible characters

Each attention head uses part of the model width. For each position, `q`
represents what it is looking for, `k` what each position can match, and `v`
what that position contributes. Their score matrix has shape `(context, context)`.
In the `para` example, row two may give weight to the first two inputs. The
mask suppresses the score of the third before softmax.

The heads contribute to a shared output through `wo`. A transformer block adds
this attention output back to its input. It then applies a two-layer network
to each position and adds that result back too.

```python
--8<-- "examples/nanogpt.py:gpt_attention"
```

## Score one sequence, then a batch

`wte[tokens]` looks up one embedding per character ID. Adding `wpe` gives
each position its own embedding as well. After the transformer blocks, the
output matrix `head` maps each position to one score per character in the
vocabulary. `sequence_loss` converts those scores to log probabilities, picks
the entry for the actual next character at each position, and averages the
negative values.

This is where Parakeet's explicit operators enter. In `loss`, both `tokens`
and `targets` have shape `(batch, context)`. The two arguments to
`loss_for_sequence` are matching rows, and all rows use the same parameters.
`dsl.map` produces one scalar loss per sequence. Their mean is the batch loss.
`value_and_grad(loss)` differentiates that scalar with respect to `params`, the
first argument of `loss`.

```python
--8<-- "examples/nanogpt.py:gpt_loss"
```

## Build the parameter arrays

`init` constructs the tuple that `sequence_loss` unpacks. In order, it contains
the token embedding table, the position embedding table, the tuple of
transformer blocks, the final normalization gain and bias, and the output
matrix. With the defaults, the token embedding table has shape `(vocab, 32)`
and the output matrix has shape `(32, vocab)`.

Inside `one_block`, the first gain and bias normalize the input to attention.
The next four matrices project queries, keys, values and the combined output.
A second gain and bias precede the two weight–bias pairs of the position-wise
network. This is the order in which `block` unpacks them.

```python
--8<-- "examples/nanogpt.py:gpt_params"
```

## Update every parameter with Adam

A call to `loss_and_grad` returns the batch loss and a gradient array for
each parameter array. The gradients follow the same nested tuple structure
as the parameters. `tree_map` walks corresponding leaves together so Adam can
update an embedding, a bias or a projection matrix with the same formula.
`m` tracks a moving average of the gradients and `v` tracks a moving average
of their squares. Both travel with the parameters into the next step.

```python
--8<-- "examples/nanogpt.py:gpt_update"
```

## Train on shifted character windows

`main` assigns one integer ID to each distinct character in `TEXT`. For a
sample starting at offset `i`, `tokens` starts at `i` and `targets` starts at
`i + 1`. Stacking those windows gives the two arrays consumed by the batch map.
The lower triangle of `causal` includes the diagonal, so a position can read
its own input when predicting the following character.

```python
--8<-- "examples/nanogpt.py:gpt_train"
```

The first training step includes compilation. The script prints losses at
steps 1, 20, 40 and 60, and checks that the final loss is below 80% of the
initial loss. In the seeded Colab run the four losses were **3.173, 2.307,
1.796 and 1.250**.

The [GPU training benchmark](../release-benchmarks.md#nanogpt-training)
uses larger four-head models and compiles Adam into the training step. The
two-head example above applies Adam in Python.
