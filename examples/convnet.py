"""A small convolutional network trained with parakeet.grad.

Convolution is written the way it is defined: for every offset of the kernel,
a matrix product between the filters and a shifted view of the image.  The
loop over offsets runs over the shape of the kernel, so it is unrolled, and
each specialization of the network is for one kernel size.
"""

import numpy as np

from parakeet import each, value_and_grad


def conv(x, w):
  """Valid cross-correlation: x is (C, H, W), w is (F, C, K, K)."""
  c, k = w.shape[1], w.shape[2]
  rows, cols = x.shape[1] - k + 1, x.shape[2] - k + 1
  out = np.zeros((w.shape[0], rows * cols))
  for i in range(k):
    for j in range(k):
      patch = x[:, i : i + rows, j : j + cols].reshape(c, rows * cols)
      out = out + w[:, :, i, j] @ patch
  return out.reshape(w.shape[0], rows, cols)


def pool(x):
  """2x2 mean pooling of (F, H, W)."""
  f, rows, cols = x.shape[0], x.shape[1] // 2, x.shape[2] // 2
  return np.mean(x[:, : 2 * rows, : 2 * cols].reshape(f, rows, 2, cols, 2), axis=(2, 4))


def logits(params, image):
  w1, b1, w2, b2, wd, bd = params
  h = pool(np.maximum(conv(image, w1) + b1[:, None, None], 0.0))
  h = pool(np.maximum(conv(h, w2) + b2[:, None, None], 0.0))
  return h.ravel() @ wd + bd


def image_loss(params, image, label):
  z = logits(params, image)
  z = z - np.max(z)
  return np.log(np.sum(np.exp(z))) - z[label]


def loss(params, images, labels):
  return np.mean(each(lambda image, label: image_loss(params, image, label), images, labels))


def predict(params, images):
  return np.array([np.argmax(logits(params, image)) for image in images])


def bars(n, size, rng):
  """Noisy images of a horizontal, vertical or diagonal bar."""
  labels = rng.integers(0, 3, n)
  images = rng.normal(0.0, 0.3, (n, 1, size, size))
  at = rng.integers(2, size - 2, n)
  for image, label, p in zip(images, labels, at):
    if label == 0:
      image[0, p, :] += 1.0
    elif label == 1:
      image[0, :, p] += 1.0
    else:
      image[0, np.arange(size), np.arange(size)] += 1.0
  return images, labels


def init(rng, size=16, filters=6, classes=3):
  after = ((size - 2) // 2 - 2) // 2
  return (
    rng.normal(0.0, 0.9, (filters, 1, 3, 3)), np.zeros(filters),
    rng.normal(0.0, 0.3, (filters, filters, 3, 3)), np.zeros(filters),
    rng.normal(0.0, 0.3, (filters * after * after, classes)), np.zeros(classes),
  )  # fmt: skip


def main(steps=60, batch=32, rate=0.3, seed=0):
  rng = np.random.default_rng(seed)
  params = init(rng)
  step = value_and_grad(loss)
  first = None
  for _ in range(steps):
    images, labels = bars(batch, 16, rng)
    value, grads = step(params, images, labels)
    params = tuple(p - rate * g for p, g in zip(params, grads))
    first = value if first is None else first
  images, labels = bars(200, 16, rng)
  accuracy = float(np.mean(predict(params, images) == labels))
  assert value < first and accuracy > 0.8, (first, value, accuracy)
  return first, value, accuracy


if __name__ == "__main__":
  print("loss {:.3f} -> {:.3f}, accuracy {:.2f}".format(*main()))
