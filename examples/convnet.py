"""A small convolutional network trained with ``par.nn.Module`` and ``par.optim.sgd_step``.

Convolution is written directly as a sum of shifted matrix products over the
``K x K`` kernel offsets. Because the kernel size is static, the inner offset
loop unrolls cleanly during compilation.
"""

import numpy as np

import parakeet as par


def conv2d(x, w):
  """Valid cross-correlation: x is (C, H, W), w is (F, C, K, K)."""
  c, k = w.shape[1], w.shape[2]
  rows, cols = x.shape[1] - k + 1, x.shape[2] - k + 1
  out = np.zeros((w.shape[0], rows * cols), dtype=x.dtype)
  for i in range(k):
    for j in range(k):
      patch = x[:, i : i + rows, j : j + cols].reshape(c, rows * cols)
      out = out + w[:, :, i, j] @ patch
  return out.reshape(w.shape[0], rows, cols)


def avg_pool2x2(x):
  """2x2 mean pooling of (F, H, W)."""
  f, rows, cols = x.shape[0], x.shape[1] // 2, x.shape[2] // 2
  return np.mean(x[:, : 2 * rows, : 2 * cols].reshape(f, rows, 2, cols, 2), axis=(2, 4))


class ConvNet(par.nn.Module):
  def __init__(self, *, size=16, filters=6, classes=3, rng=None):
    g = np.random.default_rng(0) if rng is None else rng
    after = ((size - 2) // 2 - 2) // 2
    self.w1 = g.normal(0.0, 0.9, (filters, 1, 3, 3))
    self.b1 = np.zeros(filters)
    self.w2 = g.normal(0.0, 0.3, (filters, filters, 3, 3))
    self.b2 = np.zeros(filters)
    self.head = par.nn.Linear(filters * after * after, classes, rng=g, dtype=np.float64, std=0.3)

  def __call__(self, image):
    h = avg_pool2x2(par.nn.relu(conv2d(image, self.w1) + self.b1[:, None, None]))
    h = avg_pool2x2(par.nn.relu(conv2d(h, self.w2) + self.b2[:, None, None]))
    return self.head(h.ravel())

  def image_loss(self, image, label):
    return -par.nn.log_softmax(self(image), axis=0)[label]


def loss(model, images, labels):
  return np.mean(par.map(lambda image, label: model.image_loss(image, label), images, labels))


def predict(model, images):
  return np.array([np.argmax(model(image)) for image in images])


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


def main(steps=60, batch=32, rate=0.3, seed=0):
  rng = np.random.default_rng(seed)
  model = ConvNet(size=16, filters=6, classes=3, rng=rng)
  step = par.value_and_grad(loss)
  first = None
  for _ in range(steps):
    images, labels = bars(batch, 16, rng)
    value, grads = step(model, images, labels)
    model = par.optim.sgd_step(model, grads, lr=rate)
    first = value if first is None else first
  images, labels = bars(200, 16, rng)
  accuracy = float(np.mean(predict(model, images) == labels))
  assert value < first and accuracy > 0.8, (first, value, accuracy)
  return first, value, accuracy


if __name__ == "__main__":
  print("loss {:.3f} -> {:.3f}, accuracy {:.2f}".format(*main()))
