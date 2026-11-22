import numpy as np
import parakeet
from parakeet.testing_helpers import expect, run_local_tests


def find_border(is_background):
  h, w = is_background.shape[:2]

  top, left, right, bottom = (-1, -1, -1, -1)
  # find top
  for i in range(h):
    if is_background[i, :].sum() != w and top == -1:
      top = i

  for i in range(h - 1, 0, -1):
    if is_background[i, :].sum() != w and bottom == -1:
      bottom = i

  for i in range(w):
    if is_background[:, i].sum() != h and left == -1:
      left = i

  for i in range(w - 1, 0, -1):
    if is_background[:, i].sum() != h and right == -1:
      right = i

  return top, left, right, bottom


def test_find_border():
  # (the original test used np.empty: uninitialised bytes that are not valid bools)
  is_background = np.ones((20, 20), dtype=np.bool)
  is_background[3:15, 5:17] = np.random.default_rng(0).random((12, 12)) > 0.5
  is_background[3, 8] = is_background[14, 9] = is_background[7, 5] = is_background[9, 16] = False
  expected = find_border(is_background)
  assert expected == (3, 5, 16, 14)
  expect(find_border, [is_background], expected)


if __name__ == "__main__":
  run_local_tests()
