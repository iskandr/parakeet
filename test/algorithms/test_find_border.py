import numpy as np
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
  is_background = np.empty((20, 20), dtype=np.bool)
  expect(find_border, [is_background], find_border(is_background))


if __name__ == '__main__':
  run_local_tests()