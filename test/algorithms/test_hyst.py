#
# Test for Issue #18: https://github.com/iskandr/parakeet/issues/18
#
import numpy as np 
from parakeet import jit
from parakeet.testing_helpers import expect, run_local_tests

def hyst_(mag, edge_map, labels, num_labels, high_thresh):
  for i in range(num_labels):
    if np.max(mag[labels == i]) < high_thresh:
        edge_map[labels == i] = False
  return edge_map

def hyst(n):
  mag = np.arange(n)
  labels = np.ones(n)
  labels[int(n/2)] = 0
  edge_map = np.ones(n, dtype=np.bool)
  num_labels = 2
  high_thresh = int(n/2)
  return hyst_(mag, edge_map, labels, num_labels, high_thresh)

"""
TODO: implement boolean indexing
def test_hyst():
  expect(hyst, [10], hyst(10))
"""
if __name__ == '__main__':
  run_local_tests()
  
