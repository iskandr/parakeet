import numpy as np

import parakeet
from parakeet import jit, testing_helpers 

def init(W):
  C = W.copy()
  #if E is not None:
  #  C[~E] = np.inf
  for i in xrange(C.shape[0]):
    C[i,i] = 0
  return C

def loop_dists(W):
  C = init(W)
  m,n = C.shape
  #assert m == n
  for k in range(n):
    for i in range(n):
      for j in range(n):
        C[i,j] = min(C[i,j], C[i,k] + C[k,j])
  return C

"""
def outer_dists(W,E = None):
  C = init(W,E)
  m,n = C.shape
  assert m == n
  vertices = np.arange(n)
  for k in vertices:
    from_k = C[:, k]
    to_k = C[k, :]
    A = np.add.outer(from_k, to_k)
    mask = (A<=C)
    C = mask * A + (1-mask) * C
    A = np.add.outer(from_k, to_k)
    mask = (A<=C)
    C = mask * A + ~mask * C
  return C
"""

"""
def np_dists(W, E = None):
  C = init(W,E)
  n = C.shape[0]
  for k in range(n):
    C = np.minimum(C, np.tile(C[:,k].reshape(-1,1),(1,n)) + np.tile(C[k,:],(n,1)))
  return C
"""
"""
def parakeet_dists(W, E = None):
  C = init(W,E)
  m,n = C.shape
  # assert m == n
  vertices = np.arange(n)
  def min_row(c_row, a_row):
    return map(min, c_row, a_row)
  
  for k in vertices:
    from_k = C[:, k]
    to_k = C[k, :]
    A = parakeet.allpairs(parakeet.add, from_k, to_k)
    C = parakeet.each(min_row, C, A)
  return C
"""
def test_dists():
  n = 10
  W = (np.random.randn(n,n)*2)**2

  testing_helpers.expect(loop_dists, [W], loop_dists(W))  
  """
  C1 = outer_dists(W)
  C2 = np_dists(W)
  assert np.all(np.ravel(C1 == C2)), \
      "C2 wrong: Expected %s but got %s" % (C1, C2)
      
  C3 = parakeet_dists(W)
  assert np.all(np.ravel(C1 == C3)), \
      "Parakeet wrong: Expected %s but got %s" % (C1, C3)
  """
if __name__ == '__main__':
  testing_helpers.run_local_tests()
