import numpy as np
import parakeet 
import testing_helpers

def init(W, E = None):
  C = W.copy()
  if E:
    C[~E] = np.inf
  for i in xrange(C.shape[0]):
    C[i,i] = 0
  return C
  
def loop_dists(W, E = None):
  C = init(W,E)
  m,n = C.shape
  assert m == n
  for k in range(n):
    for i in range(n):
      for j in range(n):
        C[i,j] = min(C[i,j], C[i,k] + C[k,j])
  return C

def outer_dists(W, E = None):
  C = init(W,E)
  m,n = C.shape
  assert m == n
  
  for k in range(n):
    from_k = C[:, k]
    to_k = C[k, :]
    
    A = np.add.outer(from_k, to_k)
    mask = (A<=C)
    C = mask * A + (1-mask) * C
  return C

def parakeet_dists(W, E = None):
  C = init(W,E)
  m,n = C.shape 
  assert m == n 
  vertices = np.arange(n)
  for k in vertices:
    def update(i,j):
      old_val = C[i,j]
      new_val = C[i,k] + C[k,j]
      if old_val < new_val:
        return old_val
      else:
        return new_val
    C = parakeet.allpairs(update, vertices, vertices)
  return C
    
def test_dists():
  n = 4
  W = (np.random.randn(n,n)*2)**2
  C1 = loop_dists(W)
  C2 = outer_dists(W)
  assert np.all(np.ravel(C1 == C2)), \
      "C2 wrong: Expected %s but got %s" % (C1, C2)
  C3 = parakeet_dists(W)
  assert np.all(np.ravel(C1 == C3)), \
      "Parakeet wrong: Expected %s but got %s" % (C1, C3)
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()