from parakeet import jit, testing_helpers
import numpy as np


 
def kernel(X):
  return np.array([[np.sum(np.minimum(x,y)) for y in X] for x in X]) 

def test_histogram_intersection_kernel():
  n,d = 3,4
  X = np.arange(n*d).reshape((n,d))
  K = kernel(X)
  testing_helpers.expect(kernel, [X], K)

if __name__ == "__main__":
  testing_helpers.run_local_tests


