import parakeet
import numpy as np 
parakeet.config.print_generated_code = True  

@parakeet.jit
def matmult(X,Y):
  return np.array([[np.dot(x,y) for y in Y.T] for x in X])


n, d = 120, 120
m = 120
dtype = 'float64'
X = np.random.randn(m,d).astype(dtype)
Y = np.random.randn(d,n).astype(dtype)
Z = matmult(X,Y)
