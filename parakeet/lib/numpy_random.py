import numpy as np 

from ..frontend.decorators import jit  

@jit 
def shuffle(x):
  n = len(x)
  r = np.random.randint(0, n, n)
  for i in xrange(n):
    j = np.fmod(r[i], i)
    old_xj = x[j]
    x[j] = x[i]
    x[i] = old_xj
    
@jit
def permutation(x):
  y = np.zeros_like(x)
  return shuffle(y)

  