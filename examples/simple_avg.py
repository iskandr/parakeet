from parakeet import jit  

@jit 
def avg(x,y,z):
  """
  Return the average of three scalars or arrays
  (gets optimized into single traversal)
  """
  return (x + y + z) / 3.0

import numpy as np 

N = 20
x = np.random.randn(N)
y = np.random.randn(N)
z = np.random.randn(N)

assert np.allclose(avg(x,y,z), (x+y+z)/3.0)



