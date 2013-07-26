from ..frontend import jit 

import numpy as np 


@jit
def conjugate(x):
  """
  For now we don't have complex numbers so this is just the identity function
  """
  return x


@jit
def real(x):
  """
  For now we don't have complex types, so real is just the identity function
  """
  return x  

def _scalar_sign(x):
  return x if x >= 0 else -x 

@jit
def sign(x):
  return map(_scalar_sign(x))

@jit 
def reciprocal(x):
  return 1.0 / x

@jit 
def rad2deg(rad):
  return rad * 180 / 3.141592653589793

@jit
def deg2rad(deg):
  return deg * 3.141592653589793 / 180 

@jit 
def hypot(x,y):
  return np.sqrt(x**2 + y**2)

@jit 
def square(x):
  return x * x 

#
# Copied and modified from PyPy's micronumpy sub-package.
# 
# URL: https://bitbucket.org/pypy/pypy/raw/
#      811e23458661f61c0fa54aa2f58eb5f683558576/
#      pypy/module/micronumpy/types.py
# 
# -------------------------------------------------------

@jit 
def logaddexp(self, x, y):
  diff = x - y 
  if diff > 0:
    return x + np.log1p(np.exp(-diff))
  elif diff <= 0:
    return y + np.log1p(np.exp(diff))
  else:
    return x + y 

@jit   
def log2_1p(self, x):
  return 1.0 / np.log(2) * np.log1p(x)

@jit 
def logaddexp2(self, x, y):
  diff = x - y 
  if diff > 0:
    return x + log2_1p(2 ** -diff)
  elif diff <= 0:
    return y + log2_1p(2 ** diff)
  else:
    return x + y
  
# ------------------------------------------------------- 
#
#