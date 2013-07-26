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
  return 1 / x


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
