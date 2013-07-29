from ..frontend import jit 

import numpy as np 
import math 


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
  if x > 0:
    return 1
  elif x < 0:
    return -1 
  else:
    return 0
  
@jit
def sign(x):
  return map(_scalar_sign,x)

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

@jit 
def logaddexp(x, y):
  diff = x -y 
  return np.where(diff > 0, np.log1p(np.exp(-diff)), np.log1p(np.exp(diff)))

@jit   
def log2_1p(x):
  return 1.0 / np.log(2) * np.log1p(x)


@jit 
def logaddexp2(x, y):
  diff = x - y 
  return np.where(diff > 0,  x + log2_1p(math.pow(2, -diff)) , y + log2_1p(math.pow(2, diff)))

 
@jit 
def true_divide(x, y):
  """
  Not exactly true divide, since I guess it's sometimes supposed to stay an int
  """
  return (x + 0.0) / (y + 0.0)

@jit 
def floor_divide(x, y):
  return math.floor(x / y)  
