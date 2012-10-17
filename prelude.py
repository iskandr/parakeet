
import numpy as np 
from function_mapping import remap
from adverbs import map, reduce, scan 
@remap(np.alen)
def alen(arr):
  return arr.shape[0]
    
@remap(np.shape)
def shape(arr):
  return arr.shape 

@remap(np.asarray)
def asarray(a, dtype = None, order = None):
  pass

@remap(np.ndim)
def ndim(arr):
  return len(asarray(arr).shape)

@remap(np.zeros_like)
def zeros_like(arr, dtype = None):
  arr = asarray(arr)
  if dtype is None:
    dtype = arr.dtype
  zero = dtype.type(0)
  return map(lambda _: zero, arr, axis = None)
  
