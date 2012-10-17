import numpy as np

def map(f, xs, out = None, axis = None):
  assert axis is None
  if out is None:
    out = np.zeros_like(xs)
  for i in enumerate(xs):
    out[i] = f(xs[i])
  return out

def reduce(f, xs, init  = None, axis = None):
  if init is None:
    init = f(xs[0], xs[1])
    start_idx = 2
  else:
    start_idx = 0
  n = len(xs)
  while start_idx < n:
    init = f(init, xs[start_idx])
    start_idx += 1
  return init 

def scan(f, xs, init = None, axis = None):
  pass  
    