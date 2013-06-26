import time

class timer(object):
  def __init__(self, name = None, newline = True):
    self.name = name 
    self.start_t = time.time()
    self.newline = newline
    
  def __enter__(self):
    self.start_t = time.time()
  
  def elapsed(self):
    return time.time() - self.start_t
  
  def __exit__(self,*exit_args):
    t = self.elapsed()
    if self.newline:
      print 
    if self.name is None:
      print "Elasped time %0.4f" % t 
    else:
      print "%s : %0.4f" % (self.name, t) 

from parakeet import jit
from numba import autojit
import numpy as np 

def compare_perf(fn, args, numba= True, cpython = True):
  
  parakeet_fn = jit(fn)
  
  with timer('Parakeet #1'):
    parakeet_result = parakeet_fn(*args)

  with timer('Parakeet #2'):
    parakeet_result = parakeet_fn(*args)

  if numba:
    numba_fn = autojit(fn)

    with timer('Numba #1'):
      numba_result = numba_fn(*args)

    with timer('Numba #2'):
      numba_result = numba_fn(*args)
  
    assert np.allclose(parakeet_result, numba_result)  
  
  if cpython:
    with timer('Python'):
      python_result = fn(*args)
    assert np.allclose(parakeet_result, python_result)  
