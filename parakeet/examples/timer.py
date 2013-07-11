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
  
  def __exit__(self, exc_type, exc_value, traceback):
    t = self.elapsed()
    if self.newline:
      print 
    s = "Elapsed time: " if self.name is None else "%s : " % self.name 
    if exc_type is None:
      s += "%0.4f" % t
    else:
      name = str(exc_type) if exc_type.__name__ is None else exc_type.__name__
      s += "FAILED with %s '%s'" % (name, exc_value)
    print s 
    # don't raise exceptions
    return exc_type is not KeyboardInterrupt

from parakeet import jit
from numba import autojit
import numpy as np 

def compare_perf(fn, args, numba= True, cpython = True):
  
  parakeet_fn = jit(fn)
  name = fn.__name__
  parakeet_result = None
  numba_result = None 
  cpython_result = None 

  with timer('Parakeet #1 -- %s' % name):
    parakeet_result = parakeet_fn(*args)

  with timer('Parakeet #2 -- %s' % name):
    parakeet_result = parakeet_fn(*args)

  if numba:
    numba_fn = autojit(fn)

    with timer('Numba #1 -- %s' % name):
      numba_result = numba_fn(*args)

    with timer('Numba #2 -- %s' % name):
      numba_result = numba_fn(*args)
  
  if parakeet_result is not None and numba_result is not None:  
    assert np.allclose(parakeet_result, numba_result)  
  
  if cpython:
    with timer('Python -- %s' % name):
      python_result = fn(*args)

  if cpython_result is not None and parakeet_result is not None:
    assert np.allclose(parakeet_result, python_result)  
