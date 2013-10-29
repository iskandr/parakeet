
import cStringIO
import os  
import sys 
import time
import tempfile 

class timer(object):
  def __init__(self, name = None, newline = True):
    self.name = name 
    self.start_t = time.time()
    self.newline = newline
    
  def __enter__(self):
    #self.old_stdout = sys.stdout 
    #self.old_stderr = sys.stderr 
    #sys.stdout = cStringIO.StringIO()
    #sys.stderr = cStringIO.StringIO()

    # redirect stdout to avoid seeing 
    # all of Numba's noisy LLVM code 
    
    stdout_newfile = tempfile.NamedTemporaryFile()
    stderr_newfile = tempfile.NamedTemporaryFile()
    self.prev_stdout_fd = os.dup(sys.stdout.fileno())
    self.prev_stderr_fd = os.dup(sys.stderr.fileno())
    os.dup2(stdout_newfile.fileno(), sys.stdout.fileno())
    os.dup2(stderr_newfile.fileno(), sys.stderr.fileno())
    self.prev_stdout = sys.stdout
    self.prev_stderr = sys.stderr 
    
    self.start_t = time.time()

  
  def elapsed(self):
    return time.time() - self.start_t
  
  def __exit__(self, exc_type, exc_value, traceback):
    t = self.elapsed()
    #sys.stdout = self.old_stdout
    #sys.stderr = self.old_stderr 
    os.dup2(self.prev_stdout_fd, self.prev_stdout.fileno())
    os.dup2(self.prev_stderr_fd, self.prev_stderr.fileno())
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
from numba import autojit, config 
config.print_function = False 
import numpy as np 

def compare_perf(fn, args, numba= True, cpython = True, extra = {}):
  
  parakeet_fn = jit(fn)
  name = fn.__name__
  parakeet_result = None
  numba_result = None 
  cpython_result = None 

  for backend in ('c', 'openmp'):

    with timer('Parakeet (backend = %s) #1 -- %s' % (backend, name)):
      parakeet_result = parakeet_fn(*args, _backend = backend)

    with timer('Parakeet (backend = %s) #2 -- %s' % (backend, name)):
      parakeet_result = parakeet_fn(*args, _backend = backend)

  if numba:
    numba_fn = autojit(fn)

    with timer('Numba #1 -- %s' % name):
      numba_result = numba_fn(*args)

    with timer('Numba #2 -- %s' % name):
      numba_result = numba_fn(*args)

  if cpython:
    with timer('Python -- %s' % name):
      cpython_result = fn(*args)
  
  if parakeet_result is not None and cpython_result is not None:
      assert np.allclose(cpython_result, parakeet_result), \
        "Difference between Parakeet and CPython = %s" % np.sum(np.abs(cpython_result - parakeet_result))
  
  if numba_result is not None and cpython_result is not None:
      assert np.allclose(cpython_result, numba_result), \
        "Difference between Numba and CPython = %s" % np.sum(np.abs(cpython_result - numba_result))

  
  for name, impl in extra.iteritems():
    with timer("%s #1" % name):
      impl(*args)
    with timer("%s #2" % name):
      extra_result = impl(*args)
    if parakeet_result is not None:
      assert np.allclose(parakeet_result, extra_result), \
        "Difference between Parakeet and %s = %s" % (name, np.sum(np.abs(parakeet_result - extra_result)))
