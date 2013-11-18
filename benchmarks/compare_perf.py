
from parakeet import jit
import numpy as np 

from timer import timer 


def compare_perf(fn, args, numba= True, cpython = True, 
                 extra = {}, 
                 backends = ('c', 'openmp', 'cuda'), 
                 suppress_output = True,
                 propagate_exceptions = False):

  
  parakeet_fn = jit(fn)
  name = fn.__name__
  parakeet_result = None
  numba_result = None 
  cpython_result = None 
  kwargs = {'suppress_stdout': suppress_output, 
            'suppress_stderr':suppress_output,
            'propagate_exceptions' : propagate_exceptions
           }
  backend = None 
  for backend in backends:
    with timer('Parakeet (backend = %s) #1 -- %s' % (backend, name), **kwargs):
      parakeet_result = parakeet_fn(*args, _backend = backend)

    with timer('Parakeet (backend = %s) #2 -- %s' % (backend, name), **kwargs):
      parakeet_result = parakeet_fn(*args, _backend = backend)

  if numba:
    from numba import autojit, config 
    numba_fn = autojit(fn)
    with timer('Numba #1 -- %s' % name, **kwargs):
      numba_result = numba_fn(*args)

    with timer('Numba #2 -- %s' % name, **kwargs):
      numba_result = numba_fn(*args)

  if cpython:
    with timer('Python -- %s' % name, **kwargs):
      cpython_result = fn(*args)
  

  rtol = 0.0001
  if backend in ('cuda', 'gpu'):
    atol = 0.001
  else:
    atol = 0.00001
  if parakeet_result is not None and cpython_result is not None:
      assert np.allclose(cpython_result, parakeet_result, atol = atol, rtol = rtol), \
        "Max elt difference between Parakeet and CPython = %s" % \
        np.max(np.abs(cpython_result - parakeet_result))
  
  if numba_result is not None and cpython_result is not None:
      assert np.allclose(cpython_result, numba_result, atol = atol, rtol = rtol), \
        "Max elt difference between Numba and CPython = %s" % \
        np.max(np.abs(cpython_result - numba_result))


  
  for name, impl in extra.iteritems():
    with timer("%s #1" % name, **kwargs):
      impl(*args)
    with timer("%s #2" % name, **kwargs):
      extra_result = impl(*args)
    if parakeet_result is not None:
      assert np.allclose(parakeet_result, extra_result, atol = atol, rtol = rtol), \
        "Max elt difference between Parakeet and %s = %s" % \
        (name, np.max(np.abs(parakeet_result - extra_result)))

