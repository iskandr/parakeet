import numpy as np 
import parakeet 

def wrap(pos, offset, bound):
    return ( pos + offset ) % bound

def clamp(pos, offset, bound):
    return min(bound-1,max(0,pos+offset))

def reflect(pos, offset, bound):
    idx = pos+offset
    return min(2*(bound-1)-idx,max(idx,-idx))


def python_local_maxima(data, wsize, mode=wrap):
  result = np.ones(shape=data.shape,dtype='bool')
  for pos in np.ndindex(data.shape):
    myval = data[pos]  
    for offset in np.ndindex(wsize):
      neighbor_idx = tuple(mode(p, o-w/2, w) for (p, o, w) in zip(pos, offset, wsize))
      result[pos] &= (data[neighbor_idx] <= myval)
  return result 


@parakeet.jit 
def parakeet_local_maxima(data, wsize, mode=wrap):
  def is_max(pos):
    def is_smaller_neighbor(offset):
      neighbor_idx = tuple(mode(p, o-w/2, w) for (p, o, w) in zip(pos, offset, wsize))
      return data[neighbor_idx] <= data[pos]
    return np.all(parakeet.imap(is_smaller_neighbor, wsize))
  return parakeet.imap(is_max, data.shape)
  

# not sure how to get numba to auto-jit size generic code
# get error: "FAILED with KeyError 'sized_pointer(npy_intp, 4)'"
#import numba
#numba_local_maxima = numba.autojit(python_local_maxima) 

if __name__  == '__main__':
  from timer import timer 
  shape = (30,30,20,12)
  x = np.random.randn(*shape)
  wsize = (3,3,3,3)
  with timer("Parakeet C (first run)"):
    parakeet_result_c  = parakeet_local_maxima(x, wsize, _backend = 'c')
  with timer("Parakeet C (second run)"):
    parakeet_result_c  = parakeet_local_maxima(x, wsize, _backend = 'c')
  
  with timer("Parakeet OpenMP (first run)"):
    parakeet_result_omp  = parakeet_local_maxima(x, wsize, _backend = 'openmp')
  assert np.allclose(parakeet_result_c, parakeet_result_omp), \
    "OpenMP backend differs from C: %s vs. %s" % (np.sum(parakeet_result_c), np.sum(parakeet_result_omp))
  with timer("Parakeet OpenMP (second run)"):
    parakeet_result_omp  = parakeet_local_maxima(x, wsize, _backend = 'openmp')

  with timer("Parakeet CUDA (first run)"):
    parakeet_result_cuda  = parakeet_local_maxima(x, wsize, _backend = 'cuda')
  assert np.allclose(parakeet_result_c, parakeet_result_cuda), \
    "CUDA backend differs from C: %s vs %s" % (np.sum(parakeet_result_c), np.sum(parakeet_result_cuda))
  with timer("Parakeet CUDA (second run)"):
    parakeet_result_cuda  = parakeet_local_maxima(x, wsize, _backend = 'cuda')

  #with timer("Numba (first run)"):
  #  numba_result  = numba_local_maxima(x, wsize, wrap)
  #with timer("Numba (second run)"):
  #  numba_result  = numba_local_maxima(x, wsize, wrap)
  with timer("Python"):
    python_result = python_local_maxima(x, wsize) 
  
  assert np.allclose(python_result,parakeet_result)
  
  
