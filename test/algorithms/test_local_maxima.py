import numpy as np 
import parakeet 
import parakeet.c_backend

from parakeet.testing_helpers import run_local_tests, expect 
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
def local_maxima(data, wsize, mode=wrap):
  def is_max(pos):
    def is_smaller_neighbor(offset):
      neighbor_idx = tuple(mode(p, o-w/2, w) for (p, o, w) in zip(pos, offset, wsize))
      return data[neighbor_idx] <= data[pos]
    return np.all(parakeet.imap(is_smaller_neighbor, wsize))
  return parakeet.imap(is_max, data.shape)
  

shape = (4,3,3,3)
x = np.random.randn(*shape)
wsize = (3,3,3,3)

def test_local_maxima():
  expect(local_maxima, [x, wsize], python_local_maxima(x, wsize))
  


if __name__ == '__main__':
    run_local_tests()
  
