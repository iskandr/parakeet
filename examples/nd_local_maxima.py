import numpy as np 
import parakeet 

def wrap(pos, offset, bound):
    return ( pos + offset ) % bound

def edge(pos, offset, bound):
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
      result[pos] &= (data[neighbor_idx] < myval)
  return result 

@parakeet.jit 
def local_maxima(data, wsize, mode=wrap):
  def is_max(pos):
    def is_smaller_neighbor(offset):
      neighbor_idx = tuple(mode(p, o-w/2, w) for (p, o, w) in zip(pos, offset, wsize))
      return data[neighbor_idx] < data[pos]
    return parakeet.all(parakeet.imap(is_smaller_neighbor, wsize))
  return parakeet.imap(is_max, data.shape)
  
if __name__  == '__main__':
  shape = (3,10,6,4)
  x = np.random.randn(*shape)
  wsize = (3,5,3,1)
  y = python_local_maxima(x, wsize)
  z = local_maxima(x, wsize)
  print x
  print y
  print z 
  assert x.shape == y.shape == z.shape  
  print "actual maxima", sum(y.ravel()), "/", x.size
  print "parakeet maxima", sum(z.ravel()) , "/", x.size
  assert np.allclose(y,z)
