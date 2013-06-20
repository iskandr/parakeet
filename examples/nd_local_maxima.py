import numpy as np 
import parakeet 

def wrap(pos, offset, bound):
    return ( pos + offset ) % bound

def edge(pos, offset, bound):
    return min(bound-1,max(0,pos+offset))

def reflect(pos, offset, bound):
    idx = pos+offset
    return min(2*(bound-1)-idx,max(idx,-idx))
  
@parakeet.jit 
def local_maxima(data, wsize, mode=wrap):
  def is_max(position):
    myval = data[position]
    def is_bigger_neighbor(offset):
      neighbor_idx = tuple(mode(p, o-w/2, w) for (p, o, w) in zip(position, offset, wsize))
      return data[neighbor_idx] >= myval
    return not parakeet.any(parakeet.imap(is_bigger_neighbor, wsize))
  return parakeet.imap(is_max, data.shape)


if __name__  == '__main__':
  shape = (10,20,30,4)
  x = np.random.randn(*shape)
  wsize = (3,5,3,1)
  y = local_maxima(x, wsize)
  print x
  print y
  assert x.shape == y.shape  
  print x.shape, x.dtype
  print y.shape, y.dtype 
  print "maxima", sum(y.ravel()) , "/", x.size