import parakeet
import numpy as np 
from timer import compare_perf




def clamp(i, offset, maxval):
    j = max(0, i + offset)
    return min(j, maxval)


def reflect(pos, offset, bound):
    idx = pos+offset
    return min(2*(bound-1)-idx,max(idx,-idx))
 

def conv(x, weights, mode=clamp):
    sx = x.shape
    sw = weights.shape
    result = np.zeros_like(x)
    for i in xrange(sx[0]):
        for j in xrange(sx[1]):
            for ii in xrange(sw[0]):
                for jj in xrange(sw[1]):
                    idx = mode(i,ii-sw[0]/2,sw[0]), mode(j,jj-sw[0]/2,sw[0])
                    result[i,j] += x[idx] * weights[ii,jj] 
    return result


xsize = (300,300)
x = np.random.randn(*xsize)
wsize = (5,5)
w = np.random.randn(*wsize)

compare_perf(conv, [x,w])
