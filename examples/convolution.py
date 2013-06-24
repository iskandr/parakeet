import parakeet
import numpy as np 
from timer import timer 




def clamp(i, offset, maxval):
    j = max(0, i + offset)
    return min(j, maxval)

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

fastconv = parakeet.jit(conv)


xsize = (300,300)
x = np.random.randn(*xsize)
wsize = (5,5)
w = np.random.randn(*wsize)


with timer('parakeet-first'):
    fastconv(x,w)

with timer('parakeet-second'):
    fastconv(x,w)



with timer('python'):
    conv(x, w)
