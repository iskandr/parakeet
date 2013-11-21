from parakeet import jit
import time 
import numpy as np 

def wald(v, a, rands, u_rands, sigma, accum):
    mu = a / v
    lam = a**2 / sigma**2
    y = rands[:, accum]**2
    x = mu + (mu**2. * y) / (2.*lam) - mu / (2.*lam) * np.sqrt(4.*mu*lam*y + mu**2. * y**2.)
    z = u_rands[:, accum]
    return x

waldjit = jit(wald)

rands = np.random.randn(10000, 1)
urands = np.random.rand(10000, 1)
t = time.time()
for i in xrange(100):
  wald(1,2,rands,urands,1,0)
py_t = time.time() - t 

#warmup
waldjit(1, 2, rands, urands, 1, 0)
t = time.time()
for i in xrange(100):
  wald(1,2,rands,urands,1,0)
par_t = time.time() - t 

print "Python time:", py_t 
print "Parakeet time:", par_t  
