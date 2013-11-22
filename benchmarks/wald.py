from parakeet import jit, config 
import time 
import numpy as np 
def wald(v, a, rands, u_rands, sigma, accum):
    mu = a / v
    lam = a**2 / sigma**2
    y = rands[:, accum]**2
    x = mu + (mu**2. * y) / (2.*lam) - mu / (2.*lam) * np.sqrt(4.*mu*lam*y + mu**2. * y**2.)
    z = u_rands[:, accum]
    return x


def rep(f, n = 1000, d = 10000):
  rands = np.random.randn(d, 1)
  urands = np.random.rand(d, 1)
  for i in xrange(n):
    f(1,2,rands,urands,1,0)

n = 1000

t = time.time()
rep(wald, n)
py_t = time.time() - t 


waldjit = jit(wald)
#warmup
rep(waldjit, 1)
t = time.time()
rep(waldjit, n)
par_t = time.time() - t 

config.value_specialization = False 
rep(waldjit, 1)
t = time.time()
rep(waldjit, n)
par_t_no_specialization = time.time() - t 

print "Python time per call:", 1000 * (py_t / n), "ms"
print "Parakeet time w/ value specialization:",  1000 * (par_t  / n), "ms"
print "Parakeet time w/out value specialization", 1000 * (par_t_no_specialization  / n), "ms"

