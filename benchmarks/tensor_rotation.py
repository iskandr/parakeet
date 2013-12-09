import numpy as np 

#
# Tensor rotation
# from Peter Mortensen's stackoverflow question 
# @ http://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy/18301915
#

n = 9 
def rotT_loops(T, g):
    Tprime = np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    for ii in range(n):
                        for jj in range(n):
                            for kk in range(n):
                                for ll in range(n):
                                    gg = g[ii,i]*g[jj,j]*g[kk,k]*g[ll,l]
                                    Tprime[i,j,k,l] = Tprime[i,j,k,l] + gg*T[ii,jj,kk,ll]
    return Tprime

def rotT_numpy(T, g):
  """
  Accepted response on stack overflow by phillip
  """
  gg = np.outer(g, g)
  gggg = np.outer(gg, gg).reshape(4 * g.shape)
  axes = ((0, 2, 4, 6), (0, 1, 2, 3))
  return np.tensordot(gggg, T, axes)

T = np.random.randn(n,n,n,n)
g = np.random.randn(n,n)

from compare_perf import compare_perf 
compare_perf(rotT_loops, [T, g], extra = {'numpy_tensordot': rotT_numpy}, numba= False, backends=('c', 'openmp'), cpython=True)


def rotT_par(T, g):
    def compute_elt(i,j,k,l):
      total = 0.0
      for ii in range(n):
        for jj in range(n):
          for kk in range(n):
            for ll in range(n):
              gg = g[ii,i]*g[jj,j]*g[kk,k]*g[ll,l]
              total += gg*T[ii,jj,kk,ll]
      return total 
    return np.array([[[[compute_elt(i,j,k,l) 
	                for k in xrange(n)] 
			for l in xrange(n)] 
			for j in xrange(n)]
			for i in xrange(n)])
compare_perf(rotT_par, [T, g], extra = {'numpy_tensordot': rotT_numpy}, numba= False, backends=('c', 'openmp'), cpython = True)
