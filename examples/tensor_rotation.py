import numpy as np 

#
# Tensor rotation
# from Peter Mortensen's stackoverflow question 
# @ http://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy/18301915
#


from parakeet import jit 

@jit 
def rotT_loops(T, g):
    def compute_elt(i,j,k,l):
      total = 0
      for ii in range(3):
        for jj in range(3):
          for kk in range(3):
            for ll in range(3):
              gg = g[ii,i]*g[jj,j]*g[kk,k]*g[ll,l]
              total += gg*T[ii,jj,kk,ll]
      return total 
    return np.array([[[[compute_elt(i,j,k,l) 
	                for i in xrange(3)] 
			for j in xrange(3)]
			for k in xrange(3)]
			for l in xrange(3)])

T = np.random.randn(3,3,3,3)
g = np.random.randn(3,3)
import parakeet 
parakeet.config.print_specialized_function = True 
#parakeet.config.print_indexified_function = True 
rotT_loops(T,g)
