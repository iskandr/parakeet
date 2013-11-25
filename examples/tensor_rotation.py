import numpy as np 

#
# Tensor rotation
# from Peter Mortensen's stackoverflow question 
# @ http://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy/18301915
#


from parakeet import jit 

@jit 
def rotT_loops(T, g):
    Tprime = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                for ll in range(3):
                                    gg = g[ii,i]*g[jj,j]*g[kk,k]*g[ll,l]
                                    Tprime[i,j,k,l] = Tprime[i,j,k,l] + gg*T[ii,jj,kk,ll]
    return Tprime

T = np.random.randn(3,3,3,3)
g = np.random.randn(3,3)

rotT_loops(T,g)

