import numpy as np 
from parakeet import testing_helpers

def kernel(zr, zi, cr, ci, lim, cutoff):
    ''' Computes the number of iterations `n` such that 
        |z_n| > `lim`, where `z_n = z_{n-1}**2 + c`.
    '''
    count = np.uint32(0)
    while ((zr*zr + zi*zi) < (lim*lim)) and count < cutoff:
        zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
        count += np.uint32(1)
    return count

def julia_loops(cr, ci, N, bound=1.5, lim=1000., cutoff=1e6):
    ''' Pure Python calculation of the Julia set for a given `c`.  No NumPy
        array operations are used.
    '''
    julia = np.empty((N, N), dtype=np.uint32)
    grid_x = np.linspace(-bound, bound, N)
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_x):
            julia[i,j] = kernel(x, y, cr, ci, lim, cutoff=cutoff)
    return julia

def julia(cr, ci, N, bound=1.5, lim=1000., cutoff=1e6):
   grid_x = np.linspace(-bound, bound, N)
   return np.array([[kernel(x,y,cr,ci,lim,cutoff=cutoff) 
                     for x in grid_x] 
                     for y in grid_x])
                    
from compare_perf import compare_perf 
cr=0.285
ci=0.01
N=1200
bound = 1.5 
lim = 1000
cutoff = 1e6 

from numba import autojit 


compare_perf(julia, [cr, ci, N, bound, lim, cutoff], numba = False, extra = {'numba':autojit(julia_loops)})

