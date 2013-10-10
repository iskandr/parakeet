"""
Kernel from SPH renderer, modified from Numba version at:
https://gist.github.com/rokroskar/bdcf6c6b210ff0efc738#file-gistfile1-txt-L55
"""
 
 
import numpy as np
from parakeet import testing_helpers, config 


def kernel_func(d, h) : 
    if d < 1 : 
        f = 1.-(3./2)*d**2 + (3./4.)*d**3
    elif d<2 :
        f = 0.25*(2.-d)**3
    else :
        f = 0
    return f/(np.pi*h**3)
 
 
def compute_kernel( start = 0, stop = 2.01, step = 0.01,  h = 1.0): 
    # set up the kernel values
    kernel_samples = np.arange(start, stop, step)
    return np.array([kernel_func(x, h) for x in kernel_samples])

 
print compute_kernel(0, 0.1, 0.05)

def test_kernel():
  testing_helpers.expect(compute_kernel, [0, 0.5], compute_kernel(0,0.5))

if __name__ == "__main__":
  testing_helpers.run_local_tests()
