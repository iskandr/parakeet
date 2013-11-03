import parakeet 
from  parakeet import testing_helpers
import numpy as np 

@parakeet.jit
def setidx(x, idx, y):
    x[idx] = y
    return x

def test_set_1d_simple_slice():
    x = np.array([1,2,3,4,5,6]) 
    idx = slice(2,4)
    y = [10, 20]
    x2 = x.copy()
    x[idx] = y
    testing_helpers.expect(setidx, [x2, idx, y], x)
    
def test_set_1d_simple_slice_to_const():
    x = np.array([1,2,3,4,5,6]) 
    idx = slice(2,4)
    y = 0
    x2 = x.copy()
    x[idx] = y
    testing_helpers.expect(setidx, [x2, idx, y], x)

def test_set_1d_step_slice_to_const():
    x = np.array([1,2,3,4,5,6]) 
    idx = slice(2,4,2)
    y = 0
    x2 = x.copy()
    x[idx] = y
    testing_helpers.expect(setidx, (x2, idx, y), x)

def test_set_1d_negative_slice():
    x = np.array([1,2,3,4,5,6]) 
    idx = slice(4,2,-1)
    y = [10, 20]
    x2 = x.copy()
    x[idx] = y
    testing_helpers.expect(setidx, (x2, idx, y), x)

if __name__ == '__main__':
    testing_helpers.run_local_tests()

