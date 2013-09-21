import parakeet 
from  parakeet import testing_helpers
import numpy as np 


@parakeet.jit
def setidx(x, idx, y):
    x[idx] = y

def test_set_1d_simple_slice():
    x = np.array([1,2,3,4,5,6]) 
    x2 = x.copy()
    idx = slice(2,4)
    y = [10, 20]
    print x[idx]
    x[idx] = y
    setidx(x2, idx, y)
    assert testing_helpers.eq(x, x2), "Expected %s but got %s" % (x, x2)
    
def test_set_1d_simple_slice_to_const():
    x = np.array([1,2,3,4,5,6]) 
    x2 = x.copy()
    idx = slice(2,4)
    y = 0
    x[idx] = y
    setidx(x2, idx, y)
    assert testing_helpers.eq(x, x2)

def test_set_1d_step_slice_to_const():
    x = np.array([1,2,3,4,5,6]) 
    x2 = x.copy()
    idx = slice(2,4,2)
    y = 0
    x[idx] = y
    setidx(x2, idx, y)
    assert testing_helpers.eq(x, x2)

def test_set_1d_negative_slice():
    x = np.array([1,2,3,4,5,6]) 
    x2 = x.copy()
    idx = slice(4,2,-1)
    y = [10, 20]
    x[idx] = y
    setidx(x2, idx, y)
    assert testing_helpers.eq(x, x2), "Expected %s but got %s" % (x, x2)

if __name__ == '__main__':
    testing_helpers.run_local_tests()

