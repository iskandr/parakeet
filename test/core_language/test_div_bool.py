import numpy as np 
from parakeet import jit, testing_helpers 

@jit 
def true_divided(x):
    return True / x

def test_true_divided_bool():
    testing_helpers.expect(true_divided, [True], True)

def test_true_divided_int():
    testing_helpers.expect(true_divided, [1], 1)
    testing_helpers.expect(true_divided, [2], 0)

def test_true_divided_float():
    testing_helpers.expect(true_divided, [1.0], 1.0)
    testing_helpers.expect(true_divided, [2.0], 0.5)

def test_true_divided_uint8():
    testing_helpers.expect(true_divided, [np.uint8(1)], 1)
    testing_helpers.expect(true_divided, [np.uint8(2)], 0)

if __name__ == '__main__':
    testing_helpers.run_local_tests()
