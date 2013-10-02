import numpy as np 
from parakeet.testing_helpers import expect, run_local_tests


vec = np.arange(16)
mat = vec.reshape(4,4)

def negative_stop_1d(x):
    return x[:-2]

def test_negative_stop_1d():
    expect(negative_stop_1d, [vec], vec[:-2])
    expect(negative_stop_2d, [mat], mat[:-2])

def negative_stop_2d(x):
    return x[:-2, :-2]

def test_negative_stop_2d():
    expect(negative_stop_2d, [mat], mat[:-2,:-2])

if __name__ == "__main__":
    run_local_tests()

