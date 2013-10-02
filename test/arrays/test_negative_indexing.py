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
    return x[:-2, :-3]

def test_negative_stop_2d():
    expect(negative_stop_2d, [mat], mat[:-2,:-3])


def negative_start_1d(x):
    return x[-2:]

def test_negative_start_1d():
    expect(negative_start_1d, [vec], vec[-2:])
    expect(negative_start_2d, [mat], mat[-2:])


def negative_start_2d(x):
    return x[-2:, -3:]

def test_negative_start_2d():
    expect(negative_start_2d, [mat], mat[-2:, -3:])


def negative_step_1d(x):
    return x[::-2]

def test_negative_step_1d():
    expect(negative_step_1d, [vec], vec[::-2])
    expect(negative_step_2d, [mat], mat[::-2])

def negative_step_2d(x):
    return x[::-2, ::-3]

def test_negative_step_2d():
    expect(negative_step_2d, [vec], mat[::-2, ::-3])

if __name__ == "__main__":
    run_local_tests()

