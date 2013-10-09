import numpy as np 
from parakeet.testing_helpers import expect, run_local_tests
from parakeet import config  

config.print_specialized_function = False
config.print_loopy_function = False


vec = np.arange(16)
mat = vec.reshape(4,4)

def negative_stop_1d(x):
    return x[:-2]

def test_negative_stop_1d_vec():
    expect(negative_stop_1d, [vec], negative_stop_1d(vec))

def test_negative_stop_1d_mat():
    expect(negative_stop_1d, [mat], negative_stop_1d(mat))

def negative_stop_2d(x):
    return x[:-2, :-3]

def test_negative_stop_2d():
    expect(negative_stop_2d, [mat], negative_stop_2d(mat))

def negative_start_1d(x):
    return x[-2:]

def test_negative_start_1d_vec():
    expect(negative_start_1d, [vec], vec[-2:])

def test_negative_start_1d_mat():
    expect(negative_start_1d, [mat], mat[-2:])

def negative_start_2d(x):
    return x[-2:, -3:]

def test_negative_start_2d():
    expect(negative_start_2d, [mat], mat[-2:, -3:])

def negative_step_1d(x):
    return x[::-2]

def test_negative_step_1d_vec():
    expect(negative_step_1d, [vec], vec[::-2])

def test_negative_step_1d_mat():
    expect(negative_step_1d, [mat], mat[::-2])

def negative_step_2d(x):
    return x[::-2, ::-3]

def test_negative_step_2d():
    expect(negative_step_2d, [mat], mat[::-2, ::-3])

if __name__ == "__main__":
    run_local_tests()

