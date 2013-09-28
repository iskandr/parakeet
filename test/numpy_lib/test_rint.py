import numpy as np 
import parakeet
from parakeet.testing_helpers import run_local_tests 


@parakeet.jit
def rint(x):
    return np.rint(x)

def test_rint():
    assert np.rint(1.2) == rint(1.2)
    assert np.rint(-1) == rint(-1)
    assert np.rint(-0.1) == rint(-0.1)
    x = np.array([-1, 2, 0])
    assert np.allclose(np.rint(x), rint(x))
    y = np.array([-1.1, -0.6, -0.4, 0.4, 0.6])
    assert np.allclose(np.rint(y), rint(y))

if __name__ == "__main__":
    run_local_tests()
