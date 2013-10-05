"""
Example taken from Numba discussion list 
https://groups.google.com/a/continuum.io/forum/#!topic/numba-users/pCpNGdtKlRc
"""

import math 
import numpy as np 
from parakeet import testing_helpers

def norm(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def normalize(vec):
    """ Calculate the normalized vector (norm: one). """
    return vec / norm(vec)


def test_normalize():
  v = np.array((1.0, 2.0, 3.0))
  testing_helpers.expect(normalize, [v], normalize(v))

if __name__ == "__main__":
  testing_helpers.run_local_tests()

