"""
Authors: Federico Vaggi
License: MIT
Source: https://bitbucket.org/FedericoV/numpy-tip-complex-modeling/

Source: https://github.com/numfocus/python-benchmarks/blob/master/arc_distance/arc_distance_python.py
"""

from parakeet import jit, testing_helpers
import numpy as np
from math import * 
n = 10 

a = np.random.rand(n, 2)
b = np.random.rand(n, 2)


def arc_distance_for_loops(a, b):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    a_nrows = a.shape[0]
    b_nrows = b.shape[0]

    distance_matrix = np.zeros([a_nrows, b_nrows])

    for i in range(a_nrows):
        theta_1 = a[i, 0]
        phi_1 = a[i, 1]
        for j in range(b_nrows):
            theta_2 = b[j, 0]
            phi_2 = b[j, 1]
            temp = (pow(sin((theta_2 - theta_1) / 2), 2)
                    +
                    cos(theta_1) * cos(theta_2)
                    * pow(sin((phi_2 - phi_1) / 2), 2))
            distance_matrix[i, j] = 2 * (atan2(sqrt(temp), sqrt(1 - temp)))
    return distance_matrix

def test_arc_distance_for_loops():
  testing_helpers.expect(arc_distance_for_loops, [a,b], arc_distance_for_loops(a,b))

"""
def arc_distance_tile(a, b):
    "Calculates the pairwise arc distance between all points in vector a and b."
    theta_1 = np.tile(a[:, 0], (b.shape[0], 1)).T
    phi_1 = np.tile(a[:, 1], (b.shape[0], 1)).T

    theta_2 = np.tile(b[:, 0], (a.shape[0], 1))
    phi_2 = np.tile(b[:, 1], (a.shape[0], 1))

    temp = (np.sin((theta_2 - theta_1) / 2)**2
            +
            np.cos(theta_1) * np.cos(theta_2)
            * np.sin((phi_2 - phi_1) / 2)**2)
    distance_matrix = 2 * (np.arctan2(np.sqrt(temp), np.sqrt(1 - temp)))

    return distance_matrix

def test_arc_distance_tile():
  testing_helpers.expect(arc_distance_tile, [a,b], arc_distance_tile(a,b))

"""

"""
# data-dependent broadcasting will never work with static
# map insertion so might as well drop this test since 
# it's hopeless
def arc_distance_broadcast(a, b):
    "Calculates the pairwise arc distance between all points in vector a and b."
    theta_1 = a[:, 0][:, None]
    theta_2 = b[:, 0][None, :]
    phi_1 = a[:, 1][:, None]
    phi_2 = b[:, 1][None, :]

    temp = (np.sin((theta_2 - theta_1) / 2)**2
            +
            np.cos(theta_1) * np.cos(theta_2)
            * np.sin((phi_2 - phi_1) / 2)**2)
    distance_matrix = 2 * (np.arctan2(np.sqrt(temp), np.sqrt(1 - temp)))
    return distance_matrix

def test_arc_distance_broadcast():
   testing_helpers.expect(arc_distance_broadcast, [a,b], arc_distance_broadcast(a,b))
"""

if __name__ == "__main__":
  testing_helpers.run_local_tests()
  
