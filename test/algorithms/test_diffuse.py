import numpy as np 
from parakeet import testing_helpers




mu = 0.1
Lx, Ly = 7, 7

def diffuse_loops(iter_num):
    u = np.zeros((Lx, Ly), dtype=np.float64)
    temp_u = np.zeros_like(u)
    temp_u[Lx / 2, Ly / 2] = 1000.0
    for _ in range(iter_num):
        for i in range(1, Lx - 1):
            for j in range(1, Ly - 1):
                u[i, j] = mu * (temp_u[i + 1, j] + temp_u[i - 1, j] +
                                temp_u[i, j + 1] + temp_u[i, j - 1] -
                                4 * temp_u[i, j])
        temp = u
        u = temp_u
        temp_u = temp
    return u

def test_diffuse_loops():
  testing_helpers.expect(diffuse_loops, [3], diffuse_loops(3))


def diffuse_array_expressions(iter_num):
    u = np.zeros((Lx, Ly), dtype=np.float64)
    temp_u = np.zeros_like(u)
    temp_u[Lx / 2, Ly / 2] = 1000.0

    for _ in range(iter_num):
        u[1:-1, 1:-1] = mu * (temp_u[2:, 1:-1] + temp_u[:-2, 1:-1] +
                              temp_u[1:-1, 2:] + temp_u[1:-1, :-2] -
                              4 * temp_u[1:-1, 1:-1])
        temp = u
        u = temp_u
        temp_u = temp
    return u

def test_diffuse_array_expressions():
  testing_helpers.expect(diffuse_array_expressions, [2], diffuse_array_expressions(2))

if __name__ == "__main__":
  testing_helpers.run_local_tests()

