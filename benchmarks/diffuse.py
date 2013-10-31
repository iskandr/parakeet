#
# Code taken from Numba's documentation at http://numba.pydata.org/numba-doc/0.11/arrays.html
#

import numpy as np 

mu = 0.1
Lx, Ly = 101, 101
N = 1000
import parakeet
import parakeet.c_backend
#parakeet.c_backend.config.print_module_source = True


def diffuse_loops(iter_num):
    u = np.zeros((Lx, Ly), dtype=np.float64)
    temp_u = np.zeros_like(u)
    temp_u[Lx / 2, Ly / 2] = 1000.0

    for n in range(iter_num):
        for i in range(1, Lx - 1):
            for j in range(1, Ly - 1):
                u[i, j] = mu * (temp_u[i + 1, j] + temp_u[i - 1, j] +
                                temp_u[i, j + 1] + temp_u[i, j - 1] -
                                4 * temp_u[i, j])

        temp = u
        u = temp_u
        temp_u = temp

    return u

def diffuse_array_expressions(iter_num):
    u = np.zeros((Lx, Ly), dtype=np.float64)
    temp_u = np.zeros_like(u)
    temp_u[Lx / 2, Ly / 2] = 1000.0

    for i in range(iter_num):
        u[1:-1, 1:-1] = mu * (temp_u[2:, 1:-1] + temp_u[:-2, 1:-1] +
                              temp_u[1:-1, 2:] + temp_u[1:-1, :-2] -
                              4 * temp_u[1:-1, 1:-1])

        temp = u
        u = temp_u
        temp_u = temp
    return u


from compare_perf import compare_perf 

compare_perf(diffuse_loops, [N], numba=True)
compare_perf( diffuse_array_expressions, [N], numba =True)
