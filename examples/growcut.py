
# Authors: Nathan Faggian, Stefan van der Walt, Aron Ahmadia, Olivier Grisel
# https://github.com/stefanv/growcut_py

import numpy as np

def window_floor(idx, radius):
    if radius > idx:
        return 0
    else:
        return idx - radius


def window_ceil(idx, ceil, radius):
    if idx + radius > ceil:
        return ceil
    else:
        return idx + radius


def growcut_python(image, state, state_next, window_radius):
    changes = 0
    sqrt_3 = np.sqrt(3.0)

    height = image.shape[0]
    width = image.shape[1]

    for j in xrange(width):
        for i in xrange(height):

            winning_colony = state[i, j, 0]
            defense_strength = state[i, j, 1]

            for jj in xrange(window_floor(j, window_radius),
                             window_ceil(j + 1, width, window_radius)):
                for ii in xrange(window_floor(i, window_radius),
                                 window_ceil(i + 1, height, window_radius)):
                    if ii != i or jj != j:
                        d = image[i, j, 0] - image[ii, jj, 0]
                        s = d * d
                        for k in range(1, 3):
                            d = image[i, j, k] - image[ii, jj, k]
                            s += d * d
                        gval = 1.0 - np.sqrt(s) / sqrt_3

                        attack_strength = gval * state[ii, jj, 1]

                        if attack_strength > defense_strength:
                            defense_strength = attack_strength
                            winning_colony = state[ii, jj, 0]
                            changes += 1

            state_next[i, j, 0] = winning_colony
            state_next[i, j, 1] = defense_strength

    return changes
    
N = 50
dtype = np.double
image = np.zeros((N, N, 3), dtype=dtype)
state = np.zeros((N, N, 2), dtype=dtype)
state_next = np.empty_like(state)

# colony 1 is strength 1 at position 0,0
# colony 0 is strength 0 at all other positions
state[0, 0, 0] = 1
state[0, 0, 1] = 1

window_radius = 10

from timer import compare_perf 

compare_perf(growcut_python, [image, state, state_next, window_radius])
