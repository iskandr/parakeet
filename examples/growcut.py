
# Original authors: Nathan Faggian, Stefan van der Walt, Aron Ahmadia, Olivier Grisel
# https://github.com/stefanv/growcut_py
# ...simplified and made more compact for Parakeet 

import numpy as np
import parakeet 

@parakeet.jit 
def growcut(image, state, window_radius):
    height = image.shape[0]
    width = image.shape[1]
    def attack_cell(i,j):
        winning_colony = state[i, j, 0]
        defense_strength = state[i, j, 1]
        for jj in xrange(max(0, j -  window_radius), 
                         min(width, j+window_radius+1)):
            for ii in xrange(max(0, i - window_radius), 
                             min(height, i + window_radius + 1)):
                if ii != i or jj != j:
                    ssd = np.sum( (image[i,j,:] - image[ii, jj, :]) ** 2)
                    gval = 1.0 - np.sqrt(ssd) / np.sqrt(3.0)
                    attack_strength = gval * state[ii, jj, 1]
    
                    if attack_strength > defense_strength:
                        defense_strength = attack_strength
                        winning_colony = state[ii, jj, 0]
        return np.array([winning_colony, defense_strength])
    return np.array([[attack_cell(i, j) 
                      for i in xrange(height)] 
                      for j in xrange(width)])

    
N = 50
dtype = np.double
image = np.zeros((N, N, 3), dtype=dtype)
state = np.zeros((N, N, 2), dtype=dtype)

# colony 1 is strength 1 at position 0,0
# colony 0 is strength 0 at all other positions
state[0, 0, 0] = 1
state[0, 0, 1] = 1

window_radius = 10

state_next = growcut(image, state, window_radius)

