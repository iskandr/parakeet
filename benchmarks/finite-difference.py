import numpy as np 

def fdtd(input_grid, steps):
    grid = input_grid.copy()
    old_grid = np.zeros_like(input_grid)
    previous_grid = np.zeros_like(input_grid)

    l_x = grid.shape[0]
    l_y = grid.shape[1]

    for i in range(steps):
        previous_grid[:, :] = old_grid
        old_grid[:, :] = grid 
        for x in range(l_x):
            for y in range(l_y):
                grid[x,y] = 0.0
                if x + 1 < l_x:
                    grid[x,y] += old_grid[x+1,y]
                if 0 < x-1 and x - 1 < l_x:
                    grid[x,y] += old_grid[x-1,y]
                if y+1 < l_y:
                    grid[x,y] += old_grid[x,y+1]
                if 0 < y-1 and y-1 < l_y:
                    grid[x,y] += old_grid[x,y-1]
                grid[x,y] /= 2.0
                grid[x,y] -= previous_grid[x,y]
    return grid

N = 1000
steps = 20 
input_grid = np.random.randn(N,N).astype('float64')

import parakeet
parakeet.config.print_generated_code = True 

from compare_perf import compare_perf 
compare_perf(fdtd, [input_grid, steps], backends = ('c', 'openmp', 'cuda'))


