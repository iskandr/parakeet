import numpy as np 
import time
import pylab 

import adverb_api 
import parakeet 



def each_col_sum(X):
  return parakeet.each(parakeet.sum, X)
 
def run_benchmarks(min_rows = 500, max_rows = 50000, row_step = 500, 
                     min_cols = 500, max_cols = 10000, col_step = 500, 
                     output_file = None):
  
  # warm up parakeet with a first run 
  X = np.random.random((100,100)).T
  parakeet.config.opt_tile = False
  _ = each_col_sum(X)
  parakeet.config.opt_tile = True
  _ = each_col_sum(X)



  possible_rows = range(min_rows, max_rows, row_step)
  possible_cols = range(min_cols, max_cols, col_step)
  
  # first column is numpy 
  # second column is parakeet without tiling
  # third column is parakeet with tiling
  results = np.zeros(shape = (len(possible_rows), len(possible_cols), 3))
  
  for row_idx, n_rows in enumerate(possible_rows):
    for col_idx, n_cols in enumerate(possible_cols):
      print 
      print "----------"
      print "%d x %d" % (n_rows, n_cols)      
      
      # generate the data transposed and then transpose it
      # again since Parakeet is currently cobbled by an 
      # inability to use any axis other than 0
      X = np.random.random( (n_cols, n_rows)).T
      
      start = time.time()
      np_result = np.sum(X,1)
      np_time = time.time() - start 
      
      
      print "...running without tiling..."
      parakeet.config.opt_tile = False
      start = time.time()
      parakeet_result = each_col_sum(X)
      parakeet_time = time.time() - start 
      print "...internal time: %f" % adverb_api.par_runtime
      
      print "...running with tiling..."  
      parakeet.config.opt_tile = True
      start = time.time()
      parakeet_tile_result = each_col_sum(X)
      parakeet_tile_time = time.time() - start
      print "...internal time: %ff" % adverb_api.par_runtime
      
      assert np.all(parakeet_result == np_result)
      results[row_idx, col_idx, :] = [np_time, parakeet_time, parakeet_tile_time]
      
      rmse = np.sqrt(np.mean( (parakeet_result - np_result) ** 2))
      rmse_tile = np.sqrt(np.mean( (parakeet_tile_result - np_result) ** 2))
      print "(RMSE) without tiling: %s, with tiling: %s " %(rmse, rmse_tile)
      assert rmse < 0.0001
      assert rmse_tile < 0.0001
      
      print "==> Python: %.3f" % np_time
      print "==> Parakeet (without tiling): %.3f" % parakeet_time
      print "==> Parakeet (with tiling): %.3f" % parakeet_tile_time
      
  if output_file:
    np.save(output_file, results)
  return results

def mk_plots(r, loc = 'upper left', 
             min_rows = 500, max_rows = 50000, row_step = 500):
  x = np.arange(min_rows, max_rows, row_step)
  pylab.plot(x, r[:, -1, 0], 'b--')
  pylab.plot(x, r[:, -1, 1], 'rx-')
  pylab.plot(x, r[:, -1, 2], 'g-')
  pylab.xlabel('number of rows')
  pylab.ylabel('seconds')
  pylab.legend(('NumPy', 'Parakeet (no tiling)', 'Parakeet (tiling)'), loc = loc)
  pylab.show()

    