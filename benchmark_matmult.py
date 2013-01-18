import time
import multiprocessing
import pylab 


def isolated_iter(n_rows, k, n_repeats = 3):
    import numpy as np 
    import parakeet
    import adverb_api  
    def dot(x,y):
      return sum(x*y)
    def matmult(X,Y):
      return parakeet.allpairs(dot, X, Y)
    # warm up parakeet with a first run 
    X = np.random.random((100,100)).T
    parakeet.config.opt_tile = False
    _ = matmult(X,X)
    parakeet.config.opt_tile = True
    _ = matmult(X,X)
    
    X = np.random.random( (n_rows, k))
    Y = np.random.random( (3000, k))
    times = np.array([0.0,0.0,0.0,0.0])
    
    print 
    print "----------"
    print "%d x %d multiplied with %d x 3000" % (n_rows, k, k)      


      
    for it in xrange(n_repeats):      
      # generate the data transposed and then transpose it
      # again since Parakeet is currently cobbled by an 
      # inability to use any axis other than 0

      start = time.time()
      np_result = np.dot(X,Y.T)
      times[0] += time.time() - start 
      
      parakeet.config.opt_tile = False
      start = time.time()
      parakeet_result = matmult(X,Y)
      times[1] += time.time() - start 
      print "...par_runtime without tiling: %f" % adverb_api.par_runtime
        
      # print "...running with tiling & search..."  
      parakeet.config.opt_tile = True
      parakeet.config.use_cached_tile_sizes = False
      start = time.time()
      parakeet_tile_result = matmult(X,Y)
      times[2] += time.time() - start
      print "...par_runtime with tiling & search: %f" % adverb_api.par_runtime
       
      # print "...running with tiling & cached tile sizes..."  
      parakeet.config.opt_tile = True
      parakeet.config.use_cached_tile_sizes = True
      start = time.time()
      _ = matmult(X,Y)
      times[3] += time.time() - start
      print "...par_runtime for cached tile sizes: %f" % adverb_api.par_runtime
        
        
      rmse = np.sqrt(np.mean( (parakeet_result - np_result) ** 2))
      rmse_tile = np.sqrt(np.mean( (parakeet_tile_result - np_result) ** 2))
      # print "(RMSE) without tiling: %s, with tiling: %s " %(rmse, rmse_tile)
      assert rmse < 0.0001
      assert rmse_tile < 0.0001
    return times / n_repeats 

import numpy as np
def init_process():
  import signal 
  signal.signal(signal.SIGINT, signal.SIG_IGN)

def run_benchmarks(output_file = None, 
                     min_rows = 500, max_rows = 10000, row_step = 500, 
                     min_k = 500, max_k = 4000, k_step = 500):
  possible_rows = range(min_rows, max_rows, row_step)
  possible_k = range(min_k, max_k, k_step)
  
  # first column is numpy 
  # second column is parakeet without tiling
  # third column is parakeet with tiling (search for params)
  # fourth column is parakeet with tiling (use last cached params)
  results = np.zeros(shape = (len(possible_rows), len(possible_k), 4))
  pool = multiprocessing.Pool(1, init_process)
  for row_idx, n_rows in enumerate(possible_rows):
    for k_idx, k in enumerate(possible_k):
      try:
        times = pool.apply(isolated_iter, args = (n_rows, k))
      except KeyboardInterrupt: 
        print "Killing isolated process"
        pool.close()
        pool.join()
        raise
      print "==> Python: %.3f" % times[0]
      print "==> Parakeet (without tiling): %.3f" % times[1]
      print "==> Parakeet (with tiling, search): %.3f" % times[2]
      print "==> Parakeet (with tiling, use cached tile sizes): %.3f" % times[3]
      results[row_idx, k_idx, :] = times 
  if output_file:
    np.save(output_file, results)
  pool.close()
  pool.join()
  return results
"""
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
""" 
