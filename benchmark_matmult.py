import numpy as np 
import time
import multiprocessing
import pylab 





class IsolatedIteration(multiprocessing.Process):
  def __init__(self, n_rows, k, shared_times, n_repeats = 3):
    multiprocessing.Process.__init__(self)
    self.n_rows = n_rows
    self.k = k 
    self.times = shared_times
    self.n_repeats = n_repeats

    
  def run(self):
    
    
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

    print 
    print "----------"
    print "%d x %d multiplied with %d x 3000" % (self.n_rows, self.k, self.k)      

    for _ in xrange(self.n_repeats):      
      # generate the data transposed and then transpose it
      # again since Parakeet is currently cobbled by an 
      # inability to use any axis other than 0
      X = np.random.random( (self.n_rows, self.k))
      Y = np.random.random( (3000, self.k))
      
      self.times[:] = [0.0,0.0,0.0,0.0]
        
      start = time.time()
      np_result = np.dot(X,Y.T)
      self.times[0] += time.time() - start 
        
        

      parakeet.config.opt_tile = False
      start = time.time()
      parakeet_result = matmult(X,Y)
      self.times[1] += time.time() - start 
      print "...par_runtime without tiling: %f" % adverb_api.par_runtime
        
      # print "...running with tiling & search..."  
      parakeet.config.opt_tile = True
      parakeet.config.use_cached_tile_sizes = False
      start = time.time()
      parakeet_tile_result = matmult(X,Y)
      self.times[2] += time.time() - start
      print "...par_runtime with tiling & search: %f" % adverb_api.par_runtime
       
      # print "...running with tiling & cached tile sizes..."  
      parakeet.config.opt_tile = True
      parakeet.config.use_cached_tile_sizes = True
      start = time.time()
      _ = matmult(X,Y)
      self.times[3] += time.time() - start
      print "...par_runtime for cached tile sizes: %f" % adverb_api.par_runtime
        
        
      rmse = np.sqrt(np.mean( (parakeet_result - np_result) ** 2))
      rmse_tile = np.sqrt(np.mean( (parakeet_tile_result - np_result) ** 2))
      # print "(RMSE) without tiling: %s, with tiling: %s " %(rmse, rmse_tile)
      assert rmse < 0.0001
      assert rmse_tile < 0.0001
    
    for i in xrange(len(self.times)):
      self.times[i] /= float(self.n_repeats)
    return

 
def run_benchmarks(min_rows = 500, max_rows = 10000, row_step = 500, 
                     min_k = 500, max_k = 4000, k_step = 500,  
                     output_file = None):
  possible_rows = range(min_rows, max_rows, row_step)
  possible_k = range(min_k, max_k, k_step)
  
  # first column is numpy 
  # second column is parakeet without tiling
  # third column is parakeet with tiling (search for params)
  # fourth column is parakeet with tiling (use last cached params)
  results = np.zeros(shape = (len(possible_rows), len(possible_k), 4))
  
  for row_idx, n_rows in enumerate(possible_rows):
    for k_idx, k in enumerate(possible_k):
      times = multiprocessing.Array('d',4)
      p = IsolatedIteration(n_rows, k, times)
      p.start()
      p.join()
      print "==> Python: %.3f" % times[0]
      print "==> Parakeet (without tiling): %.3f" % times[1]
      print "==> Parakeet (with tiling, search): %.3f" % times[2]
      print "==> Parakeet (with tiling, use cached tile sizes): %.3f" % times[3]
      results[row_idx, k_idx, :] = times 
  if output_file:
    np.save(output_file, results)
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