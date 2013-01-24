import multiprocessing
import numpy as np
import pylab 
import scipy.signal
import scipy.ndimage
import subprocess
import sys
import time

from numpy import array 

import adverb_api
import parakeet 

def gaussian_kernel(size):
  print "Generating gaussian kernel, size=", size 
  size = int(size)
  x, y = np.mgrid[-size:size+1, -size:size+1]
  g = np.exp(-(x**2/float(size)+y**2/float(size)))
  return g / g.sum()

n_cols = 2000

def isolated_iter(n_rows, n_cols, kernel_size, n_repeats = 3):
    print "Generating random image of size %d x %d"% (n_rows, n_cols)
    image = np.random.random((n_rows, n_cols))
    kernel = gaussian_kernel(kernel_size)
    kw, kh = kernel.shape
    print "Kernel size: %d x %d" % (kw, kh)
    row_indices = np.arange(n_rows)[kernel_size:-kernel_size]
    col_indices = np.arange(n_cols)[kernel_size:-kernel_size]

    def conv_pixel(i,j):
      window = image[i-kernel_size:i+kernel_size+1, 
                     j-kernel_size:j+kernel_size+1]
      result = 0.0
      for it in range(kw):
        for jt in range(kh):
          result = result + window[it,jt] * kernel[it,jt]
      return result 
    print "Warming up JIT (without tiling)"
    # warm up parakeet with a first run 
    parakeet.config.opt_tile = False
    _ = parakeet.allpairs(conv_pixel, row_indices, col_indices)
    print "Warming up JIT (with tiling)"
    parakeet.config.opt_tile = True
    _ = parakeet.allpairs(conv_pixel, row_indices, col_indices)
    
    times = np.array([0.0]*4)

    for _ in xrange(n_repeats):      
      # generate the data transposed and then transpose it
      # again since Parakeet is currently cobbled by an 
      # inability to use any axis other than 0

      start = time.time()
      np_result = scipy.ndimage.convolve(image, kernel)
      # scipy.signal.convolve2d(image, kernel, 'valid')
      times[0] += time.time() - start
      # trim the image to make the convolutions comparable  
      np_result = np_result[kernel_size:-kernel_size, 
                            kernel_size:-kernel_size]
      
      parakeet.config.opt_tile = False
      start = time.time()
      parakeet_result = parakeet.allpairs(conv_pixel, row_indices, col_indices) 
      times[1] += time.time() - start 
      print "...par_runtime without tiling: %f" % adverb_api.par_runtime
        
      # print "...running with tiling & search..."  
      parakeet.config.opt_tile = True
      parakeet.config.use_cached_tile_sizes = False
      start = time.time()
      parakeet_tile_result = parakeet.allpairs(conv_pixel, row_indices, col_indices) 
      times[2] += time.time() - start
      print "...par_runtime with tiling & search: %f" % adverb_api.par_runtime
       
      # print "...running with tiling & cached tile sizes..."  
      parakeet.config.opt_tile = True
      parakeet.config.use_cached_tile_sizes = True
      start = time.time()
      _ = parakeet.allpairs(conv_pixel, row_indices, col_indices) 
      times[3] += time.time() - start
      print "...par_runtime for cached tile sizes: %f" % adverb_api.par_runtime
        
        
      rmse = np.sqrt(np.mean( (parakeet_result - np_result) ** 2))
      rmse_tile = np.sqrt(np.mean( (parakeet_tile_result - np_result) ** 2))
      # print "(RMSE) without tiling: %s, with tiling: %s " %(rmse, rmse_tile)
      assert rmse < 0.0001
      assert rmse_tile < 0.0001
    return repr(times / n_repeats)

def run_benchmarks(output_file = None, 
                     min_rows = 500, max_rows = 3500, row_step = 500, 
                     min_k = 5, max_k = 35, k_step = 4):
  possible_rows = range(min_rows, max_rows, row_step)
  possible_k = range(min_k, max_k, k_step)
  
  # first column is numpy 
  # second column is parakeet without tiling
  # third column is parakeet with tiling (search for params)
  # fourth column is parakeet with tiling (use last cached params)
  results = np.zeros(shape = (len(possible_rows), len(possible_k), 4))
 
  for row_idx, n_rows in enumerate(possible_rows):
    for k_idx, k in enumerate(possible_k):
      print 
      print "%d x %d image blurred by %d x %d kernel" % \
          (n_rows, n_cols, (2*k+1), (2*k+1))
      print "-----"      
      output = \
        subprocess.check_output(["python", 
                                 __file__, 
                                 str(n_rows), 
                                 str(k)])
      last_line = output.splitlines()[-1]
      times = np.array(eval(last_line))
      print "==> Python: %.3f" % times[0]
      print "==> Parakeet (without tiling): %.3f" % times[1]
      print "==> Parakeet (with tiling, search): %.3f" % times[2]
      print "==> Parakeet (with tiling, use cached tile sizes): %.3f" % times[3]
      print "-----"
      results[row_idx, k_idx, :] = times
  if output_file:
    np.save(output_file, results)
  return results

if __name__ == '__main__':
  assert len(sys.argv) in (2,3), sys.argv
  if sys.argv[1] == 'run':
    file_name = sys.argv[2] if len(sys.argv) == 3 else None
    run_benchmarks(file_name)
  else:
    assert len(sys.argv) == 3, sys.argv
    n_rows = int(sys.argv[1])
    k = int(sys.argv[2])
    print "n_rows", n_rows 
    print "n_cols", n_cols 
    print "k", k
    print isolated_iter(n_rows, n_cols, k)
