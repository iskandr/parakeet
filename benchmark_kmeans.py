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

import test_kmeans

d = 1000
n_iters = 3

def isolated_iter(n, d, k, n_repeats = 3):
    print "Generating random data of size %d x %d"% (n, d)
    X = np.random.randn(n,d)

    assignments = np.random.randint(0, k, size = n)

    print "Warming up JIT (without tiling)"
    # warm up parakeet with a first run
    parakeet.config.opt_tile = False
    _ = test_kmeans.parakeet_kmeans(X, k, 1, assignments)

    print "Warming up JIT (with tiling)"
    parakeet.config.opt_tile = True
    _ = test_kmeans.parakeet_kmeans(X, k, 1, assignments)

    print "DONE WITH WARMUP"
    times = np.array([0.0]*4)

    for iter_num in xrange(n_repeats):
      print "iter", iter_num
      # generate the data transposed and then transpose it
      # again since Parakeet is currently cobbled by an
      # inability to use any axis other than 0

      start = time.time()
      np_result = test_kmeans.python_kmeans(X, k, n_iters, assignments)

      # scipy.signal.convolve2d(image, kernel, 'valid')
      np_time =  time.time() - start
      print "numpy time", np_time
      times[0] += np_time
      # trim the image to make the convolutions comparable

      parakeet.config.opt_tile = False
      start = time.time()
      parakeet_result = test_kmeans.parakeet_kmeans(X, k, n_iters, assignments)
      t2 = time.time() - start
      print "parakeet without tiling", t2
      times[1] += t2
      print "...par_runtime without tiling: %f" % adverb_api.par_runtime

      # print "...running with tiling & search..."
      parakeet.config.opt_tile = True
      parakeet.config.use_cached_tile_sizes = False
      start = time.time()
      parakeet_tile_result = test_kmeans.parakeet_kmeans(X, k, n_iters,
                                                         assignments)
      t3 = time.time() - start
      print "parakeet w/ tiling & search", t3
      times[2] += t3
      print "...par_runtime with tiling & search: %f" % adverb_api.par_runtime

      # print "...running with tiling & cached tile sizes..."
      parakeet.config.opt_tile = True
      parakeet.config.use_cached_tile_sizes = True
      start = time.time()
      _ = test_kmeans.parakeet_kmeans(X, k, n_iters, assignments)
      t4 = time.time() - start
      print "parakeet w/ cached tiles",t4
      times[3] += t4
      print "...par_runtime for cached tile sizes: %f" % adverb_api.par_runtime

      rmse = np.sqrt(np.mean( (parakeet_result - np_result) ** 2))
      rmse_tile = np.sqrt(np.mean( (parakeet_tile_result - np_result) ** 2))
      # print "(RMSE) without tiling: %s, with tiling: %s " %(rmse, rmse_tile)
      assert rmse < 0.001
      assert rmse_tile < 0.001
    return repr(times / n_repeats)

def run_benchmarks(output_file = None,
                   min_rows = 2500, max_rows = 15000, row_step = 2500,
                   min_k = 100, max_k = 1000, k_step = 100):
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
      print "%d x %d data clustered into %d centroids" % \
          (n_rows, d, k)
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
    print isolated_iter(n_rows, d, k)
