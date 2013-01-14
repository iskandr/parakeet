import copy
import math
import numpy as np
import random
import sys
import time

from ctypes import *

def next_power_2(n):
  n -= 1
  n = (n >> 1)|n
  n = (n >> 2)|n
  n = (n >> 4)|n
  n = (n >> 8)|n
  n = (n >> 16)|n
  n += 1
  return n

def next_smaller_power_2(n):
  return 2**int(math.log(n, 2))

class Runtime():
  def __init__(self):
    class job_t(Structure): pass
    class thread_pool_t(Structure): pass

    job_p = POINTER(job_t)
    thread_pool_p = POINTER(thread_pool_t)

    lib_name = "./libparakeetruntime.so"
    try:
      dll = cdll.LoadLibrary(lib_name)
    except:
      dll = cdll.LoadLibrary("runtime/" + lib_name)

    self.libParRuntime = dll

    # job.h
    self.libParRuntime.make_job.restype = job_p
    self.libParRuntime.reconfigure_job.restype = job_p
    self.libParRuntime.num_threads.argtypes = [job_p]
    self.libParRuntime.num_threads.restype = c_int

    # thread_pool.h
    self.libParRuntime.create_thread_pool.restype = thread_pool_p
    self.libParRuntime.launch_job.argtypes = \
        [thread_pool_p, c_void_p, c_void_p, job_p, POINTER(POINTER(c_int))]
    self.libParRuntime.launch_job.restype = None
    self.libParRuntime.job_finished.argtypes = [thread_pool_p]
    self.libParRuntime.job_finished.restype = c_int
    self.libParRuntime.get_iters_done.argtypes = [thread_pool_p]
    self.libParRuntime.get_iters_done.restype = c_int64
    self.libParRuntime.get_throughputs.argtypes = [thread_pool_p]
    self.libParRuntime.get_throughputs.restype = POINTER(c_double)
    self.libParRuntime.get_job.argtypes = [thread_pool_p]
    self.libParRuntime.get_job.restype = job_p
    self.libParRuntime.wait_for_job.argtypes = [thread_pool_p]
    self.libParRuntime.wait_for_job.restype = None
    self.libParRuntime.destroy_thread_pool.argtypes = [thread_pool_p]
    self.libParRuntime.destroy_thread_pool.restype = None

    self.MAX_THREADS = 8
    self.DEFAULT_CHUNK_LEN = 128
    self.D_PAR = 4
    self.INITIAL_TASK_SIZE = 16

    # How much of the computation should involve search?
    self.ADAPTIVE_THRESHOLD = 0.8

    # Params for setting intervals between throughput measurements
    self.SLEEP_STEP = 0.05
    self.PERCENTAGE_TO_SLEEP = 0.01
    self.SLEEP_MIN = 0.4

    # Greedy algorithm
    self.SEARCH_CUTOFF_RATIO = 0.5
    self.TILE_SEARCH_STEP = 6

    # Genetic algorithm
    self.GENETIC_NUM_FOR_CONVERGENCE = 4
    self.CROSSOVER_PROB = 0.5
    self.MUTATION_PROB = 0.1

    # Numbers for sausage
    self.L1SIZE = 2**14
    self.L2SIZE = 2**17
    self.L3SIZE = 2**21
    self.NUM_FP_REGS = 24

    self.cur_iter = 0
    self.time_per_calibration = 0.15
    self.dop = 8

    self.thread_pool = self.libParRuntime.create_thread_pool(self.MAX_THREADS)

  def run_untiled_job(self, fn, args, num_iters):
    # TODO: For now, we're assuming fn is actually a pointer to a runnable
    # function. In the future, we'll need to change that to be an AST that we
    # can compile with a particular setting of register tile sizes and loop
    # unrollings.
    dummy_tile_sizes_t = c_int * 1
    dummy_tile_sizes = dummy_tile_sizes_t()
    self.work_functions = (c_void_p * self.dop)()

    self.args = args
    self.tile_sizes = (dummy_tile_sizes_t * self.dop)()
    for i in range(self.dop):
      self.work_functions[i] = cast(fn, c_void_p)
      self.tile_sizes[i] = dummy_tile_sizes
    self.num_iters = num_iters
    self.task_size = num_iters / self.dop
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           self.dop, 1)
    self.launch_job()
    self.wait_for_job()
    self.free_job()

  def run_job_with_dummy_tiles(self, fn, args, num_iters, num_tiles):
    # TODO: For now, we're assuming fn is actually a pointer to a runnable
    # function. In the future, we'll need to change that to be an AST that we
    # can compile with a particular setting of register tile sizes and loop
    # unrollings.
    if num_tiles == 0:
      num_tiles = 1
    tile_sizes_t = POINTER(c_int64) * self.dop
    self.tile_sizes = tile_sizes_t()
    single_tile_sizes_t = c_int64 * num_tiles
    for i in range(self.dop):
      self.tile_sizes[i] = single_tile_sizes_t()
      for j in range(num_tiles-1):
        self.tile_sizes[i][j] = 50
      self.tile_sizes[i][num_tiles-1] = num_iters
    self.work_functions = (c_void_p * self.dop)()
    for i in range(self.dop):
      self.work_functions[i] = cast(fn, c_void_p)

    self.args = args
    self.num_iters = num_iters
    self.task_size = num_iters / self.dop
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           self.dop, 1)
    self.launch_job()
    self.wait_for_job()
    self.free_job()

  def run_job_with_fixed_tiles(self, fn, args, num_iters, tiles):
    tile_sizes_t = POINTER(c_int64) * self.dop
    self.tile_sizes = tile_sizes_t()
    self.work_functions = (c_void_p * self.dop)()
    for i in range(self.dop):
      self.tile_sizes[i] = tiles
      self.work_functions[i] = cast(fn, c_void_p)

    self.args = args
    self.num_iters = num_iters
    self.task_size = num_iters / self.dop
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           self.dop, 1)
    self.launch_job()
    self.wait_for_job()
    self.free_job()

  def run_compiled_job(self, fn, args, num_iters, dl_estimates, ml_estimates):
    if len(ml_estimates) == 0:
      tile_sizes_t = c_int64 * len(ml_estimates)
      tile_sizes = tile_sizes_t()
      for i in range(len(ml_estimates)):
        tile_sizes[i] = ml_estimates[i]
      self.run_job_with_fixed_tiles(fn, args, num_iters, tile_sizes)
    else:
      self.work_functions = (c_void_p * self.dop)()
      for i in range(self.dop):
        self.work_functions[i] = cast(fn, c_void_p)
      self.args = args
      self.num_iters = num_iters

      self.gaussian_find_best_tiles(dl_estimates, ml_estimates)

      self.free_job()

  def run_job(self, tiled_ast, args, num_iters,
              tiled_loop_iters, tiled_loop_parents):
    self.reg_block_sizes = \
        self.get_initial_reg_block_sizes(len(tiled_loop_iters) + 1)
    self.unroll_factor = 1
    self.work_functions = (c_void_p * self.dop)()
    self.compiled_versions = {}
    self.compiled_versions[tuple(self.reg_block_sizes) + (self.unroll_factor,)]\
        = self.compile_with_reg_blocking(tiled_ast)
    for i in range(self.dop):
      self.work_functions[i] = \
          self.compiled_versions[tuple(self.reg_block_sizes) +
                                 (self.unroll_factor,)]
    self.args = args
    self.num_iters = num_iters

    self.gradient_find_best_tiles(tiled_loop_iters, tiled_loop_parents)

    self.free_job()

  def compile_with_reg_blocking(self, tiled_ast):
    fname = "vm_a" + str(self.reg_block_sizes[0]) +\
            "_b" + str(self.reg_block_sizes[1]) + "_k0"
    return cast(getattr(tiled_ast, fname), c_void_p)

  def get_initial_reg_block_sizes(self, num_loops):
    block_sizes = [1 for _ in range(num_loops - 3)]
    block_sizes.extend([1, 6, 1])
    return block_sizes

  def gaussian_find_best_tiles(self, mins, maxes):
    num_tiled = len(mins)
    self.tile_sizes = (POINTER(c_int64) * self.dop)()
    tile_size_t = c_int64 * num_tiled
    for i in xrange(self.dop):
      self.tile_sizes[i] = tile_size_t()
    self.task_size = 1

    best = [(a+b)/2 for a,b in zip(mins, maxes)]
    sdevs = [b/3.0 for b in best]
    best_tp = -1.0

    def print_tile_sizes(tile_sizes):
      s = "("
      for t in range(num_tiled - 1):
        s += str(tile_sizes[t]) + ", "
      s += str(tile_sizes[num_tiled - 1]) + ")"
      return s

    def get_candidates(num_different = 2):
      half_dop = self.dop / 2
      for i in xrange(0, half_dop, half_dop / num_different):
        for t in xrange(num_tiled):
          #maxdiff = abs(maxes[t] - best[t])
          #mindiff = abs(mins[t] - best[t])
          #sdev = min(maxdiff, mindiff) / 2.0
          #sdev = 0.01 if sdev < 0.01 else sdev
          #new = int(round(np.random.normal(best[t], sdev)))
          #new = max(mins[t], new)
          #new = min(maxes[t], new)
          new = int(round(np.random.normal(best[t], sdevs[t])))
          new = max(mins[t], new)
          new = min(maxes[t], new)
          #new = max(1, new)
          for j in xrange(half_dop / num_different):
            self.tile_sizes[i + j][t] = new
            self.tile_sizes[i + half_dop + j][t] = new

    def check_tps(best_tp, num_different = 2):
      half_dop = self.dop / 2
      changed = False
      tps = self.get_throughputs()
      for i in xrange(0, half_dop, half_dop / num_different):
        avg = 0.0
        for j in xrange(half_dop / num_different):
          avg += tps[i + j]
          avg += tps[i + half_dop + j]
        avg /= (self.dop / num_different)
        if avg > best_tp:
          changed = True
          best_tp = avg
          for t in xrange(num_tiled):
            best[t] = self.tile_sizes[i][t]
      return changed, best_tp

    start = time.time()

    get_candidates()

    # Calibrate time to sleep between throughput measurements
    self.sleep_time = self.SLEEP_MIN
    self.task_size = self.INITIAL_TASK_SIZE
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           self.dop, 1)
    self.launch_job()
    time.sleep(self.sleep_time)
    while self.get_percentage_done() < self.PERCENTAGE_TO_SLEEP:
      self.sleep_time += self.SLEEP_STEP
      time.sleep(self.SLEEP_STEP)
    if self.sleep_time < self.SLEEP_MIN:
      self.sleep_time = self.SLEEP_MIN

    # If there's still enough work to do, enter adaptive search
    num_unchanged = 0
    pct_done = self.get_percentage_done()
    while not self.job_finished() and \
          pct_done < self.ADAPTIVE_THRESHOLD and \
          num_unchanged < 4:
      changed, best_tp = check_tps(best_tp)
      if not changed:
        num_unchanged += 1
      else:
        print "best_tp:", best_tp
        print "best tiles:", print_tile_sizes(best)
        num_unchanged = 0
      get_candidates()
      self.pause_job()
      self.relaunch_job()
      time.sleep(self.sleep_time)
      pct_done = self.get_percentage_done()
    if not self.job_finished():
      for i in xrange(self.dop):
        for t in xrange(num_tiled):
          self.tile_sizes[i][t] = best[t]
      print "time spent searching:", time.time() - start
      print "final tile sizes:", print_tile_sizes(self.tile_sizes[0])
      self.pause_job()
      self.relaunch_job()
      self.wait_for_job()
    else:
      print "time spent searching:", time.time() - start

  def genetic_find_best_tiles(self, tiled_loop_iters, tiled_loop_parents):
    num_tiled = len(tiled_loop_iters)
    tile_sizes_t = POINTER(c_int) * self.dop
    self.tile_sizes = tile_sizes_t()
    single_tile_sizes_t = c_int * num_tiled

    # Seed the population of tile settings
    for i in range(self.dop):
      self.tile_sizes[i] = cast(single_tile_sizes_t(), POINTER(c_int))
      for j in range(num_tiled):
        self.tile_sizes[i][j] = random.randint(1, tiled_loop_iters[j])

    # Sanity check to make sure we have enough work to do to make it worth
    # performing an adaptive search
    self.sleep_time = self.SLEEP_TIME
    self.task_size = self.INITIAL_TASK_SIZE
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           self.dop, 1)
    self.launch_job()

    # Calibrate time to sleep between throughput measurements
    time.sleep(self.SLEEP_TIME)
    while self.get_iters_done() < 2:
      self.sleep_time += self.SLEEP_TIME
      time.sleep(self.SLEEP_TIME)
    time.sleep(self.sleep_time)

    def is_cur_best(tps, best_tp, best_tiles, num_same):
      # Check whether we have a new best setting of sizes
      cur_best_tp = 0.0
      cur_best_tiles = self.tile_sizes[0]
      for i in range(num_tiled):
        if tps[i] > best_tp:
          cur_best_tp = tps[i]
          cur_best_tiles = self.tile_sizes[i]
      same = True
      for i in range(num_tiled):
        same = same and cur_best_tiles[i] == best_tiles[i]
      if same:
        num_same += 1
      else:
        num_same = 0
      if cur_best_tp > best_tp:
        best_tp = cur_best_tp
        best_tiles = cur_best_tiles
      return best_tp, best_tiles, num_same

    def update_population(tps):
      # Create new population of tile size settings
      total_tp = 0.0
      for i in range(self.dop):
        total_tp += tps[i]
      pps = []
      for i in range(self.dop):
        pps.append(tps[i] / total_tp)
      p = 0.0
      new_tile_sizes = tile_sizes_t()
      for i in range(self.dop - 1):
        p += pps[i] * (2.0 / (8-i))
        pps[i] = p
      pps[self.dop - 1] = 1.0
      for i in range(self.dop):
        p1 = random.random()
        for j in range(self.dop):
          if p1 < pps[j] or j == self.dop - 1:
            p1 = j
            break
        if random.random() < self.CROSSOVER_PROB:
          new_tile_sizes[i] = cast(single_tile_sizes_t(), POINTER(c_int))
          p2 = random.random()
          for j in range(self.dop):
            if p2 < pps[j] or j == self.dop - 1:
              p2 = j
              break
          cross_idx = random.randint(0, num_tiled)
          for j in range(0, cross_idx):
            new_tile_sizes[i][j] = self.tile_sizes[p1][j]
          for j in range(cross_idx, num_tiled):
            new_tile_sizes[i][j] = self.tile_sizes[p2][j]
        else:
          new_tile_sizes[i] = self.tile_sizes[i]
        for j in range(num_tiled):
          if random.random() < self.MUTATION_PROB:
            new_tile_sizes[i][j] = random.randint(1, tiled_loop_iters[j])
      return new_tile_sizes

    # If there's still enough work to do, enter adaptive search
    if self.get_percentage_done() > self.ADAPTIVE_THRESHOLD:
      self.wait_for_job()
    else:
      if not self.job_finished():
        # Find the best tile sizes
        tps = self.get_throughputs()
        best_tp = 0.0
        best_tiles = self.tile_sizes[0]
        for i in range(num_tiled):
          if tps[i] > best_tp:
            best_tp = tps[i]
            best_tiles = self.tile_sizes[i]
        num_same = 0
        self.tile_sizes = update_population(tps)
        self.pause_job()
        self.relaunch_job()
        time.sleep(self.sleep_time)
        pct_done = self.get_percentage_done()
        while not self.job_finished() and pct_done < self.SEARCH_CUTOFF_RATIO \
              and num_same < self.GENETIC_NUM_FOR_CONVERGENCE:
          tps = self.get_throughputs()
          best_tp, best_tiles, num_same =\
              is_cur_best(tps, best_tp, best_tiles, num_same)
          self.tile_sizes = update_population(tps)

          # Evaluate the new population
          self.pause_job()
          self.relaunch_job()
          time.sleep(self.sleep_time)
          pct_done = self.get_percentage_done()

        # Run the job to completion with the best found parameters
        if not self.job_finished():
          self.wait_for_job()

  def greedily_find_best_tiles(self, tiled_loop_iters, tiled_loop_parents):
        # Create the tile sizes array
    tile_sizes_t = c_int * len(tiled_loop_iters)
    self.tile_sizes = tile_sizes_t()
    for i in range(len(tiled_loop_iters)):
      #self.tile_sizes[i] = self.reg_block_sizes[i+1]
      #if self.tile_sizes[i] == 1:
        #self.tile_sizes[i] = self.TILE_SEARCH_STEP
      self.tile_sizes[i] = 4

    # Sanity check to make sure we have enough work to do to make it worth
    # performing an adaptive search
    self.sleep_time = self.SLEEP_TIME
    self.task_size = self.INITIAL_TASK_SIZE
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           self.dop, 1)
    self.launch_job()

    # Calibrate time to sleep between throughput measurements
    time.sleep(self.SLEEP_TIME)
    while self.get_iters_done() < 2:
      self.sleep_time += self.SLEEP_TIME
      time.sleep(self.SLEEP_TIME)

    # If there's still enough work to do, enter adaptive search
    if self.get_percentage_done() > self.ADAPTIVE_THRESHOLD:
      self.wait_for_job()
    else:
      if not self.job_finished():
        # Find the best L1 tile sizes
        # TODO: For now, I'm just implementing a per-loop greedy algorithm.
        tile_size_steps = copy.deepcopy(self.tile_sizes)
        best_tp = self.get_total_throughput()
        best_tile_sizes = copy.deepcopy(self.tile_sizes)
        directions = [1 for _ in best_tile_sizes]
        cur_tile = 0
        self.tile_sizes[cur_tile] += \
            (tile_size_steps[cur_tile] * directions[cur_tile])
        self.pause_job()
        self.relaunch_job()
        time.sleep(self.sleep_time)
        pct_done = self.get_percentage_done()
        while not self.job_finished() and pct_done < self.SEARCH_CUTOFF_RATIO:
          tp = self.get_total_throughput()
          print "TP with tiles", tuple(self.tile_sizes), ":", tp
          self.pause_job()
          if tp > best_tp:
            best_tp = tp
            best_tile_sizes[cur_tile] = self.tile_sizes[cur_tile]
            self.tile_sizes[cur_tile] += \
                (tile_size_steps[cur_tile] * directions[cur_tile])
          elif tile_size_steps[cur_tile] > 1:
            tile_size_steps[cur_tile] = 1
            self.tile_sizes[cur_tile] = best_tile_sizes[cur_tile] + 1
          elif directions[cur_tile] == 1:
            directions[cur_tile] = -1
            self.tile_sizes[cur_tile] = best_tile_sizes[cur_tile] - 1
          elif cur_tile < len(self.tile_sizes) - 1:
            self.tile_sizes[cur_tile] = best_tile_sizes[cur_tile]
            cur_tile += 1
            self.tile_sizes[cur_tile] += \
                (tile_size_steps[cur_tile] * directions[cur_tile])
          else:
            self.tile_sizes[cur_tile] = best_tile_sizes[cur_tile]
            self.relaunch_job()
            break
          self.relaunch_job()
          time.sleep(self.sleep_time)
          pct_done = self.get_percentage_done()

        self.pause_job()
        best_task_size = self.task_size
        self.task_size += self.TILE_SEARCH_STEP
        self.reconfigure_job()
        self.relaunch_job()

        # Now search for best task size (outer loop L1 tile size)
        # TODD: Maybe want to update sleep time here
        time.sleep(self.sleep_time)
        search_step = self.TILE_SEARCH_STEP
        searched_by_one = False
        searched_down = False
        while not self.job_finished() and pct_done < self.SEARCH_CUTOFF_RATIO:
          tp = self.get_total_throughput()
          self.pause_job()
          if tp > best_tp:
            best_tp = tp
            best_task_size = self.task_size
            self.task_size += search_step
            self.reconfigure_job()
          elif not searched_by_one:
            self.task_size = best_task_size + 1
            search_step = 1
            self.reconfigure_job()
            searched_by_one = True
          elif not searched_down:
            self.task_size = best_task_size - 1
            search_step = -1
            self.reconfigure_job()
            searched_down = True
          else:
            self.task_size = best_task_size
            self.reconfigure_job()
            self.relaunch_job()
            break
          self.relaunch_job()
          time.sleep(self.sleep_time)
          pct_done = self.get_percentage_done()

        print "Best tile sizes:", tuple(best_tile_sizes)
        print "Best task size:", self.task_size

        # Search for the best register block

        # Run the job to completion with the best found parameters
        if not self.job_finished():
          self.wait_for_job()

  def get_best_tile_sizes(self, work_function, args, num_iters,
                          tiled_loop_iters, tiled_loop_parents):
    def mod_tiles(n):
      for i in range(len(self.tile_sizes)):
        self.tile_sizes[i] += n

    times = {}
    self.task_size = 18
    bestt = float("inf")
    bests = ()
    while self.task_size < 81:
      self.tile_sizes[0] = 18
      while self.tile_sizes[0] < 81:
        self.job = self.libParRuntime.make_job(0, num_iters,
                                               self.task_size,
                                               self.dop, 1)
        start = time.time()
        self.launch_job()
        self.wait_for_job()
        t = time.time() - start
        self.free_job()
        print "Time for tile sizes", self.task_size,\
              ",", tuple(self.tile_sizes), ":", t
        times[(self.task_size,) + tuple(self.tile_sizes)] = t
        if t < bestt:
          bestt = t
          bests = (self.task_size,) + tuple(self.tile_sizes)
        self.tile_sizes[0] += 6
      self.task_size += 6

    f = open('times-6000-2400-600-vm3.txt', 'w')
    for k, v in times.iteritems():
      f.write('%s %f\n' % (str(k), v))
    f.close()
    print "Best time for sizes", bests, ":", bestt

  def calibrate_tile_sizes(self, tiled_loop_iters, tiled_loop_parents):
    # First, calibrate the outer map loop's tile/chunk size
    max_chunk = next_smaller_power_2(self.iters_per_calibration)
    print "Max chunk:", max_chunk
    cur_size = 1
    best = 1
    self.tp = self.get_total_throughput()
    while cur_size <= max_chunk:
      self.make_job(cur_size * self.dop, cur_size, self.dop, 1)
      self.launch_job()
      self.wait_for_job()
      tp = self.get_total_throughput()
      print "TP with chunk size", cur_size, ":", tp
      if tp > self.tp:
        best = cur_size
        self.tp = tp
      cur_size *= 2
    self.task_size = best
    print "Best map loop chunk size:", self.task_size

    def get_max(it):
      ret = 3
      if tiled_loop_parents[it] == -2:
        ret = tiled_loop_iters[it]
      elif tiled_loop_parents[it] == -1:
        ret = self.task_size
      else:
        ret = self.tile_sizes[tiled_loop_parents[it]]
      return next_smaller_power_2(ret)

    num_loops = len(tiled_loop_iters)
    for i in range(num_loops):
      max_size = get_max(i)
      cur_size = 2
      best = 1
      while cur_size <= max_size:
        self.tile_sizes[i] = cur_size
        self.make_job(self.task_size * self.dop, self.task_size, self.dop, 1)
        self.launch_job()
        self.wait_for_job()
        tp = self.get_total_throughput()
        if tp > self.tp:
          best = cur_size
          self.tp = tp
        cur_size *= 2
        print "TP with loop", i, "tile size", cur_size, ":", tp
      self.tile_sizes[i] = best
    for i in range(num_loops):
      print "Tile size for loop", i, ":", self.tile_sizes[i]

  def calibrate_par(self):
    self.reconfigure_job(self.d_par)
    dop = self.d_par
    dop_m1_tp = dop_p1_tp = 0.0

    dop_tp = self.get_fixed_throughput()
    print "TP with", dop, "threads:", dop_tp

    if dop < self.max_threads:
      self.reconfigure_job(dop - 1)
      dop_m1_tp = self.get_fixed_throughput()
      print "TP with", dop - 1, "threads:", dop_m1_tp
    else:
      dop_m1_tp = dop_tp + 1

    if dop > 1:
      self.reconfigure_job(dop + 1)
      dop_p1_tp = self.get_fixed_throughput()
      print "TP with", dop + 1, "threads:", dop_p1_tp
    else:
      dop_p1_tp = dop_tp + 1

    increasing = True
    if dop_m1_tp > dop_tp and dop_m1_tp > dop_p1_tp:
      increasing = False
    elif dop_p1_tp <= dop_tp:
      self.reconfigure_job(dop)
      self.launch_job()
      return dop_tp

    not_done = True
    while not_done:
      if increasing:
        self.reconfigure_job(dop + 1)
        dop_p1_tp = self.get_fixed_throughput()
        print "TP with", dop + 1, "threads:", dop_p1_tp
        if dop_p1_tp <= dop_tp:
          self.reconfigure_job(dop)
          not_done = False
        else:
          dop += 1
          dop_tp = dop_p1_tp
      else:
        self.reconfigure_job(dop - 1)
        dop_m1_tp = self.get_fixed_throughput()
        print "TP with", dop - 1, "threads:", dop_m1_tp
        if dop_m1_tp < dop_tp:
          self.reconfigure_job(dop)
          not_done = False
        else:
          dop -= 1
          dop_tp = dop_m1_tp
      not_done = not_done and dop > 1 and dop < self.max_threads

    return dop_tp

  def get_percentage_done(self):
    return self.get_iters_done() / float(self.num_iters)

  def make_job(self, num_iters, step, num_threads, chunk_len):
    self.job = self.libParRuntime.make_job(self.cur_iter,
                                           self.cur_iter + num_iters,
                                           step, num_threads, chunk_len)
    self.cur_iter += num_iters

  def get_total_throughput(self):
    tps = self.libParRuntime.get_throughputs(self.thread_pool)
    total_tp = 0.0
    for i in range(self.dop):
      total_tp += tps[i]
    return total_tp

  def reconfigure_job(self):
    self.job = self.libParRuntime.reconfigure_job(self.job, self.task_size)

  def free_job(self):
    self.libParRuntime.free_job(self.job)

  def launch_job(self):
    tile_sizes = cast(self.tile_sizes, POINTER(POINTER(c_int)))
    # print "Args:"
    # print "  thread pool", self.thread_pool
    # print "  work functions", self.work_functions
    # print "  args", self.args
    # print "  job", self.job
    # print "  tile sizes", tile_sizes
    self.libParRuntime.launch_job(
        self.thread_pool, self.work_functions, self.args, self.job,
        tile_sizes, c_int(1))

  def relaunch_job(self):
    self.libParRuntime.launch_job(
        self.thread_pool, self.work_functions, self.args, self.job,
        cast(self.tile_sizes, POINTER(POINTER(c_int))), c_int(0))

  def pause_job(self):
    self.libParRuntime.pause_job(self.thread_pool)

  def job_finished(self):
    return self.libParRuntime.job_finished(self.thread_pool) != 0

  def get_iters_done(self):
    return self.libParRuntime.get_iters_done(self.thread_pool)

  def get_throughputs(self):
    return self.libParRuntime.get_throughputs(self.thread_pool)

  def wait_for_job(self):
    self.libParRuntime.wait_for_job(self.thread_pool)

  def get_job(self):
    return self.libParRuntime.get_job(self.thread_pool)

  def cleanup(self):
    self.libParRuntime.destroy_thread_pool(self.thread_pool)
