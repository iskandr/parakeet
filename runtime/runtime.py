from ctypes import *
import copy, math, time
import numpy as np

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

    self.libParRuntime = cdll.LoadLibrary("libparakeetruntime.so")

    # job.h
    self.libParRuntime.make_job.restype = job_p
    self.libParRuntime.reconfigure_job.restype = job_p
    self.libParRuntime.num_threads.argtypes = [job_p]
    self.libParRuntime.num_threads.restype = c_int

    # thread_pool.h
    self.libParRuntime.create_thread_pool.restype = thread_pool_p
    self.libParRuntime.launch_job.argtypes = \
      [thread_pool_p, c_void_p, c_void_p, job_p, POINTER(c_int)]
    self.libParRuntime.launch_job.restype = None
    self.libParRuntime.job_finished.argtypes = [thread_pool_p]
    self.libParRuntime.job_finished.restype = c_int
    self.libParRuntime.get_iters_done.argtypes = [thread_pool_p]
    self.libParRuntime.get_iters_done.restype = c_int
    self.libParRuntime.get_throughputs.argtypes = [thread_pool_p]
    self.libParRuntime.get_throughputs.restype = POINTER(c_double)
    self.libParRuntime.free_throughputs.argtypes = [POINTER(c_double)]
    self.libParRuntime.free_throughputs.restype = None
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
    self.ADAPTIVE_THRESHOLD = 0.3
    self.SLEEP_TIME = 0.05
    self.SEARCH_CUTOFF_RATIO = 1.0
    self.TILE_SEARCH_STEP = 6

    self.cur_iter = 0
    self.time_per_calibration = 0.15
    self.dop = self.MAX_THREADS

    self.thread_pool = self.libParRuntime.create_thread_pool(self.MAX_THREADS)

  def run_job(self, tiled_ast, args, num_iters,
              tiled_loop_iters, tiled_loop_parents):
    self.reg_block_sizes = \
        self.get_initial_reg_block_sizes(len(tiled_loop_iters) + 1)
    self.work_function = self.compile_with_reg_blocking(tiled_ast)
    self.args = args
    self.num_iters = num_iters

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
    self.job = self.libParRuntime.make_job(0, num_iters, self.task_size,
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
        self.greedily_find_best_tiles()

        # Search for the best register block

        # Run the job to completion with the best found parameters
        if not self.job_finished():
          self.wait_for_job()

    self.free_job()

  def compile_with_reg_blocking(self, tiled_ast):
    fname = "vm_a" + str(self.reg_block_sizes[0]) +\
            "_b" + str(self.reg_block_sizes[1]) + "_k0"
    return cast(getattr(tiled_ast, fname), c_void_p)

  def get_initial_reg_block_sizes(self, num_loops):
    block_sizes = [1 for _ in range(num_loops - 3)]
    block_sizes.extend([2, 4, 1])
    return block_sizes

  def greedily_find_best_tiles(self):
    # Find the best L1 tile sizes
    # TODO: For now, I'm only searching over tiles (not reg blocks), and I'm
    #       just implementing a per-loop greedy algorithm.
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
    self.libParRuntime.launch_job(self.thread_pool,
                                  self.work_function,
                                  self.args,
                                  self.job,
                                  cast(self.tile_sizes, POINTER(c_int)),
                                  c_int(1))

  def relaunch_job(self):
    self.libParRuntime.launch_job(self.thread_pool,
                                  self.work_function,
                                  self.args,
                                  self.job,
                                  cast(self.tile_sizes, POINTER(c_int)),
                                  c_int(0))

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
