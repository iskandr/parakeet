from ctypes import *
import math, time

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
  class Task():
    def __init__(self, work_function, args, job, tile_sizes, num_iters):
      self.work_function = work_function
      self.args = args
      self.job = job
      self.tile_sizes = tile_sizes
      self.num_iters = num_iters

  def __init__(self):
    class job_t(Structure): pass
    class thread_pool_t(Structure): pass

    job_p = POINTER(job_t)
    thread_pool_p = POINTER(thread_pool_t)

    self.libParRuntime = cdll.LoadLibrary("libparakeetruntime.so")

    # job.h
    self.libParRuntime.make_job.restype = job_p
#    self.libParRuntime.reconfigure_job.restype = job_p
    self.libParRuntime.num_threads.argtypes = [job_p]
    self.libParRuntime.num_threads.restype = c_int

    # thread_pool.h
    self.libParRuntime.create_thread_pool.restype = thread_pool_p
    self.libParRuntime.launch_job.argtypes = \
      [thread_pool_p, c_void_p, c_void_p, job_p, POINTER(c_int)]
    self.libParRuntime.launch_job.restype = None
    self.libParRuntime.job_finished.argtypes = [thread_pool_p]
    self.libParRuntime.job_finished.restype = c_int
    self.libParRuntime.get_throughput.argtypes = [thread_pool_p]
    self.libParRuntime.get_throughput.restype = c_double
    self.libParRuntime.get_job.argtypes = [thread_pool_p]
    self.libParRuntime.get_job.restype = job_p
    self.libParRuntime.wait_for_job.argtypes = [thread_pool_p]
    self.libParRuntime.wait_for_job.restype = None
    self.libParRuntime.destroy_thread_pool.argtypes = [thread_pool_p]
    self.libParRuntime.destroy_thread_pool.restype = None

    self.MAX_THREADS = 8
    self.DEFAULT_CHUNK_LEN = 128
    self.D_PAR = 4

    self.cur_iter = 0
    self.time_per_calibration = 0.15
    self.dop = self.MAX_THREADS

    self.thread_pool = self.libParRuntime.create_thread_pool(self.MAX_THREADS)

  def run_job(self, work_function, args, num_iters,
              tiled_loop_iters, tiled_loop_parents):
    self.work_function = work_function
    self.args = args
    self.num_iters = num_iters

    # Create the tile sizes array
    tile_sizes_t = c_int * len(tiled_loop_iters)
    self.tile_sizes = tile_sizes_t()
    for i in range(len(tiled_loop_iters)):
      self.tile_sizes[i] = 1

    # Configure the iters per calibration step
    #self.calibrate_iters_per_step()
    #print "Iters per calibration:", self.iters_per_calibration

    # Configure the degree of parallelism
    #self.tp = self.calibrate_par()
    #print "Parallel version to have", self.libParRuntime.num_threads(job), \
    #      "threads"

    # Configure the tile sizes
    #self.calibrate_tile_sizes(tiled_loop_iters, tiled_loop_parents)

    # Create the final job to execute the remaining iterations
    #print "Cur iter after calibration:", self.cur_iter

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
#    while not self.job_finished():
#      time.sleep(0.02)
#      new_tp = self.get_throughput()
#      if abs(self.tp - new_tp) / self.tp > 0.2:
#        print "Throughput changed."
#        print "Old throughput:", self.tp
#        print "New throughput:", new_tp
#        self.pause_job()
#        self.tp = self.calibrate_par()
#        self.launch_job()
#
#    self.free_job()

  def calibrate_iters_per_step(self):
    self.make_job(1, 1, 1, 1)
    start = time.time()
    self.launch_job()
    self.wait_for_job()
    iter_time = time.time() - start
    self.iters_per_calibration =\
      int(round(self.time_per_calibration / iter_time))

  def calibrate_tile_sizes(self, tiled_loop_iters, tiled_loop_parents):
    # First, calibrate the outer map loop's tile/chunk size
    max_chunk = next_smaller_power_2(self.iters_per_calibration)
    print "Max chunk:", max_chunk
    cur_size = 1
    best = 1
    self.tp = self.get_throughput()
    while cur_size <= max_chunk:
      self.make_job(cur_size * self.dop, cur_size, self.dop, 1)
      self.launch_job()
      self.wait_for_job()
      tp = self.get_throughput()
      print "TP with chunk size", cur_size, ":", tp
      if tp > self.tp:
        best = cur_size
        self.tp = tp
      cur_size *= 2
    self.task_size = best
    print "Best map loop chunk size:", self.task_size

    def get_max(iter):
      ret = 3
      if tiled_loop_parents[iter] == -2:
        ret = tiled_loop_iters[iter]
      elif tiled_loop_parents[iter] == -1:
        ret = self.task_size
      else:
        ret = self.tile_sizes[tiled_loop_parents[iter]]
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
        tp = self.get_throughput()
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

  def make_job(self, num_iters, step, num_threads, chunk_len):
    self.job = self.libParRuntime.make_job(self.cur_iter,
                                           self.cur_iter + num_iters,
                                           step, num_threads, chunk_len)
    self.cur_iter += num_iters

#  def reconfigure_job(self, num_threads):
#    self.task.job =\
#      self.libParRuntime.reconfigure_job(self.task.job, num_threads)

  def free_job(self):
    self.libParRuntime.free_job(self.job)

  def launch_job(self):
    self.libParRuntime.launch_job(self.thread_pool,
                                  self.work_function,
                                  self.args,
                                  self.job,
                                  cast(self.tile_sizes, POINTER(c_int)))

  def pause_job(self):
    self.libParRuntime.pause_job(self.thread_pool)

  def job_finished(self):
    return self.libParRuntime.job_finished(self.thread_pool) != 0

  def get_throughput(self):
    return self.libParRuntime.get_throughput(self.thread_pool)

  def wait_for_job(self):
    self.libParRuntime.wait_for_job(self.thread_pool)

  def get_job(self):
    return self.libParRuntime.get_job(self.thread_pool)

  def cleanup(self):
    self.libParRuntime.destroy_thread_pool(self.thread_pool)
