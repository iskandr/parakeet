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
    self.libParRuntime.reconfigure_job.restype = job_p
    self.libParRuntime.num_threads.argtypes = [job_p]
    self.libParRuntime.num_threads.restype = c_int

    # thread_pool.h
    self.libParRuntime.create_thread_pool.restype = thread_pool_p
    self.libParRuntime.launch_job.argtypes = \
      [thread_pool_p, c_void_p, c_void_p, job_p, POINTER(c_int), c_int]
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

    self.max_threads = 8
    self.chunk_len = 128
    self.d_par = 4
    self.n_seq = 840
    self.time_per_calibration = 0.1

    self.thread_pool = self.libParRuntime.create_thread_pool(self.max_threads)

  def run_job(self, work_function, args, num_iters, tiled_loop_iters):
    # Create the tile sizes array
    tile_sizes_t = c_int * len(tiled_loop_iters)
    tile_sizes = tile_sizes_t()
    for i in range(len(tiled_loop_iters)):
      tile_sizes[i] = 1

    # Create the task
    job = self.make_job(0, num_iters, self.max_threads)
    self.task = self.Task(work_function, args, job, tile_sizes, 0)

    # Configure the iters per calibration step
    self.calibrate_iters_per_step()

    # Configure the degree of parallelism
    #self.tp = self.calibrate_par()
    #print "Parallel version to have", self.libParRuntime.num_threads(job), \
    #      "threads"
    self.tp = self.get_fixed_throughput()

    # Configure the tile sizes
    self.calibrate_tile_sizes(tiled_loop_iters)

    # Run the job to completion, monitoring its progress
    self.launch_job()
    self.wait_for_job()
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

    self.delete_task()

  def calibrate_iters_per_step(self):
    start = time.time()
    self.launch_job(1)
    self.wait_for_job()
    iter_time = time.time() - start
    self.iters_per_calibration =\
      int(round(self.time_per_calibration / iter_time))

  def calibrate_tile_sizes(self, tiled_loop_iters):
    num_loops = len(tiled_loop_iters)
    for i in range(num_loops):
      loop_iters = tiled_loop_iters[i]
      max_size = next_smaller_power_2(loop_iters)
      cur_size = 2
      best = 1
      while cur_size <= max_size:
        self.task.tile_sizes[i] = cur_size
        tp = self.get_fixed_throughput()
        if tp > self.tp:
          best = cur_size
          self.tp = tp
        cur_size *= 2
      self.task.tile_sizes[i] = best
    for i in range(num_loops):
      print "Tile size for loop", i, ":", self.task.tile_sizes[i]

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

  def get_fixed_throughput(self):
    self.launch_job(self.iters_per_calibration)
    self.wait_for_job()
    return self.get_throughput()

  def make_job(self, start, stop, num_threads):
    return self.libParRuntime.make_job(start, stop, num_threads, self.chunk_len)

  def reconfigure_job(self, num_threads):
    self.task.job =\
      self.libParRuntime.reconfigure_job(self.task.job, num_threads)

  def delete_task(self):
    self.libParRuntime.free_job(self.task.job)

  def launch_job(self, fixed_num_iters=0):
    self.libParRuntime.launch_job(self.thread_pool,
                                  self.task.work_function,
                                  self.task.args,
                                  self.task.job,
                                  cast(self.task.tile_sizes, POINTER(c_int)),
                                  fixed_num_iters)

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
