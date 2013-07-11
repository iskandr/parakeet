import copy
import math
import numpy as np
import time
import os 

from ctypes import *

print_tile_search_info = True

# Sausage machine params
L1SIZE = 2**15
L2SIZE = 2**18
L3SIZE = 2**21
NUM_FP_REGS = 24

MAX_THREADS = 8
DEFAULT_CHUNK_LEN = 128
D_PAR = 4
INITIAL_TASK_SIZE = 8

# How much of the computation should involve search?
ADAPTIVE_THRESHOLD = 0.5
NUM_UNCHANGED_STOP = 3

# Params for setting intervals between throughput measurements
SLEEP_STEP = 0.02
PERCENTAGE_TO_SLEEP = 0.01
SLEEP_MIN = 0.02
NUM_ITERS_TO_SLEEP = 1.2

time_per_calibration = 0.15
dop = 8

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

    lib_name = "_runtime.so"
    base_dir = os.path.dirname(__file__)
    lib_path = base_dir + "/" + lib_name
    dll = cdll.LoadLibrary(lib_path)


    self.libParRuntime = dll

    # job.h
    self.libParRuntime.make_job.restype = job_p
    self.libParRuntime.reconfigure_job.restype = job_p
    self.libParRuntime.num_threads.argtypes = [job_p]
    self.libParRuntime.num_threads.restype = c_int

    # thread_pool.h
    self.libParRuntime.create_thread_pool.restype = thread_pool_p
    self.libParRuntime.launch_job.argtypes = \
        [thread_pool_p, c_void_p, c_void_p, job_p, POINTER(POINTER(c_int)),
         c_int, c_int]
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

    self.cur_iter = 0

    self.thread_pool = self.libParRuntime.create_thread_pool(MAX_THREADS)

  def run_untiled_job(self, fn, args, num_iters):
    # TODO: For now, we're assuming fn is actually a pointer to a runnable
    # function. In the future, we'll need to change that to be an AST that we
    # can compile with a particular setting of register tile sizes and loop
    # unrollings.
    dummy_tile_sizes_t = c_int * 1
    dummy_tile_sizes = dummy_tile_sizes_t()
    self.work_functions = (c_void_p * dop)()

    self.args = args
    self.tile_sizes = (dummy_tile_sizes_t * dop)()
    for i in range(dop):
      self.work_functions[i] = cast(fn, c_void_p)
      self.tile_sizes[i] = dummy_tile_sizes
    self.num_iters = num_iters
    self.task_size = num_iters / dop
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           dop, 1)
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
    tile_sizes_t = POINTER(c_int64) * dop
    self.tile_sizes = tile_sizes_t()
    single_tile_sizes_t = c_int64 * num_tiles
    for i in range(dop):
      self.tile_sizes[i] = single_tile_sizes_t()
      for j in range(num_tiles-1):
        self.tile_sizes[i][j] = 50
      self.tile_sizes[i][num_tiles-1] = num_iters
    self.work_functions = (c_void_p * dop)()
    for i in range(dop):
      self.work_functions[i] = cast(fn, c_void_p)

    self.args = args
    self.num_iters = num_iters
    self.task_size = num_iters / dop
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           dop, 1)
    self.launch_job()
    self.wait_for_job()
    self.free_job()

  def run_job_with_fixed_tiles(self, fn, args, num_iters, tiles):
    tile_sizes_t = POINTER(c_int64) * dop
    self.tile_sizes = tile_sizes_t()
    self.work_functions = (c_void_p * dop)()
    for i in range(dop):
      self.tile_sizes[i] = tiles
      self.work_functions[i] = cast(fn, c_void_p)

    self.args = args
    self.num_iters = num_iters
    self.task_size = INITIAL_TASK_SIZE
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           dop, 1)
    self.launch_job()
    self.wait_for_job()
    self.free_job()

  def run_compiled_job(self, fn, args, num_iters, dl_estimates, ml_estimates):
    if len(ml_estimates) == 0:
      tile_sizes_t = c_int64 * 1
      tile_sizes = tile_sizes_t()
      tile_sizes[0] = 1
      self.run_job_with_fixed_tiles(fn, args, num_iters, tile_sizes)
    else:
      self.work_functions = (c_void_p * dop)()
      for i in range(dop):
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
    self.work_functions = (c_void_p * dop)()
    self.compiled_versions = {}
    self.compiled_versions[tuple(self.reg_block_sizes) + (self.unroll_factor,)]\
        = self.compile_with_reg_blocking(tiled_ast)
    for i in range(dop):
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
    self.tile_sizes = (POINTER(c_int64) * dop)()
    tile_size_t = c_int64 * num_tiled
    for i in xrange(dop):
      self.tile_sizes[i] = tile_size_t()
    self.task_size = INITIAL_TASK_SIZE

    best = [(a+b)/2 for a,b in zip(mins, maxes)]
    #sdevs = [(a-b)/2.0 for a,b in zip(maxes,mins)]
    sdevs = [2*(b-a) for a,b in zip(mins,maxes)]
    best_tp = -1.0
    num_different = 4

    def print_tile_sizes(tile_sizes):
      s = "("
      for t in range(num_tiled - 1):
        s += str(tile_sizes[t]) + ", "
      s += str(tile_sizes[num_tiled - 1]) + ")"
      return s

    set_half_to_best = False
    def get_candidates():
      half_dop = dop / 2
      for i in xrange(0, half_dop, half_dop / num_different):
        for t in xrange(num_tiled):
          new = -1
          while new < mins[t]:# or new > maxes[t]:
            new = int(round(np.random.normal(best[t], sdevs[t])))

          for j in xrange(half_dop / num_different):
            self.tile_sizes[i + j][t] = new
            self.tile_sizes[i + half_dop + j][t] = new

      if set_half_to_best:
        for i in xrange(0, half_dop / 2):
          for t in xrange(num_tiled):
            self.tile_sizes[i][t] = best[t]

    def check_tps(best_tp):
      half_dop = dop / 2
      changed = False
      tps = self.get_throughputs()
      for i in xrange(0, half_dop, half_dop / num_different):
        avg = 0.0
        for j in xrange(half_dop / num_different):
          avg += tps[i + j]
          avg += tps[i + half_dop + j]
        avg /= (dop / num_different)

        if avg > best_tp:
          changed = True
          best_tp = avg
          for t in xrange(num_tiled):
            best[t] = self.tile_sizes[i][t]

        if print_tile_search_info:
          print "candidate:", print_tile_sizes(self.tile_sizes[i])
          print ("avg[%d]:" % i), tps[i]
      if print_tile_search_info:
        print
      return changed, best_tp

    start = time.time()

    get_candidates()

    # Calibrate time to sleep between throughput measurements
    self.sleep_time = SLEEP_MIN
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           dop, 1)
    self.launch_job(reset_iters = True, reset_tps = True)
    time.sleep(self.sleep_time)
    while self.get_iters_done() < dop * INITIAL_TASK_SIZE * NUM_ITERS_TO_SLEEP:
      self.sleep_time += SLEEP_STEP
      time.sleep(SLEEP_STEP)
    self.pause_job()
    self.launch_job(reset_tps = True)

    # If there's still enough work to do, enter adaptive search
    num_unchanged = 0
    pct_done = self.get_percentage_done()
    while not self.job_finished() and \
          pct_done < ADAPTIVE_THRESHOLD and \
          num_unchanged < NUM_UNCHANGED_STOP:
      changed, best_tp = check_tps(best_tp)
      if not changed:
        num_unchanged += 1
      else:
        if print_tile_search_info:
          print "new best_tp:", best_tp
          print "new best tiles:", print_tile_sizes(best)
        num_unchanged = 0
      get_candidates()
      self.pause_job()
      self.launch_job(reset_tps = True)
      time.sleep(self.sleep_time)
      pct_done = self.get_percentage_done()
    if not self.job_finished():
      for i in xrange(dop):
        for t in xrange(num_tiled):
          self.tile_sizes[i][t] = best[t]
      if print_tile_search_info:
        print "time spent searching:", time.time() - start
        print "final tile sizes:", print_tile_sizes(self.tile_sizes[0])
      self.pause_job()
      self.launch_job(reset_tps = True)
      self.wait_for_job()
    else:
      if print_tile_search_info:
        print "time spent searching:", time.time() - start
        print "final tile sizes:", print_tile_sizes(best)

  def pro_find_best_tiles(self, mins, maxes):
    num_tiled = len(mins)
    self.tile_sizes = (POINTER(c_int64) * dop)()
    tile_size_t = c_int64 * num_tiled
    for i in xrange(dop):
      self.tile_sizes[i] = tile_size_t()

    # Generate initial simplex
    bs = [int(0.1 * (mx - mn)) for mn, mx in zip(mins, maxes)]
    #center = [(mx + mn) / 2 for mn, mx in zip(mins, maxes)]
    center = [m for m in maxes]
    simplex = []
    for i in range(len(bs)):
      pos = copy.copy(center)
      pos[i] = pos[i] + bs[i]
      neg = copy.copy(center)
      neg[i] = neg[i] - bs[i]
      simplex.append(pos)
      simplex.append(neg)

    num_different = 4
    half_dop = dop / 2
    num_per_step = (len(simplex) + num_different - 1) / num_different
    simplex_fs = [0.0 for _ in simplex]

    def print_tile_sizes(tile_sizes):
      s = "("
      for t in range(num_tiled - 1):
        s += str(tile_sizes[t]) + ", "
      s += str(tile_sizes[num_tiled - 1]) + ")"
      return s

    # Sets up to dop / 2 threads to have different tile sizes, in half-round-
    # robin fashion
    def set_tiles_to_simplex(idx, simplex):
      n = min(len(simplex), idx + num_different) - idx
      ps_per = dop / n
      for i in xrange(n):
        for j in xrange(ps_per / 2):
          for t in xrange(num_tiled):
            self.tile_sizes[i + j][t] = simplex[i][t]
            self.tile_sizes[i + j + half_dop][t] = simplex[i][t]

    def eval_tps(idx, fs):
      n = min(len(fs), idx + num_different) - idx
      ps_per = dop / n
      tps = self.get_throughputs()
      for i in xrange(n):
        avg = 0.0
        for j in xrange(ps_per / 2):
          avg += tps[i + j]
          avg += tps[i + j + half_dop]
        avg /= ps_per
        fs[idx + i] = avg

    def assess_simplex(simplex):
      n = (len(simplex) + num_different - 1) / num_different
      fs = [0.0 for _ in simplex]
      for i in xrange(n):
        set_tiles_to_simplex(i * num_different, simplex)
        self.pause_job()
        self.launch_job()
        time.sleep(self.sleep_time)
        eval_tps(i * num_different, fs)
      return fs

    def get_reflections():
      return [[2*a - b for a,b in zip(simplex[0], simplex[i])]
              for i in xrange(1, len(simplex))]

    def get_shrinks():
      return [[int(0.5 * (a+b)) for a,b in zip(simplex[0], simplex[i])]
              for i in xrange(1, len(simplex))]

    def get_sorted_simplex(simplex, fs):
      idxs = sorted(range(len(fs)), key = fs.__getitem__, reverse = True)
      return [simplex[idx] for idx in idxs], [fs[idx] for idx in idxs]

    def should_stop():
      return False #self.get_percentage_done() < self.ADAPTIVE_THRESHOLD

    start = time.time()

    set_tiles_to_simplex(0, simplex)

    # Calibrate time to sleep between throughput measurements
    self.sleep_time = SLEEP_MIN
    self.task_size = 8 # self.INITIAL_TASK_SIZE
    self.job = self.libParRuntime.make_job(0, self.num_iters, self.task_size,
                                           dop, 1)
    self.launch_job()
    time.sleep(self.sleep_time)
    while self.get_iters_done() < dop * self.task_size * 2:
      self.sleep_time += SLEEP_STEP
      time.sleep(SLEEP_STEP)

    eval_tps(0, simplex_fs)
    for i in xrange(1, num_per_step):
      set_tiles_to_simplex(i * num_different, simplex)
      self.pause_job()
      self.launch_job()
      time.sleep(self.sleep_time)
      eval_tps(i * num_different, simplex_fs)
    simplex, simplex_fs = get_sorted_simplex(simplex, simplex_fs)

    # Hack to remove unreliably inflated initial performance sample.
    #simplex = simplex[1:] + [simplex[0]]
    #simplex_fs = simplex_fs[1:] + [0.0]

    while not self.job_finished() and not should_stop():
      for i in xrange(2 * num_tiled):
        print simplex[i], ":", simplex_fs[i]

      reflections = get_reflections()
      fs = assess_simplex(reflections)
      reflections, fs = get_sorted_simplex(reflections, fs)

      if max(fs) > simplex_fs[0]:
        expansion = [3*a - 2*b for (a,b) in zip(simplex[0], reflections[0])]
        for i in xrange(dop):
          for t in xrange(num_tiled):
            self.tile_sizes[i][t] = expansion[t]
        self.pause_job()
        self.launch_job()
        time.sleep(self.sleep_time)
        tps = self.get_throughputs()
        tp = 0.0
        for i in xrange(dop):
          tp += tps[i]
        tp /= dop
        if tp > fs[0]:
          print "Expanding"

        else:
          print "reflecting"
          simplex, simplex_fs = get_sorted_simplex([simplex[0]] + reflections,
                                                   [simplex_fs[0]] + fs)
      else:
        # Shrink the simplex
        print "Shrinking"
        shrinks = get_shrinks()
        fs = assess_simplex(shrinks)
        simplex = [simplex[0]] + shrinks
        simplex_fs = [simplex_fs[0]] + fs

      if not self.job_finished():
        for i in xrange(dop):
          for t in xrange(num_tiled):
            self.tile_sizes[i][t] = simplex[0][t]
        self.pause_job()
        self.relaunch_job()

    self.wait_for_job()

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
    for i in range(dop):
      total_tp += tps[i]
    return total_tp

  def reconfigure_job(self):
    self.job = self.libParRuntime.reconfigure_job(self.job, self.task_size)

  def free_job(self):
    self.libParRuntime.free_job(self.job)

  def launch_job(self, reset_tps = False, reset_iters = False):
    tile_sizes = cast(self.tile_sizes, POINTER(POINTER(c_int)))
    self.libParRuntime.launch_job(
        self.thread_pool, self.work_functions, self.args, self.job,
        tile_sizes, c_int(int(reset_tps)), c_int(int(reset_iters)))

  def relaunch_job(self):
    self.libParRuntime.launch_job(
        self.thread_pool, self.work_functions, self.args, self.job,
        cast(self.tile_sizes, POINTER(POINTER(c_int))), c_int(0), c_int(0))

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
