from ctypes import *

max_threads = 8

class job_t(Structure): pass
class thread_pool_t(Structure): pass
job_p = POINTER(job_t)
thread_pool_p = POINTER(thread_pool_t)

def runtime_init():
  libParRuntime = cdll.LoadLibrary("libparakeetruntime.so")

  libParRuntime.make_job.restype = job_p
  libParRuntime.reconfigure_job.restype = job_p

  libParRuntime.create_thread_pool.restype = thread_pool_p
  libParRuntime.launch_job.argtypes = \
    [thread_pool_p, c_void_p, c_void_p, job_p, c_int]
  libParRuntime.launch_job.restype = None
  libParRuntime.job_finished.argtypes = [thread_pool_p]
  libParRuntime.job_finished.restype = c_int
  libParRuntime.get_throughput.argtypes = [thread_pool_p]
  libParRuntime.get_throughput.restype = c_double
  libParRuntime.get_job.argtypes = [thread_pool_p]
  libParRuntime.get_job.restype = job_p
  libParRuntime.wait_for_job.argtypes = [thread_pool_p]
  libParRuntime.wait_for_job.restype = None
  libParRuntime.destroy_thread_pool.argtypes = [thread_pool_p]
  libParRuntime.destroy_thread_pool.restype = None

  return libParRuntime

libParRuntime = runtime_init()
thread_pool = libParRuntime.create_thread_pool(max_threads)

def run_job():
  pass

def make_job(start, stop, num_threads):
  return libParRuntime.make_job(start, stop, num_threads)

def reconfigure_job(old_job, num_threads):
  return libParRuntime.reconfigure_job(old_job, num_threads)

def free_job(job):
  libParRuntime.free_job(job)

def launch_job(work_function, args, job, fixed_num_iters=0):
  libParRuntime.launch_job(thread_pool,
                           work_function, args, job, fixed_num_iters)

def pause_job():
  libParRuntime.pause_job(thread_pool)

def job_finished():
  return libParRuntime.job_finished(thread_pool) != 0

def get_throughput():
  return libParRuntime.get_throughput(thread_pool)

def wait_for_job():
  libParRuntime.wait_for_job(thread_pool)

def get_job():
  return libParRuntime.get_job(thread_pool)

def cleanup():
  libParRuntime.destroy_thread_pool(thread_pool)
