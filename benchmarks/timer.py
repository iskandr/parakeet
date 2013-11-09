
import cStringIO
import os  
import sys 
import time
import tempfile 

class timer(object):
  def __init__(self, name = None, newline = True, 
               suppress_stdout = True, 
               suppress_stderr = True,
               propagate_exceptions = False):
    self.name = name 
    self.start_t = time.time()
    self.newline = newline
    self.suppress_stdout = suppress_stdout
    self.suppress_stderr = suppress_stderr
    self.propagate_exceptions = propagate_exceptions 
  def __enter__(self):
    if self.suppress_stdout:
      stdout_newfile = tempfile.NamedTemporaryFile()
      self.prev_stdout_fd = os.dup(sys.stdout.fileno())
      os.dup2(stdout_newfile.fileno(), sys.stdout.fileno())
      self.prev_stdout = sys.stdout
  
    if self.suppress_stderr:
      stderr_newfile = tempfile.NamedTemporaryFile()
      self.prev_stderr_fd = os.dup(sys.stderr.fileno())
      os.dup2(stderr_newfile.fileno(), sys.stderr.fileno())
      self.prev_stderr = sys.stderr 
    
    self.start_t = time.time()

  
  def elapsed(self):
    return time.time() - self.start_t
  
  def __exit__(self, exc_type, exc_value, traceback):
    t = self.elapsed()
    if self.suppress_stdout:
      os.dup2(self.prev_stdout_fd, self.prev_stdout.fileno())
    if self.suppress_stderr:
      os.dup2(self.prev_stderr_fd, self.prev_stderr.fileno())
    if self.newline:
      print 
    s = "Elapsed time: " if self.name is None else "%s : " % self.name 
    if exc_type is None:
      s += "%0.4f" % t
    else:
      name = str(exc_type) if exc_type.__name__ is None else exc_type.__name__
      s += "FAILED with %s '%s'" % (name, exc_value)
    print s 
    # don't raise exceptions
    if self.propagate_exceptions:
      return False 
    else:
      return exc_type is not KeyboardInterrupt
    

