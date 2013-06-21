import time

class timer(object):
  def __init__(self, name = None, newline = True):
    self.name = name 
    self.start_t = time.time()
    self.newline = newline
    
  def __enter__(self):
    self.start_t = time.time()
  
  def elapsed(self):
    return time.time() - self.start_t
  
  def __exit__(self,*exit_args):
    t = self.elapsed()
    if self.newline:
      print 
    if self.name is None:
      print "Elasped time %0.4f" % t 
    else:
      print "%s : %0.4f" % (self.name, t) 