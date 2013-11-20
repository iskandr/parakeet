import os 
import subprocess
import time

import config 

class CommandFailed(Exception):
  def __init__(self, cmd, env, label):
    self.cmd = cmd 
    self.env = env 
    self.label = label 
    
  def __str__(self):
    return "CommandFailed(%s)" % self.cmd 
  
  def __repr__(self):
    return "CommandFailed(cmd=%s, env=%s, label=%s)" % (self.cmd, self.env, self.label)
  
def run_cmd(cmd, env = None, label = ""):
  if config.print_commands: 
    print " ".join(cmd)
  if config.print_command_elapsed_time: 
    t = time.time()
  
  # first compile silently
  # if you encounter an error, then recompile with output printing
  try:
    if config.suppress_compiler_output: 
      with open(os.devnull, "w") as fnull:
        subprocess.check_call(cmd, stdout = fnull, stderr = fnull, env = env)
    else:
      subprocess.check_call(cmd, env = env)
  except:
    raise CommandFailed(cmd, env, label)
    
  if config.print_command_elapsed_time: 
    if label:
      print "%s, elapsed time: %0.4f" % (label, time.time() - t)
    else:
      print "Elapsed time:", time.time() - t 
