import pycuda 

from ..c_backend import FnCompiler
from ..openmp_backend import MulticoreCompiler
from ..ndtypes import TupleT, IntT 


class CudaCompiler(MulticoreCompiler):
    
  def visit_NumCores(self, expr):
    # by default we're running sequentially 
    sm_count = None # TODO
    active_thread_blocks = 6 
    return "%d" % (sm_count * active_thread_blocks)
    
  def enter_kernel(self):
    """
    Keep a stack of adverb contexts so we know when we're global vs. block vs. thread
    """
    self.parfor_depth += 1  
  
  def exit_kernel(self):
    self.parfor_depth -= 1 
   
  
  def in_host(self):
    return self.parfor_depth == 0
  
  def in_block(self):
    return self.parfor_depth == 1
  
  def in_thread(self):
    return self.parfor_depth > 1
     
  def get_fn(self, fn, qualifier = "host"):
    """
      Valid options for qualifier = "host" | "device" | "global"
    """
    if qualifier == "host": 
      return FnCompiler.get_fn_name(self, fn)
    else:
      assert False, "GPU functions not implemented yet"
      
  def visit_ParFor(self, stmt):
    if isinstance(stmt.bounds, TupleT):
      n_indices = len(stmt.bounds) 
    else:
      assert isinstance(stmt.bounds, IntT)
      n_indices = 1
    if self.in_host():   
      fn =  self.get_fn(stmt.fn, qualifier = "device")
      # weave a kernel which will map 3D indices into whatever the kernel needs
      kernel = None 
    else:
      assert n_indices <= 3, "ParFor over %d indices not yet supported" % n_indices 
      kernel = self.get_fn(stmt.fn, qualifier = "global")
      
    if self.in_host():
      # invoke the kernel! 
      pass 
      
      
     
  def visit_IndexReduce(self, expr):
    fn =  self.get_fn(expr.fn, qualifier = "device")
    combine = self.get_fn(expr.combine, qualifier = "device")
    if self.in_host():
      #
      # weave two device functions together into a  reduce kernel
      #
      pass 
 
  
  def visit_IndexScan(self, expr):
    fn =  self.get_fn(expr.fn, qualifier = "device")
    combine = self.get_fn(expr.combine, qualifier = "device")
    emit = self.get_fn(expr.emit, qualifier = "device")
    if self.in_host():
      pass 
 


def compile_entry(fn):
  pass 