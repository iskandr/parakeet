from ..c_backend import PyModuleCompiler, FlatFnCompiler
from ..ndtypes import TupleT, IntT 
class CudaFlatFnCompiler(FlatFnCompiler):
  
  def __init__(self):
    FlatFnCompiler.__init__(self)
    self.depth = 0
    
  def enter_kernel(self):
    """
    Keep a stack of adverb contexts so we know when we're global vs. block vs. thread
    """
    pass 
  
  def exit_kernel(self):
    pass 
   
  def in_host(self):
    return self.depth == 0
  
  def in_block(self):
    return self.depth == 1
  
  def in_thread(self):
    return self.depth > 1
     
  def get_fn(self, fn, qualifier = "host"):
    """
      Valid options for qualifier = "host" | "device" | "global"
    """
    if qualifier == "host": 
      return FlatFnCompiler.get_fn(self, fn)
    else:
      assert False, "GPU functions not implemented yet"
      
  def visit_ParFor(self, stmt):
    if isinstance(stmt.bounds, TupleT):
      n_indices = len(stmt.bounds) 
    else:
      assert isinstance(stmt.bounds, IntT)
      n_indices = 1
    if n_indices > 3 or not self.in_host():   
      fn =  self.get_fn(stmt.fn, qualifier = "device")
      # weave a kernel which will map 3D indices into whatever the kernel needs
      kernel = None 
    else:
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
  

class CudaModuleCompiler(PyModuleCompiler, CudaFlatFnCompiler):
  pass  

def compile_entry(fn):
  pass 