import pycuda 

from .. import names
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
    self.depth += 1  
  
  def exit_kernel(self):
    self.depth -= 1 
   
  
  def in_host(self):
    return self.depth == 0
  
  def in_block(self):
    return self.depth == 1
  
  def in_thread(self):
    return self.depth > 1
      
  def visit_ParFor(self, stmt):
    bounds = self.tuple_to_var_list(stmt.bounds)
    n_indices = len(bounds)
    assert n_indices <= 3, "ParFor over %d indices not yet supported" % n_indices
    assert self.in_host(), "Nested ParFor not yet supported by CUDA backend"
    

    nested_fn_name, closure_args, input_types = self.get_fn_info(stmt.fn, attributes = ["__device__"])
    
    if isinstance(input_types[-1], TupleT):
      index_types = input_types[-1].elt_types 
      index_as_tuple = True
      # if function takes a tuple of 
      outer_input_types = input_types[:-1]
    else:
      index_types = input_types[-n_indices:]
      index_as_tuple = False
      outer_input_types = input_types[:-n_indices]
    outer_input_c_types = [self.to_ctype(t) for t in outer_input_types]
    
    kernel_name = names.fresh("kernel_" + nested_fn_name)
    kernel_source = "void %s() { %s() }" % (kernel_name, nested_fn_name)
    dims_with_threads = tuple(bounds) + ("1",)
    dims_str = ", ".join(dims_with_threads)
    
    return "%s<<<%s>>();" % (kernel_name, dims_str) 
    
     
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