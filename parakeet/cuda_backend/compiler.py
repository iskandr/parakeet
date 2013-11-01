import pycuda 

from .. import names
from ..builder import build_fn
# from ..c_backend import FnCompiler
from ..ndtypes import TupleT,  Int32, FnT, ScalarT, SliceT, NoneT, ArrayT
from ..openmp_backend import MulticoreCompiler
from ..syntax import SourceExpr, SourceStmt
from ..syntax.helpers import get_types, get_fn, get_closure_args



class dim3(object):
  def __init__(self, x, y, z):
    self.x = x 
    self.y = y 
    self.z = z 
  
  def __iter__(self):
    yield self.x 
    yield self.y 
    yield self.z
    
  def __getitem__(self, idx):
    if idx == 0:
      return self.x 
    elif idx == 1:
      return self.y 
    else:
      assert idx == 2, "Unexpected index %s to %s" % (idx, self)
      return self.z
    
  def __str__(self):
    return "dim3(x = %s, y = %s, z = %s)" % (self.x, self.y, self.z) 
 
# pasted literally into the C source 
blockIdx_x = SourceExpr("blockIdx.x", type=Int32)
blockIdx_y = SourceExpr("blockIdx.y", type=Int32)
blockIdx_z = SourceExpr("blockIdx.z", type=Int32)
blockIdx = dim3(blockIdx_x, blockIdx_y, blockIdx_z)

threadIdx_x = SourceExpr("threadIdx.x", type=Int32)
threadIdx_y = SourceExpr("threadIdx.y", type=Int32)
threadIdx_z = SourceExpr("threadIdx.z", type=Int32)
threadIdx = dim3(threadIdx_x, threadIdx_y, threadIdx_z)

warpSize = SourceExpr("wrapSize", type=Int32)
__syncthreads = SourceStmt("__syncthreads;")


class CudaCompiler(MulticoreCompiler):
  
  def __init__(self, *args, **kwargs):
    # keep track of which kernels we've already compiled 
    # and map the name of the nested function to its kernel name
    self._kernel_cache = {}
    MulticoreCompiler.__init__(self, *args, **kwargs)
    
  @property 
  def cache_key(self):
    return self.__class__, max(self.depth, 2)
  
    
  def get_fn_name(self, fn_expr, attributes = ["__device__"]):
    return MulticoreCompiler.get_fn_name(fn_expr, attributes = attributes)
  
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
  
  def build_kernel(self, clos, bounds):
    n_indices = len(bounds)
    fn = get_fn(clos)
    outer_closure_args = get_closure_args(clos)
    input_types = fn.input_types 
    
    #nested_fn_name, outer_closure_args, input_types = \
    # self.get_fn_info(fn, attributes = ["__device__"])
    key = fn.cache_key, self.cache_key
    if key in self._kernel_cache:
      kernel_name = self._kernel_cache[key]
      return kernel_name, outer_closure_args
    
    if isinstance(input_types[-1], TupleT):
      # index_types = input_types[-1].elt_types   
      index_as_tuple = True
      # if function takes a tuple of 
      closure_arg_types = input_types[:-1]
    else:
      # index_types = input_types[-n_indices:]
      index_as_tuple = False
      closure_arg_types = input_types[:-n_indices]
      
    n_closure_args = len(closure_arg_types)
    assert len(outer_closure_args) == n_closure_args, \
      "Mismatch between closure formal args %s and given %s" % (", ".join(closure_arg_types),
                                                                ", ".join(outer_closure_args))
    outer_input_types = tuple(closure_arg_types) + tuple(get_types(bounds))
    kernel_name = names.fresh("kernel_" + fn.name)
    parakeet_kernel, builder, input_vars  = build_fn(outer_input_types, name = kernel_name)
    
    closure_vars = input_vars[:n_closure_args]
    
    # TODO: use these to compute indices when n_indices > 3 or 
    # number of threads per block > 1  
    bounds_vars = input_vars[n_closure_args:(n_closure_args + n_indices)]
    
    indices = tuple(blockIdx)[:n_indices]
    if index_as_tuple:
      index_args = (builder.tuple(indices),)
    else:
      index_args = indices
    
    inner_args = tuple(closure_vars) + tuple(index_args)
    builder.call(fn, inner_args)
    
    c_kernel_name = self.get_fn_name(parakeet_kernel)
    
    self._kernel_cache[key] = c_kernel_name
    return c_kernel_name, outer_closure_args
  
  
  def pass_by_value(self, t):
    if isinstance(t, (ScalarT, NoneT, SliceT, FnT)):
      return True 
    if isinstance(t, TupleT):
      return all(self.pass_by_value(elt_t) for elt_t in t.elt_types)
    return False 
  
  def to_gpu(self, c_expr, t):
    if self.pass_by_value(t):
      return c_expr
    elif isinstance(t, ArrayT):
      ptr_t = "%s*" % self.to_ctype(t.elt_type)
      bytes_per_elt = t.elt_type.dtype.itemsize
      src = "%s.data.raw_ptr" % c_expr  
      dst = self.fresh_var(ptr_t, "gpu_ptr")
      nelts = "%s.size" % c_expr
      nbytes = self.fresh_var("int64_t", "nbytes", "%s * %d" % (nelts, bytes_per_elt))
      # allocate the destination pointer on the GPU
      self.append("(%s) cudaMalloc( (void**) &%s, %s);" % (ptr_t, dst, nbytes))
      
      # copy the contents of the host array to the GPU
      self.append("cudaMemcpy(%s, %s, %s);" % (dst, src, nbytes))
      
      # make an identical array descriptor but change its data pointer to the GPU location
      gpu_descriptor = self.fresh_var(self.to_ctype(t), "gpu_array", c_expr)
      self.append("%s.data.raw_ptr = %s;" % (gpu_descriptor, dst))
      return gpu_descriptor
    elif isinstance(t, TupleT):
      # copy contents of the host tuple into another struct
      gpu_tuple = self.fresh_var(self.to_ctype(t), "gpu_tuple", c_expr)
      for i, elt_t in enumerate(t.elt_types):
        gpu_elt = self.to_gpu("%s.elt%d" % (c_expr, i), elt_t)
        self.append("%s.elt%d = %s;" % (c_expr, i, gpu_elt))
      return gpu_tuple 
    else:
      assert False, "Unsupported type in CUDA backend %s" % t 
      
  
  def from_gpu(self, host_values, types):
    return [self.to_gpu(v,t) for (v,t) in zip(host_values, types)]
  
  def args_to_gpu(self, vs):
    return [self.to_gpu(v) for v in vs]
  
  def visit_ParFor(self, stmt):
    bounds = self.tuple_to_var_list(stmt.bounds)
    n_indices = len(bounds)
    if n_indices > 3 or self.in_host():
      return MulticoreCompiler.visit_ParFor(self, stmt)

    
    kernel_name, closure_args = self.build_kernel(stmt.fn, n_indices)
    
    host_closure_args = self.visit_expr_list(closure_args)
    gpu_closure_args = []
    closure_arg_types = get_types(closure_args)
    for host_value, t in zip(host_closure_args, closure_arg_types):
      gpu_value = self.to_gpu(host_value, t)
      gpu_closure_args.append(gpu_value)
    dims_with_threads = tuple(bounds) + ("1",)
    dims_str = ", ".join(dims_with_threads)
    launch = "%s<<<%s>>(%s);" % (kernel_name, dims_str, ", ".join(gpu_closure_args))
    self.append(launch)
    # copy arguments back from the GPU to the host
    return  
  
  
  def visit_NumCores(self, expr):
    # by default we're running sequentially 
    sm_count = None # TODO
    active_thread_blocks = 6 
    return "%d" % (sm_count * active_thread_blocks)  
  """
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
  """

