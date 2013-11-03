import pycuda 

from .. import names
from ..builder import build_fn
# from ..c_backend import FnCompiler
from ..ndtypes import TupleT,  Int32, FnT, ScalarT, SliceT, NoneT, ArrayT
from ..c_backend import PyModuleCompiler
from ..openmp_backend import MulticoreCompiler
from ..syntax.helpers import get_types, get_fn, get_closure_args
from cuda_syntax import threadIdx, blockIdx



class CudaCompiler(MulticoreCompiler):
  
  def __init__(self, *args, **kwargs):
    # keep track of which kernels we've already compiled 
    # and map the name of the nested function to its kernel name
    self._kernel_cache = {}
    if 'gpu_depth' in kwargs:
      self.gpu_depth = kwargs['gpu_depth'] 
      del kwargs['gpu_depth']
    else:
      self.gpu_depth = 0
    MulticoreCompiler.__init__(self, 
                               compiler_cmd = 'nvcc',
                               extra_link_flags = ['-lcudart'], 
                               src_extension = '.cu',  
                               compiler_flag_prefix = '-Xcompiler',
                               linker_flag_prefix = '-Xlinker', 
                               *args, **kwargs)
    
  @property 
  def cache_key(self):
    return self.__class__, self.depth > 0, max(self.gpu_depth, 2) 
  
  def in_host(self):
    return self.gpu_depth == 0
  
  def in_block(self):
    return self.gpu_depth == 1
  
  def in_gpu(self):
    return self.gpu_depth > 0
  
  def get_fn_name(self, fn_expr, attributes = [], inline = True):
    print "getting fn name for", fn_expr, self.depth, self.gpu_depth 
    if self.in_gpu():
      attributes = ["__device__"] + attributes 
    kwargs = {'depth':self.depth, 'gpu_depth':self.gpu_depth}
    return PyModuleCompiler.get_fn_name(self, fn_expr, 
                                        compiler_kwargs = kwargs,
                                        attributes = attributes, 
                                        inline = inline)
    
  #def get_device_fn_name(self, fn_expr):
  #  return MulticoreCompiler.get_fn_name(self, fn_expr, attributes = ["__device__"])
  
  #def get_global_fn_name(self, fn_expr):
  #  return MulticoreCompiler.get_fn_name(self, fn_expr, attributes = ["__global__"], inline = False)
  
  
  def enter_kernel(self):
    """
    Keep a stack of adverb contexts so we know when we're global vs. block vs. thread
    """
    self.depth += 1
    self.gpu_depth += 1  
  
  def exit_kernel(self):
    self.depth -= 1
    self.gpu_depth -= 1 
  
  
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
    bound_types = (Int32,) * n_indices
    outer_input_types = tuple(closure_arg_types) + bound_types
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
    
    
    self.enter_kernel()

    c_kernel_name = self.get_fn_name(parakeet_kernel, attributes = ["__global__"])
    self.exit_kernel()
    
    self._kernel_cache[key] = c_kernel_name
    return c_kernel_name, outer_closure_args
  
  
  def pass_by_value(self, t):
    if isinstance(t, (ScalarT, NoneT, SliceT, FnT)):
      return True 
    if isinstance(t, TupleT):
      return all(self.pass_by_value(elt_t) for elt_t in t.elt_types)
    return False 
  
  def memcpy_to_device(self, device_ptr, host_ptr, nbytes):
    self.append("cudaMemcpy(%s, %s, %s, cudaMemcpyHostToDevice);" % \
                 (device_ptr, host_ptr, nbytes))
  
  def memcpy_to_host(self, device_ptr, host_ptr, nbytes):
    self.append("cudaMemcpy(%s, %s, %s, cudaMemcpyDeviceToHost);" % \
                 (device_ptr, host_ptr, nbytes))
  
  
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
      self.append("cudaMemcpy(%s, %s, %s, cudaMemcpyHostToDevice);" % (dst, src, nbytes))
      
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
  
  def list_to_gpu(self, host_values, types):
    return [self.to_gpu(v,t) for (v,t) in zip(host_values, types)]
  
  
  def to_host(self, host_value, gpu_value, t):
    if self.pass_by_value(t):
      return
    elif isinstance(t, ArrayT):
      pass 
    elif isinstance(t, TupleT):
      pass 
    else:
      assert False, "Unsupported type in CUDA backend %s" % t 
      
  def list_to_host(self, host_values, gpu_values, types):
    for i, t in enumerate(types):
      host_value = host_values[i]
      gpu_value = gpu_values[i]
      self.to_host(host_value, gpu_value, t)
    
  def visit_ParFor(self, stmt):
    bounds = self.tuple_to_var_list(stmt.bounds)

    n_indices = len(bounds)
    if n_indices > 3 or not self.in_host():
      return MulticoreCompiler.visit_ParFor(self, stmt)

    
    kernel_name, closure_args = self.build_kernel(stmt.fn, bounds)
    
    host_closure_args = self.visit_expr_list(closure_args)
    closure_arg_types = get_types(closure_args)

    self.comment("copy closure arguments to the GPU")
    gpu_closure_args = \
      [self.to_gpu(c_expr, t) for (c_expr, t) in 
       zip(host_closure_args, closure_arg_types)]
    
      
    dims_with_threads = tuple(bounds) + ("1",)
    dims_str = ", ".join(dims_with_threads)
    self.comment("kernel launch")
    kernel_args = tuple(gpu_closure_args) + tuple(bounds)
    kernel_args_str = ", ".join(kernel_args)
    launch = "%s<<<%s>>>(%s);" % (kernel_name, dims_str, kernel_args_str)
    
    self.append(launch)
    self.comment("copy arguments back from the GPU to the host")
    self.list_to_host(host_closure_args, gpu_closure_args, closure_arg_types)
    return "/* done with ParFor */"
  
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

