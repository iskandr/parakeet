import numpy as np 

from .. import names
from ..builder import build_fn
# from ..c_backend import FnCompiler
from ..ndtypes import TupleT,  Int32, FnT, ScalarT, SliceT, NoneT, ArrayT, ClosureT
from ..c_backend import PyModuleCompiler
from ..openmp_backend import MulticoreCompiler
from ..syntax import PrintString, SourceExpr
from ..syntax.helpers import get_types, get_fn, get_closure_args, const_int, zero_i32, one_i32

import config 
import device_info 
from cuda_syntax import threadIdx, blockIdx, blockDim 


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
    
    self.device = device_info.best_cuda_device()
    assert self.device, "No GPU found"
    
    self.use_constant_memory_for_args = True
    MulticoreCompiler.__init__(self, 
                               compiler_cmd = ['nvcc', '-arch=%s' % config.arch],
                               extra_link_flags = ['-lcudart'], 
                               src_extension = '.cu',  
                               compiler_flag_prefix = '-Xcompiler',
                               linker_flag_prefix = '-Xlinker', 
                               *args, **kwargs)
  
  @property 
  def cache_key(self):
    return self.__class__, self.depth > 0, max(self.gpu_depth, 2) 
  
  def enter_module_body(self):
    self.append('cudaSetDevice(%d);' % device_info.device_id(self.device))
  
  def build_kernel(self, clos, bounds):
    n_indices = len(bounds)
    fn = get_fn(clos)
    outer_closure_exprs = get_closure_args(clos)
    closure_arg_types = get_types(outer_closure_exprs)
    host_closure_args = self.visit_expr_list(outer_closure_exprs)
    
    self.comment("Copying data from closure arguments to the GPU")
    
    
    read_only = [False] * len(host_closure_args)
    write_only = [False] * len(host_closure_args)
    
    gpu_closure_args = self.args_to_gpu(host_closure_args, closure_arg_types, write_only)
    input_types = fn.input_types 

    kernel_name = names.fresh("kernel_" + fn.name)
    
    if isinstance(input_types[-1], TupleT):
      index_types = input_types[-1].elt_types   
      index_as_tuple = True
      # if function takes a tuple of 
      closure_arg_types = input_types[:-1]
    else:
      index_types = input_types[-n_indices:]
      index_as_tuple = False
      closure_arg_types = input_types[:-n_indices]
      
    n_closure_args = len(closure_arg_types)
    assert len(outer_closure_exprs) == n_closure_args, \
      "Mismatch between closure formal args %s and given %s" % (", ".join(closure_arg_types),
                                                                ", ".join(outer_closure_exprs))
    bound_types = (Int32,) * n_indices
    
    if self.use_constant_memory_for_args:
      outer_input_types = tuple(bound_types)
    else:
      outer_input_types = tuple(closure_arg_types) + bound_types
      
    
        
    
    parakeet_kernel, builder, input_vars  = build_fn(outer_input_types, name = kernel_name)
    
    if self.use_constant_memory_for_args:
      inner_closure_vars = []
      for i, t in enumerate(closure_arg_types):
        raw_name = fn.arg_names[i]
        self.comment("Moving GPU arg #%d %s : %s to constant memory" % (i, raw_name, t))
        const_name = self.fresh_name("gpu_arg_"  + raw_name)
        typename = self.to_ctype(t)
        self.add_decl("__constant__ %s %s" % (typename, const_name))
        inner_closure_vars.append(SourceExpr(const_name, type=t))
        gpu_arg = gpu_closure_args[i]
        # in case this is a constant, should assign to a variable
        first_char = gpu_arg[0]
        if not first_char.isalpha():
          gpu_arg = self.fresh_var(typename, "src_" + raw_name, gpu_arg)
        self.append("cudaMemcpyToSymbolAsync(%s, &%s, sizeof(%s));" % (const_name, gpu_arg, typename))
      inner_closure_vars = tuple(inner_closure_vars)
      # need to do a cudaMemcpyToSymbol for each gpu arg   
      bounds_vars = input_vars
  
    else:
      # TODO: use these to compute indices when n_indices > 3 or 
      # number of threads per block > 1  
      inner_closure_vars = input_vars[:n_closure_args]
      bounds_vars = input_vars[n_closure_args:(n_closure_args + n_indices)]


    THREADS_PER_DIM = config.threads_per_block_dim 

    
    if n_indices == 1:
      elts_per_block = builder.div(bounds_vars[0],  
                                   const_int(THREADS_PER_DIM**2, Int32), 
                                   "elts_per_block")
      elts_per_block = builder.add(elts_per_block, one_i32, "elts_per_block")
      base_idx = builder.mul(elts_per_block, blockIdx.x, "base_x")
      start_idx = builder.add(base_idx, threadIdx.x, "start_x")  
      
      start_indices = [start_idx]
    
      stop_x = builder.mul(builder.add(one_i32, blockIdx.x, "next_base_x"), 
                           elts_per_block, "stop_x")
      stop_x = builder.min(stop_x, bounds_vars[0], "stop_x") 
      stop_indices = [stop_x]
      step_sizes = [const_int(THREADS_PER_DIM**2, Int32)]
       
    else:
      elts_per_block = builder.div(bounds_vars[0],   
                                   const_int(THREADS_PER_DIM, Int32),
                                    "elts_per_block")
      elts_per_block = builder.add(elts_per_block, one_i32, "elts_per_block")
      base_x = builder.mul(elts_per_block, blockIdx.x, "base_x")
      start_x = builder.add(base_x, threadIdx.x, "start_x")  
      start_y = builder.add(builder.mul(elts_per_block, blockIdx.y, "base_y"),
                                threadIdx.y, "start_y")
      
      start_indices = [start_x, start_y]
      
      stop_x = builder.mul(builder.add(one_i32, blockIdx.x, "next_base_x"), 
                           elts_per_block, "stop_x")
      stop_x = builder.min(stop_x, bounds_vars[0], "stop_x") 
      stop_indices = [stop_x]
      stop_y = builder.mul(builder.add(one_i32, blockIdx.y, "next_base_y"),
                           elts_per_block, "stop_y")
      stop_y = builder.min(stop_y, bounds_vars[1], "stop_y")
       
      stop_indices = [stop_x, stop_y]
      
      step_sizes = [const_int(THREADS_PER_DIM, Int32), const_int(THREADS_PER_DIM, Int32)]
      for i in xrange(2, n_indices):
        start_indices.append(zero_i32)
        step_sizes.append(one_i32)
        stop_indices.append(bounds_vars[i])
    
    
    
    def loop_body(index_vars):
      if not isinstance(index_vars, (list,tuple)):
        pass
      indices = [builder.cast(idx, t) for idx, t in zip(index_vars,index_types)]
      if index_as_tuple:
        index_args = (builder.tuple(indices),)
      else:
        index_args = indices
      inner_args = tuple(inner_closure_vars) + tuple(index_args)
      builder.call(fn, inner_args)
      
    builder.nested_loops(stop_indices, loop_body, start_indices, step_sizes, index_vars_as_list = True)
    
    self.enter_kernel()
    c_kernel_name = self.get_fn_name(parakeet_kernel, 
                                     attributes = ["__global__"], 
                                     inline = False)
    self.exit_kernel()
    
    # set cache preference of the kernel we just built 
    self.append("cudaFuncSetCacheConfig(%s, cudaFuncCachePreferL1);" % c_kernel_name)
    # self._kernel_cache[key] = c_kernel_name
    return c_kernel_name, gpu_closure_args, host_closure_args, closure_arg_types
  
  def launch_kernel(self, bounds, params, kernel_name):
    self.synchronize("Done copying arguments to GPU, prepare for kernel launch")
    
    n_bounds = len(bounds)
    sm_count = device_info.num_multiprocessors(self.device)
    n_blocks = sm_count * config.blocks_per_sm 
    
    THREADS_PER_DIM = config.threads_per_block_dim
    
    if n_bounds == 1:
      grid_dims = [n_blocks, 1, 1]
      block_dims = [THREADS_PER_DIM**2, 1, 1]
    else:
      blocks_per_axis = int(np.ceil(np.sqrt(n_blocks)))
      grid_dims = [blocks_per_axis, blocks_per_axis, 1]
      block_dims = [THREADS_PER_DIM,THREADS_PER_DIM,1]

    grid_dims_str = "dim3(%s)" % ", ".join( str(d) for d in grid_dims)
    block_dims_str = "dim3(%s)" % ", ".join( str(d) for d in block_dims)
    
    self.comment("kernel launch")
    
    kernel_args_str = ", ".join(params)
    self.append("%s<<<%s, %s>>>(%s);" % (kernel_name, grid_dims_str, block_dims_str, kernel_args_str))
  
    self.comment("After launching kernel, synchronize to make sure the computation is done")
    self.synchronize("Finished kernel launch")
    
    
  
  def visit_ParFor(self, stmt):
    bounds = self.tuple_to_var_list(stmt.bounds)

    n_indices = len(bounds)
    if n_indices > 5 or not self.in_host():
      return MulticoreCompiler.visit_ParFor(self, stmt)

    
    kernel_name, gpu_closure_args, host_closure_args, closure_arg_types  = \
      self.build_kernel(stmt.fn, bounds)
    
    if self.use_constant_memory_for_args:
      params = bounds 
    else:
      params = tuple(gpu_closure_args) + tuple(bounds)
      
    self.launch_kernel(bounds, params, kernel_name)
    
    self.comment("copy arguments back from the GPU to the host")
    self.list_to_host(host_closure_args, gpu_closure_args, closure_arg_types)
    return "/* done with ParFor */"
  
  def in_host(self):
    return self.gpu_depth == 0
  
  def in_block(self):
    return self.gpu_depth == 1
  
  def in_gpu(self):
    return self.gpu_depth > 0
  
  def get_fn_name(self, fn_expr, attributes = [], inline = True):
    if self.in_gpu() and not attributes:
      attributes = ["__device__"] 
    kwargs = {'depth':self.depth, 'gpu_depth':self.gpu_depth}
    return PyModuleCompiler.get_fn_name(self, fn_expr, 
                                        compiler_kwargs = kwargs,
                                        attributes = attributes, 
                                        inline = inline)
    

  
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
  
  def pass_by_value(self, t):
    if isinstance(t, (ScalarT, NoneT, SliceT, FnT)):
      return True 
    elif isinstance(t, TupleT):
      return all(self.pass_by_value(elt_t) for elt_t in t.elt_types)
    elif isinstance(t, ClosureT):
      return all(self.pass_by_value(elt_t) for elt_t in t.arg_types)
    return False 
  
  
  def check_gpu_error(self, context = None, error_code_var = None):
    if error_code_var is None:
      error_code_var = self.fresh_name("cuda_err")
      self.append("cudaError %s = cudaGetLastError();" % error_code_var)
    if context is None:
      context = "\"Generated CUDA source at \" __FILE__ \" : \" __LINE__"
    self.append("""
      if ( cudaSuccess != %s ) {
        printf( "Error after %s: %%s\\n",  cudaGetErrorString(%s) );
      }
    """ % (error_code_var, context, error_code_var))
  
  def synchronize(self, context = None):
    error_code_var = self.fresh_name("cuda_err")
    self.append("cudaError %s = cudaDeviceSynchronize();" % error_code_var)
    self.check_gpu_error(context, error_code_var)
    
  
  def _to_gpu(self, c_expr, t, gpu_array_dict, memcpy = True):
    if self.pass_by_value(t):
      return c_expr
    elif isinstance(t, ArrayT):
      
      ptr_t = "%s*" % self.to_ctype(t.elt_type)
      bytes_per_elt = t.elt_type.dtype.itemsize
      
      dst = self.fresh_var(ptr_t, "gpu_ptr")
      nelts = "%s.size" % c_expr
      nbytes = self.fresh_var("int64_t", "nbytes", "%s * %d" % (nelts, bytes_per_elt))
      
      if memcpy:
        src = "%s.data.raw_ptr" % c_expr   
        memcpy_stmt = "cudaMemcpyAsync(%s, %s, %s, cudaMemcpyHostToDevice);" % (dst, src, nbytes)
      else:
        memcpy_stmt = "/* no memcpy for this %s : %s */" % (c_expr, t)
      
      # allocate the destination pointer on the GPU
      malloc = "cudaMalloc( (void**) &%s, %s);" % (dst, nbytes)
      alloc_and_copy_block = "%s\n%s" % (malloc, memcpy_stmt)

      """
      TODO: do this to cut down on allocations but also
      need to track which arrays have been allocated so we 
      can delete them later 
      
      if t in gpu_array_dict:
        # if we've allocated some other args of the same type, 
        # try reusing their pointers before allocating new memory 
        for i, (prev_host, prev_gpu) in enumerate(gpu_array_dict[t]):
          cond = "%s.data.raw_ptr == %s.data.raw_ptr && %s.size == %s.size" % \
            (c_expr, prev_host, c_expr, prev_host)
          self.append("%sif (%s) {%s = %s.data.raw_ptr;}" % \
                        ("else " if i > 0 else "", cond, dst, prev_gpu))
        self.append("else {%s}" % alloc_and_copy_block)
      else:
      """
      self.append(alloc_and_copy_block)
      
      self.check_gpu_error("cudaMalloc for %s : %s" % (c_expr, t))
      
      # make an identical array descriptor but change its data pointer to the GPU location
      gpu_descriptor = self.fresh_var(self.to_ctype(t), "gpu_array", c_expr)
      self.append("%s.data.raw_ptr = %s;" % (gpu_descriptor, dst))    
      gpu_array_dict.setdefault(t, []).append( (c_expr, gpu_descriptor) )
      return gpu_descriptor
    
    elif isinstance(t, (ClosureT, TupleT)):
      # copy contents of the host tuple into another struct
      gpu_tuple = self.fresh_var(self.to_ctype(t), "gpu_tuple", c_expr)
      for i, elt_t in enumerate(t.elt_types):
        host_elt = "%s.elt%d" % (c_expr, i)
        gpu_elt = self._to_gpu(host_elt, elt_t, gpu_array_dict)
        self.append("%s.elt%d = %s;" % (c_expr, i, gpu_elt))
      return gpu_tuple 
    else:
      assert False, "Unsupported type in CUDA backend %s" % t 
  
  def args_to_gpu(self, host_values, types, write_only):
    # keep track of arrays we've already moved to the GPU, 
    # indexed by their type  
    gpu_array_dict = {}
    
    gpu_args = []
    for i,t in enumerate(types):
      gpu_arg = self._to_gpu(host_values[i], t, gpu_array_dict, memcpy = not write_only[i])
      gpu_args.append(gpu_arg)
    return gpu_args 
  
  
  def to_host(self, host_value, gpu_value, t):
    if self.pass_by_value(t):
      return
    elif isinstance(t, ArrayT):
      dst = "%s.data.raw_ptr" % host_value 
      src = "%s.data.raw_ptr"  % gpu_value 
      nelts = "%s.size" % gpu_value
      nbytes = "%s * %d" % (nelts, t.elt_type.dtype.itemsize)  
      self.append("cudaMemcpy(%s, %s, %s, cudaMemcpyDeviceToHost);" % (dst, src, nbytes) )
      self.append("cudaFree(%s);" % src) 

    elif isinstance(t, (ClosureT, TupleT)):
      for i, elt_t in enumerate(t.elt_types):
        host_elt = "%s.elt%d" % (host_value, i)
        gpu_elt = "%s.elt%d" % (gpu_value, i)
        self.to_host(host_elt, gpu_elt, elt_t)
    else:
      assert False, "Unsupported type in CUDA backend %s" % t 
      
  def list_to_host(self, host_values, gpu_values, types):
    for i, t in enumerate(types):
      host_value = host_values[i]
      gpu_value = gpu_values[i]
      self.to_host(host_value, gpu_value, t)
    
  def visit_Alloc(self, expr):
    assert self.in_host(), "Can't dynamically allocate memory in GPU code"
    return MulticoreCompiler.visit_Alloc(self, expr)
  
  def visit_AllocArray(self, expr):
    assert self.in_host(), "Can't dynamically allocate memory in GPU code"
    return MulticoreCompiler.visit_AllocArray(self, expr)

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

