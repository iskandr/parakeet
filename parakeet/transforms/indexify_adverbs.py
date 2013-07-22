from .. import names 
from ..builder import build_fn 
from ..ndtypes import Int64, repeat_tuple, NoneType 
from ..syntax import ParFor, IndexReduce, IndexScan, IndexFilter, Index, Map, OuterMap 
from ..syntax.helpers import unwrap_constant, get_types, none 
from ..syntax.adverb_helpers import max_rank_arg
from transform import Transform 

class IndexifyAdverbs(Transform):
  """
  Take all the adverbs whose parameterizing functions assume they 
  get fed slices of input data and turn them into version which take explicit
  input indices
  """
  _indexed_fn_cache = {}
  def indexify_fn(self, fn, axis, array_args, 
                   cartesian_product = False,
                   output = None):
    """
    Take a function whose last k values are slices through input data 
    and transform it into a function which explicitly extracts its arguments
    """  
    array_arg_types = tuple(get_types(array_args))
    
    # do I need fn.version *and* fn.copied_by? 
    key = (fn.name, fn.copied_by, fn.version, 
           array_arg_types, 
           cartesian_product, 
           axis, output is None)
    if key in self._indexed_fn_cache:
      return self._indexed_fn_cache[key]
    old_input_vars = self.input_vars(fn)
    n_arrays = len(array_arg_types)
    old_closure_args = old_input_vars[:-n_arrays]
    array_slice_args = old_input_vars[-n_arrays:]
    
    array_arg_vars = tuple(self.fresh_var(t, prefix="array_arg%d" % i)
                           for i,t in enumerate(array_arg_types))
    #max_array_arg = max_rank_arg(array_arg_vars)
    # max_array_rank = self.rank(max_array_arg)
    n_indices = n_arrays if cartesian_product else 1
    index_input_type = Int64 if n_indices == 1 else repeat_tuple(Int64, n_arrays) 
    
    if output is None:
      new_closure_args = tuple(old_closure_args) + array_args
      inner_input_types = get_types(new_closure_args) + tuple([index_input_type])
      new_return_type = fn.return_type 
    else:
      new_closure_args = tuple(old_closure_args) + (output,) + array_args
      inner_input_types = get_types(new_closure_args) + tuple([index_input_type])
      new_return_type = NoneType 
    new_fn, builder, input_vars = build_fn(inner_input_types, new_return_type)
    index_input_var = input_vars[-1]
    
    index_elts = self.tuple_elts(index_input_var) if n_indices > 1 else [index_input_var]
    if cartesian_product and n_indices > 1:
      index_elts = index_elts * n_arrays 

    for i, array_slice_arg in enumerate(array_slice_args):
      curr_array = array_arg_vars[i]
      # if not cartesian_product: 
      #  if self.rank(curr_array) == max_array_rank:
      #    slice_value = self.slice_along_axis(array_arg_vars[i], axis, index_elts[i])
      #  else:
      #    slice_value = curr_array
      slice_value = builder.slice_along_axis(curr_array, axis, index_elts[i]) 
      builder.assign(array_slice_arg, slice_value)
    
    elt_result = builder.call(fn, old_input_vars)
    if output is None: 
      builder.return_(elt_result)
    else:
      builder.setidx(output, index_input_var, elt_result)
      builder.return_(none)
    new_closure = self.closure(new_fn, new_closure_args)
    self._indexed_fn_cache[key] = new_closure
    return new_closure
  

  def sizes_along_axis(self, xs, axis):
    axis_sizes = [self.size_along_axis(x, axis)
                  for x in xs
                  if self.rank(x) > axis]

    assert len(axis_sizes) > 0
    # all arrays should agree in their dimensions along the
    # axis we're iterating over
    self.check_equal_sizes(axis_sizes)
    return axis_sizes

  """
  def prelude(self, map_fn, xs, axis):
    axis_sizes = self.sizes_along_axis(xs, axis)
    return axis_sizes[0]

  def acc_prelude(self, init, combine, delayed_map_result):
    zero = self.int(0)
    if init is None or self.is_none(init):
      return delayed_map_result(zero)
    else:
      # combine the provided initializer with
      # transformed first value of the data
      # in case we need to coerce up
      return self.call(combine, [init, delayed_map_result(zero)])
  """
  
  #def create_result(self, elt_type, inner_shape, outer_shape):
  #  if not self.is_tuple(outer_shape):
  #    outer_shape = self.tuple([outer_shape])
  #  result_shape = self.concat_tuples(outer_shape, inner_shape)
  #  result = self.alloc_array(elt_type, result_shape)
  #  return result

  #def create_output_array(self, fn, inputs, extra_dims):
  #  if hasattr(self, "_create_output_array"):
  #    try:
  #      return self._create_output_array(fn, inputs, extra_dims)
  #    except:
  #      pass
  #  inner_result = self.call(fn, inputs)   
  #  inner_shape = self.shape(inner_result)
  #  elt_t = self.elt_type(inner_result)
  #  res =  self.create_result(elt_t, inner_shape, extra_dims)
  #  return res 
  
  def parfor(self, bounds, fn):
    self.blocks += [ParFor(fn = fn, bounds = bounds)]
    
  def transform_Map(self, expr, output = None):
    # recursively descend down the function bodies to pull together nested ParFors
    args = self.transform_expr_list(expr.args)
    axis = unwrap_constant(expr.axis)
    old_fn = expr.fn


    if output is None:
      # outer_dims = [niters]
      # use shape inference to create output
      output = self.create_output_array(old_fn, args, axis)
        
    index_fn = self.indexify_fn(expr.fn, axis, args, cartesian_product=False, 
                                output = output)
    biggest_arg = max_rank_arg(args)
    niters = self.shape(biggest_arg, axis)
    self.parfor(niters, index_fn)
    return output 
  
  def transform_OuterMap(self, expr):
    args = self.transform_expr_list(expr.args)
    axis = unwrap_constant(expr.axis)
    fn = expr.fn 
    dimsizes = [self.shape(arg, axis) for arg in args]
    # recursively descend down the function bodies to pull together nested ParFors 
    if axis is None: 
      args = [self.ravel(arg) for arg in args]
      axis = 0
    counts = [self.size_along_axis(arg, axis) for arg in args]
    outer_shape = self.tuple(counts)
    zero = self.int(0)
    first_values = [self.slice_along_axis(arg, axis, zero) for arg in args]
    output =  self.create_output_array(fn, first_values, outer_shape)
    loop_body = self.indexify_fn(fn, axis, args, cartesian_product = True, output = output)
    self.parfor(dimsizes, loop_body)
    return output 
  
  def transform_IndexMap(self, expr):
    # recursively descend down the function bodies to pull together nested ParFors
    dims = self.tuple_elts(shape)
    if len(dims) == 1:
      shape = dims[0]
      
    if output is None:
      output = self.create_output_array(fn, [shape], shape)

    n_loops = len(dims)
    def build_loops(index_vars = ()):
      n_indices = len(index_vars)
      if n_indices == n_loops:
        if n_indices > 1:
          idx_tuple = self.tuple(index_vars)
        else:
          idx_tuple = index_vars[0]
        elt_result =  self.call(fn, (idx_tuple,))
        self.setidx(output, index_vars, elt_result)
      else:
        def loop_body(idx):
          build_loops(index_vars + (idx,))
        self.loop(self.int(0), dims[n_indices], loop_body)
    build_loops()
    return output 

    return ParFor()
  
  def transform_Reduce(self, expr):
    zero = self.int(0)
    one = self.int(1)
    
    if axis is  None or self.is_none(axis):
      assert len(values) == 1
      values = [self.ravel(values[0])]
      axis = 0
    
    niters, delayed_elts = self.map_prelude(map_fn, values, axis)
    first_acc_value = self.call(map_fn, [elt(zero) for elt in delayed_elts])
    if init is None or self.is_none(init):
      init = first_acc_value
    else:
      init = self.call(combine, [init, first_acc_value])
    def loop_body(acc, idx):
      elt = self.call(map_fn, [elt(idx) for elt in delayed_elts])
      new_acc_value = self.call(combine, [acc.get(), elt])
      acc.update(new_acc_value)
    return self.accumulate_loop(one, niters, loop_body, init)
    return IndexReduce()
  
  def transform_Scan(self, expr):
    zero = self.int(0)
    one = self.int(1)
    
    if axis is  None or self.is_none(axis):
      assert len(values) == 1
      values = [self.ravel(values[0])]
      axis = 0
    
    niters, delayed_elts = self.map_prelude(map_fn, values, axis)
    first_acc_value = self.call(map_fn, [elt(zero) for elt in delayed_elts])
    if init is None or self.is_none(init):
      init = first_acc_value
    else:
      init = self.call(combine, [init, first_acc_value])
    def loop_body(acc, idx):
      elt = self.call(map_fn, [elt(idx) for elt in delayed_elts])
      new_acc_value = self.call(combine, [acc.get(), elt])
      acc.update(new_acc_value)
    return self.accumulate_loop(one, niters, loop_body, init)
    # return IndexScan()
  
  def transform_Filter(self, expr):
    assert False, "Filter not implemented"
    # return IndexFilter(self, expr)
    
  def transform_Assign(self, stmt):
    """
    If you encounter an adverb being written to an output location, 
    then why not just use that as the output directly? 
    """
    if stmt.lhs.__class__ is Index:
      rhs_class = stmt.rhs.__class__ 
      if rhs_class is Map:
        self.transform_Map(stmt.rhs, output = stmt.lhs)
        return None 
      elif rhs_class is OuterMap:
        self.transform_OuterMap(stmt.rhs, output = stmt.lhs)
       
    return Transform.transform_Assign(self, stmt)
  
  """
  def transform_Map(self, expr, output = None):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    axis = unwrap_constant(expr.axis)
    return self.eval_map(fn, args, axis, output = output)

  def transform_Reduce(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    combine = self.transform_expr(expr.combine)
    init = self.transform_if_expr(expr.init)
    axis = unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, init, args, axis)

  def transform_Scan(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    combine = self.transform_expr(expr.combine)
    emit = self.transform_expr(expr.emit)
    init = self.transform_if_expr(expr.init)
    axis = unwrap_constant(expr.axis)
    return self.eval_scan(fn, combine, emit, init, args, axis)

  def transform_AllPairs(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    assert len(args) == 2
    x,y = self.transform_expr_list(args)
    axis = unwrap_constant(expr.axis)
    return self.eval_allpairs(fn, x, y, axis)
  """
  