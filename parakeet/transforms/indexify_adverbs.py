from .. import names 
from ..builder import build_fn 
from ..ndtypes import Int64, repeat_tuple, NoneType, ScalarT 
from ..syntax import (ParFor, IndexReduce, IndexScan, IndexFilter, Index, Map, OuterMap, 
                      Var, Return)
from ..syntax.helpers import unwrap_constant, get_types, none 
from ..syntax.adverb_helpers import max_rank_arg
from inline import Inliner 
from transform import Transform
from scipy.weave.build_tools import old_init_posix

 

class IndexifyAdverbs(Transform):
  """
  Take all the adverbs whose parameterizing functions assume they 
  get fed slices of input data and turn them into version which take explicit
  input indices
  """
  
  def fresh_input_name(self, expr):
    if expr is Var:
      return names.refresh(expr.name)
    else:
      return names.fresh("input")
    
  _indexed_fn_cache = {}
  def indexify_fn(self, fn, axis, array_args, 
                   cartesian_product = False,
                   output = None):
    """
    Take a function whose last k values are slices through input data 
    and transform it into a function which explicitly extracts its arguments
    """  
    array_args = tuple(array_args)
    array_arg_types = tuple(get_types(array_args))

    n_arrays = len(array_arg_types)
    closure_args = self.closure_elts(fn)
    closure_arg_types = tuple(get_types(closure_args))
    n_closure_args = len(closure_args)
    fn = self.get_fn(fn)
    # do I need fn.version *and* fn.copied_by? 
    key = (
             fn.name, 
             fn.copied_by,
             axis, 
             closure_arg_types, 
             array_arg_types, 
             output is None,  
             cartesian_product, 
           )
    
    def mk_closure():
      new_fn = self._indexed_fn_cache[key] 
      if output is None:
        return self.closure(new_fn, closure_args + array_args)
      else:
        return self.closure(new_fn, (output, ) + closure_args + array_args)
    
    if key in self._indexed_fn_cache:
      return mk_closure()
      
    #max_array_arg = max_rank_arg(array_arg_vars)
    # max_array_rank = self.rank(max_array_arg)
    n_indices = n_arrays if cartesian_product else 1
    index_input_type = Int64 if n_indices == 1 else repeat_tuple(Int64, n_arrays) 
    
    if output is None:
      inner_input_types = closure_arg_types + array_arg_types + (index_input_type,)
      new_return_type = fn.return_type 
    else:
      inner_input_types = (output.type,) + closure_arg_types +  array_arg_types + (index_input_type,)
      new_return_type = NoneType 
    
    
    input_names = []
    if output is not None:
      if output is Var:
        local_output_name = names.refresh(output.name)
      else:
        local_output_name = names.fresh("local_output")
      input_names.append(local_output_name) 
    
    for old_input_name in fn.arg_names:
      input_names.append(names.refresh(old_input_name)) 
    
    input_names.append(names.fresh("idx"))
   
    new_fn_name = names.fresh("idx_" + names.original(fn.name))
    
    new_fn, builder, input_vars = build_fn(inner_input_types, 
                                           new_return_type,
                                           name = new_fn_name,  
                                           input_names = input_names)

    index_input_var = input_vars[-1]
    if output is None:
      output_var = None
      closure_arg_vars = input_vars[:n_closure_args]
      array_arg_vars = input_vars[n_closure_args:-1]
    else:
      output_var = input_vars[0]
      closure_arg_vars = input_vars[1:n_closure_args+1]
      array_arg_vars = input_vars[n_closure_args+1:-1]
    
    if cartesian_product:
      index_elts = self.tuple_elts(index_input_var) if n_indices > 1 else [index_input_var]
    else:
      assert isinstance(index_input_var.type, ScalarT), \
        "Unexpected index type %s" % (index_input_var.type)
      index_elts = [index_input_var] * n_arrays 
    
    slice_values = []
    for i, curr_array in enumerate(array_arg_vars):

      # if not cartesian_product: 
      #  if self.rank(curr_array) == max_array_rank:
      #    slice_value = self.slice_along_axis(array_arg_vars[i], axis, index_elts[i])
      #  else:
      #    slice_value = curr_array
      
      curr_slice = builder.slice_along_axis(curr_array, axis, index_elts[i])
      
      slice_values.append(curr_slice) 
    
    elt_result = builder.call(fn, tuple(closure_arg_vars) + tuple(slice_values))
    if output is None: 
      builder.return_(elt_result)
    else:
      builder.setidx(output_var, index_input_var, elt_result)
      builder.return_(none)
    
    inliner = Inliner()
    new_fn = inliner.apply(new_fn)
    self._indexed_fn_cache[key] = new_fn 
    return mk_closure()
          
    
  

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
      output = self.create_map_output_array(old_fn, args, axis)
    
    index_fn = self.indexify_fn(expr.fn, axis, args, 
                                cartesian_product=False, 
                                output = output)
    biggest_arg = max_rank_arg(args)
    niters = self.shape(biggest_arg, axis)
    self.parfor(niters, index_fn)
    return output 
  
  def transform_OuterMap(self, expr):
    args = self.transform_expr_list(expr.args)
    axis = unwrap_constant(expr.axis)
    fn = expr.fn 
    # dimsizes = [self.shape(arg, axis) for arg in args]
    # recursively descend down the function bodies to pull together nested ParFors 
    if axis is None: 
      args = [self.ravel(arg) for arg in args]
      axis = 0
    counts = [self.size_along_axis(arg, axis) for arg in args]
    outer_shape = self.tuple(counts)
    zero = self.int(0)
    first_values = [self.slice_along_axis(arg, axis, zero) for arg in args]
    # self.create_output_array(fn, inner_args, outer_shape, name)
    output =  self.create_output_array(fn, first_values, outer_shape)
    loop_body = self.indexify_fn(fn, axis, args, 
                                 cartesian_product = True, 
                                 output = output)
    self.parfor(outer_shape, loop_body)
    return output 
  
  def transform_IndexMap(self, expr, output = None):
    shape = expr.shape 
    fn = expr.fn 
    
    dims = self.tuple_elts(shape)
    
    if len(dims) == 1:
      shape = dims[0]
    
    if output is None:
      output = self.create_output_array(fn, [shape], shape)
    
    
    old_closure_args = self.closure_elts(fn)
    old_closure_arg_types = get_types(old_closure_args)
    fn = self.get_fn(fn)
    
    closure_arg_names = [self.fresh_input_name(clos_arg) for clos_arg in old_closure_args] 
    new_closure_vars = [Var(name, type=t) 
                        for name, t in 
                        zip(closure_arg_names, old_closure_arg_types)]
    
    idx_name = names.refresh(fn.arg_names[-1])
    output_name = names.refresh("output")  
    
    new_input_names = [output_name] + closure_arg_names + [idx_name]            
    new_input_types =  [output.type]  + old_closure_arg_types + [fn.input_types[-1]]
    new_fn, builder, input_vars = build_fn(new_input_types, NoneType,
                                           name =  names.fresh("idx_" + names.original(fn.name)),  
                                           input_names = new_input_names)
    output_var = input_vars[0]
    idx_var = input_vars[-1]
    builder.setidx(output_var, idx_var, builder.call(fn, new_closure_vars + [idx_var]))
    builder.return_(none)
    return self.parfor(shape, self.closure(new_fn, (output,) + tuple(old_closure_args)  ))
    
  
  def transform_Reduce(self, expr):
    fn = expr.fn 
    combine = expr.combine 
    init = expr.init 
    args = expr.args

    axis = expr.axis
    if axis is  None or self.is_none(axis):
      assert len(args) == 1
      args = [self.ravel(args[0])]
      axis = 0
    else:
      axis = unwrap_constant(axis)

    if self.is_none(init):
      assert len(args) == 1, "If 'init' not specified then can't have more than 1 arg"
      init = self.index_along_axis(args[0], axis, self.int(0))
      assert init.type == fn.return_type 
      
    index_fn = self.indexify_fn(fn, 
                                axis, 
                                args, 
                                cartesian_product=False)
    max_arg = max_rank_arg(args)
    shape = self.shape(max_arg, axis) 

    return IndexReduce(fn = index_fn, 
                       init = init, 
                       combine = combine, 
                       shape = shape, 
                       type = expr.type)
  
  def transform_Scan(self, expr, output = None):
    combine = expr.combine 
    init = expr.init 
    args = expr.args

    axis = expr.axis
    if axis is  None or self.is_none(axis):
      assert len(args) == 1
      args = [self.ravel(args[0])]
      axis = 0
    else:
      axis = unwrap_constant(axis)
    index_fn = self.indexify_fn(expr.fn, 
                                axis, 
                                args, 
                                cartesian_product=False)
    assert init is not None
    max_arg = max_rank_arg(args)
    shape = self.shape(max_arg) 
    return IndexScan(fn = index_fn, 
                     init = init, 
                     combine = combine, 
                     shape = shape, 
                     type = expr.type)
  
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
  
