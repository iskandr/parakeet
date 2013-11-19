import itertools 


from .. import names 
from ..builder import build_fn 
from ..ndtypes import Int64, repeat_tuple, NoneType, ScalarT, TupleT, ArrayT 
from ..syntax import (ParFor, IndexReduce, IndexScan, Index, Map, OuterMap, Var, Const, Expr)
from ..syntax.helpers import get_types, none, zero_i64 
from ..syntax.adverb_helpers import max_rank_arg, max_rank 
from transform import Transform


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
  
  def fresh_fn_name(self, prefix, fn):
    return names.fresh(prefix + names.original(self.get_fn(fn).name))
  
  def fresh_index_names(self, n):
    name_supply = itertools.cycle(["i","j","k","l","ii","jj","kk","ll"])
    return [names.fresh(name_supply.next()) for _ in xrange(n)]
  
  _indexed_fn_cache = {}
  def indexify_fn(self, fn, 
                   axis,
                   array_args, 
                   cartesian_product = False,
                   output = None, 
                   index_offsets = None):
    """
    Take a function whose last k values are slices through input data 
    and transform it into a function which explicitly extracts its arguments
    """  
    array_args = tuple(array_args)
    
    array_arg_types = tuple(get_types(array_args))

    closure_args = self.closure_elts(fn)
    closure_arg_types = tuple(get_types(closure_args))
    n_closure_args = len(closure_args)
    fn = self.get_fn(fn)
    
    axes = self.normalize_axes(array_args, axis)
    
    
    key = (  fn.cache_key, 
             axes,
             closure_arg_types, 
             array_arg_types, 
             output is None,  
             cartesian_product, 
             index_offsets, 
           )

    def mk_closure():
      new_fn = self._indexed_fn_cache[key] 
      if output is None:
        return self.closure(new_fn, closure_args + array_args)
      else:
        return self.closure(new_fn, (output, ) + closure_args + array_args)
    
    if key in self._indexed_fn_cache:
      result = mk_closure()
      return result 
    
    if cartesian_product:
      # if we're doing a cartesian product then each argument may need 
      # a different number of indices depending on whether it's a scalar
      # and when its axis is None or given as an int 
      n_indices = 0
      for axis, arg_t in zip(axes, array_arg_types):
        if axis is None:
          n_indices += self.rank(arg_t)
        elif isinstance(arg_t, ArrayT):
          n_indices += 1
    else:
      # if we're doing an element-wise map, 
      # then either the axes are all None, in which case 
      # we need indices for the largest arg
      # or, we're just picking off one slice from 
      # every argument 
      if any(axis is None for axis in axes):
        assert all(axis is None for axis in axes), "Incompatible axes %s" % axes 
        n_indices = max_rank(array_arg_types)
      else:
        assert all(isinstance(axis, (int,long)) for axis in axes), "Invalid axes %s" % axes 
        n_indices = 1
        
    #index_input_type = Int64 if n_indices == 1 else repeat_tuple(Int64, n_arrays) 
    index_input_types = (Int64,) * n_indices  
    
    if output is None:
      inner_input_types = closure_arg_types + array_arg_types + index_input_types
      new_return_type = fn.return_type 
    else:
      inner_input_types = (output.type,) + closure_arg_types +  array_arg_types + index_input_types
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
    
    
    input_names.extend(self.fresh_index_names(n_indices))
      
    
    new_fn_name = self.fresh_fn_name("idx_", fn)
    
    new_fn, builder, input_vars = build_fn(inner_input_types, 
                                           new_return_type,
                                           name = new_fn_name,  
                                           input_names = input_names)

    index_input_vars = input_vars[-n_indices:]
    if output is None:
      output_var = None
      closure_arg_vars = input_vars[:n_closure_args]
      array_arg_vars = input_vars[n_closure_args:-n_indices]
    else:
      output_var = input_vars[0]
      closure_arg_vars = input_vars[1:n_closure_args+1]
      array_arg_vars = input_vars[n_closure_args+1:-n_indices]
    
    slice_values = \
      self.get_slices(builder, array_arg_vars, axes, index_input_vars, 
                      cartesian_product, index_offsets)


    elt_result = builder.call(fn, tuple(closure_arg_vars) + tuple(slice_values))
    if output is None: 
      builder.return_(elt_result)
    else:
      if len(index_input_vars) > 1:
        builder.setidx(output_var, builder.tuple(index_input_vars), elt_result)
      else:
        builder.setidx(output_var, index_input_vars[0], elt_result)
      builder.return_(none)

    self._indexed_fn_cache[key] = new_fn
    return mk_closure()
          
    
  
  def get_slices(self, builder, array_arg_vars, axes, index_input_vars, 
                 cartesian_product = False, index_offsets = None): 
    slice_values = []
    axes = self.normalize_axes(array_arg_vars, axes)
    # only gets incremented if we're doing a cartesian product
    if cartesian_product:
      idx_counter = 0
      for i, curr_array in enumerate(array_arg_vars):
        axis = axes[i]
        rank = curr_array.type.rank
        if rank <= 1 and axis is None:
          axis = 0
           
        if axis is None:
          start = idx_counter
          stop = idx_counter + rank
          curr_indices = index_input_vars[start:stop]
          idx_counter = stop 
          curr_slice = builder.index(curr_array, curr_indices)
        elif rank > axis:
          idx = index_input_vars[idx_counter]
          idx_counter += 1
          curr_slice = builder.slice_along_axis(curr_array, axis, idx)
        else:
          curr_slice = curr_array 
        slice_values.append(curr_slice)
    else:
      for i, curr_array in enumerate(array_arg_vars):
        axis = axes[i]
        rank = curr_array.type.rank

        if rank <= 1 and axis is None:
          axis = 0
        if axis is None:
          assert len(index_input_vars) <= rank, \
            "Insufficient indices for array arg %s : %s" % (curr_array, curr_array.type)
            
          # to be compatible with NumPy's broadcasting, we pull out the *last* r
          # indices so that Matrix + Vector will replicate the vector as columns, not rows
          curr_indices = index_input_vars[-rank:]
          curr_slice = builder.index(curr_array, curr_indices)
        elif rank > axis:
          curr_idx = index_input_vars[0]
          if index_offsets is not None:
            assert len(index_offsets) > i
            curr_offset = index_offsets[i]
            if not isinstance(curr_offset, Expr): 
              curr_offset = builder.int(curr_offset)
            curr_idx = builder.add(curr_idx, curr_offset)
          curr_slice = builder.slice_along_axis(curr_array, axis, curr_idx) 
        else:
          # if we're trying to map over axis 1 of a 1-d object, then there aren't
          # enough dims to slice anything, so it just gets passed in without modification
          
          curr_slice = curr_array 
        
          
        slice_values.append(curr_slice)
    return slice_values

  

  def create_map_output_array(self, 
                                 fn, array_args, axes, 
                                 cartesian_product = False, 
                                name = "output"):
    """
    Given a function and its argument, use shape inference to figure out the
    result shape of the array and preallocate it.  If the result should be a
    scalar, just return a scalar variable.
    """
    assert self.is_fn(fn), \
      "Expected function, got %s" % (fn,)
    assert isinstance(array_args, (list,tuple)), \
      "Expected list of array args, got %s" % (array_args,)
    axes = self.normalize_axes(array_args, axes)
    
    
    
    n_indices = 0
    for arg, axis in zip(array_args, axes):
      r = self.rank(arg)
      if r == 0:
        continue 
      if axis is None:
        if cartesian_product:
          n_indices += self.rank(arg)
        else:
          n_indices = max(n_indices, self.rank(arg))
      elif r <= axis:
        continue 
      else:
        if cartesian_product:
          n_indices += 1
        else:
          n_indices = max(n_indices, 1)
    
    
    # take the 0'th slice just to have a value in hand 
    inner_args = self.get_slices(builder = self, 
                                 array_arg_vars = array_args, 
                                 axes = axes, 
                                 index_input_vars = [zero_i64] * (n_indices),
                                 cartesian_product = cartesian_product)
     
                  
    
    if cartesian_product:
      extra_dims = []
      for array, axis in zip(array_args, axes):
        rank = self.rank(array)
        if axis is None:
          dim = 1
        elif rank > axis:
          dim = self.shape(array, axis)
        else:
          dim = 1 
        extra_dims.append(dim)
      outer_shape_tuple = self.tuple(extra_dims)
    else:
      outer_shape_tuple = self.iter_bounds(array_args, axes)
      if isinstance(outer_shape_tuple.type, ScalarT):
        outer_shape_tuple = self.tuple([outer_shape_tuple])

    return self.create_output_array(fn, inner_args, outer_shape_tuple, name)

  
  def transform_Map(self, expr, output = None):
    # TODO: 
    # - recursively descend down the function bodies to pull together nested ParFors
    

    args = self.transform_expr_list(expr.args)
    axes = self.normalize_axes(args, expr.axis)
    old_fn = expr.fn

    if output is None:
      output = self.create_map_output_array(old_fn, args, axes)

    bounds = self.iter_bounds(args, axes)
    index_fn = self.indexify_fn(expr.fn, axes, args, 
                                cartesian_product=False, 
                                output = output)
    self.parfor(index_fn, bounds)
    return output 
  
  def transform_OuterMap(self, expr):
    args = self.transform_expr_list(expr.args)
    axes = self.normalize_axes(args, expr.axis)
    
    fn = expr.fn 
    outer_shape = self.iter_bounds(args, axes, cartesian_product = True)
    # recursively descend down the function bodies to pull together nested ParFors 
    zero = self.int(0)
    
    first_values = [self.slice_along_axis(arg, axis, zero) 
                    for (arg,axis) in zip(args, axes)]
    # self.create_output_array(fn, inner_args, outer_shape, name)
    output =  self.create_output_array(fn, first_values, outer_shape)

    loop_body = self.indexify_fn(fn, axes, args, 
                                 cartesian_product = True, 
                                 output = output)
    self.parfor(loop_body, outer_shape)
    return output 
  
  def transform_IndexMap(self, expr, output = None):
    shape = expr.shape 
    fn = expr.fn 
    dims = self.tuple_elts(shape)
    n_dims = len(dims)
    if n_dims == 1: shape = dims[0]
    if output is None: output = self.create_output_array(fn, [shape], shape)
    old_closure_args = self.closure_elts(fn)
    old_closure_arg_types = get_types(old_closure_args)
    fn = self.get_fn(fn)
    
    closure_arg_names = [self.fresh_input_name(clos_arg) for clos_arg in old_closure_args] 
    new_closure_vars = [Var(name, type=t) 
                        for name, t in 
                        zip(closure_arg_names, old_closure_arg_types)]
    
    old_input_types = fn.input_types
    last_input_type = old_input_types[-1]
    index_is_tuple = isinstance(last_input_type, TupleT)
    if index_is_tuple:
      index_types = last_input_type.elt_types
    else:
      index_types = old_input_types[-n_dims:]
    
    idx_names = self.fresh_index_names(n_dims)
    assert len(index_types) == n_dims, \
        "Mismatch between bounds of IndexMap %s and %d index formal arguments" % (dims, len(index_types))
    output_name = names.refresh("output")  
    
    new_input_names = [output_name] + closure_arg_names + idx_names            
    new_input_types =  [output.type]  + old_closure_arg_types + list(index_types)
    new_fn_name = names.fresh("idx_" + names.original(fn.name))
    new_fn, builder, input_vars = build_fn(new_input_types, NoneType,
                                           name = new_fn_name,  
                                           input_names = new_input_names)
    output_var = input_vars[0]
    
    idx_vars = input_vars[-n_dims:]

    if index_is_tuple:
      elt_result = builder.call(fn, new_closure_vars + [builder.tuple(idx_vars)])
    else:
      elt_result = builder.call(fn, new_closure_vars + idx_vars)
    if len(idx_vars) == 1:
      builder.setidx(output_var, idx_vars[0], elt_result)
    else:
      builder.setidx(output_var, builder.tuple(idx_vars), elt_result)

      
    builder.return_(none)
    new_closure = self.closure(new_fn, (output,) + tuple(old_closure_args)  )
    self.parfor(new_closure, shape)
    return output
  
  def transform_Reduce(self, expr):
    fn = expr.fn 
    combine = expr.combine 
    init = expr.init 
    
    args = []
    axes = []
    raw_axes = self.normalize_axes(expr.args, expr.axis)
    for axis, arg in zip(raw_axes, expr.args):
      if self.is_none(axis):
        args.append(self.ravel(arg))
        axes.append(0)
      else:
        args.append(arg)
        axes.append(axis)
    
    max_arg = max_rank_arg(args)
    nelts = self.shape(max_arg, axis)
    
    if self.is_none(axis):
      nelts = self.prod(nelts)
      
    if init is None or self.is_none(init):
      init_args = [self.index_along_axis(arg, axis, self.int(0)) for arg, axis in zip(args, axes)]
      init = self.call(fn, init_args)
      index_offsets = (1,) * len(axes)
      assert init.type == fn.return_type
      nelts = self.sub(nelts, self.int(1), "nelts") 
    else:
      index_offsets = None
    
    index_fn = self.indexify_fn(fn, 
                                axis, 
                                args, 
                                cartesian_product=False, 
                                index_offsets = index_offsets)

    return IndexReduce(fn = index_fn, 
                       init = init, 
                       combine = combine, 
                       shape = nelts, 
                       type = expr.type)
   
  def transform_Scan(self, expr, output = None):
    
    combine = expr.combine 
    init = expr.init 
    
    
    args = []
    axes = []
    for (arg, axis)  in zip(expr.args, self.normalize_axes(expr.args, expr.axis)):
      if self.is_none(axis):
        args.append(self.ravel(arg))
        axis = 0
      else:
        args.append(arg)
      axes.append(axis)
     
    bounds = self.iter_bounds(args, axes)

    if self.is_none(init):
      assert len(args) == 1
      init_args = [self.index_along_axis(arg, axis, self.int(0))
                   for arg in args]
      init = self.call(expr.fn, init_args)  
      index_offsets = (1,) * len(bounds)
      bounds = tuple(self.sub(bound, self.int(1), "scan_niters") for bound in bounds)
    else:
      index_offsets = None
      
    index_fn = self.indexify_fn(expr.fn, 
                                axis, 
                                args, 
                                cartesian_product=False, 
                                index_offsets = index_offsets)
    

    return IndexScan(fn = index_fn,
                     init = init, 
                     combine = combine,
                     emit = expr.emit, 
                     shape = bounds,
                     type = expr.type)
  
  def transform_Filter(self, expr):
    assert False, "Filter not implemented"
    
  
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

  
