from ..builder import build_fn
from ..syntax import  Alloc,  Index, ArrayView, Const, Transpose, Ravel 
from ..syntax.helpers import zero_i64, one_i64, const_int, const_tuple, true, false   
from ..ndtypes import (ScalarT, PtrT, TupleT, ArrayT, ptr_type, Int64)

from transform import Transform


class LowerArrayOperators(Transform):
  """
  Lower first-order array operators such as Ravel, transpose
  into views or copies
  """


  def transform_Array(self, expr):
    """Array literal"""
    array_t = expr.type
    assert isinstance(array_t, ArrayT), "Expected array but got %s" % array_t
    elt_t = array_t.elt_type
    assert isinstance(elt_t, ScalarT), "Expected array to have scalar elements but got %s" % elt_t

    elts = self.transform_expr_list(expr.elts)
    n = len(elts)

    ptr_t = ptr_type(elt_t)
    alloc = Alloc(elt_t, const_int(n), type = ptr_t)
    ptr_var = self.assign_name(alloc, "data")
    array = self.array_view(ptr_var, 
                            shape = const_tuple(n), 
                            strides = const_tuple(1), 
                            offset = zero_i64,
                            nelts = const_int(n))
    for (i, elt) in enumerate(elts):
      self.setidx(array, const_int(i), elt)
    return array 
  
      
  def transform_Len(self, expr):
    return self.shape(expr, 0)

  def transform_AllocArray(self, expr):
    dims = self.transform_expr(expr.shape)
    return self.alloc_array(elt_t = expr.type.elt_type,
                            dims = dims, 
                            name = "array",
                            order = "C", 
                            array_view = True, 
                            explicit_struct = False)
    
  def transform_Reshape(self, expr):
    assert False, "Reshape not implemented"
    
  def transform_Shape(self, expr):

    return self.shape(self.transform_expr(expr.array))
    
  def transform_Strides(self, expr):
    return self.strides(self.transform_expr(expr.array))
  
  """
  Given first-order array constructors, turn them into IndexMaps
  """

  def mk_range_fn(self, start, step, output_type, _range_fn_cache = {}):
    """
    Given expressions for start, stop, and step of an iteration range
    and a desired output type. 
    
    Create a function which maps indices i \in 0...(stop-start)/step into
    f(i) = cast(start + i * step, output_type)
    """
    
      
    start_is_const = start.__class__ is Const
    step_is_const = step.__class__ is Const
    
    key = output_type, (start.type, start_is_const), (step.type, step_is_const)
    if key in _range_fn_cache:
      fn = _range_fn_cache[key]
    else:
      input_types = []
      if not start_is_const: input_types.append(start.type)
      if not step_is_const: input_types.append(step.type)
      input_types.append(Int64)
      fn, builder, input_vars = build_fn(input_types, output_type)
    
      assert len(input_vars) == (3 - start_is_const - step_is_const), \
        "Unexpected # of input vars %d" % len(input_vars)
        
      counter = 0
      if start_is_const:
        inner_start = start 
      else:
        inner_start = input_vars[counter]
        counter += 1
    
      if step_is_const:
        inner_step = step 
      else:
        inner_step = input_vars[counter]
        counter += 1
    
      idx = input_vars[counter]
    
      value = builder.add(builder.mul(idx, inner_step), inner_start)
      builder.return_(builder.cast(value, output_type))
      _range_fn_cache[key] = fn
    
    closure_args = []
    if not start_is_const: 
      closure_args.append(start)
    if not step_is_const: 
      closure_args.append(step)
    return self.closure(fn, closure_args) 
      
  def transform_Range(self, expr):
    start = expr.start 
    stop = expr.stop
    step = expr.step 
    caster = self.mk_range_fn(start, step, expr.type.elt_type)
    nelts = self.elts_in_slice(start, stop, step)
    
    return self.imap(caster, nelts)

  # TODO: Add support for np.ndindex through this operator 
  def transform_NDIndex(self, expr):
    shape = self.transform_expr(expr.shape)
    assert False, "NDIndex not yet implemented"
    
  
  def transform_Ravel(self, expr):
    
    array = expr.array 
    while array.__class__ is Ravel:
      array = expr.array 
    array = self.transform_expr(array)
    strides = self.tuple_elts(self.attr(array, 'strides'))
    shape = self.tuple_elts(self.attr(array, 'shape'))
    any_unit = false
    nelts = one_i64 
    for i in xrange(len(shape)):
      shape_elt = shape[i]
      nelts = self.mul(nelts, shape_elt)
      stride_elt = strides[i]
      any_unit = self.or_(any_unit, self.eq(stride_elt, one_i64))      
    array_result = self.fresh_var(expr.type, prefix = "raveled")
    data = self.attr(array, 'data')
    offset = self.attr(array, 'offset')
    def contiguous(x):
      view =  self.array_view(data, 
                              shape = self.tuple([nelts]), 
                              strides = self.tuple([one_i64]), 
                              offset = offset, 
                              nelts = nelts)
      self.assign(x, view)
    def not_contiguous(x):
      new_array = self.alloc_array(x.type.elt_type, 
                                   dims = self.attr(array, 'shape'),
                                   name = "fresh_array", 
                                   explicit_struct = False, 
                                   array_view = True, 
                                   order = "C")
      self.array_copy(array, new_array)
      data = self.attr(new_array, 'data')
      flat_view = self.array_view(data, 
                              shape = self.tuple([nelts]), 
                              strides = self.tuple([one_i64]), 
                              offset = offset, 
                              nelts = nelts)
      self.assign(x, flat_view)
    self.if_(any_unit, contiguous, not_contiguous, [array_result])
    return array_result 

    
  def transform_Transpose(self, expr):
    if expr.array.__class__ is Transpose:
      return self.transform_expr(expr.array.array)
    
    array = self.transform_expr(expr.array)
    if isinstance(array.type, ScalarT):
      return array 
    assert isinstance(array.type, ArrayT), "Can't transpose %s : %s" % (array, array.type)
    ndims = array.type.rank 
    if ndims <= 1:
      return array 
    assert ndims == 2, "Tranposing arrays of rank %d not yet supported" % (ndims,)
    
    data = self.attr(array, 'data')
    shape = self.shape(array)
    strides = self.strides(array)
    offset = self.attr(array, 'offset')
    stride_elts = self.tuple_elts(strides)
    shape_elts = self.tuple_elts(shape)
    new_shape = self.tuple(tuple(reversed(shape_elts)))
    new_strides = self.tuple(tuple(reversed(stride_elts)))
    size = self.attr(array, 'size')
    return self.array_view(data, new_shape, new_strides, offset, size)
    
  def transform_Where(self, expr):
    assert False, "Where not implemented"
  
  
  def mk_const_fn(self, idx_type, value, _const_fn_cache = {}):
    if isinstance(idx_type, TupleT) and len(idx_type.elt_types) == 1:
      idx_type = idx_type.elt_types[0]
    key = idx_type, value
    if value in _const_fn_cache:
      return _const_fn_cache[key]
    fn, builder, _ = build_fn([idx_type], value.type)
    builder.return_(value)
    _const_fn_cache[key] = fn 
    return fn 
    
  
  def transform_ConstArray(self, expr):
    const_fn = self.mk_const_fn(expr.shape.type, expr.value)
    return self.imap(const_fn, expr.shape)
  
  def transform_ConstArrayLike(self, expr):
    array = self.transform_expr(expr.array)
    shape = self.shape(array)
    const_fn = self.mk_const_fn(shape.type, expr.value)
    return self.imap(const_fn, shape)
  
  
  