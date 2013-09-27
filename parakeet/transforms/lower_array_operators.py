from ..builder import build_fn
from ..syntax import  Alloc,  Index, ArrayView, Const, Transpose 
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
    assert isinstance(elt_t, ScalarT), "Expected array to have scalar elemements but got %s" % elt_t

    elts = self.transform_expr_list(expr.elts)
    n = len(elts)

    ptr_t = ptr_type(elt_t)
    alloc = Alloc(elt_t, const_int(n), type = ptr_t)
    ptr_var = self.assign_name(alloc, "data")

    for (i, elt) in enumerate(elts):
      lhs = Index(ptr_var, const_int(i), type = elt_t)
      self.assign(lhs, elt)

    return self.array_view(ptr_var, 
                           const_tuple(n), 
                           const_tuple(1),
                           offset = const_int(0), 
                           nelts = const_int(n))
      
  def transform_Len(self, expr):
    return self.shape(expr, 0)

  def transform_AllocArray(self, expr):
    dims = self.transform_expr(expr.shape),
    return self.alloc_array(elt_t = expr.type.elt_type,
                            dims = dims, 
                            name = "array",
                            order = "C", 
                            array_view = True, 
                            explicit_struct = False)
    
  def transform_Reshape(self, expr):
    assert False, "Reshape not implemented"
    
  def transform_Shape(self, expr):
    return self.shape(expr)
  
  def transform_Strides(self, expr):
    return self.strides(expr)

  
  """
  Given first-order array constructors, turn them into IndexMaps
  """

  def mk_range_fn(self, start, step, output_type,   _range_fn_cache = {}):
    """
    Given expressions for start, stop, and step of an iteration range
    and a desired output type. 
    
    Create a function which maps indices i \in 0...(stop-start)/step into
    f(i) = cast(start + i * step, output_type)
    """
    
      
    start_is_const = start.__class__ is Const
    step_is_const = step.__class__ is Const
    
    key = output_type, start_is_const,  step_is_const
    if key in _range_fn_cache:
      fn = _range_fn_cache[key]
    else:
      input_types = [Int64]
      if not start_is_const: input_types.append(start.type)
      if not step_is_const: input_types.append(step.type)
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
    strides = self.strides(expr.array)
    any_unit = false 
    for stride in strides:
      any_unit = self.or_(any_unit, self.eq(stride, one_i64))
    assert False, "Ravel not implemented"
    
  def transform_Transpose(self, expr):
    if expr.array.__class__ is Transpose:
      return self.transform_expr(expr.array.array)
    array = self.transform_expr(expr.array)
    data = self.attr(array, 'data')
    shape = self.shape(array)
    strides = self.strides(array)
    offset = self.attr(array, 'offset')
    # TODO: What happens to the offset when you transpose an array? 
    assert False, "Transpose not implemented"
    
  def transform_Where(self, expr):
    assert False, "Where not implemented"
  
  
  def transform_ConstArray(self, expr):
    assert False, "ConstArray not implemented"
  
  def transform_ConstArrayLike(self, expr):
    assert False, "ConstArrayLike not implemented"
  
  
  