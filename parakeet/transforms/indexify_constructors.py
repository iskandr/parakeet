from ..builder import build_fn
from ..ndtypes import Int64 
from ..syntax import Const 

from transform import Transform


class IndexifyArrayConstructors(Transform):
  """
  Given first-order array constructors, turn them into IndexMaps
  """
  _range_fn_cache = {}
  
  def mk_range_fn(self, start, step, output_type):
    """
    Given expressions for start, stop, and step of an iteration range
    and a desired output type. 
    
    Create a function which maps indices i \in 0...(stop-start)/step into
    f(i) = cast(start + i * step, output_type)
    """
    
      
    start_is_const = start.__class__ is Const
    step_is_const = step.__class__ is Const
    
    key = output_type, start_is_const,  step_is_const
    if key in self._range_fn_cache:
      fn = self._range_fn_cache[key]
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
      self._range_fn_cache[key] = fn
    
    closure_args = []
    if not start_is_const: closure_args.append(start)
    if not step_is_const: closure_args.append(step)
    return self.closure(fn, closure_args) 
      
  def transform_Range(self, expr):
    start = expr.start 
    stop = expr.stop
    step = expr.step 
    caster = self.mk_range_fn(start, step, expr.type.elt_type)
    nelts = self.elts_in_slice(start, stop, step)
    return self.imap(caster, nelts)
    
    
    