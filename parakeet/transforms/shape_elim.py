from .. import syntax
from ..ndtypes import ArrayT, SliceT, ScalarT, TupleT, Int64, StructT, ClosureT, IntT, NoneT 
from ..syntax import Attribute, TupleProj, Var, ClosureElt
from ..shape_inference import shape_env, shape
from ..transforms import Transform


class Counter(object):
  """
  Auto-incrementing counter
  """
  def __init__(self):
    self.n = 0
    
  def get(self):
    n = self.n 
    self.n += 1
    return n 
  
class ShapeElimination(Transform):
  
   
  def pre_apply(self, fn):
    self.shape_env = shape_env(fn)
    # map from shape vars to expressions 
    self.shape_vars = {}
    self.shape_var_counter = Counter()
    input_vars = [Var(arg_name, type = arg_type)
                  for arg_name, arg_type in zip(fn.arg_names, fn.input_types)]
    self.fill_shape_vars_list(input_vars)
    
    
      
  def fill_shape_vars(self, expr):
    t = expr.type 
    c = t.__class__ 
    if c is TupleT:
      elts = [TupleProj(expr,i,type=elt_t) 
              for i, elt_t in enumerate(t.elt_types)]
      self.fill_shape_vars_list(elts)
    elif c is ClosureT:
      elts = [ClosureElt(expr,i,type=elt_t) 
              for i, elt_t in enumerate(t.arg_types)]
      self.fill_shape_vars_list(elts)
    elif c is ArrayT:
      shape = Attribute(expr, 'shape', t.shape_t)
      dims = [TupleProj(shape,i,type=Int64) for i in xrange(t.rank)]
      self.fill_shape_vars_list(dims)
    elif c is StructT:
      fields = [Attribute(expr, field_name, type = field_type)
                for field_name, field_type in zip(t.field_names, t.field_types)]
      self.fill_shape_vars_list(fields)
    elif c is SliceT:
      fields = []
      start_t = expr.type.start_type 
      if start_t != NoneT: fields.append(Attribute(expr, 'start', type = start_t))
      stop_t = expr.type.stop_type 
      if stop_t != NoneT: fields.append(Attribute(expr, 'stop', type = stop_t))
      step_t = expr.type.step_type 
      if step_t != NoneT: fields.append(Attribute(expr, 'step', type = step_t))
      self.fill_shape_vars_list(fields)
    elif isinstance(t, ScalarT):
      self.shape_vars[self.shape_var_counter.get()] = expr 
      
      
  def fill_shape_vars_list(self, arg_exprs):
    for e in arg_exprs:
      self.fill_shape_vars(e)

  def transform_Var(self, expr):
    #print expr, self.shape_env.get(expr.name) 
    if expr.name in self.shape_env:
      v = self.shape_env[expr.name]
      # don't trust shape propagation to correctly handle floating point errors
      if v.__class__ is shape.Const and isinstance(expr.type, IntT):
        return syntax.Const(value = v.value, type = expr.type)
      elif v.__class__ is shape.Var: 
        new_expr = self.shape_vars[v.num]
        
        if new_expr != expr:
          return self.cast(self.shape_vars[v.num], expr.type)
    return expr
  
  def transform_lhs(self, lhs):
    return lhs