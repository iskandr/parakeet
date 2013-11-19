
from dsltools import ScopedDict

from .. import config 
from ..analysis.verify import verify
from ..ndtypes import NoneT, SliceT, ScalarT, ArrayT, ClosureT, TupleT, PtrT, StructT 
from ..syntax import TypedFn
from transform import Transform


class RecursiveApply(Transform):  
  cache = ScopedDict()
  
  def __init__(self, transform_to_apply = None):
    Transform.__init__(self)
    self.transform = transform_to_apply
    self.cache.push()
  
  def __del__(self):
    self.cache.pop()
  
  
  first_order_types = (SliceT, NoneT, ArrayT, PtrT)
  
  def contains_function_type(self, t):
    c = t.__class__ 
    if c is ClosureT: 
      return True 
    elif c is TupleT: 
      return any(self.contains_function_type(elt_t) for elt_t in t.elt_types)
    elif c in self.first_order_types or isinstance(t, ScalarT):
      return False 
    else:
      assert isinstance(t, StructT), "Unexpected type %s" % t
      return any(self.contains_function_type(field_t) for field_t in t.field_types)
    
  def transform_type(self, t):
    c = t.__class__ 
    if c is ClosureT:
      old_fn = t.fn 
      if isinstance(old_fn, TypedFn):
        new_fn = self.transform_TypedFn(old_fn)
        if new_fn is not old_fn:
          return ClosureT(new_fn, t.arg_types)
    elif c is TupleT and self.contains_function_type(t):
      new_elt_types = []
      for elt_t in t.elt_types:
        new_elt_types.append(self.transform_type(elt_t))
      return TupleT(tuple(new_elt_types))
    # if it's neither a closure nor a structure which could contain closures, 
    # just return it 
    return t 
    
      
  def transform_TypedFn(self, expr):
    key = expr.cache_key
    if key in self.cache:
      return self.cache[key]  
    new_fn = self.transform.apply(expr)
    if config.opt_verify:
      try: 
        verify(new_fn)
      except:
        print "[RecursiveApply] Error after applying %s to function %s" % (self.transform, expr)
        raise 
    self.cache[key] = new_fn 
    return new_fn 
  
  def transform_Closure(self, expr):
    args = self.transform_expr_list(expr.args)
    new_fn = self.transform_expr(expr.fn)
    return self.closure(new_fn, args)
  
  def transform_Var(self, expr):
    expr.type = self.transform_type(expr.type)
    return expr
  
  def transform_Assign(self, stmt):
    """
    If we have an assignment like 
      a : (Fn1T, Fn2T) = (fn1, fn2)
    we might need to change it to 
      a : (Fn1T', Fn2T') = (fn1', fn2') if the RHS functions get updated
    """ 
    stmt.rhs = self.transform_expr(stmt.rhs)
    stmt.lhs.type = self.transform_type(stmt.lhs.type)
    return stmt 
  
  def pre_apply(self, fn):
    if self.transform is None:
      self.transform = fn.created_by 
    
    assert self.transform is not None, "No transform specified for RecursiveApply"
    fn.input_types = tuple(self.transform_type(t) for t in fn.input_types)
    fn.return_type = self.transform_type(fn.return_type)
    for k,t in fn.type_env.items():
      fn.type_env[k] = self.transform_type(t)
    return fn   