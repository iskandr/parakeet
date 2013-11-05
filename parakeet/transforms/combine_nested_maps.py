from ..analysis import collect_var_names_from_exprs
from ..ndtypes import TupleT 
from ..syntax import Return, Map, OuterMap, Tuple
from ..syntax.helpers import none   

from subst import subst_expr_list
from transform import Transform

class CombineNestedMaps(Transform):
  def transform_Map(self, expr):
    fn = self.get_fn(expr.fn)
    if len(fn.body) > 1:
      return expr 
    stmt = fn.body[0]
    if stmt.__class__ is not Return:
      return expr 
    nested_expr = stmt.value 
    if nested_expr.__class__ is not Map:
      return expr
    
     
    outer_axis = expr.axis
    inner_axis = expr.axis
    # either both Maps specify an axis of traversal or neither do  
    if self.is_none(outer_axis) != self.is_none(inner_axis):
      return expr 
    
    outer_args = expr.args
    n_outer_args = len(outer_args)
    
    inner_args = nested_expr.args
    n_inner_args = len(inner_args)   
    # if the two Maps are both elementwise, then make the OuterMap 
    # also elementwise  
    if self.is_none(outer_axis):
      combined_axis = none
    else:
      # combine the axes since we're going to make a single OuterMap 
      if isinstance(outer_axis.type, TupleT):
        outer_axes = self.tuple_elts(outer_axis)
      else:
        outer_axes = (outer_axis,) * n_outer_args
      if isinstance(inner_axis.type, TupleT):
        assert isinstance(inner_axis, Tuple),  \
          "Axis argument must be None, int, or literal tuple, not %s" % inner_axis 
        inner_axes = tuple(inner_axis.elts) 
      else:
        inner_axes = (inner_axis,) * n_inner_args 
      combined_axis = self.tuple(outer_axes + inner_axes)  
        
    # can only flatten two maps if their variables don't interact at all, 
    # so the only reference vars of the inner map have to be closure 
    # arguments the outer map 
    outer_closure_args = self.get_closure_args(expr.fn)
    n_outer_closure_args = len(outer_closure_args)
    closure_var_names = fn.arg_names[:n_outer_closure_args]
    inner_fn = self.get_fn(nested_expr.fn)
    
    inner_arg_names = collect_var_names_from_exprs(inner_args)
    if any(name not in closure_var_names for name in inner_arg_names):
      return expr 
    
    
    
    
    
    
    OuterMap(inner_fn)
    
     
    
    
    inner_closure_args = self.get_closure_args(inner_fn)
    n_inner_closure_args = len(inner_closure_args)
    
    
    
  