from collections import namedtuple

from ..analysis import collect_var_names_from_exprs
from ..ndtypes import TupleT 
from ..syntax import (Return, Map, OuterMap, Tuple, Var, Closure, Assign, 
                      Const, TypedFn, UntypedFn, Ravel, Shape, Strides, Transpose, 
                      Where, Reshape) 
from ..syntax.helpers import none   

from subst import subst_expr_list
from transform import Transform

class CombineNestedMaps(Transform):

  def translate_expr(self, expr, mapping, forbidden = set([])):
      c = expr.__class__ 
      if c is Var:
        if expr.name in forbidden:
          return None 
        return mapping[expr.name]
      elif c in (Const, TypedFn, UntypedFn):
        return expr 
      
      elif c is Tuple:
        return Tuple(elts = tuple(self.translate_expr(elt,mapping,forbidden) 
                                  for elt in expr.elts), type = expr.type)
      elif c is Closure:
        return Closure(fn = expr.fn, 
                       args = tuple(self.translate_expr(elt, mapping,forbidden) 
                                    for elt in expr.args), type = expr.type)
      elif c in (Ravel, Shape, Strides, Transpose, Where):
        return c(array = self.translate_expr(expr.array, mapping,forbidden), type = expr.type)
      elif c is Reshape:
        return Reshape(array = self.translate_expr(expr.array, mapping,forbidden), 
                       shape = self.translate_expr(expr.shape, mapping,forbidden), 
                       type = expr.type)
  
  def build_arg_mapping(self, fn, closure_elts, outer_args):
    
    combined_args = tuple(closure_elts) + tuple(outer_args)
    assert len(fn.arg_names) == len(combined_args), \
      "Mismatch between function with %d formals and given %d actual args" % \
      (len(fn.arg_names), len(combined_args))
    mapping = {}
    for name, expr in zip(fn.arg_names, combined_args):
      mapping[name] = expr  
  
    # don't allow transformations of array elements
    # since that's an interaction between levels of the nested
    # maps 
    forbidden = set(fn.arg_names[len(closure_elts):])
    for stmt in fn.body[:-1]:
      if stmt.__class__ is Assign and stmt.lhs.__class__ is Var:
        new_rhs = self.translate_expr(stmt.rhs, mapping,forbidden)
        if new_rhs is None:
          return None 
        else:
          mapping[stmt.lhs.name] = new_rhs
      else:
        return None 
    return mapping 
  
  def combine_maps(self, closure, outer_args, outer_axis, result_type):
    fn = self.get_fn(closure)
    closure_elts = self.closure_elts(closure)
    
    if len(fn.body) == 0: return None 
    stmt = fn.body[-1]
    if stmt.__class__ is not Return: return None 
    nested_expr = stmt.value 
    
    if nested_expr.__class__ not in (Map, OuterMap): return None
    
    arg_mapping = self.build_arg_mapping(fn, closure_elts, outer_args)
    if arg_mapping is None: return None 
    
    nested_outer_args = nested_expr.args
    n_nested_outer_args = len(nested_outer_args)
    nested_map_closure_elts = self.closure_elts(nested_expr.fn) 
    nested_fn = self.get_fn(nested_expr.fn)
    nested_axis = nested_expr.axis
    
    # either both Maps specify an axis of traversal or neither do  
    if self.is_none(outer_axis) != self.is_none(nested_axis): return None 
    
    
    n_array_args = len(outer_args)
    
    inner_array_names = fn.arg_names[-n_array_args:]
    
    if len(nested_map_closure_elts) < len(inner_array_names): return None 
    
    # if any of the last k closure arguments aren't array elements
    # then abandon ship 
    k = n_array_args
    for closure_expr in nested_map_closure_elts[-k:]:
      if closure_expr.__class__ is not Var or closure_expr.name not in inner_array_names:
        return None 
      
    # if the two Maps are both elementwise, then make the OuterMap 
    # also elementwise

    remapped_inner_outer_args = [self.translate_expr(e,arg_mapping)
                               for e in nested_outer_args]
    remapped_inner_closure_elts = [self.translate_expr(e, arg_mapping)
                                   for e in nested_map_closure_elts]
    
    new_closure_elts = remapped_inner_closure_elts[:-k]
    new_outer_args = remapped_inner_closure_elts[-k:] + remapped_inner_outer_args    
    

    if self.is_none(outer_axis):
      combined_axis = none
    else:
      # combine the axes since we're going to make a single OuterMap 
      if isinstance(outer_axis.type, TupleT):
        outer_axes = self.tuple_elts(outer_axis)
      else:
        outer_axes = (outer_axis,) * n_array_args
      if isinstance(nested_axis.type, TupleT):
        assert isinstance(nested_axis, Tuple),  \
          "Axis argument must be None, int, or literal tuple, not %s" % nested_axis 
        inner_axes = tuple(nested_axis.elts) 
      else:
        inner_axes = (nested_axis,) * n_nested_outer_args
       
      combined_axes = list(outer_axes + inner_axes)
      for i, array_arg in enumerate(new_outer_args):
        if array_arg.__class__ is Transpose:
          old_axis = combined_axes[i]
          if not self.is_none(old_axis):
            combined_axes[i] = Const(value = array_arg.type.rank - old_axis.value - 1, 
                                     type = old_axis.type)
            new_outer_args[i] = array_arg.array
      combined_axis = self.tuple(combined_axes)
    return  OuterMap(fn = self.closure(nested_fn, new_closure_elts),
                     args = tuple(new_outer_args),   
                     axis = combined_axis, 
                     type = result_type)
    
  def transform_Map(self, expr):
    # can't turn Map(-, x, y) into an OuterMap since (x,y) are at the same iteration level 
    if len(expr.args) != 1:
      return expr
    new_expr = self.combine_maps(expr.fn, expr.args, expr.axis, expr.type)
    if new_expr is None: return expr 
    else: return new_expr 
    
  def transform_OuterMap(self, expr):
    new_expr = self.combine_maps(expr.fn, expr.args, expr.axis, expr.type)
    if new_expr is None: return expr 
    else: return new_expr 
    
    