from collections import namedtuple

from ..analysis import collect_var_names_from_exprs
from ..ndtypes import TupleT
from ..syntax import (Return, Map, OuterMap, IndexMap, Tuple, TupleProj, Var, Closure, Assign,
                      Const, TypedFn, UntypedFn, Ravel, Shape, Strides, Transpose,
                      Reshape, Where, Compress)
from ..syntax.helpers import none

from subst import subst_expr_list
from clone_function import CloneFunction
from transform import Transform

class CombineFailed(Exception):
  pass

class CombineNestedMaps(Transform):

  #def translate_expr(self, expr, mapping, forbidden = set([])):
  #  try:
  #    return self._translate_expr(expr, mapping, forbidden)
  #  except CombineFailed:
  #    return None

  def translate_exprs(self, exprs, mapping, forbidden):
      results = tuple(self.translate_expr(elt, mapping, forbidden)
                      for elt in exprs)
      if any(elt is None for elt in results):
        raise CombineFailed()
      else:
        return results

  def translate_expr(self, expr, mapping, forbidden = set([])):
    c = expr.__class__
    if c is Var:
      if expr.name not in forbidden and expr.name in mapping:
        return mapping[expr.name]
      else:
        raise CombineFailed()
    elif c in (Const, TypedFn, UntypedFn):
      return expr
    elif c is Tuple:
      elts = self.translate_exprs(expr.elts, mapping, forbidden)
      return Tuple(elts = elts, type = expr.type)
    elif c is Closure:
      args = self.translate_exprs(expr.args, mapping, forbidden)
      return Closure(fn = expr.fn,
                     args = args,
                     type = expr.type)
    elif c in (Ravel, Shape, Strides, Transpose, Where):
      array = self.translate_expr(expr.array, mapping, forbidden)
      if array is None:
        raise CombineFailed()
      return c(array = array, type = expr.type)
    elif c is Compress:
      condition = self.translate_expr(expr.condition, mapping, forbidden)
      data = self.translate_expr(expr.data, mapping, forbidden)
      if condition is None or data is None:
        raise CombineFailed()
      return c(condition = condition,
               data = data,
               type = expr.type)
    elif c is TupleProj:
      tup = self.translate_expr(expr.tuple, mapping, forbidden)
      if tup is None:
        raise CombineFailed()
      return c(tuple = tup,
               index = expr.index,
               type = expr.type)
    elif c is Reshape:
      array = self.translate_expr(expr.array, mapping,forbidden)
      shape = self.translate_expr(expr.shape, mapping,forbidden)
      if array is None or shape is None:
        raise CombineFailed()
      return Reshape(array = array, shape = shape,  type = expr.type)

  def build_arg_mapping(self, fn, closure_elts, outer_args = None):
    if outer_args:
      combined_args = tuple(closure_elts) + tuple(outer_args)
    else:
      combined_args = closure_elts
    mapping = {}
    for name, expr in zip(fn.arg_names, combined_args):
      mapping[name] = expr

    # don't allow transformations of array elements
    # since that's an interaction between levels of the nested
    # maps
    forbidden = set(fn.arg_names[len(closure_elts):])
    for stmt in fn.body[:-1]:
      if stmt.__class__ is Assign and stmt.lhs.__class__ is Var:
        try:
          new_rhs = self.translate_expr(stmt.rhs, mapping,forbidden)
        except CombineFailed:
          return None
        else:
          mapping[stmt.lhs.name] = new_rhs
      else:
        return None
    return mapping

  def dissect_nested_fn(self, fn, valid_adverb_classes = (Map,OuterMap)):
    if len(fn.body) == 0:
      return None

    stmt = fn.body[-1]

    if stmt.__class__ is not Return:
      return None

    nested_expr = stmt.value
    if nested_expr.__class__ not in valid_adverb_classes:
      return None

    return nested_expr



  def combine_maps(self, closure, outer_args, outer_axis, result_type):
    fn = self.get_fn(closure)
    closure_elts = self.closure_elts(closure)

    nested_expr = self.dissect_nested_fn(fn)
    if nested_expr is None: return None

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
    if len(nested_map_closure_elts) < len(inner_array_names):  return None



    # if there's ohe outer arg and it's stuck at the back of the
    # nested args list, try permuting the arguments to make a nested fnh
    # that's compatible with OuterMap
    if n_array_args == 1 and len(nested_map_closure_elts) > 1 and \
        nested_map_closure_elts[0].__class__ is Var and \
        nested_map_closure_elts[0].name == fn.arg_names[-1]:
      permute_fn = CloneFunction().apply(nested_fn)
      permute_input_types = list(permute_fn.input_types)
      permute_arg_names = list(permute_fn.arg_names)
      permute_closure_elts = list(nested_map_closure_elts)
      first_name = permute_arg_names[0]
      first_type = permute_input_types[0]
      first_closure_elt = permute_closure_elts[0]
      permute_input_types[:-1] = permute_input_types[1:]
      permute_arg_names[:-1] = permute_arg_names[1:]
      permute_closure_elts[:-1] = permute_closure_elts[1:]
      permute_input_types[-1] = first_type
      permute_arg_names[-1] = first_name
      permute_closure_elts[-1] = first_closure_elt
      permute_fn.input_types = tuple(permute_input_types)
      permute_fn.arg_names = tuple(permute_arg_names)
      nested_fn = permute_fn
      nested_map_closure_elts = tuple(permute_closure_elts)

    # if any of the last k closure arguments aren't array elements
    # then abandon ship
    if any(closure_expr.__class__ is not Var or closure_expr.name not in inner_array_names
           for closure_expr in nested_map_closure_elts[-n_array_args:]):
        return None

    try:
      remapped_inner_outer_args = [self.translate_expr(e,arg_mapping)
                                   for e in nested_outer_args]
      remapped_inner_closure_elts = [self.translate_expr(e, arg_mapping)
                                     for e in nested_map_closure_elts]
    except CombineFailed:
      return None
    # if the two Maps are both elementwise, then make the OuterMap
    # also elementwise



    new_closure_elts = remapped_inner_closure_elts[:-n_array_args]
    new_outer_args = remapped_inner_closure_elts[-n_array_args:] + remapped_inner_outer_args

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


  def combine_index_maps(self, closure, shape, result_type):
    fn = self.get_fn(closure)
    closure_elts = self.closure_elts(closure)

    #n_outer_args = len(fn.input_types)
    n_outer_closure_args = len(closure_elts)
    #n_outer_indices = n_outer_args - n_outer_closure_args
    outer_arg_names = fn.arg_names
    outer_index_names = outer_arg_names[n_outer_closure_args:]
    #outer_index_types = fn.input_types[n_outer_closure_args:]

    nested_expr = self.dissect_nested_fn(fn, (IndexMap,))
    if nested_expr is None: return None

    inner_shape = nested_expr.shape

    arg_mapping = self.build_arg_mapping(fn, closure_elts)
    if arg_mapping is None:  return None


    nested_map_closure_elts = self.closure_elts(nested_expr.fn)


    # inner closure args must be remapped to the index args of the outer fn

    remapped_closure_args = []
    fixed_closure_args_done = False
    for i, nested_closure_elt in enumerate(nested_map_closure_elts):
      if nested_closure_elt.__class__ is not Var:
        return None
      nested_closure_name = nested_closure_elt.name

      if nested_closure_name in outer_index_names:
        fixed_closure_args_done = True
        pos = outer_index_names.index(nested_closure_name)
        # for now, can't change order of arguments
        if i != pos:
          return None
      else:
        # can't do static/data closure args, followed by indices,
        # followed by more data args
        # TODO: just a new function and permute all the arguments
        if fixed_closure_args_done:
          return None
        elif nested_closure_name not in arg_mapping:
          return None
        remapped_closure_args.append(arg_mapping[nested_closure_name])


    try:
      remapped_inner_shape = self.translate_expr(inner_shape, arg_mapping)
    except CombineFailed:
      return None


    nested_fn = self.get_fn(nested_expr.fn)

    # if the two Maps are both elementwise, then make the OuterMap
    # also elementwise

    combined_shape = self.concat_tuples(shape, remapped_inner_shape, "combined_shape")
    return  IndexMap(fn = self.closure(nested_fn, tuple(remapped_closure_args)),
                     shape = combined_shape,
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

  def transform_IndexMap(self, expr):
    return self.combine_index_maps(expr.fn, expr.shape, expr.type)
