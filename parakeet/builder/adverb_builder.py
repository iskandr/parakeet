
from ..ndtypes import TupleT, ScalarT
from ..syntax import Expr 
from ..syntax.adverb_helpers import max_rank_arg, max_rank, unwrap_constant 
from ..syntax.helpers import unwrap_constant, wrap_if_constant 
 
from loop_builder import LoopBuilder

class AdverbBuilder(LoopBuilder):

  def list_none_axes(self, args, axis):
    axes = self.normalize_expr_axes(args, axis)
    return [self.is_none(axis) for axis in axes]
  
  def normalize_expr_axes(self, args, axis):
    """
    Given a list of arguments to an adverb and some value 
    or expression representing the axis/axes of iteration, 
    return a tuple of axis expressions, one per argument
    """
    axis = wrap_if_constant(axis)
    assert isinstance(axis, Expr), "Unexpected axis %s" % axis
    
    if isinstance(axis.type, TupleT):
      axes = self.tuple_elts(axis)
    else:
      assert isinstance(axis.type, ScalarT), "Unexpected axis %s : %s" % (axis, axis.type)
      axes = (axis,) * len(args)
    
    assert len(axes) == len(args), "Wrong number of axes (%d) for %d args" % (len(axes), len(args))
    
    if self.rank(max_rank_arg(args)) < 2:
        # if we don't actually have any multidimensional arguments, 
        # might as well make the axes just 0  
      axes = tuple(self.int(0) if self.is_none(axis) else axis for axis in axes)
    return axes 
    
  def normalize_axes(self, args, axis):
    """
    Given a list of arguments to an adverb and some value 
    or expression representing the axis/axes of iteration, 
    return a tuple of axis values, one per argument
    """

    if isinstance(axis, Expr) and isinstance(axis.type, TupleT):
      axis = self.tuple_elts(axis)
          
      
    # unpack the axis argument into a tuple,  
    # if only one axis was given, then repeat it as many times as we have args 
    if isinstance(axis, (list, tuple)):
      axes = tuple([unwrap_constant(elt) for elt in axis])
    elif isinstance(axis, Expr):
      axes = (unwrap_constant(axis),) * len(args) 
    else:
      assert axis is None or isinstance(axis, (int,long)), "Invalid axis %s" % axis
      axes = (axis,) * len(args)
    
    assert len(axes) == len(args), "Wrong number of axes (%d) for %d args" % (len(axes), len(args))
    
    if self.rank(max_rank_arg(args)) < 2:
        # if we don't actually have any multidimensional arguments, 
        # might as well make the axes just 0  
      axes = tuple(0 if axis is None else axis for axis in axes)
    return axes 
  
  def iter_bounds(self, args, axes, cartesian_product = False):
    """
    Given the arguments to an adverb (higher order array operator) and 
    the axes of the adverb's iteration, 
    given back a scalar or tuple representing the iteration space bounds
    the adverb will traverse
    """
    axes = self.normalize_axes(args, axes)
    assert len(args) == len(axes), "Mismatch between args %s and axes %s" % (args, axes) 
 
    
    if cartesian_product:
      bounds = []
      for arg, axis in zip(args, axes):
        if self.rank(arg) == 0:
          continue 
        if axis is None:
          bounds.extend(self.tuple_elts(self.shape(arg)))
        else:
          bounds.append(self.shape(arg, axis))
      return self.tuple(bounds)
    
    #...otherwise, we're doing an elementwise operation across
    #possibly multiple arguments 
    if any(axis is None for axis in axes):
      # if any of the axes are None then just find the highest rank argument 
      # which is going to be fully traversed and use its shape as the bounds
      # for the generated ParFor
      best_rank = -1 
      best_arg = None 
      for curr_arg, curr_axis in zip(args,axes):
        r = self.rank(curr_arg)
        if curr_axis is None and r > best_rank:
          best_rank = r
          best_arg = curr_arg 
      return self.shape(best_arg) 
       
    else:
      # if all axes are integer values, then keep the one with highest rank, 
      # it's bad that we're not doing any error checking here to make sure that 
      # all the non-scalar arguments have compatible shapes 
      best_rank = -1  
      best_arg = None
      best_axis = None 
      for curr_arg, curr_axis in zip(args,axes):
        r = self.rank(curr_arg)
        if r > best_rank:
          best_arg = curr_arg 
          best_axis = curr_axis
          best_rank = r 
      return self.shape(best_arg, best_axis)