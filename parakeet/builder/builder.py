
from ..syntax.helpers import zero_i64, get_types 
from ..syntax.adverb_helpers import max_rank, max_rank_arg

from arith_builder import ArithBuilder 
from array_builder import ArrayBuilder
from loop_builder import LoopBuilder 
from call_builder import CallBuilder


class Builder(ArithBuilder, ArrayBuilder, CallBuilder, LoopBuilder):
  

  def create_output_array(self, fn, array_args, axis, 
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
    assert isinstance(axis, (int, long)), \
      "Expected axis to be an integer, got %s" % (axis,)
    
    # take the 0'th slice just to have a value in hand 
    inner_args = [self.slice_along_axis(array, axis, zero_i64)
                  for array in array_args]
    try:
      inner_shape_tuple = self.call_shape(fn, inner_args)
    except:
      print "Shape inference failed when calling %s with %s" % (fn, array_args)
      import sys
      print "Error %s ==> %s" % (sys.exc_info()[:2])
      raise

    
    extra_dims = []
    if cartesian_product:
      for array in array_args:
        if self.rank(array) > axis:
          dim = self.shape(array, axis)
        else:
          dim = 1 
        extra_dims.append(dim)
    else:
      biggest_arg = max_rank_arg(array_args)
      assert self.rank(biggest_arg) > axis, \
        "Can't slice along axis %d of %s (rank = %d)" % (axis, biggest_arg, self.rank(biggest_arg)) 
      extra_dims.append(self.shape(biggest_arg, axis))
      
    outer_shape_tuple = self.tuple(extra_dims)
    shape = self.concat_tuples(outer_shape_tuple, inner_shape_tuple)
    elt_t = self.elt_type(self.return_type(fn))
    if len(shape.type.elt_types) > 0:
      return self.alloc_array(elt_t, shape, name)
    else:
      return self.fresh_var(elt_t, name) 