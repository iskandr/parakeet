from ..ndtypes import make_array_type, TupleT, IntT, FnT, ClosureT, increase_rank
from ..syntax import ArrayView, Struct, Expr, ParFor, IndexMap, UntypedFn, TypedFn 
from ..syntax.helpers import zero_i64, get_types, one 

from call_builder import CallBuilder
from adverb_builder import AdverbBuilder

class Builder(AdverbBuilder, CallBuilder):
  

  
  def create_output_array(self, fn, inner_args, 
                          outer_shape = (), 
                          name = "output"):
    if isinstance(outer_shape, (list, tuple)):
      outer_shape = self.tuple(outer_shape)
      
    try:
      inner_shape_tuple = self.call_shape(fn, inner_args)
    except:
      print "Shape inference failed when calling %s with %s" % (fn, inner_args)
      import sys
      print "Error %s ==> %s" % (sys.exc_info()[:2])
      raise

    shape = self.concat_tuples(outer_shape, inner_shape_tuple)
    closure_args = self.closure_elts(fn)
    fn = self.get_fn(fn)
    if isinstance(fn, UntypedFn):
      from .. type_inference import infer_return_type 
      arg_types = get_types(tuple(closure_args) + tuple(inner_args))
      return_type = infer_return_type(fn , arg_types)
      
    else:
      assert isinstance(fn, TypedFn), "Unexpected function %s" % fn
      return_type = self.return_type(fn)
    elt_t = self.elt_type(return_type)
    if len(shape.type.elt_types) > 0:
      return self.alloc_array(elt_t, shape, name)
    else:
      return self.fresh_var(elt_t, name) 
  
  def any_eq(self, tup, target_elt):
    elts = self.tuple_elts(tup)
    is_eq = self.eq(elts[0], target_elt)
    for elt in elts[1:]:
      is_eq = self.or_(is_eq, self.eq(elt, target_elt))
    return is_eq 
  
    
  def parfor(self, fn, bounds):
    assert isinstance(bounds, Expr)
    assert isinstance(bounds.type, (TupleT, IntT))
    assert isinstance(fn, Expr)
    assert isinstance(fn.type, (FnT, ClosureT))
    self.blocks += [ParFor(fn = fn, bounds = bounds)]
  
  def imap(self, fn, bounds):

    assert isinstance(bounds, Expr), "Expected imap bounds to be expression, got %s" % bounds
    if isinstance(bounds.type, TupleT):
      tup = bounds 
      ndims = len(bounds.type.elt_types) 
    else:
      assert isinstance(bounds.type, IntT), \
        "Expected imap bounds to be tuple or int, got %s : %s" % (bounds, bounds.type)
      tup = self.tuple([bounds])
      ndims = 1 
    assert isinstance(fn, Expr), "Expected imap function to be expression, got %s" % (fn,)
    assert isinstance(fn.type, (FnT, ClosureT)), \
      "Expected imap function to have a function type but got %s : %s" % (fn, fn.type)
    elt_type = self.return_type(fn) 
    result_type = increase_rank(elt_type, ndims)
    return IndexMap(fn = fn, shape = tup, type = result_type)
    
    
  def ravel(self, x, explicit_struct = False):
    # TODO: Check the strides to see if any element is equal to 1
    # otherwise do an array_copy
    assert self.is_array(x)
    if x.type.rank == 1:
      return x
    
    nelts = self.nelts(x, explicit_struct = explicit_struct)
    assert isinstance(nelts, Expr)
    shape = self.tuple((nelts,), 'shape', explicit_struct = explicit_struct)
    strides = self.tuple((self.int(1),), "strides", explicit_struct = explicit_struct)
    data = self.attr(x, 'data', 'data_ptr')
    offset = self.attr(x, 'offset')
    t = make_array_type(x.type.elt_type, 1)
    if explicit_struct: 
      return Struct(args = (data, shape, strides, offset, nelts), type = t)
    else:
      return ArrayView(data, shape, strides, offset, nelts, type = t)
  