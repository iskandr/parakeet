from .. frontend import translate_function_value, jit, macro
 
from .. ndtypes import make_tuple_type, TupleT, ArrayT 
from .. syntax import Map, Tuple, DelayUntilTyped, Array 
 


@jit 
def _tuple_from_args(*args):
  return args

@macro
def zip(*args):

  elt_tupler = translate_function_value(_tuple_from_args)
  return Map(fn = elt_tupler, args = args)

@macro 
def _builtin_tuple(x):
  def typed_tuple(xt):
    if isinstance(xt.type, TupleT):
      return xt 
    else:
      assert isinstance(xt.type, ArrayT), "Can't create type from %s" % (xt.type,)
      assert isinstance(xt, Array), "Can only create tuple from array of const length"
      elt_types = [e.type for e in xt.elts]
      tuple_t = make_tuple_type(elt_types)
      return Tuple(xt.elts, type = tuple_t)
  return DelayUntilTyped(x, typed_tuple)