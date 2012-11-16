import names 
import syntax 
from args import Args 
import syntax_helpers 
import adverbs 
import function_registry
import ast_conversion


_adverb_wrapper_cache = {}
def untyped_wrapper(adverb_class, 
                      map_fn_name = None, 
                      combine_fn_name = None,
                      emit_fn_name = None, 
                      data_names = [], 
                      varargs_name = 'xs', 
                      axis = 0):
  """
  Given:
    - an adverb class (i.e. Map, Reduce, Scan, or AllPairs)
    - function var names (some of which can be None)
    - optional list of positional data arg names 
    - optional name for the varargs parameter
    - an axis along which the adverb operates
  Return a function which calls the desired adverb with 
  the data args and unpacked varargs tuple. 
  """
  axis = syntax_helpers.wrap_if_constant(axis)
  key = adverb_class, map_fn_name, combine_fn_name, emit_fn_name, axis
  if key in _adverb_wrapper_cache:
    return _adverb_wrapper_cache[key]
  else:
    
    positional_arg_names = []
    def mk_input(name):
      if name is None:
        return None
      else:
        assert isinstance(name, str), \
          "Expected string for fn name, got:" + str(name)
        name = names.refresh(name)
        positional_arg_names.append(name) 
        return syntax.Var(name)
    map_fn = mk_input(map_fn_name)
    combine_fn = mk_input(combine_fn_name)
    emit_fn = mk_input(emit_fn_name)
    
    value_args = map(mk_input, data_names)
    if varargs_name:
      varargs_name = names.refresh(varargs_name)
      unpack = syntax.Unpack(syntax.Var(varargs_name))
      value_args.append(unpack)
    args_obj = Args(positional = positional_arg_names, varargs = varargs_name)
    adverb = adverb_class(map_fn = map_fn, combine_fn = combine_fn, emit_fn = emit_fn,  
                          args = value_args, axis = axis)
    body = [syntax.Return(adverb)]
    fn_name = names.fresh(adverb_class.node_type() + "_wrapper")
    fundef = syntax.Fn(fn_name, args_obj, body)
    function_registry.untyped_functions[fn_name] = fundef
    _adverb_wrapper_cache[key] = fundef
    return fundef

_adverb_registry = {}
def is_registered_adverb(fn):
  return fn in _adverb_registry

def register_adverb(python_fn, wrapper):
  _adverb_registry[python_fn] = wrapper

def get_adverb_wrapper(python_fn):
  return _adverb_registry[python_fn]


def gen_arg_names(n, base_names):
  results = []
  
  m = len(base_names)
  for i in xrange(n):
    curr = base_names[i % m]
    cycle = i / m
    if cycle > 0:
      curr = "%s_%d" % (curr, cycle+1)
    results.append(curr)
  return results 

def gen_data_arg_names(n):
  return gen_arg_names(n, ['x', 'y', 'z', 'a', 'b', 'c', 'd'])

def gen_fn_arg_names(n):
  return gen_arg_names(n, ['f', 'g', 'h', 'p', 'q', 'r', 's'])


def get_fundef(fn):
  """
  Get the function definition in case 
  I want to pass in the name
  of an untyped function or an untranslated
  python fn. 
  """
  if isinstance(fn, str):
    assert fn  in function_registry.untyped_functions, \
      "Function not found: %s" % fn
    return function_registry.untyped_functions[fn]    
  elif not isinstance(fn, syntax.Fn):
    return ast_conversion.translate_function_value(fn) 
  else:
    return fn

def equiv_arg_names(fn):
  """
  Return generated arg names which match the arity and varargs
  of the given fundef 
  """
  fundef = get_fundef(fn)
  
  assert len(fundef.args.defaults) == 0
  data_names = gen_data_arg_names(len(fundef.args.positional)) 
  varargs_name = 'xs' if fundef.args.varargs else None
  return data_names, varargs_name   


  
def untyped_map_wrapper(fn, axis = 0):
  data_names, varargs_name = equiv_arg_names(fn)
  return untyped_wrapper(adverbs.Map, 
                         map_fn_name = 'f', 
                         data_names = data_names, 
                         varargs_name  = varargs_name, 
                         axis = axis)

def untyped_allpairs_wrapper(fn, axis = 0):
  data_names, varargs_name = equiv_arg_names(fn)
  assert len(data_names) == 2
  assert varargs_name is None 
  return untyped_wrapper(adverbs.AllPairs, 
                         map_fn_name = 'f', 
                         data_names = data_names, 
                         axis = axis)

def untyped_reduce_wrapper(map_fn, combine_fn, axis = 0):
  if map_fn is None: 
    map_fn_name = None
    data_names = ['x']
    varargs_name = None  
  else:
    map_fn_name = 'f'
    data_names, varargs_name = equiv_arg_names(map_fn)
    
  assert combine_fn is not None, "Combiner required" 
  combine_args, combine_varargs = equiv_arg_names(combine_fn)
  combine_arity = len(combine_args)
  assert combine_arity == 2, \
    "Expected binary operator, instead got function of %d args" % combine_arity
  assert combine_varargs is None, "Binary operator can't have varargs"
  
  return untyped_wrapper(adverbs.Reduce,
                         map_fn_name = map_fn_name,  
                         combine_fn_name = 'combine', 
                         data_names = data_names, 
                         varargs_name = varargs_name, 
                         axis = axis)


def untyped_scan_wrapper(map_fn, combine_fn, emit_fn, axis = 0):
  if map_fn is None: 
    map_fn_name = None
    data_names = ['x']
    varargs_name = None  
  else:
    map_fn_name = 'f'
    data_names, varargs_name = equiv_arg_names(map_fn)
  
  assert combine_fn is not None, "Combiner required" 
  combine_args, combine_varargs = equiv_arg_names(combine_fn)
  combine_arity = len(combine_args)
  assert combine_arity == 2, \
    "Expected binary operator, instead got function of %d args" % combine_arity
  assert combine_varargs is None, "Binary operator can't have varargs"
  
  if emit_fn is None: 
    emit_data_names = ['x']
    emit_varargs_name = None 
    emit_fn_name = None 
  else:
    emit_fn_name = ['emit'] 
    emit_data_names, emit_varargs_name = equiv_arg_names(emit_fn)
    
  assert len(emit_data_names) == 1, \
    "Expected emit to be a unary function, got %d args" % len(emit_data_names)
  assert emit_varargs_name is None, \
    "Didn't expect emit to have variable number of arguments" 
  return untyped_wrapper(adverbs.Reduce,
                         map_fn_name = map_fn_name, 
                         combine_fn_name = 'combine',
                         emit_fn_name = emit_fn_name, 
                         data_names = data_names, 
                         varargs_name = varargs_name, 
                         axis = axis)
