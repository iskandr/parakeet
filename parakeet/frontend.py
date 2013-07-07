from args import FormalArgs, ActualArgs

import config 
import names
import syntax
import syntax_helpers
import type_conv 
import type_inference 

def prepare_args(fn, args, kwargs):
  """
  Fetch the function's nonlocals and return an ActualArgs object of both the arg
  values and their types
  """
  if isinstance(fn, syntax.Fn):
    untyped = fn
  else:
    import ast_conversion
    # translate from the Python AST to Parakeet's untyped format
    untyped = ast_conversion.translate_function_value(fn)

  nonlocals = list(untyped.python_nonlocals())
  arg_values = ActualArgs(nonlocals + list(args), kwargs)

  # get types of all inputs
  arg_types = arg_values.transform(type_conv.typeof)
  return untyped, arg_values, arg_types

def specialize(fn, args, kwargs = {}):
  """
  Translate, specialize and begin to optimize the given function for the types
  of the supplies arguments.

  Return the untyped and typed representations, along with all the
  arguments in a linear order. 
  """

  untyped, arg_values, arg_types = prepare_args(fn, args, kwargs)
  
  # convert the awkward mix of positional, named, and starargs 
  # into a positional sequence of arguments
  linear_args = untyped.args.linearize_without_defaults(arg_values)
  
  # propagate types through function representation and all
  # other functions it calls
  typed_fn = type_inference.specialize(untyped, arg_types)
  
  from pipeline import high_level_optimizations
  # apply high level optimizations 
  optimized_fn = high_level_optimizations.apply(typed_fn)
  
  return untyped, optimized_fn, linear_args 
 
  
def run(fn, *args, **kwargs):
  """
  Given a python function, run it in Parakeet on the supplied args
  """
  
  if '_mode' in kwargs:
    backend_name = kwargs['_mode']
    del kwargs['_mode']
  else:
    backend_name = config.default_backend
  
  _, typed_fn, linear_args = specialize(fn, args, kwargs)
 
  import backend
  assert hasattr(backend, backend_name), "Unrecognized backend %s" % backend_name
  b = getattr(backend, backend_name) 
  return b.run(typed_fn, linear_args)
  

class jit:
  def __init__(self, f):
    self.f = f

  def __call__(self, *args, **kwargs):
    return run(self.f, *args, **kwargs)


class macro(object):
  def __init__(self, f, static_names = set([]), call_from_python = None):
    self.f = f
    self.static_names = static_names
    self.wrappers = {}
    self.call_from_python = call_from_python
    if hasattr(self.f, "__name__"):
      self.name = f.__name__
    else:
      self.name = "f"

  _macro_wrapper_cache = {}
  def _create_wrapper(self, n_pos, static_pairs, dynamic_keywords):
    args = FormalArgs()
    pos_vars = []
    keyword_vars = {}
    for i in xrange(n_pos):
      local_name = names.fresh("input_%d" % i)
      args.add_positional(local_name)
      pos_vars.append(syntax.Var(local_name))
  
    
    for visible_name in dynamic_keywords:
      local_name = names.fresh(visible_name)
      args.add_positional(local_name, visible_name)
      keyword_vars[visible_name] = syntax.Var(local_name)

    for (static_name, value) in static_pairs:
      if isinstance(value, syntax.Expr):
        assert isinstance(value, syntax.Const)
        keyword_vars[static_name] = value
      elif value is not None:
        assert syntax_helpers.is_python_constant(value), \
            "Unexpected type for static/staged value: %s : %s" % \
            (value, type(value))
        keyword_vars[static_name] = syntax_helpers.const(value)

    result_expr = self.f(*pos_vars, **keyword_vars)
    body = [syntax.Return(result_expr)]
    wrapper_name = "%s_wrapper_%d_%d" % (self.name, n_pos,
                                         len(dynamic_keywords))
    wrapper_name = names.fresh(wrapper_name)
    return syntax.Fn(name = wrapper_name, args = args, body = body)

  def as_fn(self):
    n_args = self.f.func_code.co_argcount
    n_default = 0 if not self.f.func_defaults else len(self.f.func_defaults)
    assert n_default == 0
    return self._create_wrapper(n_args,[],{})
    
  def __call__(self, *args, **kwargs):
    if self.call_from_python is not None:
      return self.call_from_python(*args, **kwargs)
    n_pos = len(args)
    keywords = kwargs.keys()

    static_pairs = ((k,kwargs.get(k)) for k in self.static_names)
    dynamic_keywords = tuple(k for k in keywords
                               if k not in self.static_names)

    static_pairs = tuple(static_pairs)
    key = (n_pos, static_pairs, dynamic_keywords)

    if key in self.wrappers:
      untyped = self.wrappers[key]
    else:
      untyped = self._create_wrapper(n_pos, static_pairs, dynamic_keywords)
      self.wrappers[key] = untyped
    dynamic_kwargs = dict( (k, kwargs[k]) for k in dynamic_keywords)
    return run(untyped, *args, **dynamic_kwargs)
    

  def transform(self, args, kwargs = {}):
    for arg in args:
      assert isinstance(arg, syntax.Expr), \
          "Macros can only take syntax nodes as arguments, got %s" % (arg,)
    for (name,arg) in kwargs.iteritems():
      assert isinstance(arg, syntax.Expr), \
          "Macros can only take syntax nodes as arguments, got %s = %s" % \
          (name, arg)
    result = self.f(*args, **kwargs)
    assert isinstance(result, syntax.Expr), \
        "Expected macro %s to return syntax expression, got %s" % \
        (self.f, result)
    return result

  def __str__(self):
    return "macro(%s)" % self.name

class staged_macro(object):
  def __init__(self, *static_names, **kwargs):
    self.static_names = tuple(static_names)

    self.call_from_python = kwargs.get('call_from_python')
    assert kwargs.keys() in [[], ['call_from_python']], \
        "Unknown keywords: %s" % kwargs.keys()

  def __call__(self, fn):
    return macro(fn, 
                 self.static_names,
                 call_from_python = self.call_from_python)
