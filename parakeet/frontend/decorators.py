
from .. import names 
  
from .. syntax import (Expr, Var, Const, Return, UntypedFn, FormalArgs, DelayUntilTyped,  
                       const, is_python_constant)

from run_function import run_python_fn, run_untyped_fn 

class jit(object):
  def __init__(self, f):
    self.f = f
    self.fn = f

  def __call__(self, *args, **kwargs):
    if '_backend' in kwargs:
      backend_name = kwargs['_backend']
      del kwargs['_backend']
    else:
      backend_name = None
    return run_python_fn(self.f, args, kwargs, backend = backend_name)


class macro(object):
  def __init__(self, f, static_names = set([]), call_from_python = None):
    self.f = f
    self.fn = f 
    self.argcount = self.f.func_code.co_argcount 
    self.varnames = self.f.func_code.co_varnames 
    self.n_args = self.f.func_code.co_argcount
    self.defaults = self.f.func_defaults
    
    n_defaults = len(self.defaults) if self.defaults else 0 
    self.defaults_dictionary = {}
    n_pos = self.n_args - n_defaults
    for i, default_name in enumerate(self.varnames[n_pos:n_pos+n_defaults]):
      self.defaults_dictionary[default_name] = self.defaults[i]
    
    self.static_names = static_names
    self.wrappers = {}
    self.call_from_python = call_from_python
    if hasattr(self.f, "__name__"):
      self.name = f.__name__
    elif hasattr(self.f, "name"):
      self.name = self.f.name 
    else:
      self.name = "macro"

  
  _macro_wrapper_cache = {}
  def _create_wrapper(self, n_pos, static_pairs, dynamic_keywords):
    static_pairs = tuple(static_pairs)
    
    key = (n_pos, static_pairs, dynamic_keywords)
    if key in self.wrappers:
      return self.wrappers[key]
    
    args = FormalArgs()
    pos_vars = []
    keyword_vars = {}
    
    for i in xrange(n_pos):
      if i <  self.argcount: 
        raw_name = self.varnames[i]
      else:
        raw_name = "input_%d" % i
      local_name = names.fresh(raw_name)
      args.add_positional(local_name)
      pos_vars.append(Var(local_name))

      
    import ast_conversion
    for visible_name in dynamic_keywords:
      local_name = names.fresh(visible_name)
      args.add_positional(local_name, visible_name)
      keyword_vars[visible_name] = Var(local_name)
      if visible_name in self.defaults_dictionary:
        default_value = self.defaults_dictionary[visible_name]
        parakeet_value = ast_conversion.value_to_syntax(default_value)
        args.defaults[local_name] = parakeet_value

    for (static_name, value) in static_pairs:
      if isinstance(value, Expr):
        assert isinstance(value, Const)
        keyword_vars[static_name] = value
      elif value is not None:
        assert is_python_constant(value), \
            "Unexpected type for static/staged value: %s : %s" % \
            (value, type(value))
        keyword_vars[static_name] = const(value)


    result_expr = self.f(*pos_vars, **keyword_vars)
    body = [Return(result_expr)]
    wrapper_name = "%s_wrapper_%d_%d" % (self.name, n_pos,
                                         len(dynamic_keywords))
    wrapper_name = names.fresh(wrapper_name)
    untyped = UntypedFn(name = wrapper_name, args = args, body = body)
    self.wrappers[key] = untyped
    return untyped 
  
  def as_fn(self):
    n_default = len(self.defaults) if self.defaults else 0 
    n_pos = self.n_args - n_default 
    arg_varnames = self.varnames[:self.argcount]
    keyword_names = arg_varnames[n_pos:]
    wrapper = self._create_wrapper(n_pos, [], keyword_names)
    return wrapper 
    
  def __call__(self, *args, **kwargs):

    if self.call_from_python is not None:
      return self.call_from_python(*args, **kwargs)
    
    if '_backend' in kwargs:
      backend_name = kwargs['_backend']
      del kwargs['_backend']
    else:
      backend_name = None

    n_pos = len(args)
    keywords = kwargs.keys()
    static_pairs = tuple((k,kwargs.get(k)) for k in self.static_names)
   
    dynamic_keywords = tuple(k for k in keywords
                               if k not in self.static_names)

     
    untyped = self._create_wrapper(n_pos, static_pairs, dynamic_keywords)

    dynamic_kwargs = dict( (k, kwargs[k]) for k in dynamic_keywords)
    return run_untyped_fn(untyped, args, dynamic_kwargs, backend = backend_name)
    

  def transform(self, args, kwargs = {}):
    for arg in args:
      assert isinstance(arg, Expr), \
          "Macros can only take syntax nodes as arguments, got %s" % (arg,)
    for (name,arg) in kwargs.iteritems():
      assert isinstance(arg, Expr), \
          "Macros can only take syntax nodes as arguments, got %s = %s" % \
          (name, arg)
    result = self.f(*args, **kwargs)
    assert isinstance(result, Expr), \
        "Expected macro %s to return syntax expression, got %s" % \
        (self.f, result)
    return result

  def __str__(self):
    return "macro(%s)" % self.name

class typed_macro(macro):
  def __init__(self, f, *args, **kwargs):
    macro.__init__(self, f, *args, **kwargs)
    def delayed(*macro_args, **macro_kwargs):
      return DelayUntilTyped(values = macro_args, keywords = macro_kwargs, fn = f)
    self.f = delayed 
    

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

axis_macro = staged_macro("axis")