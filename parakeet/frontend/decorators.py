
from .. import names 
  
from .. syntax import (Expr, Var, Const, Return, UntypedFn, FormalArgs, DelayUntilTyped,  
                       const, is_python_constant)

from run_function import run_python_fn, run_untyped_fn 

class jit(object):
  def __init__(self, f):
    self.f = f


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
    self.varnames = self.f.func_code.co_varnames 
    self.n_args = self.f.func_code.co_argcount
    self.defaults = self.f.func_defaults 
    self.static_names = static_names
    self.wrappers = {}
    self.call_from_python = call_from_python
    if hasattr(self.f, "__name__"):
      self.name = f.__name__
    else:
      self.name = "f"

  
  _macro_wrapper_cache = {}
  def _create_wrapper(self, n_pos = None, default_values = None):
      
    if n_pos is None: 
      n_pos = self.n_args 
      
    if default_values is None:
      varnames = self.varnames[:self.n_args]
      n_default = 0 if not self.defaults else len(self.defaults)
      assert len(varnames) >= n_default
      default_keys = varnames[:-n_default]
      if default_keys and self.defaults:
        default_values = dict(zip(default_keys, self.defaults))
      else:
        default_values = {}
    
    defaults_seq = tuple(default_values.items())
    key = (n_pos, defaults_seq)
    if key in self.wrappers:
      return self.wrappers[key]
    
    args = FormalArgs()
    pos_vars = []
    keyword_vars = {}
    
    
    
    from ast_conversion import value_to_syntax

    for i in xrange(n_pos):
      if i <  self.f.func_code.co_varnames: 
        raw_name = self.f.func_code.co_varnames[i] 
      else:
        raw_name = "input_%d" % i
      local_name = names.fresh(raw_name)
      if raw_name in default_values:
        default_value = default_values[raw_name]
        if not isinstance(default_value, Expr):
          v = value_to_syntax(default_value)
        else:
          v = default_value
        args.add_positional(local_name, raw_name)
        args.defaults[raw_name] = v
        keyword_vars[raw_name] = Var(local_name)  
      else:
        args.add_positional(local_name)
        pos_vars.append(Var(local_name))

    result_expr = self.f(*pos_vars, **keyword_vars)
    body = [Return(result_expr)]
    wrapper_name = "%s_wrapper" % self.name
    wrapper_name = names.fresh(wrapper_name)
    untyped = UntypedFn(name = wrapper_name, args = args, body = body)
    self.wrappers[key] = untyped
    return untyped 
  
  def as_fn(self):
    return self._create_wrapper()
        
  def __call__(self, *args, **kwargs):
    if self.call_from_python is not None:
      return self.call_from_python(*args, **kwargs)
    
    if '_backend' in kwargs:
      backend_name = kwargs['_backend']
      del kwargs['_backend']
    else:
      backend_name = None
    untyped = self._create_wrapper()
    return run_untyped_fn(untyped, args, kwargs, backend = backend_name)
    

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
