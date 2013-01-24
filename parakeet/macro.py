import names
import syntax

from args import FormalArgs

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
        assert type(value) in (int, long, float, bool), \
            "Unexpected type for static/staged value: %s : %s" % \
            (value, type(value))
        keyword_vars[static_name] = syntax.Const(value)

    result_expr = self.f(*pos_vars, **keyword_vars)
    body = [syntax.Return(result_expr)]

    wrapper_name = "%s_wrapper_%d_%d" % (self.name, n_pos,
                                         len(dynamic_keywords))
    wrapper_name = names.fresh(wrapper_name)

    return syntax.Fn(name = wrapper_name, args = args, body = body)

  def __call__(self, *args, **kwargs):
    if self.call_from_python is None:
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
      import run_function
      return run_function.run(untyped, *args, **kwargs)
    else:
      return self.call_from_python(*args, **kwargs)

  def transform(self, args, kwargs):
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
    return macro(fn, self.static_names,
                 call_from_python = self.call_from_python)
