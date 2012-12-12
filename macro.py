import syntax 

class macro:
  def __init__(self, f):
    self.f = f
  def __call__(self, *args, **kwargs):
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