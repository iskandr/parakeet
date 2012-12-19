import names
import syntax

from args import ActualArgs
from codegen import Codegen

class Transform(Codegen):
  def __init__(self, fn, verify = True, reverse = False):
    Codegen.__init__(self)
    self.fn = fn
    self.verify = verify
    self.copy = None
    self.reverse = reverse 

  def lookup_type(self, name):
    assert self.type_env is not None
    return self.type_env[name]

  def transform_TypedFn(self, fn):
    return fn

  def transform_if_expr(self, maybe_expr):
    if isinstance(maybe_expr, syntax.Expr):
      return self.transform_expr(maybe_expr)
    elif isinstance(maybe_expr, tuple):
      return tuple([self.transform_if_expr(x) for x in maybe_expr])
    elif isinstance(maybe_expr, list):
      return [self.transform_if_expr(x) for x in maybe_expr]
    elif isinstance(maybe_expr, ActualArgs):
      return maybe_expr.transform(self.transform_expr)
    else:
      return maybe_expr

  def transform_generic_expr(self, expr):
    args = {}
    changed = False 
    for member_name in expr.members():
      old_value = getattr(expr, member_name)
      new_value = self.transform_if_expr(old_value)
      args[member_name] = new_value 
      changed = changed or (old_value != new_value)
    if changed:
      return expr.__class__(**args)
    else:
      return expr 

  def find_method(self, expr, prefix = "transform_"):
    method_name = prefix + expr.node_type()
    if hasattr(self, method_name):
      return getattr(self, method_name)
    else:
      return None

  def transform_expr(self, expr):
    """
    Dispatch on the node type and call the appropriate transform method
    """
    method = self.find_method(expr, "transform_")
    if method:
      result = method(expr)
    else:
      result = self.transform_generic_expr(expr)

    assert result is not None, \
           "Transformation turned %s into None" % (expr,)
    assert result.type is not None, "Missing type for %s" % result
    return result

  def transform_lhs(self, lhs):
    """
    Overload this is you want different behavior
    for transformation of left-hand side of assignments
    """
    lhs_method = self.find_method(lhs, prefix = "transform_lhs_")
    if lhs_method:
      return lhs_method(lhs)

    method = self.find_method(lhs, prefix = "transform_")
    if method:
      return method(lhs)

    return self.transform_expr(lhs)

  def transform_expr_list(self, exprs):
    return [self.transform_expr(e) for e in exprs]

  def transform_phi_nodes(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)
      result[k] = new_left, new_right
    return result

  def transform_Assign(self, stmt):
    old_lhs = stmt.lhs 
    old_rhs = stmt.rhs 
    new_rhs = self.transform_expr(stmt.rhs)
    new_lhs = self.transform_lhs(stmt.lhs)
    if old_lhs !=  new_lhs or old_rhs != new_rhs: 
      return syntax.Assign(new_lhs, new_rhs)
    else:
      return stmt 

  def transform_Return(self, stmt):
    old_value = stmt.value 
    new_value = self.transform_expr(stmt.value)
    if old_value != new_value:
      return syntax.Return(new_value)
    else:
      return stmt 

  def transform_If(self, stmt):
    true = self.transform_block(stmt.true)
    false = self.transform_block(stmt.false)
    merge = self.transform_phi_nodes(stmt.merge)
    cond = self.transform_expr(stmt.cond)
    return syntax.If(cond, true, false, merge)

  def transform_While(self, stmt):
    body = self.transform_block(stmt.body)
    merge = self.transform_phi_nodes(stmt.merge)
    cond = self.transform_expr(stmt.cond)
    return syntax.While(cond, body, merge)

  def transform_stmt(self, stmt):
    method_name = "transform_" + stmt.node_type()
    if hasattr(self, method_name):
      result = getattr(self, method_name)(stmt)
    import types
    assert isinstance(result, (syntax.Stmt, types.NoneType)), \
        "Expected statement: %s" % result
    return result

  def transform_block(self, stmts):  
    self.blocks.push()
    for old_stmt in (reversed(stmts) if self.reverse else stmts):
      new_stmt = self.transform_stmt(old_stmt)
      if new_stmt:
        self.blocks.append_to_current(new_stmt)
    new_block = self.blocks.pop()
    if self.reverse:
      new_block.reverse()
    return new_block
    
  def pre_apply(self, old_fn):
    """
    print 
    print "Running %s" % self.__class__.__name__
    print 
    print "-- before" 
    print repr(old_fn) 
    print
    """
    pass 
  
  def post_apply(self, new_fn):
    """
    print 
    print "-- after"
    print repr(new_fn)
    print 
    """
    pass 
  
  def apply(self, copy = False):
    self.copy = copy

    old_fn = self.pre_apply(self.fn)
    if old_fn is None:
      old_fn = self.fn

    if isinstance(old_fn, syntax.TypedFn):
      self.type_env = old_fn.type_env.copy()
    else:
      self.type_env = {}
    new_body = self.transform_block(old_fn.body)
    if copy:
      new_fundef_args = dict([(m, getattr(old_fn, m)) for m in old_fn._members])
      # create a fresh function with a distinct name and the
      # transformed body and type environment
      new_fundef_args['name'] = names.refresh(self.fn.name)
      new_fundef_args['body'] = new_body
      new_fundef_args['type_env'] = self.type_env
      new_fundef = syntax.TypedFn(**new_fundef_args)

      new_fn = self.post_apply(new_fundef)
      if new_fn is None:
        new_fn = new_fundef
    else:
      old_fn.type_env = self.type_env
      old_fn.body = new_body
      new_fn = self.post_apply(old_fn)

      if new_fn is None:
        new_fn = old_fn
    if self.verify:
      import verify
      verify.verify(new_fn)
    return new_fn

class MemoizedTransform(Transform):
  _cache = {}

  def apply(self, copy = False):
    key = (self.__class__.__name__, self.fn.name)
    if key in self._cache and not copy:
      return self._cache[key]
    else:
      new_fn = Transform.apply(self, copy = copy)
      self._cache[key] = new_fn
    return new_fn

def apply_pipeline(fn, transforms, copy = False):
  for T in transforms:
    t = T(fn)
    fn = t.apply(copy = copy)

    # only copy the function on the first iteration,
    # if you're going to copy it at all
    copy = False
  return fn
