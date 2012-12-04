
import names
import syntax

from args import ActualArgs
from codegen import Codegen

class Transform(Codegen):
  def __init__(self, fn):
    Codegen.__init__(self)
    self.fn = fn
    self.copy = None

  def lookup_type(self, name):
    assert self.type_env is not None
    return self.type_env[name]

  def transform_TypedFn(self, fn):
    return fn 
    #nested_transform = self.__class__(fn)
    #return nested_transform.apply(copy = self.copy)

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
    for member_name in expr.members():
      member_value = getattr(expr, member_name)
      args[member_name] = self.transform_if_expr(member_value)
    return expr.__class__(**args)

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

    rhs = self.transform_expr(stmt.rhs)
    lhs = self.transform_lhs(stmt.lhs)
    return syntax.Assign(lhs, rhs)

  def transform_Return(self, stmt):
    return syntax.Return(self.transform_expr(stmt.value))

  def transform_If(self, stmt):
    cond = self.transform_expr(stmt.cond)
    true = self.transform_block(stmt.true)
    false = self.transform_block(stmt.false)
    merge = self.transform_phi_nodes(stmt.merge)
    return syntax.If(cond, true, false, merge)

  def transform_While(self, stmt):
    cond = self.transform_expr(stmt.cond)
    body = self.transform_block(stmt.body)
    merge = self.transform_phi_nodes(stmt.merge)
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
    for old_stmt in stmts:
      new_stmt = self.transform_stmt(old_stmt)
      if new_stmt:
        self.blocks.append_to_current(new_stmt)
    return self.blocks.pop()

  def pre_apply(self, old_fn):
    pass

  def post_apply(self, new_fn):
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
      if new_fn:
        return new_fn
      else:
        return new_fundef
    else:
      old_fn.type_env = self.type_env
      old_fn.body = new_body
      new_fn = self.post_apply(old_fn)

      if new_fn:
        return new_fn
      else:
        return old_fn

_transform_cache = {}
def cached_apply(T, fn, copy = False):
  """
  Applies the transformation, caches the result,
  and registers the new function in the global registry
  """
  key = (T, fn.name)
  if key in _transform_cache:
    return _transform_cache[key]
  else:
    new_fn = T(fn).apply(copy = copy)
    _transform_cache[key] = new_fn
    return new_fn

def apply_pipeline(fn, transforms, copy = False, memoize = False):
  for T in transforms:
    if memoize:
      fn = cached_apply(T, fn, copy = copy)
    else:
      fn = T(fn).apply(copy = copy)

    # only copy the function on the first iteration,
    # if you're going to copy it at all
    copy = False
  return fn
