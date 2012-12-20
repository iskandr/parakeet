import names
import syntax
from syntax import If, Assign, While, Return 
from syntax import Var, Tuple, Index, Attribute 
from args import ActualArgs
from codegen import Codegen
import config 
import verify

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
 
 
  """
  Common cases for expression transforms: 
  we don't need to create a method for every
  sort of expression but these run faster 
  and allocate less memory than transform_generic_expr
  """
  def transform_Var(self, expr):
    return expr 
  
  def make_Tuple(self, expr, new_elts):
    if self.copy:
      return Tuple(elts = new_elts, type = expr.type)
    if expr.elts != new_elts: 
      expr.elts = new_elts 
    return expr 
  
  def transform_Tuple(self, expr):
    new_elts = tuple(self.transform_expr(elt) for elt in expr.elts)
    return self.make_Tuple(expr, new_elts)
    
  def transform_Const(self, expr):
    return expr 
  
  def make_Index(self, expr, value, index):
    if self.copy:
      return Index(value, index, type = expr.type)
    
    if expr.value != value:
      expr.value = value
    if expr.index != index:
      expr.index = index 
    return expr 
  
  def transform_Index(self, expr):
    new_value = self.transform_expr(expr.value)
    new_index = self.transform_expr(expr.index)
    return self.make_Index(expr, new_value, new_index)
  
  
  def make_Attribute(self, expr, value):
    if self.copy:
      return Attribute(value, expr.name, type = expr.type)
    
    if expr.value != value:
      expr.value = value
    return expr 
  
  def transform_Attribute(self, expr):
    return self.make_Attribute(expr, self.transform_expr(expr.value))
  
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

  def transform_lhs_Var(self, expr):
    return self.transform_Var(expr)
  
  def transform_lhs_Tuple(self, expr):
    return self.transform_Tuple(expr)
  
  def transform_lhs_Index(self, expr):
    return self.transform_Index(expr)

  def transform_lhs_Attribute(self, expr):
    return self.transform_Attribute(self, expr)
  
  def transform_lhs(self, lhs):
    """
    Overload this is you want different behavior
    for transformation of left-hand side of assignments
    """
    lhs_class = lhs.__class__
    if lhs_class is Var:
      return self.transform_lhs_Var(lhs)
    elif lhs_class is Tuple:
      return self.transform_lhs_Tuple(lhs)
    elif lhs_class is Index:
      return self.transform_lhs_Index(lhs)
    elif lhs_class is Attribute:
      return self.transform_lhs_Attribute(lhs)
    
    lhs_method = self.find_method(lhs, prefix = "transform_lhs_")
    if lhs_method:
      return lhs_method(lhs)

    method = self.find_method(lhs, prefix = "transform_")
    assert method, "Unknown expression of type %s" % lhs_class 
    return method(lhs)
    
  def transform_expr_list(self, exprs):
    return [self.transform_expr(e) for e in exprs]

  def transform_merge(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)
      result[k] = new_left, new_right
    return result

  def make_Assign(self, stmt, lhs, rhs):
    if self.copy:
      stmt = syntax.Assign(lhs, rhs)
    else:
      if stmt.lhs !=  lhs:
        stmt.lhs = lhs 
      if stmt.rhs != rhs:
        stmt.rhs = rhs 
    return stmt
   
  def transform_Assign(self, stmt):
    new_rhs = self.transform_expr(stmt.rhs)
    new_lhs = self.transform_lhs(stmt.lhs)
    return self.make_Assign(stmt, new_lhs, new_rhs)

  
  def make_Return(self, stmt, value):
    if self.copy: 
      return syntax.Return(value)
    else:
      if stmt.value != value:
        stmt.value = value 
      return stmt

  def transform_Return(self, stmt):
    return self.make_Return(stmt, self.transform_expr(stmt.value))
  
  def make_If(self, old_stmt, true, false, merge, cond):  
    if self.copy:
      return syntax.If(cond, true, false, merge)
    else:
      old_stmt.true = true 
      old_stmt.false = false
      old_stmt.merge = merge  
      old_stmt.cond = cond
      return old_stmt 
  
  def transform_If(self, stmt):
    true = self.transform_block(stmt.true)
    false = self.transform_block(stmt.false)
    merge = self.transform_merge(stmt.merge)
    cond = self.transform_expr(stmt.cond)
    return self.make_If(stmt, true, false, merge, cond)
     
  def make_While(self, old_stmt, body, merge, cond):
    if self.copy:
      return syntax.While(cond, body, merge)
    else:
      old_stmt.body = body 
      old_stmt.merge = merge 
      old_stmt.cond = cond 
      return old_stmt 
  
  def transform_While(self, stmt):
    body = self.transform_block(stmt.body)
    merge = self.transform_merge(stmt.merge)
    cond = self.transform_expr(stmt.cond)
    return self.make_While(stmt, body, merge, cond)
  
  def transform_stmt(self, stmt):
    stmt_class = stmt.__class__ 
    if stmt_class is Assign:
      return self.transform_Assign(stmt)
    elif stmt_class is While:
      return self.transform_While(stmt)
    elif stmt_class is If:
      return self.transform_If(stmt)
    else:
      assert stmt_class is Return, \
          "Unexpected statement %s" % stmt_class 
      return self.transform_Return(stmt)
     
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
    pass 

  def post_apply(self, new_fn):
    pass 

  def apply(self, copy = False):
    if config.print_functions_before_transforms:
      print
      print "Running transform %s" % self.__class__.__name__
      print "--- before ---"
      print repr(self.fn)
      print 
    
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
     
    if config.print_functions_after_transforms:
      print
      print "Done with  %s" % self.__class__.__name__
      print "--- after ---"
      print repr(new_fn)
      print
    if self.verify:
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
