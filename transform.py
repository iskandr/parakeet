
import config
import verify

import syntax
from syntax import If, Assign, While, Return 
from syntax import Var, Tuple, Index, Attribute, Const  
from args import ActualArgs
from codegen import Codegen

class Transform(Codegen):
  def __init__(self, fn, verify = config.opt_verify, reverse = False):
    Codegen.__init__(self)
    self.fn = fn
    self.verify = verify
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
    for member_name in expr.members():
      old_value = getattr(expr, member_name)
      new_value = self.transform_if_expr(old_value)
      setattr(expr, member_name, new_value)
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
  
  def transform_Tuple(self, expr):
    expr.elts = tuple(self.transform_expr(elt) for elt in expr.elts) 
    return expr 
  
  def transform_Const(self, expr):
    return expr 
  
  def transform_Index(self, expr):
    expr.value = self.transform_expr(expr.value)
    expr.index = self.transform_expr(expr.index) 
    return expr 
  
  def transform_Attribute(self, expr):
    expr.value = self.transform_expr(expr.value)
    return expr 
  
  def transform_expr(self, expr):
    """
    Dispatch on the node type and call the appropriate transform method
    """
    expr_class = expr.__class__ 
    if expr_class is Var:
      result = self.transform_Var(expr)
    elif expr_class is Const:
      result = self.transform_Const(expr)
    elif expr_class is Tuple:
      result = self.transform_Tuple(expr)
    elif expr_class is Index:
      result = self.transform_Index(expr)
    elif expr_class is Attribute:
      result = self.transform_Attribute(expr)
    else:
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
    return self.transform_Attribute(expr)
  
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

   
  def transform_Assign(self, stmt):
    stmt.rhs = self.transform_expr(stmt.rhs)
    stmt.lhs =self.transform_lhs(stmt.lhs)
    return stmt 

  def transform_Return(self, stmt):
    stmt.value = self.transform_expr(stmt.value)
    return stmt 
  
  def transform_If(self, stmt):
    stmt.true = self.transform_block(stmt.true)
    stmt.false = self.transform_block(stmt.false)
    stmt.merge = self.transform_merge(stmt.merge)
    stmt.cond = self.transform_expr(stmt.cond)
    return stmt 
  
  def transform_While(self, stmt):
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge(stmt.merge)
    stmt.cond = self.transform_expr(stmt.cond)
    return stmt 
  
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
      if new_stmt is not None:
        self.blocks.append_to_current(new_stmt)
    new_block = self.blocks.pop()
    if self.reverse:
      new_block.reverse()
    return new_block

  def pre_apply(self, old_fn):
    pass

  def post_apply(self, new_fn):
    pass

  def apply(self):
    if config.print_functions_before_transforms:
      print
      print "Running transform %s" % self.__class__.__name__
      print "--- before ---"
      print repr(self.fn)
      print 
      
    fn = self.pre_apply(self.fn)
    if fn is None:
      fn = self.fn

    self.type_env = fn.type_env
    fn.body = self.transform_block(fn.body)
    fn.type_env = self.type_env
    new_fn = self.post_apply(fn)
    if new_fn is None:
      new_fn = fn
      
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

  def apply(self):
    key = (self.__class__.__name__, self.fn.name)
    if key in self._cache:
      return self._cache[key]
    else:
      new_fn = Transform.apply(self)
      self._cache[key] = new_fn
      return new_fn

def apply_pipeline(fn, transforms):
  for T in transforms:
    fn = T(fn).apply()
    
  return fn
