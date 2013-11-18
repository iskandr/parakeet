import time

from .. import config
from .. analysis import verify
from .. builder import Builder  
from .. syntax import (Expr, If, Assign, While, Return, ExprStmt, ForLoop, Comment, ParFor, 
                       Var, Tuple, Index, Attribute, Const, PrimCall, Struct, Alloc, Cast,  
                       TupleProj, Slice, ArrayView, Call, TypedFn,  AllocArray, Len, UntypedFn,  
                       Map, Reduce) 

transform_timings = {}
transform_counts = {}
if config.print_transform_timings:
  import atexit
  def print_timings():
    print "TRANSFORM TIMINGS"
    items = transform_timings.items()
    items.sort(key = lambda (_,t): t)
    items.reverse()
    for k, t in items:
      count = transform_counts[k]
      print "  %30s  Total = %6dms, Count = %4d, Avg = %3fms" % \
            (k.__name__, t*1000, count, (t/count)*1000)
  atexit.register(print_timings)

class Transform(Builder):
  def __init__(self, verify=config.opt_verify,
                     reverse=False,
                     require_types=True):

    Builder.__init__(self)
    self.fn = None
    self.verify = verify
    self.reverse = reverse
    self.require_types = require_types

  def __str__(self):
    return self.__class__.__name__ 
  
  def __repr__(self):
    return str(self)
  
  def __hash__(self):
    return hash(str(self))
  
  def __eq__(self, other):
    return str(self) == str(other)
  
  def lookup_type(self, name):
    assert self.type_env is not None
    return self.type_env[name]

  def transform_if_expr(self, maybe_expr):
    if isinstance(maybe_expr, Expr):
      return self.transform_expr(maybe_expr)
    elif isinstance(maybe_expr, tuple):
      return tuple([self.transform_if_expr(x) for x in maybe_expr])
    elif isinstance(maybe_expr, list):
      return [self.transform_if_expr(x) for x in maybe_expr]
    #elif hasattr(maybe_expr, 'transform'):
    #  return maybe_expr.transform(self.transform_expr)
    else:
      return maybe_expr

  def transform_generic_expr(self, expr):
    for member_name in expr.members():
      old_value = getattr(expr, member_name)
      new_value = self.transform_if_expr(old_value)
      setattr(expr, member_name, new_value)
    return expr

  def find_method(self, expr, prefix = "transform_"):
    assert expr, "Expected expression but got %s" % expr 
    method_name = prefix + expr.__class__.__name__
    if hasattr(self, method_name):
      return getattr(self, method_name)
    else:
      return None

  """
  Common cases for expression transforms: we don't need to create a method for
  every sort of expression but these run faster and allocate less memory than
  transform_generic_expr
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

  def transform_PrimCall(self, expr):
    expr.args = self.transform_expr_tuple(expr.args)
    return expr

  def transform_Call(self, expr):
    expr.fn = self.transform_expr(expr.fn)
    expr.args = self.transform_expr_tuple(expr.args)
    return expr

  def transform_Alloc(self, expr):
    expr.count = self.transform_expr(expr.count)
    return expr

  def transform_Struct(self, expr):
    expr.args = self.transform_expr_tuple(expr.args)
    return expr

  def transform_Cast(self, expr):
    expr.value = self.transform_expr(expr.value)
    return expr

  def transform_Select(self, expr):
    expr.cond = self.transform_expr(expr.cond)
    expr.true_value = self.transform_expr(expr.true_value)
    expr.false_value = self.transform_expr(expr.false_value)
    return expr 

  def transform_TupleProj(self, expr):
    expr.tuple = self.transform_expr(expr.tuple)
    return expr

  def transform_TypedFn(self, expr):
    """By default, don't do recursive transformation of referenced functions"""
    return expr
  
  def transform_UntypedFn(self, expr):
    """By default, don't do recursive transformation of referenced functions"""
    return expr


  def transform_Slice(self, expr):
    expr.start = self.transform_expr(expr.start) if expr.start else None
    expr.stop = self.transform_expr(expr.stop) if expr.stop else None
    expr.step = self.transform_expr(expr.step) if expr.step else None
    return expr
  
  def transform_Range(self, expr):
    expr.start = self.transform_expr(expr.start) if expr.start else None
    expr.stop = self.transform_expr(expr.stop) if expr.stop else None
    expr.step = self.transform_expr(expr.step) if expr.step else None
    return expr
    

  def transform_Array(self, expr):
    expr.elts = self.transform_expr_tuple(expr.elts) 
    return expr
    
  def transform_ArrayView(self, expr):
    expr.data = self.transform_expr(expr.data)
    expr.shape = self.transform_expr(expr.shape)
    expr.strides = self.transform_expr(expr.strides)
    expr.offset = self.transform_expr(expr.offset)
    expr.size = self.transform_expr(expr.size)
    return expr
  
  def transform_AllocArray(self, expr):
    expr.shape = self.transform_expr(expr.shape)
    return expr 
  
  def transform_ConstArray(self, expr):
    expr.shape = self.transform_expr(expr.shape)
    expr.value = self.transform_expr(expr.value)
    
  def transform_ConstArrayLike(self, expr):
    expr.array = self.transform_expr(expr.array)
    expr.value = self.transform_expr(expr.value)

  def transform_Ravel(self, expr):
    expr.array = self.transform_expr(expr.array)
    return expr 
  
  def transform_Reshape(self, expr):
    expr.array = self.transform_expr(expr.array)
    expr.shape = self.transform_expr(expr.shape)
    return expr
  
  def transform_Transpose(self, expr):
    expr.array = self.transform_expr(expr.array)
    return expr 
  
  def transform_Shape(self, expr):
    expr.array = self.transform_expr(expr.array)
    return expr 
  
  def transform_Len(self, expr):
    expr.value = self.transform_expr(expr.value) 
    return expr

  def transform_IndexMap(self, expr):
    expr.fn = self.transform_expr(expr.fn)
    expr.shape = self.transform_expr(expr.shape)
    return expr 
  
  def transform_IndexReduce(self, expr):
    expr.fn = self.transform_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    expr.shape = self.transform_expr(expr.shape)
    expr.init = self.transform_if_expr(expr.init)
    return expr
  
  def transform_IndexScan(self, expr):
    expr.fn = self.transform_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    expr.shape = self.transform_expr(expr.shape)
    expr.init = self.transform_if_expr(expr.init)
    return expr
  
  def transform_Map(self, expr):
    expr.axis = self.transform_if_expr(expr.axis)
    expr.fn = self.transform_expr(expr.fn)
    expr.args = self.transform_expr_list(expr.args)
    return expr
  
  def transform_Reduce(self, expr):
    expr.axis = self.transform_if_expr(expr.axis)
    expr.init = self.transform_if_expr(expr.init)
    expr.fn = self.transform_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    expr.args = self.transform_expr_list(expr.args)
    return expr
  
  def transform_Scan(self, expr):
    expr.axis = self.transform_if_expr(expr.axis)
    expr.init = self.transform_if_expr(expr.init)
    expr.fn = self.transform_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    expr.args = self.transform_expr_list(expr.args)
    expr.emit = self.transform_expr(expr.emit)
    return expr
  
  def transform_OuterMap(self, expr):
    expr.axis = self.transform_if_expr(expr.axis)
    expr.fn = self.transform_expr(expr.fn)
    expr.args = self.transform_expr_tuple(expr.args)
    return expr
  
  def transform_Closure(self, expr):
    expr.args = self.transform_expr_tuple(expr.args)
    expr.fn = self.transform_expr(expr.fn)
    return expr
  
  
  def transform_ClosureElt(self, expr):
    expr.closure = self.transform_expr(expr.closure)
    return expr

  def transform_TypeValue(self, expr):
    pass 
  
  def transform_DelayUntilTyped(self, expr):
    expr.values = self.transform_expr_tuple(expr.values)
    return expr 
  
  def transform_expr(self, expr):
    """Dispatch on the node type and call the appropriate transform method"""

    expr_class = expr.__class__
    if expr_class is Var:
      result = self.transform_Var(expr)
    elif expr_class is Const:
      result = self.transform_Const(expr)
    elif expr_class is Tuple:
      result = self.transform_Tuple(expr)
    elif expr_class is TupleProj:
      result = self.transform_TupleProj(expr)
    elif expr_class is Index:
      result = self.transform_Index(expr)
    elif expr_class is Slice:
      result = self.transform_Slice(expr)
    elif expr_class is Attribute:
      result = self.transform_Attribute(expr)
    elif expr_class is PrimCall:
      result = self.transform_PrimCall(expr)
    elif expr_class is Struct:
      result = self.transform_Struct(expr)
    elif expr_class is AllocArray:
      result = self.transform_AllocArray(expr)
    elif expr_class is Alloc:
      result = self.transform_Alloc(expr)
    elif expr_class is Cast:
      result = self.transform_Cast(expr)
    elif expr_class is ArrayView:
      result = self.transform_ArrayView(expr)
    elif expr_class is TypedFn:
      result = self.transform_TypedFn(expr)
    elif expr_class is UntypedFn:
      result = self.transform_UntypedFn(expr)
    elif expr_class is Call:
      result = self.transform_Call(expr)
    elif expr_class is Map:
      result = self.transform_Map(expr)
    elif expr_class is Reduce:
      result = self.transform_Reduce(expr)
    elif expr_class is Len:
      result = self.transform_Len(expr)
    else:
      method = self.find_method(expr, "transform_")
      if method:
        result = method(expr)

      else:
        assert False, "Unsupported expr %s" % (expr,)
        result = self.transform_generic_expr(expr)
    if result is None:
      return expr 
    else:
      assert isinstance(result, Expr), \
        "Invalid result type in transformation: %s" % (type(result),)
      result.source_info = expr.source_info 
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
    Overload this is you want different behavior for transformation of left-hand
    side of assignments
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

  def transform_expr_tuple(self, exprs):
    return tuple(self.transform_expr_list(exprs))

  

  def transform_merge(self, phi_nodes):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)
      result[k] = new_left, new_right
    return result

  def transform_merge_before_loop(self, phi_nodes):
    return phi_nodes 
  
  def transform_merge_after_loop(self, phi_nodes):
    return self.transform_merge(phi_nodes)
  
  def transform_Assign(self, stmt):
    stmt.rhs = self.transform_expr(stmt.rhs)
    stmt.lhs = self.transform_lhs(stmt.lhs)
    return stmt

  def transform_ExprStmt(self, stmt):
    stmt.value = self.transform_expr(stmt.value)
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
    stmt.merge = self.transform_merge_before_loop(stmt.merge)
    stmt.cond = self.transform_expr(stmt.cond)
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge_after_loop(stmt.merge)
    return stmt
  
  def transform_ForLoop(self, stmt):
    stmt.var = self.transform_expr(stmt.var)
    stmt.merge = self.transform_merge_before_loop(stmt.merge)
    stmt.start = self.transform_expr(stmt.start)
    stmt.stop = self.transform_expr(stmt.stop)
    stmt.step = self.transform_expr(stmt.step)
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge_after_loop(stmt.merge)
    return stmt 
  
  def transform_Comment(self, stmt):
    return stmt 
  
  def transform_ParFor(self, stmt):
    stmt.fn = self.transform_expr(stmt.fn)
    stmt.bounds= self.transform_expr(stmt.bounds)
    return stmt 
  
  def transform_stmt(self, stmt):
    
    stmt_class = stmt.__class__
    if stmt_class is Assign:
      return self.transform_Assign(stmt)
    elif stmt_class is ForLoop:
      return self.transform_ForLoop(stmt)
    elif stmt_class is While:
      return self.transform_While(stmt)
    elif stmt_class is If:
      return self.transform_If(stmt)
    elif stmt_class is Return:
      return self.transform_Return(stmt)
    elif stmt_class is ExprStmt:
      return self.transform_ExprStmt(stmt)
    elif stmt_class is ParFor:
      return self.transform_ParFor(stmt)
    elif stmt_class is Comment:
      return self.transform_Comment(stmt)
    else:
      assert False, "Unexpected statement %s" % stmt_class

  def transform_block(self, stmts):
    
    self.blocks.push()
    
    if self.reverse: stmts = reversed(stmts)
    
    for old_stmt in stmts:
      new_stmt = self.transform_stmt(old_stmt)
      if new_stmt is not None:
        self.blocks.append_to_current(new_stmt)
    new_block = self.blocks.pop()
    if self.reverse: new_block.reverse()
    return new_block

  def pre_apply(self, old_fn):
    pass  

  def post_apply(self, new_fn):
    pass 

  def apply(self, fn):
    if config.print_transform_timings:
      start_time = time.time()

    transform_name = self.__class__.__name__
    
      
    if config.print_functions_before_transforms == True or \
        (isinstance(config.print_functions_before_transforms, list) and
         transform_name in config.print_functions_before_transforms):
      print
      print "Running transform %s" % transform_name
      print "--- before ---"
      print repr(fn)
      print
    
    self.fn = fn
    self.type_env = fn.type_env

    # push an extra block onto the stack just in case
    # one of the pre_apply methods want to put statements somewhere
    self.blocks.push()
    pre_fn = self.pre_apply(self.fn)
    pre_block = self.blocks.pop()

    if pre_fn is not None:
      fn = pre_fn

    self.fn = fn
    self.type_env = fn.type_env

    # pop the outermost block, which have been written to by
    new_body = self.transform_block(fn.body)
    
    if len(pre_block) > 0:
      new_body = pre_block  + new_body

    fn.body = new_body
    fn.type_env = self.type_env

    self.blocks.push()
    new_fn = self.post_apply(fn)
    post_block = self.blocks.pop()
    if new_fn is None:
      new_fn = fn

    if len(post_block) > 0:
      new_fn.body = new_fn.body + post_block

    if config.print_functions_after_transforms == True or \
        (isinstance(config.print_functions_after_transforms, list) and
         transform_name in config.print_functions_after_transforms):
      print
      print "Done with  %s" % transform_name
      print "--- after ---"
      print repr(new_fn)
      print

    if self.verify:
      try:
        verify(new_fn)
      except:
        print "ERROR after running %s on %s" % (transform_name , new_fn)
        raise

    if config.print_transform_timings:
      end_time = time.time()
      c = self.__class__
      total_time = transform_timings.get(c, 0)
      transform_timings[c] = total_time + (end_time - start_time)
      transform_counts[c] = transform_counts.get(c, 0) + 1
    return new_fn

