from adverbs import Map, Reduce, Scan
from syntax import Assign, ExprStmt, ForLoop, If, Return, While
from syntax import Comment
from syntax import Attribute, Const, Index, PrimCall, Tuple, Var
from syntax import Alloc, Call, Struct, TypedFn
from syntax import AllocArray, ArrayView, Cast, Slice, TupleProj

class SyntaxVisitor(object):
  """
  Traverse the statement structure of a syntax block, optionally collecting
  values
  """

  def visit_Var(self, expr):
    pass

  def visit_Const(self, expr):
    pass

  def visit_Tuple(self, expr):
    for elt in expr.elts:
      self.visit_expr(elt)

  def visit_PrimCall(self, expr):
    for arg in expr.args:
      self.visit_expr(arg)

  def visit_Attribute(self, expr):
    self.visit_expr(expr.value)

  def visit_Index(self, expr):
    self.visit_expr(expr.value)
    self.visit_expr(expr.index)

  def visit_TypedFn(self, expr):
    pass

  def visit_Alloc(self, expr):
    self.visit_expr(expr.count)

  def visit_Struct(self, expr):
    for arg in expr.args:
      self.visit_expr(arg)

  def visit_ArrayView(self, expr):
    self.visit_expr(expr.data)
    self.visit_expr(expr.shape)
    self.visit_expr(expr.strides)
    self.visit_expr(expr.offset)
    self.visit_expr(expr.total_elts)

  def visit_AllocArray(self, expr):
    self.visit_expr(expr.shape)

  def visit_Slice(self, expr):
    self.visit_expr(expr.start)
    self.visit_expr(expr.stop)
    self.visit_expr(expr.step)

  def visit_Map(self, expr):
    self.visit_expr(expr.fn)
    for arg in expr.args:
      self.visit_expr(arg)
    if expr.out:
      self.visit_expr(expr.out)

  def visit_AllPairs(self, expr):
    self.visit_expr(expr.fn)
    for arg in expr.args:
      self.visit_expr(arg)
    if expr.out:
      self.visit_expr(expr.out)

  def visit_Reduce(self, expr):
    self.visit_expr(expr.fn)
    if expr.init:
      self.visit_expr(expr.init)
    for arg in expr.args:
      self.visit_expr(arg)
    if expr.out:
      self.visit_expr(expr.out)

  def visit_Scan(self, expr):
    self.visit_expr(expr.fn)
    if expr.init:
      self.visit_expr(expr.init)
    for arg in expr.args:
      self.visit_expr(arg)
    if expr.out:
      self.visit_expr(expr.out)

  def visit_TupleProj(self, expr):
    return self.visit_expr(expr.tuple)

  def visit_Call(self, expr):
    self.visit_expr(expr.fn)
    for arg in expr.args:
      self.visit_expr(arg)

  def visit_Cast(self, expr):
    return self.visit_expr(expr.value)

  def visit_generic_expr(self, expr):
    for v in expr.children():
      self.visit_expr(v)

  def visit_expr(self, expr):
    c = expr.__class__
    if c is Var:
      return self.visit_Var(expr)
    elif c is Const:
      return self.visit_Const(expr)
    elif c is PrimCall:
      return self.visit_PrimCall(expr)
    elif c is Attribute:
      return self.visit_Attribute(expr)
    elif c is Index:
      return self.visit_Index(expr)
    elif c is Tuple:
      return self.visit_Tuple(expr)
    elif c is TupleProj:
      return self.visit_TupleProj(expr)
    elif c is Slice:
      return self.visit_Slice(expr)
    elif c is Struct:
      return self.visit_Struct(expr)
    elif c is AllocArray:
      return self.visit_AllocArray(expr)
    elif c is ArrayView:
      return self.visit_ArrayView(expr)
    elif c is Alloc:
      return self.visit_Alloc(expr)
    elif c is Cast:
      return self.visit_Cast(expr)
    elif c is Call:
      return self.visit_Call(expr)
    elif c is Map:
      return self.visit_Map(expr)
    elif c is Reduce:
      return self.visit_Reduce(expr)
    elif c is Scan:
      return self.visit_Scan(expr)
    elif c is TypedFn:
      return self.visit_TypedFn(expr)
    else:
      method_name = 'visit_' + expr.node_type()
      method = getattr(self, method_name, None)
      if method:
        return method(expr)
      else:
        self.visit_generic_expr(expr)

  def visit_expr_list(self, exprs):
    return [self.visit_expr(expr) for expr in exprs]

  def visit_lhs_Var(self, lhs):
    self.visit_Var(lhs)

  def visit_lhs_Tuple(self, lhs):
    self.visit_Tuple(lhs)

  def visit_lhs_Index(self, lhs):
    self.visit_Index(lhs)

  def visit_lhs_Attribute(self, lhs):
    self.visit_Attribute(lhs)

  def visit_lhs(self, lhs):
    c = lhs.__class__
    if c is Var:
      return self.visit_lhs_Var(lhs)
    elif c is Tuple:
      return self.visit_lhs_Tuple(lhs)
    elif c is Index:
      return self.visit_lhs_Index(lhs)
    elif c is Attribute:
      return self.visit_lhs_Attribute(lhs)
    else:
      assert False, "LHS not implemented: %s" % (lhs,)

  def visit_block(self, stmts):
    for s in stmts:
      self.visit_stmt(s)

  def visit_Assign(self, stmt):
    self.visit_lhs(stmt.lhs)
    self.visit_expr(stmt.rhs)

  def visit_merge(self, phi_nodes):
    for (_, (l,r)) in phi_nodes.iteritems():
      self.visit_expr(l)
      self.visit_expr(r)

  def visit_merge_if(self, phi_nodes):
    self.visit_merge(phi_nodes)

  def visit_If(self, stmt):
    self.visit_expr(stmt.cond)
    self.visit_block(stmt.true)
    self.visit_block(stmt.false)
    self.visit_merge_if(stmt.merge)

  def visit_ExprStmt(self, stmt):
    self.visit_expr(stmt.value)

  def visit_Return(self, stmt):
    self.visit_expr(stmt.value)

  def visit_merge_loop_start(self, phi_nodes):
    pass

  def visit_merge_loop_repeat(self, phi_nodes):
    self.visit_merge(phi_nodes)

  def visit_While(self, stmt):
    self.visit_merge_loop_start(stmt.merge)
    self.visit_expr(stmt.cond)
    self.visit_block(stmt.body)
    self.visit_merge_loop_repeat(stmt.merge)

  def visit_ForLoop(self, stmt):
    self.visit_lhs(stmt.var)
    self.visit_expr(stmt.start)
    self.visit_merge_loop_start(stmt.merge)
    self.visit_expr(stmt.stop)
    self.visit_block(stmt.body)
    self.visit_expr(stmt.step)
    self.visit_merge_loop_repeat(stmt.merge)

  def visit_Comment(self, stmt):
    pass

  def visit_stmt(self, stmt):
    c = stmt.__class__
    if c is Assign:
      self.visit_Assign(stmt)
    elif c is Return:
      self.visit_Return(stmt)
    elif c is ForLoop:
      self.visit_ForLoop(stmt)
    elif c is While:
      self.visit_While(stmt)
    elif c is If:
      self.visit_If(stmt)
    elif c is ExprStmt:
      self.visit_ExprStmt(stmt)
    elif c is Comment:
      self.visit_Comment(stmt)
    else:
      assert False, "Statement not implemented: %s" % stmt

  def visit_fn(self, fn):
    self.visit_block(fn.body)
