from .. syntax import (Expr, Assign, ExprStmt, ForLoop, If, Return, While, Comment, ParFor, 
                       TypedFn, UntypedFn,  Closure, ClosureElt, Select,  
                       Attribute, Const, Index, PrimCall, Tuple, Var, 
                       Alloc, Array, Call, Struct, Shape, Strides, Range, Ravel, Transpose,
                       AllocArray, ArrayView, Cast, Slice, TupleProj, TypeValue,  
                       Map, Reduce, Scan, OuterMap, IndexMap, IndexReduce, IndexScan )

class SyntaxVisitor(object):
  """
  Traverse the statement structure of a syntax block, optionally collecting
  values
  """

  def visit_if_expr(self, expr):
    if isinstance(expr, Expr):
      self.visit_expr(expr)
      
  def visit_Var(self, expr):
    pass

  def visit_Const(self, expr):
    pass
  
  def visit_Fn(self, expr):
    pass 
  
  def visit_ClosureElt(self, expr):
    self.visit_expr(expr.closure)
  
  def visit_Closure(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr_list(expr.args)
    
    
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

  def visit_UntypedFn(self, expr):
    pass 
  
  def visit_TypedFn(self, expr):
    pass

  def visit_Alloc(self, expr):
    self.visit_expr(expr.count)

  def visit_Struct(self, expr):
    for arg in expr.args:
      self.visit_expr(arg)

  def visit_Array(self, expr):
    for elt in expr.elts:
      self.visit_expr(elt)
      
  def visit_ArrayView(self, expr):
    self.visit_expr(expr.data)
    self.visit_expr(expr.shape)
    self.visit_expr(expr.strides)
    self.visit_expr(expr.offset)
    self.visit_expr(expr.size)

  def visit_AllocArray(self, expr):
    self.visit_expr(expr.shape)

  def visit_Ravel(self, expr):
    self.visit_expr(expr.array)

  def visit_Reshape(self, expr):
    self.visit_expr(expr.array)
    self.visit_expr(expr.shape)

  def visit_Shape(self, expr):
    self.visit_expr(expr.array)
  
  def visit_ConstArray(self, expr):
    self.visit_expr(expr.shape)
    self.visit_expr(expr.value)
    
  def visit_ConstArrayLike(self, expr):
    self.visit_expr(expr.array)
    self.visit_expr(expr.value)
  
  def visit_Slice(self, expr):
    self.visit_expr(expr.start)
    self.visit_expr(expr.stop)
    self.visit_expr(expr.step)

  def visit_Transpose(self, expr):
    self.visit_expr(expr.array)
    
  def visit_IndexMap(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr(expr.shape)

  def visit_IndexReduce(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr(expr.combine)
    self.visit_expr(expr.shape)
    self.visit_expr(expr.init)

  def visit_IndexScan(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr(expr.combine)
    self.visit_expr(expr.shape)
    self.visit_expr(expr.init)

  def visit_Map(self, expr):
    self.visit_expr(expr.fn)
    self.visit_if_expr(expr.axis)
    for arg in expr.args:
      self.visit_expr(arg)

  def visit_OuterMap(self, expr):
    self.visit_expr(expr.fn)
    self.visit_if_expr(expr.axis)
    for arg in expr.args:
      self.visit_expr(arg)

  def visit_Reduce(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr(expr.combine)
    self.visit_if_expr(expr.axis)
    self.visit_if_expr(expr.init)
    for arg in expr.args:
      self.visit_expr(arg)

  def visit_Scan(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr(expr.combine)
    self.visit_if_expr(expr.axis)
    self.visit_if_expr(expr.init)
    for arg in expr.args:
      self.visit_expr(arg)
      
  def visit_TupleProj(self, expr):
    return self.visit_expr(expr.tuple)

  def visit_Call(self, expr):
    self.visit_expr(expr.fn)
    for arg in expr.args:
      self.visit_expr(arg)

  def visit_Cast(self, expr):
    return self.visit_expr(expr.value)

  def visit_Range(self, expr):
    self.visit_expr(expr.start)
    self.visit_expr(expr.stop)
    self.visit_expr(expr.step)
  
  def visit_Select(self, expr):
    self.visit_expr(expr.cond)
    self.visit_expr(expr.true_value)
    self.visit_expr(expr.false_value)
    
  def visit_TypeValue(self, expr):
    pass 
  
  def visit_generic_expr(self, expr):
    for v in expr.children():
      self.visit_expr(v)

  _expr_method_names = {
    Var : 'visit_Var', 
    Const : 'visit_Const', 
    PrimCall : 'visit_PrimCall', 
    Attribute : 'visit_Attribute', 
    Index : 'visit_Index', 
    Tuple : 'visit_Tuple', 
    TupleProj : 'visit_TupleProj', 
    Slice : 'visit_Slice', 
    Struct : 'visit_Struct', 
    AllocArray : 'visit_AllocArray', 
    ArrayView : 'visit_ArrayView', 
    Array : 'visit_Array',
    Range : 'visit_Range',
    Ravel : 'visit_Ravel',   
    Transpose : 'visit_Transpose', 
    Shape : 'visit_Shape', 
    Strides : 'visit_Strides', 
    Alloc : 'visit_Alloc', 
    Cast : 'visit_Cast', 
    Call : 'visit_Call', 
    Select : 'visit_Select', 
    Map : 'visit_Map',
    OuterMap : 'visit_OuterMap',
    IndexMap : 'visit_IndexMap', 
    Reduce : 'visit_Reduce',
    IndexReduce : 'visit_IndexReduce',
    Scan : 'visit_Scan', 
    IndexScan : 'visit_IndexScan',
    Closure : 'visit_Closure', 
    ClosureElt : 'visit_ClosureElt', 
    UntypedFn : 'visit_UntypedFn',  
    TypedFn : 'visit_TypedFn', 
    TypeValue : 'visit_TypeValue', 
  }
  
  def visit_expr(self, expr):   
    c = expr.__class__
    if c is Var:  
      return self.visit_Var(expr)
    elif c is Const: 
      return self.visit_Const(expr)
    elif c is PrimCall:
      return self.visit_PrimCall(expr)
    elif c is Tuple:
      return self.visit_Tuple(expr)
    elif c is TupleProj:
      return self.visit_TupleProj(expr) 
    elif c is Index: 
      return self.visit_Index(expr)
    
    # try looking up the method in fast-path dictionary
    method_name = self._expr_method_names.get(c)
    if not method_name:
      # ...otherwise, just construct a new string following the visit_Expr pattern 
      method_name = "visit_" + c.__name__
    
    # if we found a method, call it 
    if method_name:
      method = getattr(self, method_name)
      
    if method:
      return method(expr)

    for child in expr.children():
      self.visit_expr(child)
      
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

  def visit_ParFor(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr(expr.bounds)
  
  _stmt_method_names = {
    Assign : 'visit_Assign', 
    Return : 'visit_Return', 
    While : 'visit_While', 
    ForLoop : 'visit_ForLoop', 
    If : 'visit_If', 
    ExprStmt : 'visit_ExprStmt', 
    ParFor : 'visit_ParFor', 
    Comment : 'visit_Comment',                    
  }
  
  def visit_stmt(self, stmt):
    c = stmt.__class__
    if c is Assign: 
      self.visit_Assign(stmt)
    else:
      getattr(self, self._stmt_method_names[c])(stmt)
    

  def visit_fn(self, fn):
    self.visit_block(fn.body)
