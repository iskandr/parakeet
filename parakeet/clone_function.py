import names 
import syntax 
from syntax import TypedFn, Var, Const, Attribute, Index, PrimCall
from syntax import If, Assign, While, ExprStmt, Return, ForLoop 
from syntax import  Slice, Struct
from syntax import Tuple, TupleProj, Cast, Alloc     
from transform import Transform 

class CloneFunction(Transform):
  """
  Copy all the objects in the AST of a function
  """
  def __init__(self, rename = False, recursive = False):
    Transform.__init__(self)
    self.rename = rename 
    self.recursive = recursive
     
  def transform_Var(self, expr):
    return Var(expr.name, type = expr.type)
  
  def transform_expr(self, expr):
    c = expr.__class__
    if c is Var:
      return self.transform_Var(expr)  
    elif c is Const:
      return Const(expr.value, type = expr.type)
    elif c is Tuple:
      new_elts = tuple(self.transform_expr(elt) for elt in expr.elts)
      return Tuple(elts = new_elts, type = expr.type)
    elif c is Attribute:
      value = self.transform_expr(expr.value)
      return Attribute(value, expr.name, type = expr.type)
    elif c is Index:
      value = self.transform_expr(expr.value)
      index = self.transform_expr(expr.index)
      return Index(value, index, type = expr.type)
    elif c is PrimCall:
      args = tuple(self.transform_expr(elt) for elt in expr.args)
      return PrimCall(expr.prim, args, type = expr.type)
    elif c is TypedFn:
      if self.recursive:
        cloner = CloneFunction(rename = self.rename, recursive = True)
        return cloner.apply(expr)
      else:
        return expr 
    elif c is Slice: 
      start = self.transform_if_expr(expr.start)
      stop = self.transform_if_expr(expr.stop)
      step = self.transform_if_expr(expr.step)
      return Slice(start, stop, step, type = expr.type)
    elif c is Struct: 
      new_args = self.transform_expr_list(expr.args)
      return Struct(args = new_args, type = expr.type)
    elif c is TupleProj: 
      new_tuple = self.transform_expr(expr.tuple)
      return TupleProj(new_tuple, expr.index, type = expr.type)
    elif c is Cast:
      return Cast(self.transform_expr(expr.value), expr.type) 
    elif c is Alloc:
      return Alloc(count = self.transform_expr(expr.count), 
                   elt_type = expr.elt_type, 
                   type = expr.type)
    else:
      args = {}
      for member_name in expr.members():
        old_value = getattr(expr, member_name)
        new_value = self.transform_if_expr(old_value)
        args[member_name] = new_value
      return expr.__class__(**args)
  
  def transform_Assign(self, stmt):
    new_lhs = self.transform_expr(stmt.lhs)
    new_rhs = self.transform_expr(stmt.rhs)
    return Assign(new_lhs, new_rhs)

  def transform_ExprStmt(self, stmt):
    return ExprStmt(self.transform_expr(stmt.value))

  def transform_Return(self, stmt):
    res = Return(self.transform_expr(stmt.value))
    return res 
  
  def transform_If(self, stmt):
    new_true = self.transform_block(stmt.true)
    new_false = self.transform_block(stmt.false)
    new_merge = self.transform_merge(stmt.merge)
    new_cond = self.transform_expr(stmt.cond)
    return If(new_cond, new_true, new_false, new_merge)
  
  def transform_While(self, stmt):
    new_body = self.transform_block(stmt.body)
    new_merge = self.transform_merge(stmt.merge)
    new_cond = self.transform_expr(stmt.cond)
    return While(new_cond, new_body, new_merge)
  
  def transform_ForLoop(self, stmt):
    new_var = self.transform_expr(stmt.var)
    new_start = self.transform_expr(stmt.start)
    new_stop = self.transform_expr(stmt.stop)
    new_step = self.transform_expr(stmt.step)
    new_body = self.transform_block(stmt.body)
    new_merge = self.transform_merge(stmt.merge)
    return ForLoop(new_var, new_start, new_stop, new_step, new_body, new_merge)  
  
  def pre_apply(self, old_fn):
    new_fundef_args = dict([(m, getattr(old_fn, m)) for m in old_fn._members])
    # create a fresh function with a distinct name and the
    # transformed body and type environment
    if self.rename: 
      new_fundef_args['name'] = names.refresh(old_fn.name)
    else:
      new_fundef_args['name'] = old_fn.name
      new_fundef_args['version'] = old_fn.next_version(old_fn.name) 
    new_fundef_args['type_env'] = old_fn.type_env.copy()
    # don't need to set a new body block since we're assuming 
    # that transform_block will at least allocate a new list 
    new_fundef = syntax.TypedFn(**new_fundef_args)
    return new_fundef 

