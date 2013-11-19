from .. import names 
from .. syntax import (TypedFn, Var, Const, Attribute, Index, PrimCall, 
                       If, Assign, While, ExprStmt, Return, ForLoop, ParFor,  
                       Slice, Struct, Tuple, TupleProj, Cast, Alloc, Closure, 
                       Map, Reduce, Scan, IndexMap, IndexReduce, IndexScan, 
                       UntypedFn )      
from transform import Transform 

class CloneFunction(Transform):
  """
  Copy all the objects in the AST of a function
  """
  def __init__(self, parent_transform = None, 
                      rename = False):
    Transform.__init__(self)
    self.rename = rename 
    # self.recursive = recursive
    self.parent_transform = parent_transform 
     
  def transform_Var(self, expr):
    return Var(expr.name, type = expr.type)
  
  def transform_expr(self, expr):
    new_expr = self._transform_expr(expr)
    new_expr.type = expr.type 
    new_expr.source_info = expr.source_info 
    return new_expr 
  
  def _transform_expr(self, expr):
    c = expr.__class__
    if c is Var:
      return Var(expr.name)
      
    elif c is Const:
      return Const(expr.value)
    
    elif c is Tuple:
      new_elts = tuple(self.transform_expr(elt) for elt in expr.elts)
      return Tuple(elts = new_elts)
    
    elif c is Attribute:
      value = self.transform_expr(expr.value)
      return Attribute(value, expr.name)
    
    elif c is Index:
      value = self.transform_expr(expr.value)
      index = self.transform_expr(expr.index)
      return Index(value, index)
    
    elif c is PrimCall:
      args = tuple(self.transform_expr(elt) for elt in expr.args)
      return PrimCall(expr.prim, args)
    
    elif c is TypedFn:
      #if self.recursive:
      #  cloner = CloneFunction(rename = self.rename, recursive = True)
      #  return cloner.apply(expr)
      #else:
      return expr 
    elif c is UntypedFn:
      return expr 
    elif c is Closure: 
      args = self.transform_expr_tuple(expr.args)
      return Closure(fn = expr.fn, args = args)
    
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
                   elt_type = expr.elt_type)
    elif c is Map:
      return Map(fn = self.transform_expr(expr.fn), 
                  args = self.transform_expr_list(expr.args), 
                  axis = self.transform_if_expr(expr.axis), 
                 )
    elif c is Reduce: 
      return Reduce(fn = self.transform_expr(expr.fn), 
                    combine = self.transform_expr(expr.combine), 
                    args = self.transform_expr_list(expr.args), 
                    axis = self.transform_if_expr(expr.axis), 
                    init = self.transform_if_expr(expr.init), 
                  )
   
    else:
      args = {}  
      for k,v in expr.__dict__.iteritems():
        args[k] = self.transform_if_expr(v)
      return c(**args)
  
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
  
  def transform_ParFor(self, stmt):
    new_bounds = self.transform_expr(stmt.bounds)
    new_fn = self.transform_expr(stmt.fn)
    return ParFor(fn = new_fn, bounds = new_bounds)
  
  def pre_apply(self, old_fn):
    new_fundef_args = old_fn.__dict__.copy()
    del new_fundef_args['type']
    # new_fundef_args = dict([(m, getattr(old_fn, m)) for m in old_fn._members])
    # create a fresh function with a distinct name and the
    # transformed body and type environment
    if self.rename: 
      new_fundef_args['name'] = names.refresh(old_fn.name)
    else:
      new_fundef_args['name'] = old_fn.name
       
    new_fundef_args['type_env'] = old_fn.type_env.copy()
    new_fundef_args['transform_history'] = old_fn.transform_history.copy()
    new_fundef_args['created_by'] = self.parent_transform
    # don't need to set a new body block since we're assuming 
    # that transform_block will at least allocate a new list 
    new_fundef = TypedFn(**new_fundef_args)
    return new_fundef 

