
from ..analysis import collect_var_names
from ..builder import build_fn
from ..ndtypes import ArrayT 
from ..syntax import Var, ArrayExpr, Adverb, Range, Const, IndexMap 
from ..syntax.helpers import is_zero, is_one, is_false, is_true, is_none
from transform import Transform

class SimplifyArrayOperators(Transform):
  """
  Algebraic rewrite rules involving first-order array operations
  such as Range, ConstArray, etc.. 
  
  We do this conservatively by collecting all the array bindings
  and wiping them from the dictionary if there's even any chance they 
  might get modified
  """
  
  def pre_apply(self, fn):
    self.bindings = []
    return fn 
  
  def push(self):
    self.bindings.append({})
    
  def pop(self):
    self.bindings.pop()
    
  def mark_dirty(self, name):
    if isinstance(self.type_env[name], ArrayT):
      for d in self.bindings:
        if name in d:
          del d[name]
  def bind(self, name, v):
    self.bindings[-1][name] = v 
    
  def lookup(self, name):
    return self.bindings[-1].get(name)
  
  def transform_Attribute(self, expr):
    return expr
  
  def transform_Var(self, expr):
    """
    If we're using an array variable *anywhere* assume it might get modified
    """
    self.mark_dirty(expr.name)
    return expr  
  
  def transform_Map(self, expr):
    """
    Map(f, Range(start,stop)) = IndexMap(f', (stop - start))
    Map(f, NDIndex(shape)) = IndexMap(f, shape)
    """
     
    if len(expr.args) != 1:
      return expr 
    arg = expr.args[0]
    if arg.__class__ is not Var:
      return expr 
    name = arg.name
    prev = self.lookup(name)
    if prev is None:
      return expr 
    if prev.__class__ is Range:
      step = prev.step   
      if step.__class__ is not Const:
        return expr 
      if step.value is not None and step.value != 1:
        return expr 
      diff = self.sub(prev.stop, prev.start, "niters")
      return IndexMap(fn = expr.fn, shape = diff, start_index = prev.start, type = expr.type) 
    return expr 
      
  def transform_Assign(self, stmt):
    
    if isinstance(stmt.lhs, Var):
      if isinstance(stmt.rhs, (ArrayExpr, Adverb)):
        self.bind(stmt.lhs.name, stmt.rhs)
      stmt.rhs = self.transform_expr(stmt.rhs)
    else:
      for name in collect_var_names(stmt.lhs):
        self.mark_dirty(name)
      for name in collect_var_names(stmt.rhs):
        self.mark_dirty(name)
    return stmt 
  
  def transform_block(self, stmts):
    self.push()
    stmts = Transform.transform_block(self, stmts)
    self.pop()
    return stmts 