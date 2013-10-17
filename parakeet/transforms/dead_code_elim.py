

from .. import syntax
from .. analysis.collect_vars import collect_var_names_list
from .. analysis.use_analysis import use_count
from .. syntax import Assign, Const, Index, PrimCall, Tuple, TupleProj, Var, Closure, ClosureElt 
from transform import Transform


class DCE(Transform):
  def __init__(self):
    Transform.__init__(self, reverse = True)

  def pre_apply(self, fn):
    self.use_counts = use_count(fn)
    
  def is_live(self, name):
    count = self.use_counts.get(name)
    return count and count > 0

  def is_live_lhs(self, lhs):
    c = lhs.__class__
    if c is Var:
      return self.is_live(lhs.name)
    elif c is Tuple:
      return any(self.is_live_lhs(elt) for elt in lhs.elts)
    elif isinstance(lhs, (str, tuple)):
      assert False, "Raw data? This ain't the stone age, you know."
    else:
      return True

  def decref(self, expr):
    for var_name in collect_var_names_list(expr):
      self.use_counts[var_name] -= 1


  def transform_merge(self, phi_nodes):
    new_merge = {}
    for (var_name, (l,r)) in phi_nodes.iteritems():
      if self.is_live(var_name):
        new_merge[var_name] = (l,r)
      else:
        self.decref(l)
        self.decref(r)
    return new_merge

  
  def save_lhs_tuple(self, lhs):
    """
    If there's a Tuple assignment on the LHS 
    then all the variables must be kept alive
    together if any of them survive
    """
    for elt in lhs.elts:
      if elt.__class__ is Tuple:
        self.save_lhs_tuple(elt)
      else:
        assert elt.__class__ is Var 
        if not self.is_live(elt.name):
          self.use_counts[elt.name] = 1 
    
  def transform_Assign(self, stmt):
    if self.is_live_lhs(stmt.lhs):
      if stmt.lhs.__class__ is Tuple: 
        self.save_lhs_tuple(stmt.lhs)
      return stmt
    self.decref(stmt.rhs)

    return None

  def is_dead_loop(self, cond, body, merge):
    # pattern match to find loops which only
    # increment a counter without any visible effects
    if syntax.helpers.is_false(cond):
      return True

    rhs_counts = {}
    
    if cond.__class__ is not Var:
      return False
    rhs_counts[cond.name] = 1
 
    def process_rhs(expr):
      klass = expr.__class__
      if klass is Var:
        rhs_counts[expr.name] = rhs_counts.get(expr.name, 0) + 1
      elif klass is Const:
        pass
      elif klass is PrimCall:
        for arg in expr.args:
          process_rhs(arg)
      elif klass is Tuple:
        for elt in expr.elts:
          process_rhs(elt)
      elif klass is TupleProj:
        process_rhs(expr.tuple)
      elif klass is Index:
        process_rhs(expr.value)
        process_rhs(expr.index)
      elif klass is Closure: 
        process_rhs(expr.fn)
        for arg in expr.args:
          process_rhs(arg)
      elif klass is ClosureElt:
        process_rhs(expr.closure)
        process_rhs(expr.index)
      else:
        return False

    for stmt in body:
      # if statements are anything other than x = safe_expr then
      # there might be externally visible effects to this loop
      if stmt.__class__ is not Assign or stmt.lhs.__class__ is not Var:
        return False
      process_rhs(stmt.rhs)

    for output_name in merge.iterkeys():
      if self.use_counts[output_name] != rhs_counts.get(output_name, 0):
        return False
    return True

  def transform_While(self, stmt):
    new_merge = self.transform_merge(stmt.merge)
    new_body = self.transform_block(stmt.body)

    if self.is_dead_loop(stmt.cond, new_body, new_merge):
      return None
    stmt.body = new_body
    stmt.merge = new_merge
    return stmt

  def transform_If(self, stmt):
    cond = stmt.cond

    # Process the phi-merge first 
    # so that variables dead after the If 
    # statement can become dead inside it 
    stmt.merge = self.transform_merge(stmt.merge)

    stmt.true = self.transform_block(stmt.true)

    stmt.false = self.transform_block(stmt.false)

    if len(stmt.merge) == 0 and len(stmt.true) == 0 and \
        len(stmt.false) == 0:
      return None
    elif syntax.helpers.is_true(cond):
      for name, (v, _) in stmt.merge.iteritems():
        self.assign(Var(name, type = v.type), v)
      self.blocks.extend_current(reversed(stmt.true))
      return None
    elif syntax.helpers.is_false(cond):
      for name, (_, v) in stmt.merge.items():
        self.assign(Var(name, type = v.type), v)
      self.blocks.extend_current(reversed(stmt.false))
      return None
    return stmt

  def transform_Return(self, stmt):
    return stmt
  
  def transform_ExprStmt(self, stmt):
    if self.is_pure(stmt.value):
      return None 
    else:
      return stmt 

  def transform_ForLoop(self, stmt):
    stmt.merge = self.transform_merge(stmt.merge)
    stmt.body = self.transform_block(stmt.body)

    if len(stmt.body) > 0 or len(stmt.merge) > 0:
      return stmt

  def post_apply(self, fn):
    type_env = {}
    for (name,t) in fn.type_env.iteritems():
      if self.is_live(name):
        type_env[name] = t

    fn.type_env = type_env
    Transform.post_apply(self, fn)
    return fn

def dead_code_elim(fn):
  return DCE().apply(fn)
