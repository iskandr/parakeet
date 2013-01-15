import syntax_helpers

from collect_vars import collect_var_names_list
from syntax import Assign, Const, Index, PrimCall, Tuple, TupleProj, Var
from transform import Transform
from use_analysis import use_count

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

  def transform_Assign(self, stmt):
    if self.is_live_lhs(stmt.lhs):
      return stmt
    self.decref(stmt.rhs)
    return None

  def is_dead_loop(self, cond, body, merge):
    # pattern match to find loops which only
    # increment a counter without any visible effects
    if syntax_helpers.is_false(cond):
      return True

    rhs_counts = {}
    if cond.__class__ is not Var:
      return False
    rhs_counts[cond.name] = 1
    def process_rhs(expr):
      if expr.__class__ is Var:
        rhs_counts[expr.name] = rhs_counts.get(expr.name, 0) + 1
      elif expr.__class__ is Const:
        pass
      elif expr.__class__ is PrimCall:
        for arg in expr.args:
          process_rhs(arg)
      elif expr.__class__ is Tuple:
        for elt in expr.elts:
          process_rhs(elt)
      elif expr.__class__ is TupleProj:
        process_rhs(expr.tuple)
      elif expr.__class__ is Index:
        process_rhs(expr.value)
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
    # expressions don't get changed by this transform
    new_body = self.transform_block(stmt.body)
    new_merge = self.transform_merge(stmt.merge)
    if self.is_dead_loop(stmt.cond, new_body, new_merge):
      return None
    stmt.body = new_body
    stmt.merge = new_merge
    return stmt

  def transform_If(self, stmt):
    cond = stmt.cond
    stmt.true = self.transform_block(stmt.true)

    stmt.false = self.transform_block(stmt.false)

    new_merge = self.transform_merge(stmt.merge)

    if len(new_merge) == 0 and len(stmt.true) == 0 and \
        len(stmt.false) == 0:
      return None
    elif syntax_helpers.is_true(cond):
      for name, (_, v) in new_merge.iteritems():
        self.assign(Var(name, type = v.type), v)
      self.blocks.extend_current(reversed(stmt.true))
      return None
    elif syntax_helpers.is_false(cond):
      for name, (v, _) in new_merge.items():
        self.assign(Var(name, type = v.type), v)
      self.blocks.extend_current(reversed(stmt.false))
      return None
    return stmt

  def transform_Return(self, stmt):
    return stmt

  def transform_ForLoop(self, stmt):
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge(stmt.merge)
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
