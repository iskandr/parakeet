
from syntax import Var, Tuple 
from transform import Transform
import syntax_helpers  
from use_analysis import use_count
from collect_vars import collect_var_names_list


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
  
  def transform_While(self, stmt):
    # expressions don't get changed by this transform
    new_body = self.transform_block(stmt.body) 
    new_merge = self.transform_merge(stmt.merge)
    if len(new_merge) == 0 and len(new_body) == 0:
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
  