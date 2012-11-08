import transform
import syntax 
import adverbs

def var_names(expr_list):
  return [e.name for e in expr_list if isinstance(e, syntax.Var)]

def can_fuse(map1, map1_result_names, map2):
  return map1.axis == map2.axis and \
    all(n in map1_result_names for n in var_names(map2.args))   

def flatten_lhs(lhs):
  if isinstance(lhs, syntax.Tuple):
    result = []
    for elt in lhs.elts:
      result.extend(flatten_lhs(elt))
    return result
  else:
    return [lhs]    

class FusionCandidate(object):
  
  def __init__(self, pred_var, pred_map, succ_var, succ_adverb):
    self.pred_var = pred_var
    self.pred_map = pred_map 
    self.succ_var = succ_var
    self.succ_adverb = succ_adverb  
 
  def fuse(self):
    pass 
   
class ApplyFusion(transform.Transform):
  
  def __init__(self):
    pass 
     
 
class Fusion(transform.Transform):
  def __init__(self):
    # map names of variables to their adverb expression 
    self.maps = {}
    
    # map each variable to 
    self.use_counts = {}
    
    # map each variable name to a pair of adverbs
    self.fusion_candidates = {}
    
  
  def transform_Var(self, expr):
    last_count = self.use_counts.get(expr.name, 0)
    self.use_counts[expr.name] = last_count + 1
    return expr 
    
  def transform_Assign(self, stmt):
    rhs = self.transform_expr(stmt.rhs)
    lhs_elts = flatten_lhs(stmt.lhs)
    if all(isinstance(lhs_elt, syntax.Var) for lhs_elt in lhs_elts):
      if len(lhs_elts) == 1:
        lhs_var = lhs_elts[0].name 
        if isinstance(rhs, adverbs.Map):
          self.maps[lhs_var] = rhs 
        used_vars = var_names(rhs.args)
        if len(used_vars) == 1:
          used_var = used_vars[0]
          if used_var in self.maps:
            pred_map = self.maps[used_var]
            candidate = FusionCandidate(used_var, pred_map, lhs_var, rhs)
            self.candidates[lhs_var] = candidate
            # how do I mark that the pred var might get used by a fusion candidate? 
            # or just rely on dead code elim to clean up after?
            # do the latter for now, by creating a live_vars set 
    return stmt 