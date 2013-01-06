import config 

from core_types import ScalarT 
from syntax import Var, Attribute, Tuple 
from syntax_visitor import SyntaxVisitor

empty = set([])

def collect_nonscalar_names(expr):
  if expr is None or isinstance(expr.type, ScalarT):
    return []
  elif expr.__class__ is Var:
    return [expr.name]
  else:
    return collect_nonscalar_names_from_list(expr.children())

def collect_nonscalar_names_from_list(exprs):
  result = []
  for expr in exprs:
    result.extend(collect_nonscalar_names(expr))
  return result

class EscapeAnalysis(SyntaxVisitor):
  """
  A very imprecise combined escape and alias analysis. 
  Rough idea: whenever you assign some 
     x = expr(y,z)
  then take all the non-scalar values and unify them into 
  a single alias set along with all the previous variable
  names in the alias sets of y and z.  
  """
  
  def visit_fn(self, fn):
    self.scalars = set([])
    self.may_alias = {}
    all_scalars = True 
    # every name at least aliases it selfcollect_var_names
    for (name,t) in fn.type_env.iteritems():
      if isinstance(t, ScalarT):
        self.scalars.add(name)
      else:
        self.may_alias[name] = set([name])
        all_scalars = False 
    
    self.may_escape = set([])
    
    if all_scalars:
      return  
    
    self.visit_block(fn.body)
    
    # once we've accumulated all the aliases
    # anyone who points into the input data 
    # is also considered to be escaping 

    for name in fn.arg_names:
      if name not in self.scalars:
        self.may_escape.update(self.may_alias[name])
    
    if config.print_escape_analysis: 
      print "[EscapeAnalysis] In function %s (version %d)" % \
         (fn.name, fn.version) 
      print "-------"
      print fn 
      print 
      print "aliases"
      print "-------"
      for (k,aliases) in sorted(self.may_alias.items(), key = lambda (k,_): k):
        print "  %s => %s" % (k, aliases)
      print 
      print "escape set"
      print "----------"
      print sorted(self.may_escape)

  def mark_escape(self, name):
    if name not in self.scalars:
      for alias in self.may_alias[name]:
        self.may_escape.add(alias)

  def mark_escape_list(self, names):
    for name in names:
        self.mark_escape(name)
  
  def update_aliases(self, lhs_name, rhs_names):
    """
    Once we've gotten all the RHS names being assigned to the 
    LHS var, we do a recursively lookup into may_alias since
    we may have code like:
      a = tuple(b,c)
      d = tuple(a,e) 
    and we want the alias set of d to be {a,e,b,c}
    """
    if lhs_name not in self.scalars:
      combined_set = self.may_alias[lhs_name]
      for rhs_name in rhs_names:
        if rhs_name not in self.scalars:
          combined_set.update(self.may_alias[rhs_name])
      for alias in combined_set:
        self.may_alias[alias] = combined_set
      return combined_set  
  
  def update_escaped(self, lhs_name, rhs_alias_set):
    if lhs_name not in self.scalars and \
       any(alias in self.may_escape for alias in rhs_alias_set):
        self.may_escape.update(rhs_alias_set.difference(self.scalars))
        self.may_escape.add(lhs_name)
  
  def visit_Call(self, expr):
    self.mark_escape_list(collect_nonscalar_names_from_list(expr.args))

  def collect_lhs_names(self, expr):
    if expr.__class__ is Var:
      return [expr.name]
    elif expr.__class__ is Attribute:
      return self.collect_lhs_names(expr.value)
    elif expr.__class__ is Tuple:
      combined = []
      for elt in expr.elts:
        combined.extend(self.collect_lhs_names(elt))
    else:
      return []

  def visit_Assign(self, stmt):
    lhs_names = set(self.collect_lhs_names(stmt.lhs))
    rhs_names = set(collect_nonscalar_names(stmt.rhs))
    for lhs_name in lhs_names:
      self.update_aliases(lhs_name, rhs_names)

  def visit_Return(self, expr):
    self.mark_escape_list(collect_nonscalar_names(expr.value))

  def visit_merge(self, merge):
    for (name, (l,r)) in merge.iteritems():
      if l.__class__ is Var: 
        left_aliases = self.may_alias.get(l.name, empty)
      else:
        left_aliases = empty 
      if r.__class__ is Var: 
        right_aliases = self.may_alias.get(r.name, empty)
      else:
        right_aliases = empty
      combined_set = self.update_aliases(name, left_aliases.union(right_aliases))
      self.update_escaped(name, combined_set)

_cache = {}
def run(fundef):
  key = (fundef.name, fundef.copied_by, fundef.version)
  if key in _cache:
    return _cache[key]
  else: 
    analysis = EscapeAnalysis()
    analysis.visit_fn(fundef)
    _cache[key] = analysis
    return analysis

def may_alias(fundef):
  return run(fundef).may_alias 

def may_escape(fundef):
  return run(fundef).may_escape
