
from .. import config
from .. ndtypes import ScalarT, ArrayT, PtrT, TupleT, ClosureT, FnT, SliceT, NoneT
from .. syntax import Var, Attribute, Tuple 
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
  
  def __init__(self, fresh_alloc_args = set([])):
    self.fresh_alloc_args = fresh_alloc_args
    self.immutable = set([])
    self.may_alias = {}
    self.may_escape = set([])
    self.may_return = set([])
    
  
  def nested_mutable_types(self, t):
    if isinstance(t, (PtrT, ArrayT)):
      return set([t])
    elif isinstance(t, TupleT):
      result = set([])
      for elt_t in t.elt_types:
        result.update(self.nested_mutable_types(elt_t))
      return result 
    elif isinstance(t, ClosureT):
      result = set([])
      for elt_t in t.arg_types:
        result.update(self.nested_mutable_types(elt_t))
      return result
    else:
      return set([])
    
  def immutable_type(self, t):
    return isinstance(t, (ScalarT, NoneT, SliceT, FnT)) or len(self.nested_mutable_types(t)) == 0
  
  def immutable_name(self, name):
    if name in self.immutable:
      return True 
    return self.immutable_type(self.type_env.get(name))
  
  def visit_fn(self, fn):

    all_scalars = True 
    self.type_env = fn.type_env 
    # every name at least aliases it self
    for (name,t) in fn.type_env.iteritems():

      if self.immutable_type(t):
        self.immutable.add(name)
      else:
        self.may_alias[name] = set([name])
        all_scalars = False

    if all_scalars:
      return  
    
    # every arg also aliases at least all the other input of the same type 
    # unless we were told it's a freshly allocated input
    
      
    reverse_type_mapping = {}
      
    for name in fn.arg_names:
      if name not in self.fresh_alloc_args:
        t = fn.type_env[name]
        for nested_t in self.nested_mutable_types(t):
          reverse_type_mapping.setdefault(nested_t, set([])).add(name)

      
    for name in fn.arg_names:
      # 
      # Every input argument should alias the other inputs of the same type 
      # if they are both arrays or tuples which contain arrays 
      # We must exclude, however, arguments which we're told explicitly were
      # freshly allocated before we entered the function 
      if name not in self.fresh_alloc_args:
        t = fn.type_env[name]
        for nested_t in self.nested_mutable_types(t):
          self.may_alias[name].update(reverse_type_mapping[nested_t])

    self.visit_block(fn.body)
    
    # once we've accumulated all the aliases
    # anyone who points into the input data 
    # is also considered to be escaping 
    for name in fn.arg_names:
      if name not in self.immutable and name not in self.fresh_alloc_args:
        self.may_escape.update(self.may_alias[name])
    
    if config.print_escape_analysis: 
      print "[EscapeAnalysis] In function %s" % (fn.cache_key, )
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
    if name not in self.immutable:
      for alias in self.may_alias[name]:
        self.may_escape.add(alias)
  
  def mark_escape_list(self, names):
    for name in names:
        self.mark_escape(name)
  
  
  def mark_return(self, name):
    if name not in self.immutable:
      for alias in self.may_alias[name]:
        self.may_return.add(alias)
  
  
  def mark_return_list(self, names):
    for name in names:
        self.mark_return(name)
  
  def update_aliases(self, lhs_name, rhs_names):
    """
    Once we've gotten all the RHS names being assigned to the 
    LHS var, we do a recursively lookup into may_alias since
    we may have code like:
      a = tuple(b,c)
      d = tuple(a,e) 
    and we want the alias set of d to be {a,e,b,c}
    """
    if lhs_name not in self.immutable:
      combined_set = self.may_alias[lhs_name]
      for rhs_name in rhs_names:
        if not self.immutable_name(rhs_name):
          combined_set.update(self.may_alias[rhs_name])
      for alias in combined_set:
        self.may_alias[alias] = combined_set
      return combined_set  
  
  def update_escaped(self, lhs_name, rhs_alias_set):
    if lhs_name not in self.immutable and \
       any(alias in self.may_escape for alias in rhs_alias_set):
        self.may_escape.update(rhs_alias_set.difference(self.immutable))
        self.may_escape.add(lhs_name)
        
  def update_return(self, lhs_name, rhs_alias_set):
    if lhs_name not in self.immutable and \
       any(alias in self.may_return for alias in rhs_alias_set):
        self.may_return.update(rhs_alias_set.difference(self.immutable))
        self.may_return.add(lhs_name)
    
  
  def visit_Call(self, expr):
    self.visit_expr(expr.fn)
    self.mark_escape_list(collect_nonscalar_names_from_list(expr.args))

  def visit_Map(self, expr):
    self.visit_expr(expr.fn)
    self.visit_if_expr(expr.axis)
    self.mark_escape_list(collect_nonscalar_names_from_list(expr.args))

  def visit_Reduce(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr(expr.combine)
    self.visit_if_expr(expr.axis)
    self.visit_if_expr(expr.init)
    self.mark_escape_list(collect_nonscalar_names_from_list(expr.args))

  def visit_Scan(self, expr):
    self.visit_expr(expr.fn)
    self.visit_expr(expr.combine)
    self.visit_if_expr(expr.axis)
    self.visit_if_expr(expr.init)
    self.mark_escape_list(collect_nonscalar_names_from_list(expr.args))
  
  def visit_OuterMap(self, expr):
    self.visit_expr(expr.fn)
    self.visit_if_expr(expr.axis)
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
  
  def visit_Closure(self, expr):
    self.mark_escape_list(collect_nonscalar_names_from_list(expr.args))
  
  def visit_Assign(self, stmt):
    lhs_names = set(self.collect_lhs_names(stmt.lhs))
    rhs_names = set(collect_nonscalar_names(stmt.rhs))
 
    for lhs_name in lhs_names:
      self.update_aliases(lhs_name, rhs_names)

  def visit_Return(self, stmt):
    arrays = collect_nonscalar_names(stmt.value)
    self.mark_escape_list(arrays)
    self.mark_return_list(arrays)
 
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
      self.update_return(name, combined_set)

_cache = {}
def escape_analysis(fundef, fresh_alloc_args = set([])):
  key = fundef.cache_key, frozenset(fresh_alloc_args)
  if key in _cache:
    return _cache[key]
  else: 
    analysis = EscapeAnalysis(fresh_alloc_args = fresh_alloc_args)
    analysis.visit_fn(fundef)
    _cache[key] = analysis
    return analysis

def may_alias(fundef):
  return escape_analysis(fundef).may_alias 

def may_escape(fundef):
  return escape_analysis(fundef).may_escape


# TODO: 
# actually generate all this info! 
from collections import namedtuple 
FunctionInfo = namedtuple("FunctionInfo", 
  ("pure", # nothing in this function will ever write to any array's data
   "allocates",        # does this function allocate new arrays?
   "unused", # which local variables never get used?
   
   "may_alias",       # alias relationships between local variables
   "may_escape",      # which variables may alias a returned array's data?
    
   "may_read_arrays",  # which arrays may have their data read in this function? 
   "may_write_arrays",   # which arrays may have their data written to in this function?
   "always_returns_fresh_array", # is the array returned always something locally allocated?
  ))

