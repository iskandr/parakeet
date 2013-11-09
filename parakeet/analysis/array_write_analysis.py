
from collect_vars import collect_var_names, collect_var_names_from_exprs
from escape_analysis import escape_analysis
from syntax_visitor import SyntaxVisitor


class ArrayWriteAnalysis(SyntaxVisitor):
  def __init__(self, fn, fresh_alloc_args = set([])):
    self.fn = fn 
    self.fresh_alloc_args = fresh_alloc_args
    
  def visit_fn(self, fn):
    escape_info = escape_analysis(fn, self.fresh_alloc_args)
    self.may_alias = escape_info.may_alias 
    SyntaxVisitor.visit_fn(self, fn)
    self.writes = set([])
    
  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Tuple:
      for elt in stmt.lhs.elts:
        self.visit_Assign(elt)
    elif stmt.lhs.__class__ is Index:
      for name in collect_var_names(stmt.lhs.value):
        self.writes.append(name)
        for alias_name in self.may_alias[name]:
          self.writes.append(alias_name)
      
    
  def visit_Index(self):
   
