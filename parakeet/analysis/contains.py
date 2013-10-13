from ..ndtypes import ScalarT, PtrT, NoneT, TupleT, FnT, ClosureT 
from .. syntax import Adverb, ParFor, Closure, UntypedFn, TypedFn 

from syntax_visitor import SyntaxVisitor

def contains_structs(fn):
  for t in fn.type_env.itervalues():
    if not isinstance(t, (ScalarT, PtrT, NoneT)):
      return True
  return False


class Yes(Exception):
  pass 

class ContainsSlices(SyntaxVisitor):
  def visit_Index(self, expr):
    if isinstance(expr.index.type, ScalarT):
      idx_types = [expr.index.type]
    elif isinstance(expr.index.type, TupleT):
      idx_types = expr.index.type.elt_types
    else: 
      raise Yes()
    if not all(isinstance(t, ScalarT) for t in idx_types):
      raise Yes()
  
def contains_slices(fn):
  try:
    ContainsSlices().visit_fn(fn)
    return False 
  except Yes:
    return True
  
class ContainsLoops(SyntaxVisitor):
  def visit_ForLoop(self, expr):
    raise Yes()
  
  def visit_While(self, expr):
    raise Yes()

def contains_loops(fn):
  try:
    ContainsLoops().visit_fn(fn)
    return False 
  except Yes:
    return True
  

class ContainsCalls(SyntaxVisitor):
  def visit_Call(self, expr):
    raise Yes()

def contains_calls(fn):
  try: 
    ContainsCalls().visit_fn(fn)
    return False 
  except:
    return True 

class ContainsAdverbs(SyntaxVisitor):
  def visit_expr(self, expr):
    if isinstance(expr, Adverb):
      raise Yes()
    SyntaxVisitor.visit_expr(self, expr)
  

def contains_adverbs(fn):
  try:
    ContainsAdverbs().visit_fn(fn)
    return False
  except Yes:
    return True

class ContainsParFor(SyntaxVisitor):
  def visit_stmt(self, stmt):
    if stmt.__class__ is ParFor:
      raise Yes()
    SyntaxVisitor.visit_stmt(self, stmt)

def contrains_parfor(fn):
  try:
    ContainsParFor.visit_fn(fn)
    return False 
  except Yes:
    return True 

class ContainsFunctions(SyntaxVisitor):
  def visit_expr(self, expr):
    if isinstance(expr, (UntypedFn, TypedFn, Closure)) or isinstance(expr.type, (FnT, ClosureT)):
      raise Yes()
    SyntaxVisitor.visit_expr(expr)
    
def contains_functions(fn):
  try:
    ContainsFunctions().visit_fn(fn)
    return False 
  except Yes:
    return True 
      