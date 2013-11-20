from ..ndtypes import ScalarT, PtrT, NoneT, TupleT, FnT, ClosureT 
from .. syntax import Adverb, ParFor, Closure, UntypedFn, TypedFn, ArrayExpr  

from syntax_visitor import SyntaxVisitor

def memoize(analyzer):
  _cache = {}
  def memoized(fn_arg): 
    key = analyzer, fn_arg.cache_key
    if key in _cache:
      return _cache[key]
    result = analyzer(fn_arg)
    _cache[key] = result 
    return result 
  return memoized  
  

@memoize 
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

@memoize 
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
  
  def visit_TypedFn(self, expr):
    if contains_calls(expr):
      raise Yes()
  
@memoize 
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
  
  def visit_TypedFn(self, expr):
    if contains_adverbs(expr):
      raise Yes()
  
@memoize 
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
  
  
  def visit_TypedFn(self, expr):
    if contains_parfor(expr):
      raise Yes()

@memoize 
def contains_parfor(fn):
  try:
    ContainsParFor().visit_fn(fn)
    return False 
  except Yes:
    return True 

class ContainsFunctions(SyntaxVisitor):
  def visit_expr(self, expr):
    if isinstance(expr, (UntypedFn, TypedFn, Closure)) or isinstance(expr.type, (FnT, ClosureT)):
      raise Yes()
    SyntaxVisitor.visit_expr(expr)

@memoize 
def contains_functions(fn):
  try:
    ContainsFunctions().visit_fn(fn)
    return False 
  except Yes:
    return True 

class ContainsAlloc(SyntaxVisitor):
  def visit_Alloc(self, _):
    raise Yes()
  
  def visit_AllocArray(self, _):
    raise Yes()
  
  def visit_TypedFn(self, fn):
    if contains_alloc(fn):
      raise Yes()  
      
@memoize 
def contains_alloc(fn):
  try:
    ContainsAlloc().visit_fn(fn) 
    return False 
  except Yes:
    return True

class ContainsArrayOperators(SyntaxVisitor):
  def visit_expr(self, expr):
    """
    We're assuming that array expressions only occur on RHS of assignments
    and aren't nested. It's up to Simplify to de-nest them
    """
    if isinstance(expr, ArrayExpr):
      raise Yes()
  
@memoize 
def contains_array_operators(fn):
  try:
    ContainsArrayOperators().visit_fn(fn) 
    return False 
  except Yes:
    return True        