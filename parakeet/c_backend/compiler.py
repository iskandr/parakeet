from treelike import NestedBlocks

from .. import names 
from ..analysis import SyntaxVisitor
from ..syntax import Var
from ..ndtypes import (Int8, Int16, Int32, Int64, Float32, Float64, 
                       TupleT, ScalarT, ArrayT, 
                       elt_type) 

from type_converters import to_ctype, to_dtype

class Compiler(SyntaxVisitor):
  
  def __init__(self):
    self.blocks = []
    self.name_versions = {}
    self.name_mappings = {}
  
  def push(self):
    self.blocks.append([])
  
  def pop(self):
    stmts = self.blocks.pop()
    return "\n".join("  " + stmt for stmt in stmts)
  
  def append(self, stmt):
    self.blocks[0].append(stmt)
  
  def stmt(self, s):
    self.append("%s;" % s)
  
  def fresh_name(self, prefix):
    version = self.name_versions.get(prefix, 1)
    if version == 1:
      return prefix 
    else:
      return "%s_%d" % (prefix, version)
  
  def name(self, ssa_name):
    """
    Convert from ssa names, which might have large version numbers and contain 
    syntactically invalid characters to valid local C names
    """
    if ssa_name in self.name_mappings:
      return self.name_mappings[ssa_name]
    prefix = names.original(ssa_name)
    prefix = prefix.replace(".", "_")
    name = self.fresh_name(prefix)
    self.name_mappings[ssa_name] = name 
    return name 
  
  def tuple_to_stack_array(self, expr):
    assert expr.type.__class__ is TupleT 
    t0 = expr.type.elt_types
    assert all(t == t0 for t in expr.type.elt_types[1:])
    array_name = self.fresh_name("array_from_tuple")
    n = len(expr.type.elt_types)
    self.stmt("%s %s[%d]" % (to_ctype(t0), array_name, n))
    
  def visit_AllocArray(self, expr):
    shape = self.tuple_to_stack_array(expr.shape)
    t = to_dtype(elt_type(expr.type))
    return "PyArray_SimpleNew(%d, %s, %s)" % (expr.type.rank, shape, t)
  
  def visit_Const(self, expr):
    return "%s" % expr.value 
  
  def visit_Var(self, expr):
    return self.name(expr.name)
  
  def visit_Assign(self, stmt):
    assert stmt.lhs.__class__ is Var
    lhs_name = self.name(stmt.lhs.name)
    rhs = self.visit_expr(stmt.rhs)
    self.stmt("%s = %s" % (lhs_name, rhs))
    
  def visit_PrimCall(self, expr):