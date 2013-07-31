from treelike import NestedBlocks

from .. import names, prims 
from ..analysis import SyntaxVisitor
from ..syntax import Var, Const, TypedFn 
from ..ndtypes import (TupleT, ScalarT, ArrayT, 
                       elt_type) 

from type_converters import to_ctype, to_dtype

common_headers = """
#include <stdint.h>
#include <Python.h>
"""

class Compiler(SyntaxVisitor):
  function_defs = {}
  function_names = {}
   
  def __init__(self):
    self.blocks = []
    self.name_versions = {}
    self.name_mappings = {}

  
  def push(self):
    self.blocks.append([])
  
  def pop(self):
    stmts = self.blocks.pop()
    return "\n".join("  " + stmt for stmt in stmts)
  
  def indent(self, block_str):
    return block_str.replace("\n", "\n  ")
  
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
  
  def visit_Cast(self, expr):
    x = self.visit_expr(expr.value)
    ct = to_ctype(expr.type)
    if isinstance(expr, (Const, Var)):
      return "(%s) %s" % (ct, x)
    else:
      return "((%s) (%s))" % (ct, x)
  
  def visit_PrimCall(self, expr):
    args = self.visit_expr_list(expr.args)
    p = expr.prim 
    if p == prims.add:
      return "%s + %s" % (args[0], args[1])
    if p == prims.subtract:
      return "%s + %s" % (args[0], args[1])
    elif p == prims.multiply:
      return "%s * %s" % (args[0], args[1])
    elif p == prims.divide:
      return "%s / %s" % (args[0], args[1])
    elif p == prims.abs:
      x  = args[0]
      return "%{x}s ? %{x}s >= 0 : -%{x}s" % {'x': x}
    elif p == prims.bitwise_and:
      return "%s & %s" % (args[0], args[1])
    elif p == prims.bitwise_or:
      return "%s | %s" % (args[0], args[1])
    elif p == prims.bitwise_or:
      return "%s | %s" % (args[0], args[1])
     
  def visit_Assign(self, stmt):
    assert stmt.lhs.__class__ is Var
    lhs_name = self.name(stmt.lhs.name)
    rhs = self.visit_expr(stmt.rhs)
    self.stmt("%s = %s" % (lhs_name, rhs))
    
  def visit_block(self, stmts):
    self.push()
    for stmt in stmts:
      self.visit_stmt(stmt)
    return self.pop()
  
  def visit_ForLoop(self, stmt):
    start = self.visit_expr(stmt.start)
    stop = self.visit_expr(stmt.stop)
    step = self.visit_expr(stmt.step)
    var = self.visit_expr(stmt.var)
    vartype = to_ctype(stmt.var.type)
    body = self.indent("\n" + self.visit_block(stmt.body)) 
    return "for (%s %s = %s; %s < %s; %s += %s) {%s}" % \
      (vartype, var, start, var, stop, var, step, body)
      
  def visit_TypedFn(self, expr):
    name = expr.name 
    if name not in self.function_names:
      new_compiler = Compiler()
      new_compiler.visit_fn(expr)
    assert name in self.function_names
    return self.function_names[name]
  
  def visit_UntypedFn(self, expr):
    assert False, "Unexpected UntypedFn %s in C backend, should have been specialized" % expr.name
     
  def visit_fn(self, fn):
    c_input_names = [self.name(argname) for argname in fn.arg_names]
    c_input_types = [to_ctype(t) for t in fn.input_names]
    input_str = ", ".join(ct + " " + n for (ct, n) in zip(c_input_types, c_input_names))
    c_return_type = to_ctype(fn.return_type)
    c_fn_name = self.name(fn.name)
    self.function_names[fn.name] = c_fn_name
    c_body = self.indent("\n" + self.visit_block(fn.body))
    fndef = "%{c_return_type}s %{c_fn_name}s (%{input_str}s) {%s}" % locals()
    self.function_defs[fn.name] = fndef 
    return fndef 
    
    
    
    
    
    