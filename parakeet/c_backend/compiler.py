from treelike import NestedBlocks

from .. import names, prims 
from ..analysis import SyntaxVisitor
from ..syntax import Var, Const, TypedFn 
from ..ndtypes import (TupleT, ScalarT, ArrayT, 
                       elt_type) 

from boxing import box_scalar, unbox_scalar
from c_types import to_ctype, to_dtype

common_headers = """
#include <math.h>
#include <stdint.h>
#include <Python.h>
"""


_function_names = {}
_function_defs = {}
  
class Compiler(object):
   
  def __init__(self):
    self.blocks = []
    self.name_versions = {}
    self.name_mappings = {}

  
  def visit_expr(self, expr):
    expr_class_name = expr.__class__.__name__
    method_name = "visit_" + expr_class_name
    assert hasattr(self, method_name), "Unsupported expression %s" % expr_class_name  
    result = getattr(self, method_name)(expr)
    assert result is not None, "Compilation method for expression %s return None" % expr_class_name
    return result 
  
  def visit_expr_list(self, exprs):
    return [self.visit_expr(e) for e in exprs]
  
  def visit_stmt(self, stmt):
    stmt_class_name = stmt.__class__.__name__
    method_name = "visit_" + stmt_class_name
    assert hasattr(self, method_name), "Unsupported statemet %s" % stmt_class_name  
    result = getattr(self, method_name)(stmt)
    assert result is not None, "Compilation method for statement %s return None" % stmt_class_name
    return result 
  
  def push(self):
    self.blocks.append([])
  
  def pop(self):
    stmts = self.blocks.pop()
    return "\n".join("  " + stmt for stmt in stmts)
  
  def indent(self, block_str):
    return block_str.replace("\n", "\n  ")
  
  def append(self, stmt):
    self.blocks[-1].append(stmt)
  
  def fresh_name(self, prefix):
    prefix = names.original(prefix)
    prefix = prefix.replace(".", "_")
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
  
  def visit_Tuple(self, expr):
    elts = self.visit_expr_list(expr.elts)
    elt_str = ", ".join(elts) 
    n = len(expr.elts)
    return "(PyTupleObject*) (%d, %s)" % (n, elt_str)
  

  
  def visit_TupleProj(self, expr):
    t = expr.type 
    tup = self.visit_expr(expr.tuple)
    is_scalar = isinstance(t, ScalarT)
    cast_type = "PyObject*" if is_scalar else to_ctype(t) 
    proj_str = "(%s) PyTuple_GET_ITEM(%s, %d)" % (cast_type, tup, expr.index)
    if is_scalar:
      return unbox_scalar(proj_str, t)
    else:
      return proj_str 
  
  def visit_Index(self, expr):
    return "<INDEXING>: SHOULD BE REPLACED WITH FLATTENED CODE"
  
  def visit_Attribute(self, expr):
    attr = expr.name
    v = self.visit_expr(expr.value) 
    if attr == "data":
      return "PyArray_DATA (%s)" % v
    elif attr == "shape":
      return "PyArray_SHAPE(%s)" % v
    elif attr == "strides":
      return "PyArray_STRIDES(%s)" % v
    else:
      assert False, "Unsupported attribute %s" % attr 
      
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
      return "%(x)s ? %(x)s >= 0 : -%(x)s" % {'x': x}
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
    return "%s = %s;" % (lhs_name, rhs)
  
  def visit_merge_left(self, merge):
    pass 
  
  def visit_merge_right(self, merge):
    pass 
    
  def visit_ForLoop(self, stmt):
    start = self.visit_expr(stmt.start)
    stop = self.visit_expr(stmt.stop)
    step = self.visit_expr(stmt.step)
    var = self.visit_expr(stmt.var)
    t = to_ctype(stmt.var.type)
    
    body = self.indent("\n" + self.visit_block(stmt.body))
    s = "for (%(t)s %(var)s = %(start)s; %(var)s < %(stop)s; %(var)s += %(step)s) {%(body)s}"
    return s % locals()
  
  def visit_Return(self, stmt):
    return "return %s;" % self.visit_expr(stmt.value)
  
  def visit_block(self, stmts):
    self.push()
    for stmt in stmts:
      s = self.visit_stmt(stmt)
      self.append(s)
    self.append("\n")
    return self.pop()
      
  def visit_TypedFn(self, expr):
    return function_name(expr)

  def visit_UntypedFn(self, expr):
    assert False, "Unexpected UntypedFn %s in C backend, should have been specialized" % expr.name
     
  def visit_fn(self, fn):
    c_input_names = [self.name(argname) for argname in fn.arg_names]
    c_input_types = [to_ctype(t) for t in fn.input_types]
    input_str = ", ".join(ct + " " + n for (ct, n) in zip(c_input_types, c_input_names))
    c_return_type = to_ctype(fn.return_type)
    c_fn_name = self.fresh_name(fn.name)
    c_body = self.indent("\n" + self.visit_block(fn.body))
    fndef = "%(c_return_type)s %(c_fn_name)s (%(input_str)s) {%(c_body)s}" % locals()
    return c_fn_name, fndef 

def function_source(fn):
  key = fn.name, fn.copied_by 
  if key in _function_defs:
    return _function_defs[key]
  
  new_compiler = Compiler()
  name, src = new_compiler.visit_fn(fn)
  _function_names[key] = name
  _function_defs[key] = src
  return src

def function_name(fn):
  key = fn.name, fn.copied_by 
  if key in _function_names:
    return _function_names[key]
  
  new_compiler = Compiler()
  name, src = new_compiler.visit_fn(fn)
  _function_names[key] = name
  _function_defs[key] = src
  return name

 
    
  
    
    
    
    
    