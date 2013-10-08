from collections import namedtuple
 
from .. import prims, names 
from ..analysis import use_count
from ..syntax import (Const, Tuple, TypedFn, Var, TupleProj, ArrayView, 
                      PrimCall, Attribute, Expr, Closure)  
from ..syntax.helpers import get_types 
from ..ndtypes import (TupleT,  ArrayT, NoneT, 
                       elt_type, ScalarT, 
                       FloatT, Float32, Float64, 
                       BoolT, Bool, 
                       IntT,  Int64, SignedT,
                       PtrT, NoneType, 
                       FnT, ClosureT, Type) 

import type_mappings
from base_compiler import BaseCompiler

from compile_util import compile_module
import config 

CompiledFlatFn = namedtuple("CompiledFlatFn", 
                            ("name", "sig", "src",
                             "extra_objects",
                             "extra_functions",
                             "extra_function_signatures", 
                             "declarations"))


def compile_flat_source(fn, _struct_type_cache = {}, _compile_cache = {}):
  # make sure compiled source uses consistent names for tuple types 
  relevant_types = set(t for t in fn.type_env.itervalues() if isinstance(t, TupleT))
  declared_tuples = set(t for t in _struct_type_cache.iterkeys() if t in relevant_types)
  
  key = fn.cache_key, frozenset(declared_tuples)
  
  if key in _compile_cache:
    return _compile_cache[key]
  
  compiler = FlatFnCompiler(struct_type_cache = _struct_type_cache)
  name, sig, src = compiler.visit_fn(fn)
  result = CompiledFlatFn(name = name, 
                          sig = sig, 
                          src = src,
                          extra_objects = compiler.extra_objects, 
                          extra_functions = compiler.extra_functions,
                          extra_function_signatures = compiler.extra_function_signatures,
                          declarations = compiler.declarations)
  _compile_cache[key] = result
  return result


def compile_entry(fn, _compile_cache = {}):
  key = fn.cache_key
  if key in _compile_cache:
    return _compile_cache[key]
  compiler = PyModuleCompiler()
  name, sig, src = compiler.visit_fn(fn)
  if config.print_function_source: 
    print "Generated C source for %s: %s" %(name, src)
  ordered_function_sources = [compiler.extra_functions[extra_sig] for 
                              extra_sig in compiler.extra_function_signatures]

  compiled_fn = compile_module(src, 
                               fn_name = name,
                               fn_signature = sig, 
                               extra_objects = set(compiler.extra_objects),
                               extra_function_sources = ordered_function_sources, 
                               declarations =  compiler.declarations, 
                               print_source = config.print_module_source)
  _compile_cache[key]  = compiled_fn
  return compiled_fn


def entry_function_source(fn):
  return compile_entry(fn).src 

def entry_function_name(fn):
  return compile_entry(fn).fn_name 

def entry_function_signature(fn):
  return compile_entry(fn).fn_signature 

class FlatFnCompiler(BaseCompiler):
  
  def __init__(self, struct_type_cache = None):
    BaseCompiler.__init__(self)
    
    
    self.declarations = []
    
    # depends on these .o files
    self.extra_objects = set([]) 
    
    # to avoid adding the same function's source twice 
    # we use its signature as a key  
    self.extra_functions = {}
    self.extra_function_signatures = []
    
    if struct_type_cache is None:
      # don't create more than one struct type per tuple type
      self._tuple_struct_cache = {}
    else:
      self._tuple_struct_cache = struct_type_cache
  
  def add_decl(self, decl):
    if decl not in self.declarations:
      self.declarations.append(decl)
  
  def struct_type_from_fields(self, field_types):
    
    if any(not isinstance(t, str) for t in field_types):
      field_types = tuple(self.to_ctype(t) if isinstance(t, Type) else t 
                          for t in field_types)
    else:
      field_types = tuple(field_types)
    
    if field_types in self._tuple_struct_cache:
      return self._tuple_struct_cache[field_types]
    
    typename = names.fresh("tuple_type").replace(".", "_")

    field_decls = ["  %s %s;" % (t, "elt%d" % i) for i,t in enumerate(field_types)]
    decl = "typedef struct %s {\n%s\n} %s;" % (typename, "\n".join(field_decls), typename)
    self.add_decl(decl)
    self._tuple_struct_cache[field_types] = typename
    return typename 
  
  
  def to_ctypes(self, ts):
    return tuple(self.to_ctype(t) for t in ts)
  
  def to_ctype(self, t):
    if isinstance(t, TupleT):
      elt_types = self.to_ctypes(t.elt_types)
      return self.struct_type_from_fields(elt_types) 
    else:
      return type_mappings.to_ctype(t)
    
  def visit_Alloc(self, expr):
    elt_t =  expr.elt_type
    nelts = self.fresh_var("npy_intp", "nelts", self.visit_expr(expr.count))

    return "(PyArrayObject*) PyArray_SimpleNew(1, &%s, %s)" % (nelts, type_mappings.to_dtype(elt_t))
  
  
  
  def visit_Const(self, expr):
    if isinstance(expr.type, BoolT):
      return "1" if expr.value else "0"
    elif isinstance(expr.type, NoneT):
      return "0"
    return "%s" % expr.value 
  
  def visit_Var(self, expr):
    return self.name(expr.name)
  
  def visit_Cast(self, expr):
    x = self.visit_expr(expr.value)
    ct = self.to_ctype(expr.type)
    if isinstance(expr, (Const, Var)):
      return "(%s) %s" % (ct, x)
    else:
      return "((%s) (%s))" % (ct, x)
  
  
  def not_(self, x):
    if x == "1":
      return "0"
    elif x == "0":
      return "1"
    return "!%s" % x
  
  def and_(self, x, y):
    if x == "0" or y == "0":
      return "0"
    elif x == "1" and y == "1":
      return "1"
    elif x == "1":
      return y 
    elif y == "1":
      return x
    return "%s && %s" % (x,y) 
  
  def or_(self, x, y):
    if x == "1" or y == "1":
      return "1"
    elif x == "0":
      return y
    elif y == "0":
      return x 
    return "%s || %s" % (x,y) 
  
  def gt(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "0"
    return "%s > %s" % (x, y)
  
  def gte(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "1"
    return "%s >= %s" % (x,y) 
  
  def lt(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "0"
    return "%s < %s" % (x,y)
  
  def lte(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "1"
    return "%s <= %s" % (x, y) 
  
  def neq(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "0"
    return "%s != %s" % (x, y) 
  
  def eq(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "1"
    return "%s == %s" % (x, y)
  
  
  
  def visit_PrimCall(self, expr):
    t = expr.type
    args = self.visit_expr_list(expr.args)
    
    # parenthesize any compound expressions 
    for i, arg_expr in enumerate(expr.args):
      if not isinstance(arg_expr, (Var, Const)):
        args[i] = "(" + args[i] + ")"
        
    p = expr.prim 
    if p == prims.add:
      return "%s + %s" % (args[0], args[1])
    if p == prims.subtract:
      return "%s - %s" % (args[0], args[1])
    elif p == prims.multiply:
      return "%s * %s" % (args[0], args[1])
    elif p == prims.divide:
      return "%s / %s" % (args[0], args[1])
    elif p == prims.negative:
      if t == Bool:
        return "1 - %s" % args[0]
      else:
        return "-%s" % args[0]
    elif p == prims.abs:
      x  = args[0]
      return " %s >= 0 ? %s  : -%s" % (x,x,x)
    
    elif p == prims.bitwise_and:
      return "%s & %s" % (args[0], args[1])
    elif p == prims.bitwise_or:
      return "%s | %s" % (args[0], args[1])
    elif p == prims.bitwise_not:
      return "~%s" % args[0]
    
    elif p == prims.logical_and:
      return self.and_(args[0], args[1])
      
    elif p == prims.logical_or:
      return self.or_(args[0], args[1])
    
    elif p == prims.logical_not:
      return self.not_(args[0])
      
    elif p == prims.equal:
      return self.eq(args[0], args[1], t)
    
    elif p == prims.not_equal:
      return self.neq(args[0], args[1], t)
    
    elif p == prims.greater:
      return self.gt(args[0], args[1], t)
      
    elif p == prims.greater_equal:
      return self.gte(args[0], args[1], t)
    
    elif p == prims.less:
      return self.lt(args[0], args[1], t)
    
    elif p == prims.less_equal:
      return self.lte(args[0], args[1], t)
    
    elif p == prims.remainder:
      x,y = args
      if t == Float32: return "fmod(%s, %s)" % (x,y)
      elif t == Float64: return "fmod(%s, %s)" % (x,y)
      assert isinstance(t, (BoolT, IntT)), "Modulo not implemented for %s" % t
      rem = self.fresh_var(t, "rem", "%s %% %s" % (x,y))
      y_is_negative = self.fresh_var(t, "y_is_negative", "%s < 0" % y)
      rem_is_negative = self.fresh_var(t, "rem_is_negative", "%s < 0" % rem)
      y_nonzero = self.fresh_var(t, "y_nonzero", "%s != 0" % y)
      rem_nonzero = self.fresh_var(t, "rem_nonzero", "%s != 0" % rem)
      neither_zero = self.fresh_var(t, "neither_zero", "%s && %s" % (y_nonzero, rem_nonzero))
      diff_signs = self.fresh_var(t, "diff_signs", "%s ^ %s" % (y_is_negative, rem_is_negative))
      should_flip = self.fresh_var(t, "should_flip", "%s && %s" % (neither_zero, diff_signs))
      flipped_rem = self.fresh_var(t, "flipped_rem", "%s + %s" % (y, rem))
      return "%s ? %s : %s" % (should_flip, flipped_rem, rem)
    elif p == prims.fmod:
      if t == Float32: return "fmodf(%s, %s)" % (args[0], args[1])
      elif t == Float64: return "fmod(%s, %s)" % (args[0], args[1])
      return "%s %% %s" % (args[0], args[1])
    elif p == prims.maximum:
      x,y = args
      return "(%s > %s) ? %s : %s" % (x,y,x,y)
    elif p == prims.minimum:
      x,y = args
      return "(%s < %s) ? %s : %s" % (x,y,x,y)
    
    elif p == prims.power:
      if t == Float32: 
        return "powf(%s, %s)" % (args[0], args[1])
      else:
        return "pow(%s, %s)" % (args[0], args[1])
    
    elif isinstance(t, FloatT):
      # many float prims implemented using the same name in math.h
      name = p.name
      if name.startswith("arc"):
        # arccos -> acos
        name = "a" + name[3:]
      if t == Float32: name = name + "f" 
      if len(args) == 1:
        return "%s(%s)" % (name, args[0])
      else:
        assert len(args) == 2, "Unexpected prim %s with %d args (%s)" % (p, len(args), args)
        return "%s(%s, %s)" % (name, args[0], args[1])
  
    else:
      assert False, "Prim not yet implemented: %s" % p
  
  def visit_Index(self, expr):
    arr = self.visit_expr(expr.value)
    idx = self.visit_expr(expr.index)
    elt_t = expr.value.type.elt_type
    ptr_t = "%s*" % self.to_ctype(elt_t)
    return "( (%s) (PyArray_DATA(%s)))[%s]" % (ptr_t, arr, idx)
  
  def visit_Call(self, expr):
    fn_name = self.get_fn(expr.fn)
    closure_args = self.get_closure_args(expr.fn)
    args = self.visit_expr_list(expr.args)
    return "%s(%s)" % (fn_name, ", ".join(tuple(closure_args) + tuple(args)))
  
  def visit_Select(self, expr):
    cond = self.visit_expr(expr.cond)
    true = self.visit_expr(expr.true_value)
    false = self.visit_expr(expr.false_value)
    return "%s ? %s : %s" % (cond, true, false) 
  
  def is_pure(self, expr):
    return expr.__class__ in (Var, Const, PrimCall, Attribute, TupleProj, Tuple, ArrayView)
  
  def visit_Assign(self, stmt):
    rhs = self.visit_expr(stmt.rhs)

    if stmt.lhs.__class__ is Var:
      lhs = self.visit_expr(stmt.lhs)
      return "%s %s = %s;" % (self.to_ctype(stmt.lhs.type), lhs, rhs)
    elif stmt.lhs.__class__ is Tuple:
      struct_value = self.fresh_var(self.to_ctype(stmt.lhs.type), "lhs_tuple")
      self.assign(struct_value, rhs)
      
      for i, lhs_var in enumerate(stmt.lhs.elts):
        assert isinstance(lhs_var, Var), "Expected LHS variable, got %s" % lhs_var
        c_name = self.visit_expr(lhs_var)
        self.append("%s %s = %s.elt%d;" % (self.to_ctype(lhs_var.type), c_name, struct_value, i ))
      return "" 
    else:
      lhs = self.visit_expr(stmt.lhs)
      return "%s = %s;" % (lhs, rhs)
  
  def declare(self, parakeet_name, parakeet_type, init_value = None):
    c_name = self.name(parakeet_name)
    t = self.to_ctype(parakeet_type)
    if init_value is None:
      self.append("%s %s;" % (t, c_name))
    else: 
      self.append("%s %s = %s;" % (t, c_name, init_value))
  
  def declare_merge_vars(self, merge):
    """ 
    Declare but don't initialize
    """
    for (name, (left, _)) in merge.iteritems():
      self.declare(name, left.type)
      
  def visit_merge_left(self, merge, fresh_vars = True):
    
    if len(merge) == 0:
      return ""
    
    self.push()
    self.comment("Merge Phi Nodes (left side) " + str(merge))
    for (name, (left, _)) in merge.iteritems():
      c_left = self.visit_expr(left)
      if fresh_vars:
        self.declare(name, left.type, c_left)
      else:
        c_name = self.name(name)
        self.append("%s = %s;" % (c_name, c_left))
        
    return self.pop()
  
  def visit_merge_right(self, merge):
    
    if len(merge) == 0:
      return ""
    self.push()
    self.comment("Merge Phi Nodes (right side) " + str(merge))
    
    for (name, (_, right)) in merge.iteritems():
      c_right = self.visit_expr(right)
     
      self.append("%s = %s;"  % (self.name(name), c_right))
    return self.pop()
  
  def visit_If(self, stmt):
    self.declare_merge_vars(stmt.merge)
    cond = self.visit_expr(stmt.cond)
    true = self.visit_block(stmt.true) + self.visit_merge_left(stmt.merge, fresh_vars = False)
    false = self.visit_block(stmt.false) + self.visit_merge_right(stmt.merge)
    return self.indent("if(%s) {\n%s\n} else {\n%s\n}" % (cond, self.indent(true), self.indent(false))) 
  
  def visit_While(self, stmt):
    decls = self.visit_merge_left(stmt.merge, fresh_vars = True)
    cond = self.visit_expr(stmt.cond)
    body = self.visit_block(stmt.body) + self.visit_merge_right(stmt.merge)
    return decls + "while (%s) {%s}" % (cond, body)
  
  def visit_ForLoop(self, stmt):
    s = self.visit_merge_left(stmt.merge, fresh_vars = True)
    start = self.visit_expr(stmt.start)
    stop = self.visit_expr(stmt.stop)
    step = self.visit_expr(stmt.step)
    var = self.visit_expr(stmt.var)
    t = self.to_ctype(stmt.var.type)
    body =  self.visit_block(stmt.body)
    body += self.visit_merge_right(stmt.merge)
    body = self.indent("\n" + body) 
    
    
    s += "\n %(t)s %(var)s;"
    s += "\nfor (%(var)s = %(start)s; %(var)s < %(stop)s; %(var)s += %(step)s) {%(body)s}"
    return s % locals()

  def visit_Return(self, stmt):
    assert not self.return_by_ref, "Returning multiple values by ref not yet implemented: %s" % stmt
    if self.return_void:
      return "return;"
    elif isinstance(stmt.value, Tuple):
      # if not returning multiple values by reference, then make a struct for them
      field_types = get_types(stmt.value.elts) 
      struct_type = self.struct_type_from_fields(field_types)
      result_elts = ", ".join(self.visit_expr(elt) for elt in stmt.value.elts)
      result_value = "{" + result_elts + "}"
      result = self.fresh_var(struct_type, "result", result_value)
      return "return %s;" % result 
    else:
      v = self.visit_expr(stmt.value)
      return "return %s;" % v
      
  def visit_block(self, stmts, push = True):
    if push: self.push()
    for stmt in stmts:
      s = self.visit_stmt(stmt)
      self.append(s)
    self.append("\n")
    return self.indent("\n" + self.pop())
  
  def tuple_to_var_list(self, expr):
    assert isinstance(expr, Expr)
    if isinstance(expr, Tuple):
      elts = expr.elts 
    else:
      assert isinstance(expr.type, ScalarT), "Unexpected expr %s : %s" % (expr, expr.type)
      elts = [expr]
    return self.visit_expr_list(elts)
      
  
  def get_fn(self, expr):
    if expr.__class__ is  TypedFn:
      fn = expr 
    elif expr.__class__ is Closure:
      fn = expr.fn 
    else:
      assert isinstance(expr.type, (FnT, ClosureT)), \
        "Expected function or closure, got %s : %s" % (expr, expr.type)
      fn = expr.type.fn
    #compiled_fn = compile_flat(result)
    #self.extra_objects.add(compiled_fn.object_filename)
    #self.declarations.add(compiled_fn.fn_signature)
    compiled = compile_flat_source(fn, _struct_type_cache = self._tuple_struct_cache)
    if compiled.sig not in self.extra_function_signatures:
      # add any declarations it depends on 
      for decl in compiled.declarations:
        self.add_decl(decl)
      
      #add any external objects it wants to be linked against 
      self.extra_objects.update(compiled.extra_objects)
      
      # first add the new function's dependencies
      for extra_sig in compiled.extra_function_signatures:
        if extra_sig not in self.extra_function_signatures:
          self.extra_function_signatures.append(extra_sig)
          extra_src = compiled.function_sources[extra_sig]
          self.extra_functions[extra_sig] = extra_src 
      # now add the function itself 
      self.extra_function_signatures.append(compiled.sig)
      self.extra_functions[compiled.sig] = compiled.src
    return compiled.name

  def get_closure_args(self, fn):
    if isinstance(fn.type, FnT):
      return []
    else:
      assert isinstance(fn, Closure), "Expected closure, got %s : %s" % (fn, fn.type)
      return self.visit_expr_list(fn.args)
      
  def build_loops(self, loop_vars, bounds, body):
    if len(loop_vars) == 0:
      return body
    var = loop_vars[0]
    bound = bounds[0]
    nested = self.build_loops(loop_vars[1:], bounds[1:], body)
    return """
    for (%s = 0; %s < %s; ++%s) {
      %s
    }""" % (var, var, bound, var, nested )
    

      
  def visit_TypedFn(self, expr):
    return self.get_fn(expr)

  def visit_UntypedFn(self, expr):
    assert False, "Unexpected UntypedFn %s in C backend, should have been specialized" % expr.name
  
  
  def return_types(self, fn):
    if isinstance(fn.return_type, TupleT):
      return fn.return_type.elt_types
    elif isinstance(fn.return_type, NoneT):
      return []
    else:
      assert isinstance(fn.return_type, (PtrT, ScalarT))
      return [fn.return_type]
    
  
  def visit_fn(self, fn, return_by_ref = False):
    
    c_fn_name = self.fresh_name(fn.name)
    arg_types = [self.to_ctype(t) for t in fn.input_types]
    arg_names = [self.name(old_arg) for old_arg in fn.arg_names]
    return_types = self.return_types(fn)
    n_return = len(return_types)
    
    if n_return == 1:
      return_type = self.to_ctype(return_types[0])
      self.return_void = (return_type == NoneType)
      self.return_by_ref = False
    elif n_return == 0:
      return_type = "void"
      self.return_void = True
      self.return_by_ref = False
    elif return_by_ref:
      return_type = "void"
      self.return_void = True
      self.return_by_ref = True
      self.return_var_types = [self.to_ctype(t) for t in return_types]
      self.return_var_names = [self.fresh_name("return_value%d" % i) for i in xrange(n_return)]
      arg_types = arg_types + ["%s*" % t for t in self.return_var_types] 
      arg_names = arg_names + self.return_var_names
    else:
      return_type = self.struct_type_from_fields(return_types)
      self.return_void = False 
      self.return_by_ref = False 
    args_str = ", ".join("%s %s" % (t, name) for (t,name) in zip(arg_types,arg_names))
    
    body_str = self.visit_block(fn.body) 

    sig = "%s %s(%s)" % (return_type, c_fn_name, args_str)
    src = "%s { %s }" % (sig, body_str) 
    return c_fn_name, sig, src
    

class PyModuleCompiler(FlatFnCompiler):
   
  def unbox_scalar(self, x, t, target = None):
    assert isinstance(t, ScalarT), "Expected scalar type, got %s" % t
    if target is None:
      target = "scalar_value"
    
    result = self.fresh_var(t, target)
    if isinstance(t, IntT):
      check = "PyInt_Check"
      if isinstance(t, SignedT):
        get = "PyInt_AsLong"
      else:
        get = "PyInt_AsUnsignedLongMask"
    elif isinstance(t, FloatT):
      check = "PyFloat_Check"
      get = "PyFloat_AsDouble"
    else:
      assert isinstance(t, BoolT), "Unexpected type %s" % t 
      check = "PyBool_Check"
      get = "PyObject_IsTrue"
    
    self.append("""
      if (%(check)s(%(x)s)) { %(result)s = %(get)s(%(x)s); }
      else { PyArray_ScalarAsCtype(%(x)s, &%(result)s); }
    """ % locals())
    return result 
      
  def box_scalar(self, x, t):  
    if isinstance(t, BoolT):
      return "PyBool_FromLong(%s)" % x
    elif isinstance(t, NoneT):
      self.append("Py_INCREF(Py_None);")
      return "Py_None"
    if x.replace("_", "").isalpha():
      scalar = x
    else:
      scalar = self.fresh_name("scalar");
      self.append("%s %s = %s;" % (self.to_ctype(t), scalar, x))
    return "PyArray_Scalar(&%s, PyArray_DescrFromType(%s), NULL)" % (scalar, type_mappings.to_dtype(t) )
  
  def box_tuple(self, x, t):
    if isinstance(t, ClosureT):
      elt_types = t.arg_types 
    else:
      assert isinstance(t, TupleT)
      elt_types = t.elt_types
    unboxed_elts = self.tuple_elts(x, elt_types)
    boxed_elts = [self.box(elt, elt_t) for elt, elt_t in zip(unboxed_elts, elt_types)]
    n = len(boxed_elts)
    if n == 0:
      return "PyTuple_Pack(0)"
    else:
      return "PyTuple_Pack(%d, %s)" % (n, ", ".join(boxed_elts))
  
  def box(self, x, t):
    if isinstance(t, (NoneT, ScalarT)):
      return self.box_scalar(x, t)
    elif isinstance(t, (ClosureT, TupleT)):
      return self.box_tuple(x, t)
    else:
      # all other types already boxed 
      return x
      
      
  def as_pyobj(self, expr):
    """
    Compile the expression and if necessary box it up as a PyObject
    """
    x = self.visit_expr(expr)
    if isinstance(expr.type, (NoneT, ScalarT, TupleT, ClosureT)):
      return self.box(x, expr.type)
    elif isinstance(expr.type, ArrayT):
      return "(PyObject*) " + x 
    else:
      return x
  
  def as_pyobj_list(self, exprs):
    return [self.as_pyobj(expr) for expr in exprs]
  
  def c_str(self, obj):
    return "PyString_AsString(PyObject_Str(%s))" % obj
  
  def c_type_str(self, obj):
    return self.c_str("PyObject_Type((PyObject*) %s)" % obj)
   
  def print_pyobj(self, obj, text = ""):
    text = '"%s"' % text
    self.append('printf("%%s%%s\\n", %s, %s);' % (text, self.c_str(obj)))
  
  def print_pyobj_type(self, obj, text=""):
    text = '"%s"' % text
    self.append('printf("%%s%%s\\n", %s, %s);' % (text, self.c_type_str(obj)))
  
  def unbox(self, boxed, t, target = None):
    if isinstance(t, (NoneT, PtrT, ScalarT)):
      return self.unbox_scalar(boxed, t, target = target)
    elif isinstance(t, (ClosureT, TupleT)):
      return self.unbox_tuple(boxed, t, target = target)
    else:
      return boxed 
  
  def unbox_tuple(self, boxed_tuple, tuple_t, target = "tuple_value"):
    if isinstance(tuple_t, TupleT):
      elt_types = tuple(tuple_t.elt_types)
    elif isinstance(tuple_t, ClosureT):
      elt_types = tuple(tuple_t.arg_types)
    else:
      assert False, "Expected tuple type, got %s" % tuple_t
     
    c_struct_t = self.struct_type_from_fields(elt_types)
    tuple_elts = self.tuple_elts(boxed_tuple, elt_types, boxed=True)
    elts_str = ", ".join(tuple_elts)
    return self.fresh_var(c_struct_t, target, "{" + elts_str + "}")
     
  def tuple_to_stack_array(self, expr, name = "array_from_tuple", elt_type = None):
    t0 = expr.type.elt_types[0]
    
    assert expr.type.__class__ is TupleT 
    assert all(t == t0 for t in expr.type.elt_types[1:])
    
    if expr.__class__ is Tuple:
      elts = [self.visit_expr(elt_expr) for elt_expr in expr.elts]
    else:
      tup = self.visit_expr(expr)
      self.check_tuple(tup)
      elts = self.tuple_elts(tup, expr.type.elt_types)
    
    array_name = self.fresh_name(name)
    n = len(expr.type.elt_types)
    if elt_type is None:
      elt_type = self.to_ctype(t0)
    self.append("%s %s[%d];" % (elt_type, array_name, n))
    for i, elt in enumerate(elts):
      self.append("%s[%d] = %s;" % (array_name, i, elt))
    return array_name
    
  def array_to_tuple(self, arr, n, elt_t, boxed = False):
    raw_elts = ["%s[%d]" % (arr,i) for i in xrange(n)]
    if boxed:
      if n == 0: 
        return "PyTuple_Pack(0);"
      else:
        boxed_elts = [self.box_scalar(raw_elt, elt_t) for raw_elt in raw_elts]
        elt_str = ", ".join(boxed_elts)
        return "PyTuple_Pack(%d, %s)" % (n, elt_str)
    else:
      field_types = (self.to_ctype(elt_t),) * n 
      tuple_struct_t = self.struct_type_from_fields(field_types)
      init = "{%s}" % ", ".join(raw_elts)
      return self.fresh_var(tuple_struct_t, "tuple_value", init)
      
  
  def tuple_elts(self, tup, ts, boxed = False):
    result = []
    for i,t in enumerate(ts):
      result.append(self.tuple_elt(tup, i, t, boxed = boxed))
    return result
  
  def mk_tuple(self, elts, boxed = False):
    n = len(elts)
    if boxed:
      if n == 0: 
        return "PyTuple_Pack(0);"
      else:
        elt_str = ", ".join(self.as_pyobj_list(elts)) 
        return "PyTuple_Pack(%d, %s)" % (n, elt_str)
    else:
      field_types = tuple(elt.type for elt in elts)
      tuple_struct_t = self.struct_type_from_fields(field_types)
      elts_str = ", ".join(self.visit_expr(elt) for elt in elts)
      init = "{" + elts_str +  "}"
      return self.fresh_var(tuple_struct_t, "tuple_value", init)
    
  
  def check_tuple(self, tup):
    if not config.check_pyobj_types: return 
    self.newline()
    self.comment("Checking tuple type for %s" % tup)
    self.append("""
      if (!PyTuple_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be tuple, got %%s", 
                    %s); 
        return NULL;
      }""" % (tup, tup, self.c_type_str(tup)))
 
  def check_slice(self, obj):
    if not config.check_pyobj_types: return 
    self.newline()
    self.comment("Checking slice type for %s" % obj)
    self.append("""
      if (!PySlice_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be slice, got %%s", 
                    %s); 
        return NULL;
      }""" % (obj, obj, self.c_type_str(obj)))
  
  def check_array(self, arr):
    if not config.check_pyobj_types: return 
    self.newline()
    self.comment("Checking array type for %s" % arr)
    self.append("""
      if (!PyArray_Check(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                    "Expected %s to be array, got %%s : %%s", 
                    %s, %s); 
        return NULL;
      }""" % (arr, arr, self.c_str(arr), self.c_type_str(arr)))
  
  
  def check_bool(self, x):
    if not config.check_pyobj_types: return 
    self.newline()
    self.comment("Checking bool type for %s" % x)
    self.append("""
      if (!PyArray_IsScalar(%s, Bool)) { 
        PyErr_Format(PyExc_AssertionError, 
                     "Expected %s to be bool, got %%s", 
                     %s); 
        return NULL;
      }""" % (x, x, self.c_type_str(x)))
  
  def check_int(self, x):
    if not config.check_pyobj_types: return 
    self.newline()
    self.comment("Checking int type for %s" % x)
    self.append("""
      if (!PyArray_IsIntegerScalar(%s)) { 
        PyErr_Format(PyExc_AssertionError, 
                     "Expected %s to be int, got %%s", 
                     %s); 
        return NULL;
      }""" % (x, x, self.c_type_str(x)))
  
  def check_type(self, v, t):
    if not config.check_pyobj_types: return 
    if isinstance(t, (ClosureT, TupleT)):
      self.check_tuple(v)
    elif isinstance(t, BoolT):
      self.check_bool(v)
    elif isinstance(t, IntT):
      self.check_int(v)
    elif isinstance(t, ArrayT):
      self.check_array(v)
      
  def tuple_elt(self, tup, idx, t, boxed = False):
    if boxed: 
      self.check_tuple(tup)
      proj_str = "PyTuple_GetItem(%s, %d)" % (tup, idx)
      if isinstance(t, (TupleT, ScalarT)):
        elt_obj = self.fresh_var("PyObject*", "%s_elt" % tup, proj_str)
        result = self.unbox(elt_obj, t)
        if config.debug and t == Int64:
          self.append(""" printf("tupleproj %s[%d] = %%" PRId64 "\\n", %s);""" % (tup, idx, result))
        return result
      
      else:
        return proj_str
    else:
      return "%s.elt%d" % (tup, idx)  
 
  
  def strides(self, array_expr):
    arr_t = array_expr.type
    assert isinstance(arr_t, ArrayT), \
      "Can only get strides of array, not %s : %s" % (array_expr, arr_t)
    elt_t = arr_t.elt_type
    arr = self.visit_expr(array_expr)
    
    bytes_per_elt = elt_t.dtype.itemsize

    strides_tuple_t = arr_t.strides_t
    stride_t = strides_tuple_t.elt_types[0]
    
    assert all(t == stride_t for t in strides_tuple_t)
    n = len(strides_tuple_t.elt_types)
    strides_bytes = self.fresh_name("strides_bytes")
    self.append("npy_intp* %s = PyArray_STRIDES( (PyArrayObject*) %s);" % (strides_bytes, arr))
    strides_elts = self.fresh_name("strides_elts")
    self.append("npy_intp %s[%d];" % (strides_elts, n))
    for i in xrange(n):
      if config.debug:
        self.printf("converting strides %s[%d] = %%ld to %%ld" % (strides_bytes, i), 
                    "%s[%d]" % (strides_bytes, i), "%s[%d] / %d" % (strides_bytes, i, bytes_per_elt))   
      self.append("%s[%d] = %s[%d] / %d;" % (strides_elts, i, strides_bytes, i, bytes_per_elt))
    strides_tuple = self.array_to_tuple(strides_elts, n, stride_t)
    return strides_tuple
    
  def decref(self, obj):
    self.append("Py_DECREF(%s);" % obj)
    
  def attribute(self, v, attr, t):
    if attr == "data":
      self.check_array(v)

      result = "(PyArrayObject*) PyArray_Ravel((PyArrayObject*) %s, 0)" % (v,)

      
      return result   

    elif attr == "shape":
      self.check_array(v)
      elt_types = t.elt_types
      n = len(elt_types)
      elt_t = elt_types[0]
      assert all(t == elt_t for t in elt_types)
      shape_name = self.fresh_name("strides")
      shape_array = "PyArray_DIMS( (PyArrayObject*) %s)" % v
      self.append("npy_intp* %s = %s;" % (shape_name, shape_array))
      return self.array_to_tuple(shape_name, n, elt_t)
      
    elif attr == "strides":
      assert False, "Can't directly use NumPy strides without dividing by itemsize"
      
    elif attr == 'offset':
      return "0"
    elif attr in ('size', 'nelts'):
      return "PyArray_Size(%s)" % v
    
    elif attr in ('start', 'stop', 'step'):
      self.check_slice(v)
      obj = "((PySliceObject*)%s)->%s" % (v, attr)
      return self.unbox_scalar(obj, t, attr)
    else:
      assert False, "Unsupported attribute %s" % attr 
   
    
  
  def visit_AllocArray(self, expr):
    shape = self.tuple_to_stack_array(expr.shape)
    t = type_mappings.to_dtype(elt_type(expr.type))
    return "(PyArrayObject*) PyArray_SimpleNew(%d, %s, %s)" % (expr.type.rank, shape, t)
    
  def visit_Tuple(self, expr):
    return self.mk_tuple(expr.elts, boxed = False)
  
  def visit_Closure(self, expr):
    return self.mk_tuple(expr.args)
  
  def visit_TupleProj(self, expr):
    tup = self.visit_expr(expr.tuple)
    result = self.tuple_elt(tup, expr.index, expr.type)

    return result
  
  def visit_ClosureElt(self, expr):
    clos = self.visit_expr(expr.closure)
    return self.tuple_elt(clos, expr.index, expr.type)
  
  
  def visit_ArrayView(self, expr):
    ndims = expr.type.rank 
    vec = self.visit_expr(expr.data)
    offset = self.visit_expr(expr.offset)
    
    if expr.strides.__class__ is Tuple:
      strides_elts = self.visit_expr_list(expr.strides.elts)  
    else:
      strides_var = self.visit_expr(expr.strides)
      strides_array = self.tuple_to_stack_array(strides_var, "strides_array", "npy_intp")
      strides_elts = ["%s[%d]" % (strides_array, i) for i in xrange(ndims)]
    
    if expr.shape.__class__ is Tuple:
      shape_elts = self.visit_expr_list(expr.shape.elts) 
    else:
      shape_var = self.visit_expr(expr.strides)
      shape_array = self.tuple_to_stack_array(shape_var, "shape_array", "npy_intp")
      shape_elts = ["%s[%d]" % (shape_array, i) for i in xrange(ndims)]
    
    # slice out the 1D data array if there's an offset 
    size =  "PySequence_Size( (PyObject*) %(vec)s)" % locals()

    cond = self.gt(offset, "0", Int64)
    if cond == "1":
      self.append("""
        %(vec)s = (PyArrayObject*) PySequence_GetSlice( (PyObject*)  %(vec)s, %(offset)s, %(size)s);
        """ % locals())
    elif cond != "0":
      self.append("""
        if (%(cond)s) {
          %(vec)s = (PyArrayObject*) PySequence_GetSlice( (PyObject*)  %(vec)s, %(offset)s, %(size)s);
        }""" % locals())
    self.check_array(vec)

    count = self.fresh_var("int64_t", "count", "PySequence_Size( (PyObject*) %s)" % vec)

    
      
    
    # increase the rank of the 1D nd-array to whatever the rank of the result 
    # should be so that the strides and shape arrays are of the appropriate size 
    if ndims > 1:
  
      uprank_elts =  (count,) + ("1",) * (ndims-1) 
      uprank_elts_as_pyobj = ["PyInt_FromLong(%s)"  % elt for elt in uprank_elts]
      uprank_elts_str = ", ".join(uprank_elts_as_pyobj)
      if ndims == 0: 
        uprank_shape = "PyTuple_Pack(0)"
      else:
        uprank_shape = "PyTuple_Pack(%d, %s)" % (ndims, uprank_elts_str)
      self.append("%s = (PyArrayObject*) PyArray_Reshape( (PyArrayObject*) %s, %s);" % (vec, vec, uprank_shape))
      self.return_if_null(vec)
    numpy_strides = self.fresh_var("npy_intp*", "numpy_strides")
    self.append("%s = PyArray_STRIDES(  (PyArrayObject*) %s);" % (numpy_strides, vec))
    numpy_shape = self.fresh_var("npy_intp*", "numpy_shape")
    self.append("%s = PyArray_DIMS(  (PyArrayObject*) %s);" % (numpy_shape, vec))
    bytes_per_elt = expr.type.elt_type.dtype.itemsize
    
    for i, _ in enumerate(expr.strides.type.elt_types):
      self.append("%s[%d] = %s * %d;" % (numpy_strides, i, strides_elts[i], bytes_per_elt) )
      self.append("%s[%d] = %s;" % (numpy_shape, i, shape_elts[i]))
    
    self.append("""
      // clear both fortran and c layout flags 
      ((PyArrayObject*) %(vec)s)->flags &= ~NPY_F_CONTIGUOUS;
      ((PyArrayObject*) %(vec)s)->flags &= ~NPY_C_CONTIGUOUS;
    """ % locals())
    
    f_layout_strides = ["1"]
    for shape_elt in shape_elts[1:]:
      f_layout_strides.append(f_layout_strides[-1] + " * " + shape_elt)
    
    c_layout_strides = ["1"]
    for shape_elt in list(reversed(shape_elts))[:-1]:
      c_layout_strides = [c_layout_strides[-1] + " * " + shape_elt] + c_layout_strides
    
    
    
    is_c_layout = "&& ".join(self.eq(actual, ideal, Int64) 
                             for actual, ideal 
                             in zip(strides_elts, c_layout_strides))
    is_f_layout = " && ".join(self.eq(actual, ideal, Int64) 
                             for actual, ideal 
                             in zip(strides_elts, f_layout_strides))
    
       
    # make sure the contiguity flags are set correctly 
    self.append("""
      // it's possible that *neither* of the above flags should be on
      // which is why we enable them separately here 
      if (%(is_f_layout)s) { ((PyArrayObject*)%(vec)s)->flags |= NPY_F_CONTIGUOUS; }
      if (%(is_c_layout)s) { ((PyArrayObject*)%(vec)s)->flags |= NPY_C_CONTIGUOUS; }
    """ % locals())
    return vec
  
    
  def visit_Attribute(self, expr):
    attr = expr.name
    if attr == 'strides':
      return self.strides(expr.value)
    v = self.visit_expr(expr.value) 
    return self.attribute(v, attr, expr.type)
  
  
    
  def visit_Return(self, stmt):
    v = self.as_pyobj(stmt.value)
    if config.debug: 
      self.print_pyobj_type(v, "Return type: ")
      self.print_pyobj(v, "Return value: ")

    return "return %s;" % v
  
  def visit_block(self, stmts, push = True):
    if push: self.push()
    for stmt in stmts:
      s = self.visit_stmt(stmt)
      self.append(s)
    self.append("\n")
    return self.pop()
      
  def visit_TypedFn(self, expr):
    return self.get_fn(expr)

  def visit_UntypedFn(self, expr):
    assert False, "Unexpected UntypedFn %s in C backend, should have been specialized" % expr.name
  
         
  def visit_fn(self, fn):
    if config.print_input_ir:
      print "=== Compiling to C (entry function) ==="
      print fn
    c_fn_name = self.fresh_name(fn.name)
    uses = use_count(fn)
    self.push()
    
    
    dummy = self.fresh_name("dummy")
    args = self.fresh_name("args")
    
    if config.debug: 
      self.newline()
      self.printf("\\nStarting %s : %s..." % (c_fn_name, fn.type))
      
    for i, argname in enumerate(fn.arg_names):
      assert argname in uses, "Couldn't find arg %s in use-counts" % argname
      if uses[argname] <= 1:
        self.comment("Skipping unused argument %s" % argname)
        continue
      self.comment("Unpacking argument %s"  % argname)
      c_name = self.name(argname)
      t = fn.type_env[argname]
      self.comment("Getting arg #%d: %s : %s => %s" % (i+1, argname,  t, c_name) )
      self.append("PyObject* %s = PyTuple_GetItem(%s, %d);" % (c_name, args, i))
      
      self.check_type(c_name, t)
      if config.debug:
        self.printf("Printing arg #%d %s" % (i,c_name))
        self.print_pyobj_type(c_name, text = "Type: ")
        self.print_pyobj(c_name, text = "Value: ")
      
      if isinstance(t, (TupleT, NoneT, PtrT, ClosureT, ScalarT)):
        #new_name = self.name(argname, overwrite = True)

        var = self.unbox(c_name, t, target = argname)

        self.name_mappings[argname] = var
      
        
    c_body = self.visit_block(fn.body, push=False)
    c_body = self.indent("\n" + c_body )#+ "\nPyGILState_Release(gstate);")
    c_args = "PyObject* %s, PyObject* %s" % (dummy, args) #", ".join("PyObject* %s" % self.name(n) for n in fn.arg_names)
    c_sig = "PyObject* %(c_fn_name)s (%(c_args)s)" % locals() 
    fndef = "%s {%s}" % (c_sig, c_body)
    return c_fn_name, c_sig, fndef 
