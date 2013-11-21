from collections import namedtuple
import numpy as np 

from .. import names, prims  
from ..ndtypes import (IntT, FloatT, TupleT, FnT, Type, BoolT, NoneT, Float32, Float64, Bool, 
                       ClosureT, ScalarT, PtrT, NoneType, ArrayT, SliceT, TypeValueT)    
from ..syntax import (Const, Var,  PrimCall, Attribute, TupleProj, Tuple, ArrayView,
                      Expr, Closure, TypedFn)
# from ..syntax.helpers import get_types   
import type_mappings
from base_compiler import BaseCompiler


CompiledFlatFn = namedtuple("CompiledFlatFn", 
                            ("name", "sig", "src",
                             "extra_objects",
                             "extra_functions",
                             "extra_function_signatures", 
                             "declarations"))


# mapping from (field_types, struct_name, field_names) to type names 
_struct_type_names = {}

# mapping from struct name to decl 
_struct_type_decls = {}

class FnCompiler(BaseCompiler):
  
  def __init__(self,  
               module_entry = False, 
               struct_type_cache = None, 
               **kwargs):
    BaseCompiler.__init__(self, **kwargs)
    
    
    self.declarations = []
    
    # depends on these .o files
    self.extra_objects = set([]) 
    
    # to avoid adding the same function's source twice 
    # we use its signature as a key  
    self.extra_functions = {}
    self.extra_function_signatures = []
    
      
    # are we actually compiling the entry-point into a Python module?
    # if so, expect some of the methods like visit_Return to be overloaded 
    # to return PyObjects
    self.module_entry = module_entry
     
  def add_decl(self, decl):
    if decl not in self.declarations:
      self.declarations.append(decl)
  
  def ptr_struct_type(self, elt_t):
    # need both an actual data pointer 
    # and an optional PyObject base
    field_types = ["%s*" % self.to_ctype(elt_t), "PyObject*"]
    return self.struct_type_from_fields(field_types, 
                                        struct_name = "%s_ptr_type" % elt_t,
                                        field_names = ["raw_ptr", "base"])
  
  def array_struct_type(self, elt_t, rank):
    ptr_field_t = self.ptr_struct_type(elt_t)
    field_types = [ptr_field_t, "npy_intp", "npy_intp", "int64_t", "int64_t"]
    field_names = ["data", "shape", "strides", "offset", "size"]
    field_repeats = {}
    field_repeats["shape"] = rank 
    field_repeats["strides"] = rank
    return self.struct_type_from_fields(field_types, "array_type", field_names, field_repeats)
  
  def slice_struct_type(self, start_t = "int64_t", stop_t = "int64_t", step_t = "int64_t"):
    field_types = [start_t, stop_t, step_t]
    field_names = ["start", "stop", "step"]
    return self.struct_type_from_fields(field_types, "slice_type", field_names)
       
  def struct_type_from_fields(self, 
                                 field_types, 
                                 struct_name = "tuple_type", 
                                 field_names = None, 
                                 field_repeats = {}):
    
    if any(not isinstance(t, str) for t in field_types):
      field_types = tuple(self.to_ctype(t) if isinstance(t, Type) else t 
                          for t in field_types)
    else:
      field_types = tuple(field_types)
    
    if field_names is None:
      field_names = tuple("elt%d" % i for i in xrange(len(field_types)))
    else:
      assert len(field_names) == len(field_types), \
        "Mismatching number of types %d and field names %d" % (len(field_types), len(field_names))
      field_names = tuple(field_names)
    
    repeat_set =  frozenset(sorted(field_repeats.items()))
    key = field_types, struct_name, field_names, repeat_set
     
    if key in _struct_type_names:
      typename = _struct_type_names[key]
      decl = _struct_type_decls[typename]
      if decl not in self.declarations:
        self.declarations.append(decl)
      return typename 
    
    typename = names.fresh(struct_name).replace(".", "")
    
    field_decls = []
    for t, field_name in zip(field_types, field_names):
      if field_name in field_repeats:
        field_decl = "  %s %s[%d];" % (t, field_name, field_repeats[field_name])
      else:
        field_decl = "  %s %s;" % (t, field_name)
      field_decls.append(field_decl)
    decl = "typedef struct %s {\n%s\n} %s;" % (typename, "\n".join(field_decls), typename)
    
    _struct_type_names[key] = typename
    _struct_type_decls[typename] = decl
    self.add_decl(decl)
    return typename 
  

  def to_ctype(self, parakeet_type):
    if isinstance(parakeet_type, (NoneT, ScalarT)):
      return type_mappings.to_ctype(parakeet_type)
    
    elif isinstance(parakeet_type, TupleT):
      return self.struct_type_from_fields(parakeet_type.elt_types)
    elif isinstance(parakeet_type, PtrT):
      return self.ptr_struct_type(parakeet_type.elt_type)
    elif isinstance(parakeet_type, ArrayT):
      elt_t = parakeet_type.elt_type 
      rank = parakeet_type.rank 
      return self.array_struct_type(elt_t, rank)
    
    elif isinstance(parakeet_type, SliceT):
      return self.slice_struct_type()
    elif isinstance(parakeet_type, ClosureT):
      return self.struct_type_from_fields(parakeet_type.arg_types)
    elif isinstance(parakeet_type, TypeValueT):
      return "int"
    else:
      assert False, "Don't know how to make C type for %s" % parakeet_type
    
  
  def to_ctypes(self, ts):
    return tuple(self.to_ctype(t) for t in ts)
  
  
  
  def visit_Slice(self, expr):
    typename = self.to_ctype(expr.type)
    start = self.visit_expr(expr.start)
    stop = self.visit_expr(expr.stop)
    step = self.visit_expr(expr.step)
    return self.fresh_var(typename, "slice", "{%s, %s, %s}" % (start,stop,step))
    
  def visit_Alloc(self, expr):
    elt_t =  expr.elt_type
    nelts = self.fresh_var("npy_intp", "nelts", self.visit_expr(expr.count))
    bytes_per_elt = elt_t.nbytes
    nbytes = self.mul(nelts, bytes_per_elt)#"%s * %d" % (nelts, bytes_per_elt)
    raw_ptr = "(%s) malloc(%s)" % (type_mappings.to_ctype(expr.type), nbytes)
    struct_type = self.to_ctype(expr.type)
    return self.fresh_var(struct_type, "new_ptr", "{%s, NULL}" % raw_ptr)
    
  def visit_Const(self, expr):
    t = expr.type 
    c = t.__class__ 
    if c == BoolT:
      return "1" if expr.value else "0"
    elif c == NoneT:
      return "0"
    
    assert isinstance(t, ScalarT), "Don't know how to translate Const %s : %s" % (expr,t)
    v = expr.value 
    if np.isinf(v):
      return "INFINITY"
    elif np.isnan(v):
      return "NAN"
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
  
  

  
  
  def visit_PrimCall(self, expr):
    t = expr.type
    args = self.visit_expr_list(expr.args)
    
    # parenthesize any compound expressions 
    for i, arg_expr in enumerate(expr.args):
      if not isinstance(arg_expr, (Var, Const)):
        args[i] = "(" + args[i] + ")"
        
    p = expr.prim 
    if p == prims.add:
      #return "%s + %s" % (args[0], args[1])
      return self.add(args[0], args[1])
    if p == prims.subtract:
      #return "%s - %s" % (args[0], args[1])
      return self.sub(args[0],args[1])
    elif p == prims.multiply:
      return self.mul(args[0], args[1])
      # return "%s * %s" % (args[0], args[1])
    elif p == prims.divide:
      return self.div(args[0], args[1])
      # return "%s / %s" % (args[0], args[1])
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
    if isinstance(expr.index.type, ScalarT):
      index_exprs = [expr.index]
    else:
      assert isinstance(expr.index.type, TupleT), \
        "Unexpected index %s : %s" % (expr.index, expr.index.type)
      if isinstance(expr.index, Tuple):
        index_exprs = expr.index.elts 
      else:
        index_exprs = [TupleProj(expr.index, i, type = t) 
                       for i, t in enumerate(expr.index.type.elt_types) ]
    assert all(isinstance(idx_expr.type, ScalarT) for idx_expr in index_exprs), \
      "Expected all indices to be scalars but got %s" % (index_exprs,)
    indices = [self.visit_expr(idx_expr) for idx_expr in index_exprs]
    if isinstance(expr.value.type, PtrT):
      assert len (indices) == 1, \
        "Can't index into pointer using %d indices (%s)" % (len(indices), index_exprs)
      raw_ptr = "%s.raw_ptr" % arr
      offset = indices[0]
    else:
      assert isinstance(expr.value.type, ArrayT)
      offset = self.fresh_var("int64_t", "offset", "%s.offset" % arr)
      for i, idx in enumerate(indices):
        stride = "%s.strides[%d]" % (arr, i)
        self.append("%s += %s * %s;" % (offset, idx, stride))
      raw_ptr = "%s.data.raw_ptr" % arr 

    return "%s[%s]" % (raw_ptr, offset)
  
  def visit_Call(self, expr):
    fn_name = self.get_fn_name(expr.fn)
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
  
  def visit_NumCores(self, expr):
    # by default we're running sequentially 
    return "1"
  
  def visit_Comment(self, stmt):
    return "// " + stmt.text
    
  def visit_PrintString(self, stmt):
    self.printf(stmt.text)
    return "// done with printf"
    
  def visit_SourceExpr(self, expr):
    return expr.text 
  
  def visit_SourceStmt(self, stmt):
    return stmt.text 
  
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
  
  def visit_ExprStmt(self, stmt):
    return self.visit_expr(stmt.value) + ";"
  
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
    up_loop = \
        "\nfor (%(var)s = %(start)s; %(var)s < %(stop)s; %(var)s += %(step)s) {%(body)s}"
    down_loop = \
        "\nfor (%(var)s = %(start)s; %(var)s > %(stop)s; %(var)s += %(step)s) {%(body)s}"
      
    if stmt.step.__class__ is Const:
      if stmt.step.value >= 0:
        s += up_loop
      else:
        s += down_loop
    else:
      s += "if(%(step)s >= 0) {\n"
      s += up_loop
      s += "\n} else {\n"
      s += down_loop
      s += "\n}"
    return s % locals()

  def visit_Return(self, stmt):
    assert not self.return_by_ref, "Returning multiple values by ref not yet implemented: %s" % stmt
    if self.return_void:
      return "return;"
    elif isinstance(stmt.value, Tuple):
      # if not returning multiple values by reference, then make a struct for them
      struct_type = self.to_ctype(stmt.value.type)
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
      
  
  def get_fn_name(self, expr, compiler_kwargs = {}, attributes = [], inline = True):
    if expr.__class__ is  TypedFn:
      fn = expr 
    elif expr.__class__ is Closure:
      fn = expr.fn 
    else:
      assert isinstance(expr.type, (FnT, ClosureT)), \
        "Expected function or closure, got %s : %s" % (expr, expr.type)
      fn = expr.type.fn

    compiler = self.__class__(module_entry = False, **compiler_kwargs)
    compiled = compiler.compile_flat_source(fn, attributes = attributes, inline = inline)
    
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
          self.extra_functions[extra_sig] = compiled.extra_functions[extra_sig] 
      # now add the function itself 
      self.extra_function_signatures.append(compiled.sig)
      self.extra_functions[compiled.sig] = compiled.src
      
    
    for link_flag in compiler.extra_link_flags:
      if link_flag not in self.extra_link_flags:
        self.extra_link_flags.append(link_flag)
    
    for compile_flag in compiler.extra_compile_flags:
      if compile_flag not in self.extra_compile_flags:
        self.extra_compile_flags.append(compile_flag)
  
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
    return self.get_fn_name(expr)

  def visit_UntypedFn(self, expr):
    return "{}"
  
  def return_types(self, fn):
    if isinstance(fn.return_type, TupleT):
      return fn.return_type.elt_types
    elif isinstance(fn.return_type, NoneT):
      return []
    else:
      # assert isinstance(fn.return_type, (PtrT, ScalarT)), "Unexpected return type %s" % fn.return_type
      return [fn.return_type]
    
  
  def visit_flat_fn(self, fn, return_by_ref = False, attributes = None, inline = True):
    if attributes is None:
      attributes = []
    
    c_fn_name = names.refresh(fn.name).replace(".", "_")
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
    
    if inline:
      # "__attribute__((always_inline))",
      attributes = attributes + ["inline"]
    attr_str = " ".join(attributes)
    sig = "%s %s(%s)" % (return_type, c_fn_name, args_str)
    src = "%s %s {\n\n%s}" % (attr_str, sig, body_str) 
    return c_fn_name, sig, src
  
  @property 
  def cache_key(self):
    """
    If we ever need to differentiate compiled function by *how* they were compiled,
    we can use this cache key to track the class of the compiler or other
    relevant meta-data
    """ 
    return self.__class__ 
  
  _flat_compile_cache = {}
  def compile_flat_source(self, parakeet_fn, attributes = [], inline = True):
      
    # make sure compiled source uses consistent names for tuple and array types, 
    # which both need declarations for their C struct representations  
    struct_types = set(t for t in parakeet_fn.type_env.itervalues() 
                         if isinstance(t, (ArrayT, TupleT)))
    
    # include your own class in the cache key so that we get distinct code 
    # for derived compilers like OpenMP and CUDA 
    key = parakeet_fn.cache_key, frozenset(struct_types), self.cache_key, tuple(attributes)
    
    if key in self._flat_compile_cache:
      return self._flat_compile_cache[key]
    
    name, sig, src = self.visit_flat_fn(parakeet_fn, attributes = attributes, inline = inline)
      
    
    result = CompiledFlatFn(
      name = name, 
      sig = sig, 
      src = src,
      extra_objects = self.extra_objects, 
      extra_functions = self.extra_functions,
      extra_function_signatures = self.extra_function_signatures,
      declarations = self.declarations)
    self._flat_compile_cache[key] = result
    return result
