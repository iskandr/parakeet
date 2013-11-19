import __builtin__
import ast
import inspect
import types
from collections import OrderedDict

import numpy as np
from dsltools import NestedBlocks, ScopedDict
 
from .. import config, names, prims, syntax

from ..names import NameNotFound
from ..ndtypes import Type
from ..prims import Prim 
from ..syntax import (Expr, 
                      Assign, If, ForLoop, Return,  
                      Var, PrimCall, Cast,  Select, 
                      Map, Reduce, 
                      Enumerate, Zip, Len, Range,    
                      Slice,  Tuple, Array,
                      Const, Call, Index, 
                      FormalArgs, ActualArgs, 
                      UntypedFn, Closure, 
                      SourceInfo)
 
from ..syntax.helpers import (none, true, false, one_i64, zero_i64, zero_i24,  
                              is_python_constant, const)
from ..syntax.wrappers import build_untyped_prim_fn, build_untyped_expr_fn, build_untyped_cast_fn


 
from ..transforms import subst_expr, subst_stmt_list

from decorators import jit, macro  
from python_ref import GlobalValueRef,  ClosureCellRef


class UnsupportedSyntax(Exception):
  def __init__(self, node, function_name = None, filename = None):
    self.function_name = function_name 
    self.filename = filename 
    self.node = node
    
  def __str__(self):
    if self.function_name is not None and \
       self.filename is not None and \
       self.node.lineno is not None:
      return "Parakeet doesn't support %s from function '%s' in file %s on line %d" % \
        (self.node.__class__.__name__, self.function_name, self.filename, self.node.lineno)
    elif self.function_name is not None:
      return "Parakeet doesn't support %s in '%s'" % \
        (self.node.__class__.__name__, self.function_name)
    else:
      return "Parakeet doesn't support %s" % self.node.__class__.__name__



  
class ExternalValue(object):
  """
  Wrap up references global values with this class
  """ 
  def __init__(self, python_value):
    self.value = python_value
    
  def __str__(self):
    return "ExternalValue(%s)" % self.value


def mk_reduce_call(fn, positional, init = None):
  init = none if init is None else init
  axis = zero_i64
  from .. import lib 
  return Reduce(fn = translate_function_value(lib.identity), 
                combine = fn, 
                args = positional, 
                axis = axis, 
                init = init)

def mk_simple_fn(mk_body, input_name = "x", fn_name = "cast"):
  unique_arg_name = names.fresh(input_name)
  unique_fn_name = names.fresh(fn_name)
  var = Var(unique_arg_name)
  formals = FormalArgs()
  formals.add_positional(unique_arg_name, input_name)
  body = mk_body(var)
  return UntypedFn(unique_fn_name, formals, body)
  
       
def is_hashable(x):
  try:
    hash(x)
    return True
  except:
    return False 

def is_prim(v):
  return isinstance(v, (Prim)) or (is_hashable(v) and v in prims.prim_lookup_by_value) 

def is_builtin_function(v):
  return isinstance(v, (types.TypeType, types.BuiltinFunctionType))
  
def is_user_function(v):
  return isinstance(v, (types.FunctionType, jit, macro))

def is_function_value(v):
  return is_user_function(v) or is_builtin_function(v) or is_prim(v) 
    
def is_static_value(v):
  return is_python_constant(v) or \
         type(v) is np.dtype  or \
         is_function_value(v)

def value_to_syntax(v):
  if isinstance(v, Expr):
    return v 
  elif is_python_constant(v):
    return const(v)
  else:
    return translate_function_value(v)
    
class AST_Translator(ast.NodeVisitor):
  def __init__(self, 
               globals_dict=None, 
               closure_cell_dict=None,
               parent=None, 
               function_name = None, 
               filename = None):
    # assignments which need to get prepended at the beginning of the
    # function
    self.globals = globals_dict
    self.blocks = NestedBlocks()

    self.parent = parent 
    self.scopes = ScopedDict()
    
    self.globals_dict = globals_dict 
    self.closure_cell_dict = closure_cell_dict 
    
    # mapping from names/paths to either a closure cell reference or a 
    # global value 
    self.python_refs = OrderedDict()
    
    self.original_outer_names = []
    self.localized_outer_names = []
    self.filename = filename 
    self.function_name = function_name 
    self.push()

  def push(self, scope = None, block = None):
    if scope is None:
      scope = {}
    if block is None:
      block = []
    self.scopes.push(scope)
    self.blocks.push(block)


  def pop(self):
    scope = self.scopes.pop()
    block = self.blocks.pop()
    return scope, block
  
  def fresh_name(self, original_name):
    fresh_name = names.fresh(original_name)
    self.scopes[original_name] = fresh_name
    return fresh_name
    
  def fresh_names(self, original_names):
    return map(self.fresh_name, original_names)

  def fresh_var(self, name):
    return Var(self.fresh_name(name))

  def fresh_vars(self, original_names):
    return map(self.fresh_var, original_names)
  
  def current_block(self):
    return self.blocks.top()

  def current_scope(self):
    return self.scopes.top()

  def ast_to_value(self, expr):
    if isinstance(expr, ast.Num):
      return expr.n
    elif isinstance(expr, ast.Tuple):
      return tuple(self.ast_to_value(elt) for elt in expr.elts)
    elif isinstance(expr, ast.Name):
      return self.lookup_global(expr.id) 
    elif isinstance(expr, ast.Attribute):
      left = self.ast_to_value(expr.value)
      if isinstance(left, ExternalValue):
        left = left.value 
      return getattr(left, expr.attr) 

  def lookup_global(self, key):
    if isinstance(key, (list, tuple)):
      assert len(key) == 1
      key = key[0]
    else:
      assert isinstance(key, str), "Invalid global key: %s" % (key,)

    if self.globals:
      if key in self.globals:
        return self.globals[key]
      elif key in __builtin__.__dict__:

        return __builtin__.__dict__[key]
      else:
        assert False, "Couldn't find global name %s" % key
    else:
      assert self.parent is not None
      return self.parent.lookup_global(key)
    
  def is_global(self, key):
    if isinstance(key, (list, tuple)):
      key = key[0]
    if key in self.scopes:
      return False
    elif self.closure_cell_dict and key in self.closure_cell_dict:
      return False
    if self.globals:
      return key in self.globals or key in __builtins__  
    assert self.parent is not None 
    return self.parent.is_global(key)
  
  
  def local_ref_name(self, ref, python_name):
    for (local_name, other_ref) in self.python_refs.iteritems():
      if ref == other_ref:
        return Var(local_name)
    local_name = names.fresh(python_name)
    self.scopes[python_name] = local_name
    self.original_outer_names.append(python_name)
    self.localized_outer_names.append(local_name)
    self.python_refs[local_name] = ref
    return Var(local_name)
  
  def lookup(self, name):
    #if name in reserved_names:
    #  return reserved_names[name]
    if name in self.scopes:
      return Var(self.scopes[name])
    elif self.parent:
      # don't actually keep the outer binding name, we just
      # need to check that it's possible and tell the outer scope
      # to register any necessary python refs
      local_name = names.fresh(name)
      self.scopes[name] = local_name
      self.original_outer_names.append(name)
      self.localized_outer_names.append(local_name)
      return Var(local_name)
    elif self.closure_cell_dict and name in self.closure_cell_dict:
      ref = ClosureCellRef(self.closure_cell_dict[name], name)
      return self.local_ref_name(ref, name)
    elif self.is_global(name):
      value = self.lookup_global(name)
      if is_static_value(value):
        return value_to_syntax(value)
      elif isinstance(value, np.ndarray): 
        ref = GlobalValueRef(value)
        return self.local_ref_name(ref, name)
      else:
        # assume that this is a module or object which will have some 
        # statically convertible value pulled out of it 
        return ExternalValue(value)
      #else:
      #  assert False, "Can't use global value %s" % value
    else:
      raise NameNotFound(name)
      
      
  def visit_list(self, nodes):
    return map(self.visit, nodes)


  def tuple_arg_assignments(self, elts, var):
    """
    Recursively decompose a nested tuple argument like
      def f((x,(y,z))):
        ...
    into a single name and a series of assignments:
      def f(tuple_arg):
        x = tuple_arg[0]
        tuple_arg_elt = tuple_arg[1]
        y = tuple_arg_elt[0]
        z = tuple_arg_elt[1]
    """

    assignments = []
    for (i, sub_arg) in enumerate(elts):
      if isinstance(sub_arg, ast.Tuple):
        name = "tuple_arg_elt"
      else:
        assert isinstance(sub_arg, ast.Name)
        name = sub_arg.id
      lhs = self.fresh_var(name)
      stmt = Assign(lhs, Index(var, Const(i)))
      assignments.append(stmt)
      if isinstance(sub_arg, ast.Tuple):
        more_stmts = self.tuple_arg_assignments(sub_arg.elts, lhs)
        assignments.extend(more_stmts)
    return assignments

  def translate_args(self, args):
    assert not args.kwarg
    formals = FormalArgs()
    assignments = []
    for arg in args.args:
      if isinstance(arg, ast.Name):
        visible_name = arg.id
        local_name = self.fresh_name(visible_name)
        formals.add_positional(local_name, visible_name)
      else:
        assert isinstance(arg, ast.Tuple)
        arg_name = self.fresh_name("tuple_arg")
        formals.add_positional(arg_name)
        var = Var(arg_name)
        stmts = self.tuple_arg_assignments(arg.elts, var)
        assignments.extend(stmts)

    n_defaults = len(args.defaults)
    if n_defaults > 0:
      local_names = formals.positional[-n_defaults:]
      for (k,expr) in zip(local_names, args.defaults):
        v = self.ast_to_value(expr)
        
        # for now we're putting literal python 
        # values in the defaults dictionary of
        # a function's formal arguments
        formals.defaults[k] = v 

    if args.vararg:
      assert isinstance(args.vararg, str)
      formals.starargs = self.fresh_name(args.vararg)

    return formals, assignments


  def visit_Name(self, expr):
    assert isinstance(expr, ast.Name), "Expected AST Name object: %s" % expr
    return self.lookup(expr.id)

  def create_phi_nodes(self, left_scope, right_scope, new_names = {}):
    """
    Phi nodes make explicit the possible sources of each variable's values and
    are needed when either two branches merge or when one was optionally taken.
    """
    merge = {}
    for (name, ssa_name) in left_scope.iteritems():
      left = Var(ssa_name)
      if name in right_scope:
        right = Var(right_scope[name])
      else:
        try:
          right = self.lookup(name)
        except NameNotFound:
          continue 
          
      if name in new_names:
        new_name = new_names[name]
      else:
        new_name = self.fresh_name(name)
      merge[new_name] = (left, right)

    for (name, ssa_name) in right_scope.iteritems():
      if name not in left_scope:
        try:
          left = self.lookup(name)
          right = Var(ssa_name)

          if name in new_names:
            new_name = new_names[name]
          else:
            new_name = self.fresh_name(name)
          merge[new_name] = (left, right)
        except names.NameNotFound:
          # for now skip over variables which weren't defined before
          # a control flow split, which means that loop-local variables
          # can't be used after the loop.
          # TODO: Fix this. Maybe with 'undef' nodes?
          pass
    return merge


  def visit_Index(self, expr):
    return self.visit(expr.value)

  def visit_Ellipsis(self, expr):
    raise RuntimeError("Ellipsis operator unsupported")

  def visit_Slice(self, expr):
    """
    x[l:u:s]
    Optional fields
      expr.lower
      expr.upper
      expr.step
    """

    start = self.visit(expr.lower) if expr.lower else none
    stop = self.visit(expr.upper) if expr.upper else none
    step = self.visit(expr.step) if expr.step else none
    return Slice(start, stop, step)

  def visit_ExtSlice(self, expr):
    slice_elts = map(self.visit, expr.dims)
    if len(slice_elts) > 1:
      return Tuple(slice_elts)
    else:
      return slice_elts[0]

  def visit_UnaryOp(self, expr):
    ssa_val = self.visit(expr.operand)
    # UAdd doesn't do anything!
    
    if expr.op.__class__.__name__ == 'UAdd':
      return ssa_val 
    prim = prims.find_ast_op(expr.op)
    return PrimCall(prim, [ssa_val])

  def visit_BinOp(self, expr):
    ssa_left = self.visit(expr.left)
    ssa_right = self.visit(expr.right)
    prim = prims.find_ast_op(expr.op)
    return PrimCall(prim, [ssa_left, ssa_right])

  def visit_BoolOp(self, expr):
    values = map(self.visit, expr.values)
    prim = prims.find_ast_op(expr.op)
    # Python, strangely, allows more than two arguments to
    # Boolean operators
    result = values[0]
    for v in values[1:]:
      result = PrimCall(prim, [result, v])
    return result

  def visit_Compare(self, expr):
    lhs = self.visit(expr.left)
    assert len(expr.ops) == 1
    prim = prims.find_ast_op(expr.ops[0])
    assert len(expr.comparators) == 1
    rhs = self.visit(expr.comparators[0])
    return PrimCall(prim, [lhs, rhs])

  def visit_Subscript(self, expr):
    value = self.visit(expr.value)
    index = self.visit(expr.slice)
    return Index(value, index)

  def generic_visit(self, expr):
    raise UnsupportedSyntax(expr, 
                            function_name = self.function_name, 
                            filename = self.filename)
  
  def visit(self, node):
    res = ast.NodeVisitor.visit(self, node)
    source_info = SourceInfo(filename = self.filename, 
                             line = getattr(node, 'lineno', None),
                             col = getattr(node, 'e.col_offset', None), 
                             function = self.function_name, )
    res.source_info = source_info 
    return res 
    
  def translate_value_call(self, value, positional, keywords_dict= {}, starargs_expr = None):
    if value is sum:
      return mk_reduce_call(build_untyped_prim_fn(prims.add), positional, zero_i24)
    
    elif value is max:
      if len(positional) == 1:
        return mk_reduce_call(build_untyped_prim_fn(prims.maximum), positional)
      else:
        assert len(positional) == 2
        return PrimCall(prims.maximum, positional)
    
    elif value is min:
      if len(positional) == 1:
        return mk_reduce_call(build_untyped_prim_fn(prims.minimum), positional)
      else:
        assert len(positional) == 2
        return PrimCall(prims.minimum, positional)
    
    elif value is map:
      assert len(keywords_dict) == 0
      assert len(positional) > 1
      axis = keywords_dict.get("axis", None)
      return Map(fn = positional[0], args = positional[1:], axis = axis)
    
    elif value is enumerate:
      assert len(positional) == 1, "Wrong number of args for 'enumerate': %s" % positional 
      assert len(keywords_dict) == 0, \
        "Didn't expect keyword arguments for 'enumerate': %s" % keywords_dict
      return Enumerate(positional[0])
    
    elif value is len:
      assert len(positional) == 1, "Wrong number of args for 'len': %s" % positional 
      assert len(keywords_dict) == 0, \
        "Didn't expect keyword arguments for 'len': %s" % keywords_dict
      return self.len(positional[0])
    
    elif value is zip:
      assert len(positional) > 1, "Wrong number of args for 'zip': %s" % positional 
      assert len(keywords_dict) == 0, \
        "Didn't expect keyword arguments for 'zip': %s" % keywords_dict
      return Zip(values = positional)
    
    from ..mappings import function_mappings
    if value in function_mappings:
      value = function_mappings[value]
    
    if isinstance(value, macro):
      return value.transform(positional, keywords_dict)
      
    fn = translate_function_value(value)
    return Call(fn, ActualArgs(positional, keywords_dict, starargs_expr))
    
  def visit_Call(self, expr):
    """
    TODO: 
    The logic here is broken and haphazard, eventually try to handle nested
    scopes correctly, along with globals, cell refs, etc..
    """
    
    fn, args, keywords_list, starargs, kwargs = \
        expr.func, expr.args, expr.keywords, expr.starargs, expr.kwargs
    assert kwargs is None, "Dictionary of keyword args not supported"

    positional = self.visit_list(args)

    keywords_dict = {}
    for kwd in keywords_list:
      keywords_dict[kwd.arg] = self.visit(kwd.value)

    if starargs:
      starargs_expr = self.visit(starargs)
    else:
      starargs_expr = None
    
    def is_attr_chain(expr):
      return isinstance(expr, ast.Name) or \
             (isinstance(expr, ast.Attribute) and is_attr_chain(expr.value))
    def extract_attr_chain(expr):
      if isinstance(expr, ast.Name):
        return [expr.id]
      else:
        base = extract_attr_chain(expr.value)
        base.append(expr.attr)
        return base
    
    def lookup_attr_chain(names):
      value = self.lookup_global(names[0])
    
      for name in names[1:]:
        if hasattr(value, name):
          value = getattr(value, name)
        else:
          try:
            value = value[name]
          except:
            assert False, "Couldn't find global name %s" % ('.'.join(names))
      return value
          
    if is_attr_chain(fn):
      names = extract_attr_chain(fn)
      
      if self.is_global(names):
        return self.translate_value_call(lookup_attr_chain(names), 
                                         positional, keywords_dict, starargs_expr)
    fn_node = self.visit(fn)    
    if isinstance(fn_node, syntax.Expr):
      actuals = ActualArgs(positional, keywords_dict, starargs_expr)
      return Call(fn_node, actuals)
    else:
      assert isinstance(fn_node, ExternalValue)
      return self.translate_value_call(fn_node.value, 
                                       positional, keywords_dict, starargs_expr)

  def visit_List(self, expr):
    return Array(self.visit_list(expr.elts))
    
  def visit_Expr(self, expr):
    # dummy assignment to allow for side effects on RHS
    lhs = self.fresh_var("dummy")
    if isinstance(expr.value, ast.Str):
      return Assign(lhs, zero_i64)
      # return syntax.Comment(expr.value.s.strip().replace('\n', ''))
    else:
      rhs = self.visit(expr.value)
      return syntax.Assign(lhs, rhs)

  def visit_GeneratorExp(self, expr):
    return self.visit_ListComp(expr)
    
  def visit_ListComp(self, expr):
    gens = expr.generators
    assert len(gens) == 1
    gen = gens[0]
    target = gen.target
    if target.__class__ is ast.Name:
      arg_vars = [target]
    else:
      assert target.__class__ is ast.Tuple and all(e.__class__ is ast.Name for e in target.elts),\
       "Expected comprehension target to be variable or tuple of variables, got %s" % ast.dump(target)
      arg_vars = [ast.Tuple(elts = target.elts)]
    # build a lambda as a Python ast representing 
    # what we do to each element 

    args = ast.arguments(args = arg_vars, 
                         vararg = None,  
                         kwarg = None,  
                         defaults = ())

    fn = translate_function_ast(name = "comprehension_map", 
                                args = args, 
                                body = [ast.Return(expr.elt)], 
                                parent = self)
    seq = self.visit(gen.iter)
    ifs = gen.ifs
    assert len(ifs) == 0, "Parakeet: Conditions in array comprehensions not yet supported"
    return Map(fn = fn, args=(seq,), axis = zero_i64)
      
  def visit_Attribute(self, expr):
    # TODO:
    # Recursive lookup to see if:
    #  (1) base object is local, if so-- create chain of attributes
    #  (2) base object is global but an adverb primitive-- use it locally
    #      without adding it to nonlocals
    #  (3) not local at all-- in which case, add the whole chain of strings
    #      to nonlocals
    #
    #  AN IDEA:
    #     Allow external values to be brought into the syntax tree as 
    #     a designated ExternalValue node
    #     and then here check if the LHS is an ExternalValue and if so, 
    #     pull out the value. If it's a constant, then make it into syntax, 
    #     if it's a function, then parse it, else raise an error. 
    #
    
    from ..mappings import property_mappings, method_mappings
    value = self.visit(expr.value)
    attr = expr.attr
    if isinstance(value, ExternalValue):
      value = value.value 
      assert hasattr(value, attr), "Couldn't find attribute '%s' in %s" % (attr, value)
      value = getattr(value, attr)
      if is_static_value(value):
        return value_to_syntax(value)
      else:
        return ExternalValue(value) 
    elif attr in property_mappings:
      fn = property_mappings[attr]
      if isinstance(fn, macro):
        return fn.transform( [value] )
      else:
        return Call(translate_function_value(fn),
                    ActualArgs(positional = (value,)))  
    elif attr in method_mappings:
      fn_python = method_mappings[attr]
      fn_syntax = translate_function_value(fn_python)
      return Closure(fn_syntax, args=(value,))
    else:
      assert False, "Attribute %s not supported" % attr 


  def visit_Num(self, expr):
    return Const(expr.n)

  def visit_Tuple(self, expr):
    return syntax.Tuple(self.visit_list(expr.elts))

  def visit_IfExp(self, expr):
    cond = self.visit(expr.test)
    if_true = self.visit(expr.body)
    if_false = self.visit(expr.orelse)
    return Select(cond, if_true, if_false)
    
  def visit_lhs(self, lhs):
    if isinstance(lhs, ast.Name):
      return self.fresh_var(lhs.id)
    elif isinstance(lhs, ast.Tuple):
      return syntax.Tuple( map(self.visit_lhs, lhs.elts))
    else:
      # in case of slicing or attributes
      res = self.visit(lhs)
      return res

  def visit_Assign(self, stmt):
    # important to evaluate RHS before LHS for statements like 'x = x + 1'
    ssa_rhs = self.visit(stmt.value)
    ssa_lhs = self.visit_lhs(stmt.targets[0])
    return Assign(ssa_lhs, ssa_rhs)
  
  def visit_AugAssign(self, stmt):
    ssa_incr = self.visit(stmt.value)
    ssa_old_value = self.visit(stmt.target)
    ssa_new_value = self.visit_lhs(stmt.target)
    prim = prims.find_ast_op(stmt.op) 
    return Assign(ssa_new_value, PrimCall(prim, [ssa_old_value, ssa_incr]))

  def visit_Return(self, stmt):
    return syntax.Return(self.visit(stmt.value))

  def visit_If(self, stmt):
    cond = self.visit(stmt.test)
    true_scope, true_block  = self.visit_block(stmt.body)
    false_scope, false_block = self.visit_block(stmt.orelse)
    merge = self.create_phi_nodes(true_scope, false_scope)
    return syntax.If(cond, true_block, false_block, merge)

  def visit_loop_body(self, body, *exprs):
    merge = {}
    substitutions = {}
    curr_scope = self.current_scope()
    exprs = [self.visit(expr) for expr in exprs]
    scope_after, body = self.visit_block(body)
    for (k, name_after) in scope_after.iteritems():
      if k in self.scopes:
        name_before = self.scopes[k]
        new_name = names.fresh(k + "_loop")
        merge[new_name] = (Var(name_before), Var(name_after))
        substitutions[name_before]  = new_name
        curr_scope[k] = new_name
    
    exprs = [subst_expr(expr, substitutions) for expr in exprs]
    body = subst_stmt_list(body, substitutions)
    return body, merge, exprs 

  def visit_While(self, stmt):
    assert not stmt.orelse
    body, merge, (cond,) = self.visit_loop_body(stmt.body, stmt.test)
    return syntax.While(cond, body, merge)

  def assign(self, lhs, rhs):
    self.current_block().append(Assign(lhs,rhs))
 
  def assign_to_var(self, rhs, name = None):
    if isinstance(rhs, (Var, Const)):
      return rhs 
    if name is None:
      name = "temp"
    var = self.fresh_var(name)
    self.assign(var, rhs)
    return var 
  
    
  def add(self, x, y, temp = True):
    expr = PrimCall(prims.add, [x,y])
    if temp:
      return self.assign_to_var(expr, "add")
    else:
      return expr 
  
  def sub(self, x, y, temp = True):
    expr = PrimCall(prims.subtract, [x,y])
    if temp:
      return self.assign_to_var(expr, "sub")
    else:
      return expr 
  
  def mul(self, x, y, temp = True):
    expr = PrimCall(prims.multiply, [x,y])
    if temp:
      return self.assign_to_var(expr, "mul")
    else:
      return expr 
  
  def div(self, x, y, temp = True):
    expr = PrimCall(prims.divide, [x,y])
    if temp:
      return self.assign_to_var(expr, "div")
    else:
      return expr 
  
  
  def len(self, x):
    if isinstance(x, Enumerate):
      return self.len(x.value)
    elif isinstance(x, Zip):
      elt_lens = [self.len(v) for v in x.values]
      result = elt_lens[0]
      for n in elt_lens[1:]:
        result = PrimCall(prims.minimum, [result, n])
      return result 
    elif isinstance(x, (Array, Tuple)):
      return Const(len(x.elts))
    
    elif isinstance(x, Range):
      # if it's a range from 0..len(x), then just return len(x)
      if isinstance(x.stop, Len):
        if isinstance(x.start, Const) and x.start.value == 0:
          if isinstance(x.step, Const) and x.stop.value in (1,-1, None):
            return x.stop
    seq_var = self.assign_to_var(x, "len_input")
    return self.assign_to_var(Len(seq_var), "len_result")


  def is_none(self, v):
    return v is None or isinstance(v, Const) and v.value is None 
  
  def for_loop_bindings(self, idx, lhs, rhs):
    if isinstance(rhs, Enumerate):
      array = rhs.value 
      elt = Index(array, idx)
      if isinstance(lhs, Tuple):
        var_names = ", ".join(str(elt) for elt in lhs.elts)
        if len(lhs.elts) < 2:
          raise SyntaxError("Too many values to unpack: 'enumerate' expects 2 but given %s" % var_names)
        elif len(lhs.elts) > 2:
          raise SyntaxError("Need more than 2 values to unpack for LHS of %s" % var_names)
        idx_var, seq_var = lhs.elts
        other_bindings = self.for_loop_bindings(idx, seq_var, array)
        return [Assign(idx_var, idx)] +  other_bindings 
      elif isinstance(lhs, Var):
        seq_var = self.fresh_var("seq_elt")
        other_bindings = self.for_loop_bindings(idx, seq_var, array)
        return [Assign(lhs, Tuple(idx, seq_var))] + other_bindings
      else:
        raise SyntaxError("Unexpected binding in for loop: %s = %s" % (lhs,rhs)) 
      
    elif isinstance(rhs, Zip):
      values_str = ", ".join(str(v) for v in rhs.values)
      if len(rhs.values) < 2:
        raise SyntaxError("'zip' must take at least two arguments, given: %s" % values_str)
      if isinstance(lhs, Tuple):
        if len(lhs.elts) < len(rhs.values):
          raise SyntaxError("Too many values to unpack in %s = %s" % (lhs, rhs))
        elif len(lhs.elts) > len(rhs.values):
          raise SyntaxError("Too few values on LHS of bindings in %s = %s" % (lhs,rhs))
        result = []
        for lhs_var, rhs_value in zip(lhs.elts, rhs.values):
          result.extend(self.for_loop_bindings(idx, lhs_var, rhs_value))
        return result 
      elif isinstance(lhs, Var):
        lhs_vars = [self.fresh_var("elt%d" % i) for i in xrange(len(rhs.values))]
        result = []
        for lhs_var, rhs_value in zip(lhs_vars, rhs.values):
          result.extend(self.for_loop_bindings(idx, lhs_var, rhs_value))
        result.append(Assign(lhs, Tuple(elts=lhs_vars)))
        return result  
      else:
        raise SyntaxError("Unexpected binding in for loop: %s = %s" % (lhs,rhs)) 
      
    elif isinstance(rhs, Range):
      if isinstance(lhs, Tuple):
        raise SyntaxError("Too few values in unpack in for loop binding %s = %s" % (lhs,rhs))
      elif isinstance(lhs, Var):
        start = rhs.start
        if self.is_none(start): 
          start = zero_i64
        step = rhs.step
        if self.is_none(step): 
          step = one_i64 
        return [Assign(lhs, self.add(start, self.mul(idx, step, temp = False), temp= False))]
      else:
        raise SyntaxError("Unexpected binding in for loop: %s = %s" % (lhs,rhs)) 
      
    else:
      return [Assign(lhs, Index(rhs,idx))]
      
      
  def visit_For(self, stmt):
    assert not stmt.orelse 
    var = self.visit_lhs(stmt.target)
    seq = self.visit(stmt.iter)
    body, merge, _ = self.visit_loop_body(stmt.body)

    if isinstance(seq, Range):
      assert isinstance(var, Var), "Expect loop variable to be simple but got '%s'" % var
      return ForLoop(var, seq.start, seq.stop, seq.step, body, merge)
    else:
      idx = self.fresh_var("idx")
      n = self.len(seq)
      bindings = self.for_loop_bindings(idx, var, seq)
      return ForLoop(idx, zero_i64, n, one_i64, bindings + body, merge)
    
  def visit_block(self, stmts):
    self.push()
    curr_block = self.current_block()
    for stmt in stmts:
      parakeet_stmt = self.visit(stmt)
      curr_block.append(parakeet_stmt)
    return self.pop()

  def visit_FunctionDef(self, node):
    """
    Translate a nested function
    """
    fundef = translate_function_ast(node.name, node.args, node.body, parent = self)
    local_var = self.fresh_var(node.name)
    return Assign(local_var, fundef)

  def visit_Lambda(self, node):
    return translate_function_ast("lambda", node.args, [ast.Return(node.body)], parent = self)
    
def translate_function_ast(name, args, body, 
                              globals_dict = None,
                              closure_vars = [], 
                              closure_cells = [],
                              parent = None, 
                              filename = None):
  """
  Helper to launch translation of a python function's AST, and then construct
  an untyped parakeet function from the arguments, refs, and translated body.
  """
  
  assert len(closure_vars) == len(closure_cells)
  closure_cell_dict = dict(zip(closure_vars, closure_cells))
  
  if filename is None and parent is not None:
    filename = parent.filename 
    
  translator = AST_Translator(globals_dict, closure_cell_dict, 
                              parent, function_name = name, filename = filename)

  assert not args.kwarg, "Parakeet doesn't support **kwargs, found in %s%s(%s)" % \
    (filename +":" if filename else "", name, args) 
  ssa_args, assignments = translator.translate_args(args)
  
  doc_string = None
  if len(body) > 0 and isinstance(body[0], ast.Expr):
    if isinstance(body[0].value, ast.Str):
      doc_string = body[0].value.s
      body = body[1:]
     
    
  _, body = translator.visit_block(body)
  body = assignments + body

  ssa_fn_name = names.fresh(name)

  # if function was nested in parakeet, it can have references to its
  # surrounding parakeet scope, which can't be captured with a python ref cell
  original_outer_names = translator.original_outer_names
  localized_outer_names = translator.localized_outer_names
  python_refs = translator.python_refs 
  ssa_args.prepend_nonlocal_args(localized_outer_names)
  if globals_dict:
    assert parent is None
    assert len(original_outer_names) == len(python_refs)
    return UntypedFn(ssa_fn_name, ssa_args, body, python_refs.values(), [])
  else:
    assert parent
    fn = UntypedFn(ssa_fn_name, ssa_args, body, [], original_outer_names, doc_string = doc_string)
    if len(original_outer_names) > 0:
      outer_ssa_vars = [parent.lookup(x) for x in original_outer_names]
      return syntax.Closure(fn, outer_ssa_vars)
    else:
      return fn

def strip_leading_whitespace(source):
  lines = source.splitlines()
  assert len(lines) > 0
  first_line = lines[0]
  n_removed = len(first_line) - len(first_line.lstrip())
  if n_removed > 0:
    return '\n'.join(line[n_removed:] for line in lines)
  else:
    return source

def translate_function_source(source, 
                              globals_dict, 
                              closure_vars = [],
                              closure_cells = [],
                              filename = None):
  assert len(closure_vars) == len(closure_cells)
  syntax_tree = ast.parse(strip_leading_whitespace(source))

  if isinstance(syntax_tree, (ast.Module, ast.Interactive)):
    assert len(syntax_tree.body) == 1
    syntax_tree = syntax_tree.body[0]
  elif isinstance(syntax_tree, ast.Expression):
    syntax_tree = syntax_tree.body
  
  assert isinstance(syntax_tree, ast.FunctionDef), \
      "Unexpected Python syntax node: %s" % ast.dump(syntax_tree)
  return translate_function_ast(syntax_tree.name, 
                                syntax_tree.args, 
                                syntax_tree.body, 
                                globals_dict, 
                                closure_vars,
                                closure_cells, 
                                filename = filename)

# python value of a user-defined function mapped to its
# untyped representation
_known_python_functions = {}

# keep track of which functions are being translated at this moment 
# to check for recursive calls 
_currently_processing = set([])


import threading 
def translate_function_value(fn, _lock = threading.RLock()):
  """
  Prevent two threads from clobbering the recursion logic by both entering 
  the translation code
  """
  with _lock: 
    return _translate_function_value(fn)

def _translate_function_value(fn):
  
  # if it's already a Parakeet function, just return it 
  if isinstance(fn, UntypedFn):
    return fn 
  
  # short-circuit logic for turning dtypes and Python types into 
  # functions for casting from any value to those types 
  if isinstance(fn, (np.dtype, int, long, float, bool)): 
    return build_untyped_cast_fn(fn)
      
  # any builtin or numpy library function should have an entry here
  from ..mappings import function_mappings 
  if fn in function_mappings:
    fn = function_mappings[fn]
 
  # ...unless we forgot to add it to mappings but some equivalent primitive 
  # got registered 
  if fn in prims.prim_lookup_by_value:
    fn = prims.prim_lookup_by_value[fn]
  
  # if the function has been wrapped with a decorator, unwrap it 
  while isinstance(fn, jit):
    fn = fn.f

  if fn in _known_python_functions:
    return _known_python_functions[fn]
  
  assert is_hashable(fn), "Can't convert unhashable value: %s" % (fn,)
  assert fn not in _currently_processing, \
    "Recursion detected through function value %s" % (fn,)

  original_fn = fn
  
  if isinstance(fn, Prim):
    fundef = build_untyped_prim_fn(fn)
  elif isinstance(fn, (Type, np.dtype, int, bool, long, float)):
    fundef = build_untyped_cast_fn(fn)
  elif isinstance(fn, type) and Expr in fn.mro():
    fundef = build_untyped_expr_fn(fn)  
  elif isinstance(fn, macro):
    fundef = fn.as_fn()
    
  else:
    # if it's not a macro or some sort of internal expression
    # then we're really dealing with a Python function
    # so get to work pulling apart its AST and translating
    # it into Parakeet IR
    assert type(fn) not in (types.BuiltinFunctionType, types.TypeType, np.ufunc, types.MethodType), \
      "Unsupported function: %s" % (fn,) 
    
    _currently_processing.add(fn) 
    try:
      assert hasattr(fn, 'func_globals'), "Expected function to have globals: %s" % fn
      assert hasattr(fn, 'func_closure'), "Expected function to have closure cells: %s" % fn
      assert hasattr(fn, 'func_code'), "Expected function to have code object: %s" % fn

      source = inspect.getsource(fn)
      filename = inspect.getsourcefile(fn)
    except:
      _currently_processing.remove(fn)
      assert False, "Parakeet couldn't access source of function %s" % fn 
    globals_dict = fn.func_globals

    free_vars = fn.func_code.co_freevars
    closure_cells = fn.func_closure
    if closure_cells is None: 
      closure_cells = ()
    try: 
      fundef = translate_function_source(source,
                                        globals_dict,
                                        free_vars,
                                        closure_cells, 
                                        filename = filename)
    except:
      _currently_processing.remove(fn)
      raise 
    if config.print_untyped_function:
      print "[ast_conversion] Translated %s into untyped function:\n%s" % (fn, repr(fundef))
  
    _currently_processing.remove(fn)              
    _known_python_functions[original_fn] = fundef
     
  _known_python_functions[fn] = fundef 
  
  return fundef 
