import __builtin__
import ast
import inspect
import types

import numpy as np

import config
import core_types 
import lib 
import names
import nested_blocks 
import prims
import scoped_dict
import syntax
import syntax_helpers 

from args import FormalArgs, ActualArgs
from collections import OrderedDict
from function_registry import already_registered_python_fn
from function_registry import register_python_fn, lookup_python_fn
from decorators import macro, jit 
from names import NameNotFound
from prims import Prim, prim_wrapper
from python_ref import GlobalValueRef, GlobalNameRef, ClosureCellRef

from subst import subst_expr, subst_stmt_list
from syntax import Assign, If, ForLoop, Var, PrimCall, Map
from syntax_helpers import none, true, false, one_i64, zero_i64

from mappings import function_mappings, method_mappings, property_mappings 

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
  init = syntax_helpers.none if init is None else init
  axis = syntax_helpers.zero_i64
  return syntax.Reduce(fn = translate_function_value(lib.identity), 
                       combine = fn, 
                       args = positional, 
                       axis = axis, 
                       init = init)

def mk_simple_fn(mk_body, input_name = "x", fn_name = "cast"):
  unique_arg_name = names.fresh(input_name)
  unique_fn_name = names.fresh(fn_name)
  var = syntax.Var(unique_arg_name)
  formals = FormalArgs()
  formals.add_positional(unique_arg_name, input_name)
  body = mk_body(var)
  return syntax.Fn(unique_fn_name, formals, body)
  
_function_wrapper_cache = {}
def mk_wrapper_function(p):
  """
  Generate wrappers for builtins and primitives
  """
  if p in _function_wrapper_cache:
    return _function_wrapper_cache[p]
  if isinstance(p, prims.Prim):
    f = prims.prim_wrapper(p)
  elif p in prims.prim_lookup_by_value:
    f = prims.prim_wrapper(prims.prim_lookup_by_value[p]) 
  else:
    assert isinstance(p, types.BuiltinFunctionType)
    assert p.__name__ in lib.__dict__, "Unsupported builtin: %s" % (p,)
    f = translate_function_value(lib.__dict__[p.__name__]) 
  _function_wrapper_cache[p] = f
  return f
  
       
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
  return syntax_helpers.is_python_constant(v) or \
         type(v) is np.dtype  or \
         is_function_value(v)

def value_to_syntax(v):
  if syntax_helpers.is_python_constant(v):
    return syntax_helpers.const(v)
  elif isinstance(v, np.dtype):
    x = names.fresh("x")
    fn_name = names.fresh("cast") 
    formals = FormalArgs()
    formals.add_positional(x, "x")
    body = [syntax.Return(syntax.Cast(syntax.Var(x), type=core_types.from_dtype(v)))]
    return syntax.Fn(fn_name, formals, body)
  else:
    assert is_function_value(v), "Can't make value %s : %s into static syntax" % (v, type(v))
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
    self.blocks = nested_blocks.NestedBlocks()

    self.parent = parent 
    self.scopes = scoped_dict.ScopedDictionary()
    
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
    return syntax.Var(self.fresh_name(name))

  def fresh_vars(self, original_names):
    return map(self.fresh_var, original_names)
  
  def current_block(self):
    return self.blocks.top()

  def current_scope(self):
    return self.scopes.top()


  def syntax_to_value(self, expr):
    if isinstance(expr, ast.Num):
      return expr.n
    elif isinstance(expr, ast.Tuple):
      return tuple(self.syntax_to_value(elt) for elt in expr.elts)
    elif isinstance(expr, ast.Name):
      return self.lookup_global(expr.id)
    elif isinstance(expr, ast.Attribute):
      left = self.syntax_to_value(expr.value)
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
      for (local_name, other_ref) in self.python_refs.iteritems():
        if ref == other_ref:
          return Var(local_name)
      local_name = names.fresh(name)
      self.scopes[name] = local_name
      self.original_outer_names.append(name)
      self.localized_outer_names.append(local_name)
      self.python_refs[local_name] = ref
      return Var(local_name)
    elif self.is_global(name):
      value = self.lookup_global(name)
      if is_static_value(value): 
        return value_to_syntax(value)
      else:
        return ExternalValue(value)
      
        
      #else:
      #  assert False, "External values must be scalars or functions"
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
      stmt = Assign(lhs, syntax.Index(var, syntax.Const(i)))
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
        formals.defaults[k] = self.syntax_to_value(expr)


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
        right = self.lookup(name)

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
    return syntax.Slice(start, stop, step)

  def visit_ExtSlice(self, expr):
    slice_elts = map(self.visit, expr.dims)
    if len(slice_elts) > 1:
      return syntax.Tuple(slice_elts)
    else:
      return slice_elts[0]
    

  def visit_UnaryOp(self, expr):
    ssa_val = self.visit(expr.operand)
    prim = prims.find_ast_op(expr.op)
    return syntax.PrimCall(prim, [ssa_val])

  def visit_BinOp(self, expr):
    ssa_left = self.visit(expr.left)
    ssa_right = self.visit(expr.right)
    prim = prims.find_ast_op(expr.op)
    return syntax.PrimCall(prim, [ssa_left, ssa_right])

  def visit_BoolOp(self, expr):
    values = map(self.visit, expr.values)
    prim = prims.find_ast_op(expr.op)
    # Python, strangely, allows more than two arguments to
    # Boolean operators
    result = values[0]
    for v in values[1:]:
      result = syntax.PrimCall(prim, [result, v])
    return result

  def visit_Compare(self, expr):
    lhs = self.visit(expr.left)
    assert len(expr.ops) == 1
    prim = prims.find_ast_op(expr.ops[0])
    assert len(expr.comparators) == 1
    rhs = self.visit(expr.comparators[0])
    return syntax.PrimCall(prim, [lhs, rhs])

  def visit_Subscript(self, expr):
    value = self.visit(expr.value)
    index = self.visit(expr.slice)
    return syntax.Index(value, index)

  def generic_visit(self, expr):
    raise UnsupportedSyntax(expr, 
                            function_name = self.function_name, 
                            filename = self.filename)
  
  def translate_builtin_call(self, value, positional, keywords_dict):
    if value is sum:
      return mk_reduce_call(prims.prim_wrapper(prims.add), positional, zero_i64)
    elif value is max:
      if len(positional) == 1:
        return mk_reduce_call(prims.prim_wrapper(prims.maximum), positional)
      else:
        assert len(positional) == 2
        return syntax.PrimCall(prims.maximum, positional)
    elif value is min:
      if len(positional) == 1:
        return mk_reduce_call(prims.prim_wrapper(prims.minimum), positional)
      else:
        assert len(positional) == 2
        return syntax.PrimCall(prims.minimum, positional)
    elif value is map:
      assert len(keywords_dict) == 0
      assert len(positional) > 1
      axis = keywords_dict.get("axis", None)
      return Map(fn = positional[0], args = positional[1:], axis = axis)
    elif isinstance(value, (types.BuiltinFunctionType, types.TypeType)) and \
         value.__name__ in lib.__dict__:
      parakeet_equiv = lib.__dict__[value.__name__]
      if isinstance(parakeet_equiv, macro):
        return parakeet_equiv.transform(positional, keywords_dict)
    fn = value_to_syntax(value)
    return syntax.Call(fn, ActualArgs(positional, keywords_dict))
    
      
  def visit(self, node):
    res = ast.NodeVisitor.visit(self, node)
    return res 
    
  def translate_value_call(self, value, positional, keywords_dict= {}, starargs_expr = None):
    if value in function_mappings:
      value = function_mappings[value]
      
    if isinstance(value, macro):
      return value.transform(positional, keywords_dict)
    elif is_user_function(value):
      return syntax.Call(translate_function_value(value), 
                           ActualArgs(positional, keywords_dict, starargs_expr))
    else:
      return self.translate_builtin_call(value, positional, keywords_dict)
    
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
      return syntax.Call(fn_node, actuals)
    else:
      assert isinstance(fn_node, ExternalValue)
      return self.translate_value_call(fn_node.value, 
                                       positional, keywords_dict, starargs_expr)

  def visit_List(self, expr):
    return syntax.Array(self.visit_list(expr.elts))
    
  def visit_Expr(self, expr):
    # dummy assignment to allow for side effects on RHS
    lhs = self.fresh_var("dummy")
    if isinstance(expr.value, ast.Str):
      return syntax.Assign(lhs, zero_i64)
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
    value = self.visit(expr.value)
    attr = expr.attr
    if isinstance(value, ExternalValue):
      value = value.value 
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
        return syntax.Call(translate_function_value(fn),
                            ActualArgs(positional = (value,)))  
    elif attr in method_mappings:
      fn_python = method_mappings[attr]
      fn_syntax = translate_function_value(fn_python)
      return syntax.Closure(fn_syntax, args=(value,))
    else:
      assert False, "Attribute %s not supported" % attr 
    # return syntax.Attribute(value, expr.attr)

  def visit_Num(self, expr):
    return syntax.Const(expr.n)

  def visit_Tuple(self, expr):
    return syntax.Tuple(self.visit_list(expr.elts))

  def visit_IfExp(self, expr):
    temp1, temp2, result = self.fresh_vars(["if_true", "if_false", "if_result"])
    cond = self.visit(expr.test)
    true_block = [Assign(temp1, self.visit(expr.body))]
    false_block = [Assign(temp2, self.visit(expr.orelse))]
    merge = {result.name : (temp1, temp2)}
    if_stmt = If(cond, true_block, false_block, merge)
    self.current_block().append(if_stmt)
    return result

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

  def visit_For(self, stmt):
    assert not stmt.orelse 
    var = self.visit_lhs(stmt.target)
    assert isinstance(var, Var)
    seq = self.visit(stmt.iter)
    body, merge, _ = self.visit_loop_body(stmt.body)
    if isinstance(seq, syntax.Range):
      return ForLoop(var, seq.start, seq.stop, seq.step, body, merge)
    else:
      seq_name = self.fresh_name("seq")
      seq_var = Var(seq_name)
      self.current_block().append(Assign(seq_var, seq))
      len_fn = translate_function_value(len)
      n = syntax.Call(len_fn, ActualArgs([seq_var]))
      start = zero_i64
      step = one_i64
      loop_counter_name = self.fresh_name('loop_counter')
      loop_var = Var(loop_counter_name)
      body = [Assign(var, syntax.Index(seq_var, loop_var))] + body
      return ForLoop(loop_var, start, n, step, body, merge)
    
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
    
def translate_function_ast(name, 
                           args, 
                           body, 
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

  ssa_args, assignments = translator.translate_args(args)
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
    return syntax.Fn(ssa_fn_name, ssa_args, body, python_refs.values(), [])
  else:
    assert parent
    fn = syntax.Fn(ssa_fn_name, ssa_args, body, [], original_outer_names)
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
  syntax = ast.parse(strip_leading_whitespace(source))

  if isinstance(syntax, (ast.Module, ast.Interactive)):
    assert len(syntax.body) == 1
    syntax = syntax.body[0]
  elif isinstance(syntax, ast.Expression):
    syntax = syntax.body
  
  assert isinstance(syntax, ast.FunctionDef), \
      "Unexpected Python syntax node: %s" % ast.dump(syntax)
  return translate_function_ast(syntax.name, 
                                syntax.args, 
                                syntax.body, 
                                globals_dict, 
                                closure_vars,
                                closure_cells, 
                                filename = filename)

def translate_function_value(fn, _currently_processing = set([])):
  if fn in function_mappings:
    fn = function_mappings[fn]
  
  if fn in prims.prim_lookup_by_value:
    fn = prims.prim_lookup_by_value[fn]
  
  # if the function has been wrapped with a decorator, unwrap it 
  while isinstance(fn, jit):
    fn = fn.f 
  
  # print ">>", fn   
  assert type(fn) not in (types.BuiltinFunctionType, types.TypeType), \
    "Unsupported primitive: %s" % (fn,) 

  if already_registered_python_fn(fn):
    return lookup_python_fn(fn)
  
  assert is_hashable(fn), "Can't convert unhashable value: %s" % (fn,)
  assert fn not in _currently_processing, \
    "Recursion detected through function value %s" % (fn,)
  _currently_processing.add(fn)
  original_fn = fn
  
  if isinstance(fn, Prim):
    fundef = prim_wrapper(fn)
  
    
  # TODO: Right now we can only deal with adverbs over a fixed axis and fixed
  # number of args due to lack of support for a few language constructs:
  # - variable number of args packed as a tuple i.e. *args
  # - keyword arguments packed as a...? ...struct of some kind? i.e. **kwds
  # - unpacking tuples and unpacking structs

  elif isinstance(fn, macro):
    fundef = fn.as_fn()
  else:
    assert hasattr(fn, 'func_globals'), "Expected function to have globals: %s" % fn
    assert hasattr(fn, 'func_closure'), "Expected function to have closure cells: %s" % fn
    assert hasattr(fn, 'func_code'), "Expected function to have code object: %s" % fn
    source = inspect.getsource(fn)
    filename = inspect.getsourcefile(fn)
    globals_dict = fn.func_globals

    free_vars = fn.func_code.co_freevars
    closure_cells = fn.func_closure
    if closure_cells is None: closure_cells = ()
    fundef = translate_function_source(source,
                                        globals_dict,
                                        free_vars,
                                        closure_cells, 
                                        filename = filename)
    if config.print_untyped_function:
      print "[ast_conversion] Translated %s into untyped function:\n%s" % (fn, repr(fundef))
                
  register_python_fn(original_fn, fundef)
  register_python_fn(fn, fundef)
  _currently_processing.remove(original_fn)
  return fundef