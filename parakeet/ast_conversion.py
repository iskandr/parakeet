import ast
import inspect
import types

import numpy as np

import config
import core_types 
import names
import nested_blocks 
import prims
import scoped_dict
import syntax
import syntax_helpers 

from adverbs import Map
from args import FormalArgs, ActualArgs
from collections import OrderedDict
from common import dispatch
from function_registry import already_registered_python_fn
from function_registry import register_python_fn, lookup_python_fn
from decorators import macro, jit 
from names import NameNotFound
from prims import Prim, prim_wrapper
from python_ref import GlobalRef, ClosureCellRef
from scoped_env import ScopedEnv
from subst import subst_expr, subst_stmt_list
from syntax import Assign, If, ForLoop, Var, PrimCall
from syntax_helpers import none, true, false, one_i64, zero_i64



reserved_names = {
  'True' : true,
  'False' : false,
  'None' : none,
}

def translate_default_arg_value(arg):
  if isinstance(arg, ast.Num):
    return arg.n # syntax.Const(arg.n)
  else:
    assert isinstance(arg, ast.Name)
    name = arg.id
    assert name in reserved_names
    return reserved_names[name].value

class AST_Translator(ast.NodeVisitor):
  def __init__(self, globals_dict=None, closure_cell_dict=None,
               parent=None):
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

  def lookup_global(self, key):
    if self.globals:
      if isinstance(key, str):
        return self.globals[key]
      else:
        assert isinstance(key, (list, tuple))  
        value = self.globals_dict[key[0]]
        for elt in key[1:]:
          value = getattr(value, elt)
        return value 
    else:
      return self.parent.lookup_global(key)
    
  def is_global(self, key):

    if isinstance(key, str) and key in self.scopes:
      return False 
    if isinstance(key, (list,tuple)) and key[0] in self.scopes:
      return False 
    
    if self.globals:
      if isinstance(key, str):
        return key in self.globals 
      else:
        assert isinstance(key, (list, tuple))
        return key[0] in self.globals 
    else:
      return self.parent.is_global(key)
  
  def lookup(self, name):

    if name in reserved_names:
      return reserved_names[name]
    elif name in self.scopes:
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
    elif self.is_global(name):
      ref = GlobalRef(self.globals_dict, name)
    else:
      raise NameNotFound(name)
    
    value = ref.deref()
    
    try:
      # if a value is a function then immediately parse it 
      # and return its Parakeet representation 
      
      return translate_function_value(value)
    except:
      # on the other hand, don't inline constants but rather keep them in an indirect 
      # form 
      
      # make sure we haven't already recorded a reference to this value under a different name
      for (local_name, other_ref) in self.python_refs.iteritems():
        if ref == other_ref:
          return Var(local_name)
      local_name = names.fresh(name)
      self.scopes[name] = local_name
      self.original_outer_names.append(name)
      self.localized_outer_names.append(local_name)
      self.python_refs[local_name] = ref
      return Var(local_name)
  
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
      for (k,v) in zip(local_names, args.defaults):
        formals.defaults[k] = translate_default_arg_value(v)

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

  def visit_slice(self, expr):
    def visit_Index():
      return self.visit(expr.value)

    def visit_Ellipsis():
      raise RuntimeError("Ellipsis operator unsupported")

    def visit_Slice():
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

    def visit_ExtSlice():
      slice_elts = map(self.visit_slice, expr.dims)
      if len(slice_elts) > 1:
        return syntax.Tuple(slice_elts)
      else:
        return slice_elts[0]
    result = dispatch(expr, 'visit')
    return result

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
    index = self.visit_slice(expr.slice)
    return syntax.Index(value, index)

  def generic_visit(self, expr):
    raise RuntimeError("Unsupported: %s : %s" % (ast.dump(expr),
                                                 expr.__class__.__name__))

  def build_attribute_chain(self, expr):
    assert isinstance(expr, (ast.Name, ast.Attribute))
    if isinstance(expr, ast.Name):
      return [expr.id]
    else:
      left = self.build_attribute_chain(expr.value)
      left.append (expr.attr)
      return left

  def lookup_attribute_chain(self, attr_chain):
    assert len(attr_chain) > 0
    value = self.globals
    for name in attr_chain:
      if hasattr(value, '__getitem__') and name in value:
        value = value[name]
      elif hasattr(value, '__dict__') and name in value.__dict__:
        value = value.__dict__[name]
      else:
        value = getattr(value, name)
    return value

  def translate_builtin(self, value, positional, keywords_dict):
    def mk_reduction(fn, positional, init = None):
      import adverb_wrapper
      wrapper = adverb_wrapper.untyped_reduce_wrapper(None, fn)
      if init:
        keywords = keywords = {'init': init}
      else:
        keywords = {}
      positional = [fn] + list(positional)
      args = ActualArgs(positional = positional, keywords = keywords) 
      return syntax.Call(wrapper, args)
    if value == sum:
      return mk_reduction(prims.add, positional, zero_i64)
    elif value == max:
      if len(positional) == 1:
        return mk_reduction(prims.maximum, positional)
      else:
        assert len(positional) == 2
        return syntax.PrimCall(prims.maximum, positional)
    elif value == min:
      if len(positional) == 1:
        return mk_reduction(prims.minimum, positional)
      else:
        assert len(positional) == 2
        return syntax.PrimCall(prims.minimum, positional)
    elif value == abs:
      assert len(keywords_dict) == 0
      assert len(positional) == 1
      return syntax.PrimCall(prims.abs, positional)
    elif value == map:
      assert len(keywords_dict) == 0
      assert len(positional) > 1
      return Map(fn = positional[0], args = positional[1:], axis = 0)
    elif value == range or value == np.arange:
      assert len(keywords_dict) == 0
      n_args = len(positional)
      
      if n_args == 1:
        positional = [zero_i64] + positional + [one_i64]
      elif n_args == 2:
        positional.extend([one_i64])
      else:
        assert n_args == 3
      return syntax.Range(*positional)
    elif value == len:
      assert len(keywords_dict) == 0
      assert len(positional) == 1
      return syntax.Len(positional[0])
    elif value == float:
      return syntax.Cast(value = positional[0], type = core_types.Float64)
    elif value == int:
      return syntax.Cast(value = positional[0], type = core_types.Int64)
    else:
      assert value == slice, "Builtin not implemented: %s" % value 
      assert len(keywords_dict) == 0
      return syntax.Slice(*positional)

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
      
    attr_chain = self.build_attribute_chain(fn)
    root = attr_chain[0]
    if root in self.scopes:
      assert len(attr_chain) == 1
      fn_node = Var(self.scopes[root])
      actuals = ActualArgs(positional, keywords_dict, starargs_expr)
      return syntax.Call(fn_node, actuals)
    else:
      if self.is_global(attr_chain):
        value = self.lookup_global(attr_chain)
        # value = self.lookup_attribute_chain(attr_chain)
        if isinstance(value, macro):
          return value.transform(positional, keywords_dict)
        elif isinstance(value, prims.Prim):
          return syntax.PrimCall(value, positional)
        elif isinstance(value, types.BuiltinFunctionType):
          return self.translate_builtin(value, positional, keywords_dict)
        
        elif hasattr(value, '__call__'):
          # if it's already been wrapped, extract the underlying function value
          if isinstance(value, jit):
            value = value.f
          fn_node = translate_function_value(value)
          
          actuals = ActualArgs(positional, keywords_dict, starargs_expr)
          return syntax.Call(fn_node, actuals)
        else:
          assert False, "depends on global %s" % attr_chain 
           
      elif len(attr_chain) == 1 and root in __builtins__:
        value = __builtins__[root]
        return self.translate_builtin(value, positional, keywords_dict)
  # assume that function must be locally defined 
    assert isinstance(expr.func, ast.Name)
    fn_node = self.lookup(expr.func.id) 
    actuals = ActualArgs(positional, keywords_dict, starargs_expr)
    return syntax.Call(fn_node, actuals)
    
    
  def visit_List(self, expr):
    return syntax.Array(self.visit_list(expr.elts))
    
  def visit_Expr(self, expr):
    assert False, "Expression statement not supported: %s" % ast.dump(expr)
    
  def visit_ListComp(self, expr):
    gens = expr.generators
    assert len(gens) == 1
    gen = gens[0]
    target = gen.target
    assert target.__class__ is ast.Name
    # build a lambda as a Python ast representing 
    # what we do to each element 

    py_fn = ast.FunctionDef(name = "comprehension_map", 
                                args = ast.arguments(
                                  args = [target], 
                                  vararg = None, 
                                  kwarg = None, 
                                  defaults = ()),
                                body = [ast.Return(expr.elt)], 
                                decorator_list = ())
    
    fn = translate_function_ast(py_fn, parent = self)
    print str(fn)
    seq = self.visit(gen.iter)
    ifs = gen.ifs
    assert len(ifs) == 0
    return Map(fn, args=(seq,), axis = 0)
      
  def visit_Attribute(self, expr):
    # TODO:
    # Recursive lookup to see if:
    #  (1) base object is local, if so-- create chain of attributes
    #  (2) base object is global but an adverb primitive-- use it locally
    #      without adding it to nonlocals
    #  (3) not local at all-- in which case, add the whole chain of strings
    #      to nonlocals
    value = self.visit(expr.value)
    return syntax.Attribute(value, expr.attr)

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
      return self.visit(lhs)

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
      n = syntax.Len(seq)
      start = zero_i64
      stop = n # syntax.PrimCall(prims.subtract, [n, one_i64])
      step = one_i64
      loop_counter_name = self.fresh_name('i')
      loop_var = Var(loop_counter_name)
      body = [Assign(var, syntax.Index(seq, loop_var))] + body
      return ForLoop(loop_var, start, stop, step, body, merge)
    
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
    fundef = translate_function_ast(node, parent = self)
    local_var = self.fresh_var(node.name)
    return Assign(local_var, fundef)

def translate_function_ast(function_def_ast, globals_dict = None,
                           closure_vars = [], closure_cells = [],
                           parent = None):
  """
  Helper to launch translation of a python function's AST, and then construct
  an untyped parakeet function from the arguments, refs, and translated body.
  """
  
  assert len(closure_vars) == len(closure_cells)
  closure_cell_dict = dict(zip(closure_vars, closure_cells))

  translator = AST_Translator(globals_dict, closure_cell_dict, parent)

  ssa_args, assignments = translator.translate_args(function_def_ast.args)
  _, body = translator.visit_block(function_def_ast.body)
  body = assignments + body

  ssa_fn_name = names.fresh(function_def_ast.name)

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

def translate_function_source(source, globals_dict, closure_vars = [],
                              closure_cells = []):
  assert len(closure_vars) == len(closure_cells)
  syntax = ast.parse(strip_leading_whitespace(source))

  if isinstance(syntax, (ast.Module, ast.Interactive)):
    assert len(syntax.body) == 1
    syntax = syntax.body[0]
  elif isinstance(syntax, ast.Expression):
    syntax = syntax.body
  assert isinstance(syntax, ast.FunctionDef), \
      "Unexpected Python syntax node: %s" % syntax
  return translate_function_ast(syntax, globals_dict, closure_vars,
                                closure_cells)

import adverb_registry
def translate_function_value(fn):
  # if the function has been wrapped with a decorator, unwrap it 
  while isinstance(fn, jit):
    fn = fn.f 
  
  if fn in prims.prim_lookup_by_value:
    fn = prims.prim_lookup_by_value[fn]
    
  if already_registered_python_fn(fn):
    return lookup_python_fn(fn)
  elif isinstance(fn, Prim):
    return prim_wrapper(fn)
    

  # TODO: Right now we can only deal with adverbs over a fixed axis and fixed
  # number of args due to lack of support for a few language constructs:
  # - variable number of args packed as a tuple i.e. *args
  # - keyword arguments packed as a...? ...struct of some kind? i.e. **kwds
  # - unpacking tuples and unpacking structs
  elif adverb_registry.is_registered(fn):
    return adverb_registry.get_wrapper(fn)
  else:
    assert hasattr(fn, 'func_globals'), \
        "Expected function to have globals: %s" % fn
    assert hasattr(fn, 'func_closure'), \
        "Expected function to have closure cells: %s" % fn
    assert hasattr(fn, 'func_code'), \
        "Expected function to have code object: %s" % fn

    source = inspect.getsource(fn)

    globals_dict = fn.func_globals

    free_vars = fn.func_code.co_freevars
    closure_cells = fn.func_closure
    if closure_cells is None:
      closure_cells = ()

    #print "[translate_function_value] Translating: ", source 
    fundef = translate_function_source(source, globals_dict, free_vars, closure_cells)
    #print "[translate_function_value] Produced:", repr(fundef)
    register_python_fn(fn, fundef)

    if config.print_untyped_function:
      print "[ast_conversion] Translated %s into untyped function:\n%s" % \
            (fn, repr(fundef))
    return fundef
