import ast
import inspect
from collections import OrderedDict

import syntax

import prims
from prims import Prim, prim_wrapper

from function_registry import untyped_functions, already_registered_python_fn
from function_registry import register_python_fn, lookup_python_fn

import names
from scoped_env import ScopedEnv

from common import dispatch
from args import FormalArgs, ActualArgs

from subst import subst, subst_list

reserved_names = {
  'True' : syntax.Const(True),
  'False' : syntax.Const(False),
  'None' : syntax.Const(None),
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

  def __init__(self, globals_dict = None, closure_cell_dict = None,
               outer_env = None):
    # assignments which need to get prepended at the beginning of the 
    # function 
    self.env = \
      ScopedEnv(outer_env = outer_env,
                closure_cell_dict = closure_cell_dict,
                globals_dict = globals_dict)
    
  def fresh_name(self, original_name):
    return self.env.fresh(original_name)

  def fresh_names(self, original_names):
    return map(self.env.fresh, original_names)

  def fresh_var(self, original_name):
    return self.env.fresh_var(original_name)

  def fresh_vars(self, original_names):
    return map(self.fresh_var, original_names)

  def current_block(self):
    return self.env.current_block()

  def current_scope(self):
    return self.env.current_scope()

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
      stmt = syntax.Assign(lhs, syntax.TupleProj(var, i))
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
        formals.add_positional(local_name)
        var = syntax.Var(arg_name)
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

  def get_name(self, name):
    if name in reserved_names:
      return reserved_names[name]
    else:
      return syntax.Var(self.env[name])

  def visit_Name(self, expr):
    assert isinstance(expr, ast.Name), "Expected AST Name object: %s" % expr
    old_name = expr.id
    return self.get_name(old_name)

  def create_phi_nodes(self, left_scope, right_scope, new_names = {}):
    """
    Phi nodes make explicit the possible sources of each variable's values and
    are needed when either two branches merge or when one was optionally taken.
    """
    merge = {}
    for (name, ssa_name) in left_scope.iteritems():
      left = syntax.Var(ssa_name)
      if name in right_scope:
        right = syntax.Var (right_scope[name])
      else:
        right = self.get_name(name)

      if name in new_names:
        new_name = new_names[name]
      else:
        new_name = self.env.fresh(name)
      merge[new_name] = (left, right)

    for (name, ssa_name) in right_scope.iteritems():
      if name not in left_scope:
        try:
          left = self.get_name(name)
          right = syntax.Var(ssa_name)

          if name in new_names:
            new_name = new_names[name]
          else:
            new_name = self.env.fresh(name)
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
      start = self.visit_expr(expr.lower)
      stop = self.visit_expr(expr.upper)
      step = self.visit_expr(expr.step)
      return syntax.Slice(start, stop, step) 

    def visit_ExtSlice():
      slice_elts = map(self.translate_slice, expr.dims)
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
    return syntax.PrimCall(prim, [ssa_left, ssa_right] )

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

  def visit_Call(self, expr):
    
    fn, args, keywords_list, starargs, kwargs = \
      expr.func, expr.args, expr.keywords, expr.starargs, expr.kwargs
    assert kwargs is None, "Dictionary of keyword args not supported"

    fn_val = self.visit(fn)
    
    positional = self.visit_list(args)
    
    keywords_dict = {}
    for kwd in keywords_list:
      keywords_dict[kwd.arg] = self.visit(kwd.value) 
    
    if starargs:
      starargs_expr = self.visit(starargs)
    else:
      starargs_expr = None
      
    actuals = ActualArgs(positional, keywords_dict, starargs_expr) 
       
    return syntax.Invoke(fn_val, actuals)

  def visit_List(self, expr):
    return syntax.Array(self.visit_list(expr.elts))

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
    true_block = [syntax.Assign(temp1, self.visit(expr.body))]
    false_block = [syntax.Assign(temp2, self.visit(expr.orelse))]
    merge = {result.name :  (temp1, temp2)}
    if_stmt = syntax.If(cond, true_block, false_block, merge)
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
    return syntax.Assign(ssa_lhs, ssa_rhs)

  def visit_Return(self, stmt):
    return syntax.Return(self.visit(stmt.value))

  def visit_If(self, stmt):
    cond = self.visit(stmt.test)
    true_scope, true_block  = self.visit_block(stmt.body)
    false_scope, false_block = self.visit_block(stmt.orelse)
    merge = self.create_phi_nodes(true_scope, false_scope)
    return syntax.If(cond, true_block, false_block, merge)

  def visit_While(self, stmt, counter = [0]):
    counter[0] = counter[0] + 1
    cond = self.visit(stmt.test)
    scope_after, body = self.visit_block(stmt.body)
    merge = {}
    substitutions = {}
    curr_scope = self.env.current_scope()
    for (k, name_after) in scope_after.iteritems():
      if k in self.env:
        name_before = self.env[k]
        new_name = names.fresh(k)
        merge[new_name] = (syntax.Var(name_before), syntax.Var(name_after))
        substitutions[name_before]  = new_name
        curr_scope[k] = name_after
    cond = subst(cond, substitutions)
    body = subst_list(body, substitutions)
    return syntax.While(cond, body, merge)

  def visit_block(self, stmts):
    self.env.push()
    curr_block = self.current_block()
    for stmt in stmts:
      parakeet_stmt = self.visit(stmt)
      curr_block.append(parakeet_stmt)
    return self.env.pop()

  def visit_FunctionDef(self, node):
    """
    Translate a nested function
    """
    fundef = \
      translate_function_ast(node, outer_env = self.env)
    closure_args = map(self.get_name, fundef.parakeet_nonlocals)
    local_name = self.env.fresh_var(node.name)
    closure = syntax.Closure(fundef.name, closure_args)
    return syntax.Assign(local_name, closure)


def translate_function_ast(function_def_ast, globals_dict = None,
                           closure_vars = [], closure_cells = [],
                           outer_env = None):
  """
  Helper to launch translation of a python function's AST, and then construct
  an untyped parakeet function from the arguments, refs, and translated body.
  """

  assert len(closure_vars) == len(closure_cells)
  closure_cell_dict = dict(*zip(closure_vars, closure_cells))

  translator = AST_Translator(globals_dict, closure_cell_dict, outer_env)

  ssa_args, assignments = translator.translate_args(function_def_ast.args)
  _, body = translator.visit_block(function_def_ast.body)
  body = assignments + body 
  ssa_fn_name = names.fresh(function_def_ast.name)

  refs = []
  ref_names = []
  for (ssa_name, ref) in translator.env.python_refs.iteritems():
    refs.append(ref)
    ref_names.append(ssa_name)

  # if function was nested in parakeet, it can have references to its
  # surrounding parakeet scope, which can't be captured with a python ref cell
  original_outer_names = translator.env.original_outer_names
  localized_outer_names = translator.env.localized_outer_names

  
  ssa_args.prepend_nonlocal_args(localized_outer_names + ref_names)
  
  fundef = syntax.Fn(ssa_fn_name, ssa_args, 
                     body,  
                     refs, original_outer_names)
  
  untyped_functions[fundef.name]  = fundef

  return fundef

def translate_function_source(source, globals_dict, closure_vars = [],
                              closure_cells = []):
  assert len(closure_vars) == len(closure_cells)
  syntax = ast.parse(source)
  if isinstance(syntax, ast.Module):
    assert len(syntax.body) == 1
    syntax = syntax.body[0]
  assert isinstance(syntax, ast.FunctionDef)
  return translate_function_ast(syntax, globals_dict, closure_vars,
                                closure_cells)

import adverb_registry 

def translate_function_value(fn):

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

    fundef = translate_function_source(source, globals_dict, free_vars, closure_cells)

    register_python_fn(fn, fundef)

    return fundef

