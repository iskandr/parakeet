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
from args import Args 

reserved_names = { 
  'True' : syntax.Const(True), 
  'False' : syntax.Const(False), 
  'None' : syntax.Const(None), 
}

def translate_default_arg_value(arg):
  if isinstance(arg, ast.Num):
    return syntax.Const (arg.n)
  else:
    assert isinstance(arg, ast.Name)
    name = arg.id 
    assert name in reserved_names
    return reserved_names[name]
  
def translate_positional(arg):
  if isinstance(arg, ast.Name):
    return syntax.Var(arg.id)
  elif isinstance(arg, ast.Tuple):
    return syntax.Tuple(translate_positional_args(arg.elts))  
  
def translate_positional_args(args):
  return map(translate_positional, args) 

def translate_args(args):
  assert not args.vararg
  assert not args.kwarg
  
  positional = translate_positional_args(args.args)
  defaults = OrderedDict()
  for (k,v) in args.defaults:
    assert isinstance(k, ast.Name)
    defaults[k] = translate_default_arg_value(v)
  return Args(positional, defaults)

#def collect_defs_from_node(node):
#  """
#  Recursively traverse nested statements and collect 
#  variable names from the left-hand-sides of assignments,
#  ignoring variables if they appear within a slice or index
#  """
#  if isinstance(node, ast.While):
#    return collect_defs_from_list(node.body + node.orelse)
#  elif isinstance(node, ast.If):
#    return collect_defs_from_list(node.body + node.orelse)
#  if isinstance(node, ast.Assign):
#    return collect_defs_from_list(node.targets)
#  elif isinstance(node, ast.Name):
#    return set([node.id])
#  elif isinstance(node, ast.Tuple):
#    return collect_defs_from_list(node.elts)
#  elif isinstance(node, ast.List):
#    return collect_defs_from_list(node.elts)
#  else:
#    return set([])
        
#def collect_defs_from_list(nodes):
#  assert isinstance(nodes, list)
#  defs = set([])
#  for node in nodes:
#    defs.update(collect_defs_from_node(node))
#  return defs 


def subst(node, rename_dict):
  if isinstance(node, syntax.Var):
    return syntax.Var(rename_dict.get(node.name, node.name) )
  if isinstance(node, (syntax.Expr, syntax.Stmt)):
    new_values = {}
    for member_name in node.members:
      old_v = getattr(node, member_name)
      new_values[member_name] = subst(old_v, rename_dict)
    new_node = node.__class__(**new_values)
    return new_node

  elif isinstance(node, list):
    
    return subst_list(node, rename_dict)
  elif isinstance(node, tuple):
    return subst_tuple(node, rename_dict)
  elif isinstance(node, dict):
    return subst_dict(node, rename_dict)
  else:
    return node 

def subst_dict(old_dict, rename_dict):
  new_dict = {}
  for (k,v) in old_dict.iteritems():
    new_dict[subst(k, rename_dict)] = subst(v, rename_dict)
  return new_dict 

def subst_list(nodes, rename_dict):
  return [subst(node, rename_dict) for node in nodes]

def subst_tuple(elts, rename_dict):
  return tuple(subst_list(elts, rename_dict))

class AST_Translator(ast.NodeVisitor):

  def __init__(self, globals_dict = None, closure_cell_dict = None, outer_env = None):
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
      raise RuntimeError("Slice unsupported")
    
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
    raise RuntimeError("Unsupported: %s" % expr) 
  
  def visit_Call(self, expr):
    fn, args, kwargs, starargs = \
      expr.func, expr.args, expr.kwargs, expr.starargs
    assert kwargs is None, "Dictionary of keyword args not supported"
    assert starargs is None, "List of varargs not supported"
    fn_val = self.visit(fn)
    arg_vals = self.visit_list(args)
    return syntax.Invoke(fn_val, arg_vals) 
    
  def visit_List(self, expr):
    return syntax.Array(self.visit_list(expr.elts))
  
  def visit_Attribute(self, expr):
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
                           closure_vars = [], closure_cells = [], outer_env = None):
  """
  Helper to launch translation of a python function's AST, and then construct 
  an untyped parakeet function from the arguments, refs, and translated body.
  """
  
  assert len(closure_vars) == len(closure_cells)
  closure_cell_dict = dict(*zip(closure_vars, closure_cells))
  
  translator = AST_Translator(globals_dict, closure_cell_dict, outer_env)

  direct_args = translate_args(function_def_ast.args)
  ssa_args = direct_args.transform(translator.env.fresh, extract_name = True) 
  _, body = translator.visit_block(function_def_ast.body)
  ssa_fn_name = names.fresh(function_def_ast.name)
  
  refs = []
  ref_names = []
  for (ssa_name, ref) in translator.env.python_refs.iteritems():
    refs.append(ref)
    ref_names.append(ssa_name)
    
  # if function was nested in parakeet, it can have references to its surrounding parakeet 
  # scope, which can't be captured with a python ref cell 
  original_outer_names = translator.env.original_outer_names
  localized_outer_names = translator.env.localized_outer_names
  
  nonlocal_ssa_args = ref_names + localized_outer_names
  full_args = Args(ssa_args.positional, ssa_args.defaults, nonlocals = nonlocal_ssa_args)
  
  fundef = syntax.Fn(ssa_fn_name, full_args, body,  refs, original_outer_names)
  untyped_functions[fundef.name]  = fundef
  return fundef   

def translate_function_source(source, globals_dict, closure_vars = [], closure_cells = []):
  assert len(closure_vars) == len(closure_cells)
  syntax = ast.parse(source)
  if isinstance(syntax, ast.Module):
    assert len(syntax.body) == 1
    syntax = syntax.body[0]
  assert isinstance(syntax, ast.FunctionDef)
  return translate_function_ast(syntax, globals_dict, closure_vars, closure_cells)
 
import adverb_helpers
 
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
  elif adverb_helpers.is_registered_adverb(fn):
    return adverb_helpers.get_adverb_wrapper(fn)
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
    # print fundef 
    register_python_fn(fn, fundef)
    # print "Translated", fundef 
    return fundef   
