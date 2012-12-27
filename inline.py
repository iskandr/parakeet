import names

import syntax
from syntax import If, Assign, While, Return, Var, TypedFn, Const 
import syntax_visitor

from subst import subst_stmt_list
from transform import Transform

class FoundCall(Exception):
  pass

class ContainsCalls(syntax_visitor.SyntaxVisitor):
  def visit_expr(self, expr):
    if expr.__class__ is syntax.Call:
      raise FoundCall

  def visit_fn(self, fn):
    try:
      self.visit_block(fn.body)
    except FoundCall:
      return True
    return False

def contains_calls(fn):
  return ContainsCalls().visit_fn(fn)

def replace_returns(stmts, output_var):
  """
  Change any returns at the outer scope into assignments to the output var
  """
  for (i,stmt) in enumerate(stmts):
    if stmt.__class__ is Return:
      stmts[i] = syntax.Assign(output_var, stmt.value)
    
def can_inline_block(stmts, outer = False):
  for stmt in stmts:
    stmt_class = stmt.__class__
    if stmt_class is If:
      return can_inline_block(stmt.true) and can_inline_block(stmt.false)
    elif stmt_class is While:
      if not can_inline_block(stmt.body):
        return False
    elif stmt_class is Return:
      if not outer:
        return False
    else:
      assert stmt_class is Assign
  return True

def can_inline(fundef):
  return can_inline_block(fundef.body, outer = True)

def replace_return_with_var(body, type_env, return_type):
  result_name = names.fresh("result")
  type_env[result_name] = return_type
  result_var = Var(result_name, type = return_type)
  replace_returns(body, result_var)
  return result_var

def do_inline(src_fundef, args, dest_type_env, dest_block):

  arg_names = src_fundef.arg_names
  n_expected = len(arg_names)
  n_given = len(args)
  assert n_expected ==  n_given, \
      "Function %s expects %d args (%s) but given %d" % \
      (src_fundef, n_expected, ",".join(src_fundef.arg_names), n_given)
  
  rename_dict = {}
  
  def rename_var(old_name):
    t = src_fundef.type_env[old_name]
    new_name = names.refresh(old_name)
  
    new_var = Var(new_name, type = t)
    rename_dict[old_name] = new_var 
    dest_type_env[new_name] = t
    return new_var 
    
  for (old_formal, actual) in zip(arg_names, args):
    if actual.__class__ in (Var, Const):
      rename_dict[old_formal] = actual 
    else: 
      new_var = rename_var(old_formal)
      dest_block.append(Assign(new_var, actual))
  
  for old_name in src_fundef.type_env.iterkeys():
    if old_name not in arg_names:
      rename_var(old_name)

  
  new_body = subst_stmt_list(src_fundef.body, rename_dict)
  result_var = replace_return_with_var(new_body, dest_type_env,
                                       src_fundef.return_type)
  dest_block.extend(new_body)

  return result_var

class Inliner(Transform):
  def __init__(self):
    Transform.__init__(self)
    self.count = 0

  def transform_Call(self, expr):
    if isinstance(expr.fn, str):
      target = syntax.TypedFn.registry[expr.fn]
    else:
      target = expr.fn
    if target.__class__ is TypedFn and can_inline(target):
      self.count += 1
      curr_block = self.blocks.current()
      result_var = do_inline(target, expr.args, self.type_env, curr_block)
      return result_var
    else:
      return expr

  def apply(self, fn):
    if contains_calls(fn):
      return  Transform.apply(self, fn)
    else:
      return fn
