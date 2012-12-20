

import transform


import syntax 
import names 
from subst import subst_list

def replace_return(stmt, output_var):
  """
  Change any returns into assignments to the output var
  """
  if isinstance(stmt, syntax.Return):
    return syntax.Assign(output_var, stmt.value)
  else:
    return stmt

def replace_returns(stmts, output_var):
  return [replace_return(stmt, output_var) for stmt in stmts]

def can_inline_block(stmts, outer = False):
  for stmt in stmts:
    if isinstance(stmt, syntax.If):
      return can_inline_block(stmt.true) and can_inline_block(stmt.false)
    elif isinstance(stmt, syntax.While):
      if not can_inline_block(stmt.body):
        return False
    elif isinstance(stmt, syntax.Return):
      if not outer:
        return False
    else:
      assert isinstance(stmt, syntax.Assign)
  return True

def can_inline(fundef):
  return can_inline_block(fundef.body, outer = True)

def do_inline(src_fundef, args, dest_type_env, dest_block):
  rename_dict = {}
  for (name, t) in src_fundef.type_env.iteritems():
    new_name = names.refresh(name)
    rename_dict[name] = new_name
    dest_type_env[new_name] = t
  arg_names = src_fundef.arg_names
  n_expected = len(arg_names)
  n_given = len(args)
  assert n_expected ==  n_given, \
      "Function %s expects %d args (%s) but given %d" % \
      (src_fundef, n_expected, ",".join(src_fundef.arg_names), n_given)
  new_formal_names = [rename_dict[x] for x in arg_names]

  for (arg_name, actual) in zip(new_formal_names, args):
    t = dest_type_env[arg_name]
    var = syntax.Var(arg_name, type = t )
    dest_block.append(syntax.Assign(var, actual))
    
  renamed_body = subst_list(src_fundef.body, rename_dict)
  result_name = names.fresh("result")
  dest_type_env[result_name] = src_fundef.return_type 
  result_var = syntax.Var(result_name, type = src_fundef.return_type)
  new_body = replace_returns(renamed_body, result_var)
  dest_block.extend(new_body) 
  return result_var 
  

class Inliner(transform.Transform):
  
  def __init__(self, fn):
    transform.Transform.__init__(self, fn)
    self.count = 0 

  def transform_Call(self, expr):
    if isinstance(expr.fn, str):
      target = syntax.TypedFn.registry[expr.fn]
    else:
      target = expr.fn
    if can_inline(target):
      self.count += 1
      curr_block = self.blocks.current()
      result_var = do_inline(target, expr.args, self.type_env, curr_block)
      return result_var 
    else:
      return expr
