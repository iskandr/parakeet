import names 
from subst import subst_list
import syntax 
import transform 



from syntax_helpers import get_types 
from tuple_type import make_tuple_type
import function_registry 
import type_inference 

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
      return can_inline_block(stmt.body)
    elif isinstance(stmt, syntax.Return):
      return outer
    else:
      assert isinstance(stmt, syntax.Assign)
      return True  
  
def can_inline(fundef):
  return can_inline_block(fundef.body, outer = True)

class Inliner(transform.Transform):
  
  def wrap_formal(self, arg):
    """
    Args might be strings & tuples, whereas 
    we want them to be syntax nodes
    """
    if isinstance(arg, str):
      return syntax.Var(arg, self.type_env[arg])
    elif isinstance(arg, tuple):
      elts = map(self.wrap_formal, arg)
      elt_types = get_types(elts)
      tuple_t = make_tuple_type(elt_types)
      return syntax.Tuple(elts, type = tuple_t)
    else:
      return arg 

  def do_inline(self, fundef, args):
    rename_dict = {}
    for (name, t) in fundef.type_env.iteritems():
      new_name = names.refresh(name)
      rename_dict[name] = new_name
      self.type_env[new_name] = t
    arg_names = fundef.arg_names 
    n_expected = len(arg_names)
    n_given = len(args)
    assert n_expected ==  n_given, \
      "Function to be inlined expects %d args but given %d" % \
      (n_expected, n_given)
    new_formal_names = [rename_dict[x] for x in arg_names]
    
    for (arg_name, actual) in zip(new_formal_names, args):
      self.assign(self.wrap_formal(arg_name), actual)
    renamed_body = subst_list(fundef.body, rename_dict)
    result_var = self.fresh_var(fundef.return_type, "result")
    inlined_body = replace_returns(renamed_body, result_var)
    self.blocks.current().extend(inlined_body)
    return result_var
  
  def transform_Call(self, expr):
    if isinstance(expr.fn, str):
      target = function_registry.typed_functions[expr.fn]
    else:
      target = expr.fn 
    if can_inline(target):
      return self.do_inline(target, expr.args)
    else:
      print "CAN'T INLINE", expr 
      return expr
  
  

#def inline(fn):
#  return transform.cached_apply(Inliner, fn)