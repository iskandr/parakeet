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
    # to lift this restriction we'd have to track which branches 
    # returned of If statements and Loops, and deal with merging
    # returned values and/or simulating unstructured control flow
    # with guard values like "if !returned: ..." 
    assert isinstance(stmt, syntax.Assign), \
      "Only straight-line code can be inlined (for now)"
    return stmt   

def replace_returns(stmts, output_var):
  return [replace_return(stmt, output_var) for stmt in stmts]  
  
def no_control_flow(stmt):
  return isinstance(stmt, (syntax.Assign, syntax.Return))
  
def can_inline(fundef):
  return all(map(no_control_flow, fundef.body))

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
      
    old_formals = fundef.args
    new_formals = old_formals.transform(
      name_fn = lambda k: rename_dict[k], extract_name = True)
    arg_slots = map(self.wrap_formal, new_formals.arg_slots)
    
    for (formal, actual) in zip(arg_slots, args):
      self.assign(formal, actual)
    renamed_body = subst_list(fundef.body, rename_dict)
    result_var = self.fresh_var(fundef.return_type, "result")
    inlined_body = replace_returns(renamed_body, result_var)
    self.blocks.current().extend(inlined_body)
    return result_var
  
  def transform_Call(self, expr):
    target = function_registry.typed_functions[expr.fn]
    if can_inline(target):
      return self.do_inline(target, expr.args)
    else:
      return expr
    
  def transform_Invoke(self, expr):
    closure_t = expr.closure.type 
    
    if len(closure_t.args) == 0:
      arg_types = get_types(expr.args)
      typed_fundef = type_inference.specialize(closure_t.fn, arg_types)
      if can_inline(typed_fundef):
        return self.do_inline(typed_fundef, expr.args)
    return expr 
  
  
_inline_cache = {}
def inline(fn):
  if fn.name in _inline_cache:
    return _inline_cache[fn.name]
  else:
    return Inliner(fn).apply()    