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
    print "fundef before inline", fundef
    print "args", args  
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
    # print "inlined_body", inlined_body
    self.blocks.current().extend(inlined_body)
    return result_var
  
  def transform_Call(self, expr):
    target = function_registry.typed_functions[expr.fn]
    if can_inline(target):
      return self.do_inline(target, expr.args)
    else:
      print "CAN'T INLINE", expr 
      return expr
  
  def pre_apply(self, old_fn):
    #print "Before inlining", old_fn 
    return old_fn 
  
  def post_apply(self, new_fn):
    # print "After inlining", new_fn
    return new_fn 
  

#def inline(fn):
#  return transform.cached_apply(Inliner, fn)