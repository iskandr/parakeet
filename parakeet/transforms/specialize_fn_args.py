from .. import names
from ..builder import build_fn
from ..syntax import Const, Var
from ..syntax.helpers import is_constant
from inline import Inliner
from transform import Transform

class SpecializeFnArgs(Transform):
  """
  Specialize the arguments to Map, Reduce, Scan, etc..  in two ways (including closure args):
    - constant arguments get moved into the function and are removed
      from the arguments list 
    - if an argument is repeated, only keep the first one. Be careful to not remove two instances
      of an array if it's being traversed along two different axes.
  
  TODO:
    - also get rid of unused arguments 
  """
  
  
  def build_arg_mapping(self, outer_closure_args, outer_array_args, inner_arg_names):
    """
    This function examines a list of actual args to a function (including its closure values), 
    and constructs:
      - a list of which closure and array args to keep
      - a list of which inner names to keep 
      - a dictionary associating discarded inner names with constants
      - a dictionary associating discarded inner names with kept names 
    """
    new_closure_args = []      
    remap = {}
    const = {}
    new_inner_names = []
      
    for pos, arg in enumerate(outer_closure_args):
      inner_name = inner_arg_names[pos]
      if arg.__class__ is Const:
        const[inner_name] = arg 
      elif arg in new_closure_args: 
        prev_pos = new_closure_args.index(arg)
        remap[inner_name] = new_inner_names[prev_pos]
      else:
        new_inner_names.append(inner_name)
        new_closure_args.append(arg)
      
    
    new_array_args = []
    for array_arg_pos, arg in enumerate(outer_array_args):
      pos = array_arg_pos + len(outer_closure_args)
      inner_name = inner_arg_names[pos]
      if arg.__class__ is Const:
        const[inner_name] = arg 
      elif arg in new_array_args:
        prev_pos = len(new_closure_args) + new_array_args.index(arg)
        remap[inner_name] = new_inner_names[prev_pos]
      else:
        new_inner_names.append(inner_name)
        new_array_args.append(arg)
        
    return new_closure_args, new_array_args, remap, const, new_inner_names

  def specialize_closure(self, clos, array_args):    
    closure_args = self.closure_elts(clos)
    fn = self.get_fn(clos)
    array_args = self.transform_expr_list(array_args)
    new_closure_args, new_array_args, remap, const, new_inner_names = \
      self.build_arg_mapping(closure_args, array_args, fn.arg_names)
    if len(new_inner_names) == len(fn.arg_names):
      return clos, array_args 
    
    new_input_types = [fn.type_env[name] for name in new_inner_names]
    
    new_name = "specialized_" + names.original(fn.name)
    new_fn, builder, _  = \
      build_fn(new_input_types,  
               return_type = fn.return_type, 
               name = new_name, 
               input_names = new_inner_names)
      
    new_fn.created_by = fn.created_by
     
    # call the original function 
    call_args = [] 
    for old_name, t  in zip(fn.arg_names, fn.input_types):
      if old_name in const:
        call_args.append(const[old_name])
      elif old_name in remap:
        other_name = remap[old_name]
        call_args.append(Var(name = other_name, type = t))
      else:
        assert old_name in new_inner_names, \
          "Expected %s to be in list of kept function arguments: %s" % (old_name, new_inner_names)
        call_args.append(Var(name = old_name, type = t))
    builder.return_(builder.call(fn, call_args))
    new_fn = Inliner().apply(new_fn)
    new_closure = self.closure(new_fn, new_closure_args)
    return new_closure, new_array_args

  def transform_Map(self, expr):
    if self.is_none(expr.axis) or is_constant(expr.axis):
      new_closure, new_array_args = self.specialize_closure(expr.fn, expr.args)
      expr.fn = new_closure
      expr.args = new_array_args
    return expr 
    
      