import names 
from syntax import Var,  Return, TypedFn    
from adverbs import Adverb, Scan, Reduce, Map, AllPairs 
from transform import Transform 
from use_analysis import use_count
import inline 
import dead_code_elim

def fuse(prev_fn, next_fn):
  type_env = prev_fn.type_env.copy()
  body = [stmt for stmt in prev_fn.body]
  prev_return_var = inline.replace_return_with_var(body, type_env, prev_fn.return_type)
  # for now we're restricting both functions to have a single return at the outermost scope 
  next_return_var = inline.do_inline(next_fn, [prev_return_var], type_env, body)
  body.append(Return(next_return_var))
  
  # we're not renaming variables that originate from the predecessor function 
  return TypedFn(name = names.fresh('fused'),
                        arg_names = prev_fn.arg_names, 
                        body = body, 
                        input_types = prev_fn.input_types, 
                        return_type = next_fn.return_type, 
                        type_env = type_env)

class Fusion(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    # name of variable -> Map or Scan adverb  
    self.adverb_bindings = {}
    
    # map each variable to 
    self.use_counts = use_count(fn)
      
  def transform_Assign(self, stmt):
    rhs = stmt.rhs
    if stmt.lhs.__class__ is Var and isinstance(rhs, Adverb) and \
        rhs.__class__ is not AllPairs:
      args = rhs.args 
      if len(args) == 1 and args[0].__class__ is Var:
        arg_name = args[0].name 
        if self.use_counts[arg_name] == 1 and arg_name in self.adverb_bindings:
          prev_adverb = self.adverb_bindings[arg_name]
          if prev_adverb.__class__ is Map and rhs.axis == prev_adverb.axis and \
              inline.can_inline(prev_adverb.fn) and inline.can_inline(rhs.fn):
            # since we're modifying the RHS of the assignment
            # we better make sure the caller doesn't expect us 
            # to return a fresh copy of the AST 
            assert not self.copy
            rhs.fn = fuse(prev_adverb.fn, rhs.fn)
            rhs.args = prev_adverb.args

      self.adverb_bindings[stmt.lhs.name] = rhs 
    return stmt
  
  def post_apply(self, fn):
    return dead_code_elim.dead_code_elim(fn)

    