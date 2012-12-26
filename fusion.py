import names
from syntax import Var, Const,  Return, TypedFn
from adverbs import Adverb, Scan, Reduce, Map, AllPairs
from transform import Transform, apply_pipeline
from use_analysis import use_count
import inline
from dead_code_elim import DCE
from simplify import Simplify

def fuse(prev_fn, next_fn, const_args = [None]):
  """
  Expects the prev_fn's returned value to be one or more of the arguments to
  next_fn. Any element in 'const_args' which is None gets replaced by the
  returned Var
  """

  type_env = prev_fn.type_env.copy()
  body = [stmt for stmt in prev_fn.body]
  prev_return_var = inline.replace_return_with_var(body, type_env,
                                                   prev_fn.return_type)
  # for now we're restricting both functions to have a single return at the
  # outermost scope
  next_args = [prev_return_var if arg is None else arg for arg in const_args]
  next_return_var = inline.do_inline(next_fn, next_args, type_env, body)
  body.append(Return(next_return_var))

  # we're not renaming variables that originate from the predecessor function
  return TypedFn(name = names.fresh('fused'),
                        arg_names = prev_fn.arg_names,
                        body = body,
                        input_types = prev_fn.input_types,
                        return_type = next_fn.return_type,
                        type_env = type_env)

class Fusion(Transform):
  def __init__(self):
    Transform.__init__(self)
    # name of variable -> Map or Scan adverb
    self.adverb_bindings = {}

  def pre_apply(self, fn):
    # map each variable to
    self.use_counts = use_count(fn)

  def transform_TypedFn(self, fn):
    return apply_pipeline(fn, [Simplify, DCE, Fusion])

  def transform_Assign(self, stmt):
    stmt.rhs = self.transform_expr(stmt.rhs)
    rhs = stmt.rhs
    if stmt.lhs.__class__ is Var and isinstance(rhs, Adverb) and \
       rhs.__class__ is not AllPairs:
      args = rhs.args
      if all(arg.__class__ in (Var, Const) for arg in args):
        arg_names = [arg.name for arg in args if arg.__class__ is Var]
        n_unique_vars = len(set(arg_names))
        n_occurrences = len(arg_names)
        if n_unique_vars == 1:
          arg_name = arg_names[0]

          if self.use_counts[arg_name] == n_occurrences and \
             arg_name in self.adverb_bindings:
            prev_adverb = self.adverb_bindings[arg_name]
            if prev_adverb.__class__ is Map and \
               rhs.axis == prev_adverb.axis and \
               inline.can_inline(prev_adverb.fn) and inline.can_inline(rhs.fn):
              const_args = [None if arg.__class__ is Var else arg
                            for arg in args]
              rhs.fn = fuse(prev_adverb.fn, rhs.fn, const_args)
              rhs.args = prev_adverb.args
      self.adverb_bindings[stmt.lhs.name] = rhs
    return stmt
