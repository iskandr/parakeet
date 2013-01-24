import inline
import names
import syntax_helpers

from adverbs import Adverb, Map, AllPairs
from syntax import Var, Const,  Return, TypedFn
from transform import Transform
from use_analysis import use_count

def fuse(prev_fn, prev_fixed_args, next_fn, next_fixed_args, fusion_args):
  if syntax_helpers.is_identity_fn(next_fn):
    assert len(next_fixed_args) == 0
    return prev_fn, prev_fixed_args

  """
  Expects the prev_fn's returned value to be one or more of the arguments to
  next_fn. Any element in 'const_args' which is None gets replaced by the
  returned Var
  """
  fused_formals = []
  fused_input_types = []
  fused_type_env = prev_fn.type_env.copy()
  fused_name = names.fresh('fused')

  prev_closure_formals = prev_fn.arg_names[:len(prev_fixed_args)]

  for prev_closure_arg_name in prev_closure_formals:
    t = prev_fn.type_env[prev_closure_arg_name]
    fused_formals.append(prev_closure_arg_name)
    fused_input_types.append(t)

  next_closure_formals = next_fn.arg_names[:len(next_fixed_args)]
  for next_closure_arg_name in next_closure_formals:
    t = next_fn.type_env[next_closure_arg_name]
    new_name = names.refresh(next_closure_arg_name)
    fused_type_env[new_name] = t
    fused_formals.append(new_name)
    fused_input_types.append(t)

  prev_direct_formals = prev_fn.arg_names[len(prev_fixed_args):]
  for arg_name in prev_direct_formals:
    t = prev_fn.type_env[arg_name]
    fused_formals.append(arg_name)
    fused_input_types.append(t)

  prev_return_var, fused_body = \
      inline.replace_return_with_var(prev_fn.body,
                                     fused_type_env,
                                     prev_fn.return_type)
  # for now we're restricting both functions to have a single return at the
  # outermost scope
  inline_args = list(next_closure_formals)
  for arg in fusion_args:
    if arg is None:
      inline_args.append(prev_return_var)
    elif isinstance(arg, int):
      # positional arg which is not being fused out
      inner_name = next_fn.arg_names[arg]

      inner_type = next_fn.type_env[inner_name]
      new_name = names.refresh(inner_name)
      fused_formals.append(new_name)
      fused_type_env[new_name] = inner_type
      fused_input_types.append(inner_type)
      var = Var(new_name, inner_type)
      inline_args.append(var)
    else:
      assert arg.__class__ is Const, \
         "Only scalars can be spliced as literals into a fused fn: %s" % arg
      inline_args.append(arg)
  next_return_var = inline.do_inline(next_fn, inline_args,
                                     fused_type_env, fused_body)
  fused_body.append(Return(next_return_var))

  # we're not renaming variables that originate from the predecessor function
  new_fn = TypedFn(name = fused_name,
                   arg_names = fused_formals,
                   body = fused_body,
                   input_types = tuple(fused_input_types),
                   return_type = next_fn.return_type,
                   type_env = fused_type_env)

  return new_fn, prev_fixed_args + next_fixed_args

class Fusion(Transform):
  def __init__(self, recursive=True):
    Transform.__init__(self)
    # name of variable -> Map or Scan adverb
    self.adverb_bindings = {}
    self.recursive = True

  def pre_apply(self, fn):
    # map each variable to
    self.use_counts = use_count(fn)

  def transform_TypedFn(self, fn):
    if self.fn.copied_by is not None:
      return self.fn.copied_by.apply(fn)
    else:
      # at the very least do high level optimizations
      import pipeline
      return pipeline.high_level_optimizations(fn)

  def transform_Assign(self, stmt):
    if self.recursive:
      stmt.rhs = self.transform_expr(stmt.rhs)
    rhs = stmt.rhs

    if isinstance(rhs, Adverb) and \
       rhs.__class__ is not AllPairs:
      args = rhs.args
      if all(arg.__class__ in (Var, Const) for arg in args):
        arg_names = [arg.name for arg in args if arg.__class__ is Var]
        unique_vars = set(arg_names)
        adverb_vars = [name for name in unique_vars
                       if name in self.adverb_bindings]

        for arg_name in adverb_vars:
          n_occurrences = sum((name == arg_name for name in arg_names))
          if self.use_counts[arg_name] == n_occurrences:
            prev_adverb = self.adverb_bindings[arg_name]
            if prev_adverb.__class__ is Map and \
               rhs.axis == prev_adverb.axis and \
               inline.can_inline(self.get_fn(prev_adverb.fn)) and \
               inline.can_inline(self.get_fn(rhs.fn)):
              surviving_array_args = []
              fusion_args = []
              for (pos, arg) in enumerate(rhs.args):
                c = arg.__class__
                if c is Var and arg.name == arg_name:
                  fusion_args.append(None)
                elif c is Const:
                  fusion_args.append(arg)
                else:
                  surviving_array_args.append(arg)
                  fusion_args.append(pos)
              new_fn, clos_args = \
                  fuse(self.get_fn(prev_adverb.fn),
                       self.closure_elts(prev_adverb.fn),
                       self.get_fn(rhs.fn),
                       self.closure_elts(rhs.fn),
                       fusion_args)
              assert new_fn.return_type == self.return_type(rhs.fn)
              del self.adverb_bindings[arg_name]
              if self.fn.copied_by:
                new_fn = self.fn.copied_by.apply(new_fn)
              rhs.fn = self.closure(new_fn, clos_args)
              rhs.args = prev_adverb.args + surviving_array_args

    if stmt.lhs.__class__ is Var and isinstance(rhs, Adverb):
      self.adverb_bindings[stmt.lhs.name] = rhs
    return stmt
