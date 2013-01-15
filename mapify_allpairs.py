import array_type
import names

from adverbs import AllPairs, Map
from syntax import Assign, Return, TypedFn, Var
from transform import Transform

class MapifyAllPairs(Transform):
  def transform_stmt(self, stmt):
    """
    Assume that all adverbs occur only at the outer level of bindings, so skip
    recursive evaluation of expressions
    """

    if stmt.__class__ is Assign and stmt.rhs.__class__ is AllPairs:
      stmt.rhs = self.transform_AllPairs(stmt.rhs)
    elif stmt.__class__ is Return and stmt.value.__class__ is AllPairs:
      stmt.value = self.transform_AllPairs(stmt.value)
    return stmt

  def transform_AllPairs(self, expr):
    """
    Transform each AllPairs(f, X, Y) operation into a pair of nested maps:
      def g(x_elt):
        def h(y_elt):
          return f(x_elt, y_elt)
      return map(g, X)
    """

    if expr.out is not None:
      return expr

    # if the adverb function is a closure, give me all the values it
    # closes over
    closure_elts = self.closure_elts(expr.fn)
    n_closure_elts = len(closure_elts)

    # strip off the closure wrappings and give me the underlying TypedFn
    fn = self.get_fn(expr.fn)

    # the two array arguments to this AllPairs adverb
    x, y_outer = expr.args

    x_elt_name = names.fresh('x_elt')
    x_elt_t = fn.input_types[n_closure_elts]
    x_elt_var = Var(x_elt_name, type = x_elt_t)
    y_inner_name = names.fresh('y')
    y_inner = Var(y_inner_name, type = y_outer.type)

    inner_closure_args = []
    for (i, elt) in enumerate(closure_elts):
      t = elt.type
      if elt.__class__ is Var:
        name = names.refresh(elt.name)
      else:
        name = names.fresh('closure_arg%d' % i)
      inner_closure_args.append(Var(name, type = t))

    inner_arg_names = []
    inner_input_types = []
    type_env = {}

    for var in inner_closure_args + [y_inner, x_elt_var]:
      type_env[var.name] = var.type
      inner_arg_names.append(var.name)
      inner_input_types.append(var.type)

    inner_closure_rhs = self.closure(fn, inner_closure_args + [x_elt_var])

    inner_result_t = array_type.lower_rank(expr.type, 1)
    inner_fn = TypedFn(
      name = names.fresh('allpairs_into_maps_wrapper'),
      arg_names = tuple(inner_arg_names),
      input_types = tuple(inner_input_types),
      return_type = inner_result_t,
      type_env = type_env,
      body = [
        Return(Map(inner_closure_rhs,
                   args=[y_inner],
                   axis = expr.axis,
                   type = inner_result_t))
      ]
    )
    closure = self.closure(inner_fn, closure_elts + [y_outer])
    return Map(closure, [x], axis = expr.axis, type = expr.type)
