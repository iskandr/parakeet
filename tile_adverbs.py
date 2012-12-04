import adverb_helpers
import adverbs
import array_type
import copy
import names
import syntax
import syntax_helpers
import tuple_type

from core_types import Int32, Int64
from lower_adverbs import LowerAdverbs
from transform import Transform

int32_array_t = array_type.make_array_type(Int32, 1)

def free_vars_list(expr_list):
  rslt = set()
  for expr in expr_list:
    rslt.update(free_vars(expr))
  return rslt

def free_vars(expr):
  if isinstance(expr, syntax.Var):
    return set([expr])
  elif isinstance(expr, (syntax.PrimCall,syntax.Call)):
    return free_vars_list(expr.args)
  elif isinstance(expr, syntax.Index):
    return free_vars(expr.value).union(free_vars(expr.index))
  elif isinstance(expr, syntax.Tuple):
    return free_vars_list(expr.elts)
  else:
    assert isinstance(expr, syntax.Const), ("%s is not a Const" % expr)
    return set()

class FindAdverbs(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    self.has_adverbs = False

  def transform_Adverb(self, expr):
    self.has_adverbs = True

class TileAdverbs(Transform):
  def __init__(self, fn, adverbs_visited=[], expansions={}):
    Transform.__init__(self, fn)
    self.adverbs_visited = adverbs_visited
    self.expansions = expansions
    self.exp_stack = []

  def push_exp(self, adv):
    self.exp_stack.append(self.expansions)
    self.expansions = copy.deepcopy(self.expansions)
    self.adverbs_visited.append(adv)

  def pop_exp(self):
    self.expansions = self.exp_stack.pop()
    self.adverbs_visited.pop()

  def get_expansions(self, arg):
    if arg in self.expansions:
      return self.expansions[arg]
    else:
      return []

  def gen_unpack_tree(self, adverb_tree, depths, v_names, block, type_env):
    exps_left = {}
    for arg in v_names:
      exps_left[arg] = len(self.get_expansions(arg))

    def order_args(depth):
      cur_depth_args = []
      other_args = []
      for arg in v_names:
        arg_exps = self.get_expansions(arg)
        if depth in arg_exps:
          cur_depth_args.append(arg)
        else:
          other_args.append(arg)
      return (cur_depth_args, other_args)

    def gen_unpack_fn(depth_idx):
      def make_args(type_env, cur_depth_args, other_args):
        # We know that the caller is always going to be an adverb (in which case
        # the type needs to increase its rank by 1) or a Tiled adverb (in which
        # case the types are ignored).
        cur_arg_types = [array_type.increase_rank(type_env[arg], 1)
                         for arg in cur_depth_args]
        cur_vars = [syntax.Var(name, type=t)
                    for name, t in zip(cur_depth_args, cur_arg_types)]

        # Get the other arg types and make a tuple to pass them into the
        # nested fn through the Adverb
        other_arg_types = [type_env[arg.type] for arg in other_args]
        fixed_args_t = tuple_type.make_tuple_type(other_arg_types)
        other_vars = [syntax.Var(name, type=t)
                      for name, t in zip(other_args, other_arg_types)]
        fixed_args = syntax.Tuple(other_vars, type=fixed_args_t)
        return cur_vars, cur_arg_types, fixed_args, fixed_args_t, other_vars

      if depth_idx >= len(depths):
        # Create type env for innermost fn - just the original types
        inner_type_env = {}
        for arg in v_names:
          inner_type_env[arg] = type_env[arg]

        # For each stmt in body, add its lhs free vars to the type env
        return_t = Int32 # Dummy type
        for s in block:
          if isinstance(s, syntax.Assign):
            lhs_vars = free_vars(s.lhs)
            lhs_names = [var.name for var in lhs_vars]
            lhs_types = [type_env[name] for name in lhs_names]
            for name, t in zip(lhs_names, lhs_types):
              inner_type_env[name] = t
          elif isinstance(s, syntax.Return):
            return_t = type_env[s.value.name]

        # The innermost function always uses all the variables
        cur_vars, cur_arg_types, fixed_args, fixed_args_t, _ = \
            make_args(inner_type_env, v_names, [])

        args = [fixed_args] + cur_vars
        arg_types = [fixed_args_t] + cur_arg_types
        inner_type_env["$fixed_args"] = fixed_args_t

        arg_names = ["$fixed_args"] + v_names
        fn = syntax.TypedFn(name=names.fresh("expanded_assign"),
                            arg_names=arg_names,
                            body=block,
                            input_types=arg_types,
                            return_type=return_t,
                            type_env=inner_type_env)
        return (args, fn)
      else:
        # Get the current depth
        depth = depths[depth_idx]

        # Order the arguments for the current depth, i.e. for the nested fn
        cur_depth_args, other_args = order_args(depth)

        # Make a type env for this function based on the number of expansions
        # left for each arg
        new_type_env = {}
        for arg in cur_depth_args + other_args:
          new_type_env[arg] = array_type.increase_rank(type_env[arg],
                                                       exps_left[arg])
        print "new_type_env:", new_type_env
        for arg in cur_depth_args:
          exps_left[arg] -= 1

        # Get the current arg types and great a list of Vars for them
        cur_vars, cur_arg_types, fixed_args, fixed_args_t, other_vars = \
            make_args(new_type_env, cur_depth_args, other_args)
        args = [fixed_args] + cur_vars
        new_type_env["$fixed_args"] = fixed_args_t

        # Pull out all of the fixed args from the current fn's tuple arg and
        # place them in local Vars
        body = []
        for i, v in enumerate(other_vars):
          indexed_val = syntax.Index(syntax.Var("$fixed_args"),
                                     syntax_helpers.const_int(i))
          body += [syntax.Assign(v, indexed_val)]

        # Generate the nested fn
        nested_args, nested_fn = gen_unpack_fn(depth_idx+1)

        # Make an adverb that wraps the nested fn
        axis = 0 # When unpacking a non-adverb assignment, all axes are 0
        return_t = array_type.increase_rank(nested_fn.return_type, 1)
        new_adverb = adverb_tree[depth_idx](nested_fn, nested_args, axis,
                                            type=return_t)

        # Add the adverb to the body of the current fn and return the fn
        body += [syntax.Return(new_adverb)]
        fn = syntax.TypedFn(name=names.fresh("expanded_assign"),
                            arg_names=["$fixed_args"] + cur_depth_args,
                            body=body,
                            input_types=[fixed_args_t] + cur_arg_types,
                            return_type=return_t,
                            type_env=new_type_env)
        return (args, fn)

    return gen_unpack_fn(0)

  def get_depths_list(self, v_names):
    depths = set()
    for name in v_names:
      for e in self.get_expansions(name):
        depths.add(e)
    depths = list(depths)
    depths.sort()
    return depths

  def transform_Assign(self, stmt):
    # Do nothing unless we're inside a tree of adverbs being tiled
    if len(self.adverbs_visited) < 1:
      return stmt

    if isinstance(stmt.rhs, adverbs.Adverb):
      new_rhs = self.transform_expr(stmt.rhs)
      return syntax.Assign(stmt.lhs, new_rhs)
    else:
      fv = free_vars(stmt.rhs)
      fv_names = [v.name for v in fv]
      depths = self.get_depths_list(fv_names)
      map_tree = [adverbs.Map for _ in depths]
      inner_body = [stmt, syntax.Return(stmt.lhs)]
      nested_args, unpack_fn = \
          self.gen_unpack_tree(map_tree, depths, fv_names, inner_body,
                               self.fn.type_env)
      new_rhs = syntax.Call(unpack_fn, nested_args)
      return syntax.Assign(stmt.lhs, new_rhs)

  def transform_Return(self, stmt):
    if isinstance(stmt.value, adverbs.Adverb):
      return syntax.Return(self.transform_expr(stmt.value))

    return stmt

  def transform_Map(self, expr):
    # TODO: Have to handle naming collisions in the expansions dict
    depth = len(self.adverbs_visited)
    self.push_exp(adverbs.Map)
    for fn_arg, map_arg in zip(expr.fn.arg_names, expr.args):
      new_expansions = copy.copy(self.get_expansions(map_arg))
      new_expansions.append(depth)
      self.expansions[fn_arg] = new_expansions

    new_fn = syntax.TypedFn
    nested_args = []
    find_adverbs = FindAdverbs(expr.fn)
    find_adverbs.apply(copy=False)
    depths = self.get_depths_list(expr.fn.arg_names)
    print "depths:", depths
    print "expansions:", self.expansions
    if find_adverbs.has_adverbs:
      # I think we probably want to generate our own Fn here rather than calling
      # gen_unpack_tree
      new_body = self.transform_block(expr.fn.body)
      nested_args, new_fn = self.gen_unpack_tree([], depths, expr.fn.arg_names,
                                                 new_body, expr.fn.type_env)
    else:
      nested_args, new_fn = self.gen_unpack_tree(self.adverbs_visited, depths,
                                                 expr.fn.arg_names,
                                                 expr.fn.body, expr.fn.type_env)
    #TODO: below is for when we have multiple axes
    #axis = [len(self.get_expansions(arg)) + a
    #        for arg, a in zip(expr.args, expr.axis)]
    self.pop_exp()
    return adverbs.TiledMap(new_fn, [nested_args[0]]+expr.args,
                            expr.axis, type=expr.type)

class LowerTiledAdverbs(LowerAdverbs):
  def __init__(self, fn):
    LowerAdverbs.__init__(self, fn)
    self.tile_params = []
    self.num_tiled_adverbs = 0

  def transform_TiledMap(self, expr):
    fn = expr.fn # Should never be a Closure
    args = expr.args
    axis = syntax_helpers.unwrap_constant(expr.axis)

    # TODO: Should make sure that all the shapes conform here,
    # but we don't yet have anything like assertions or error handling
    max_arg = adverb_helpers.max_rank_arg(args[1:])
    print "max_arg:", max_arg
    print "max_arg.type:", max_arg.type
    niters = self.shape(max_arg, axis)
    print "niters:", niters

    # Create the tile size variable and find the number of tiles
    tile_size = self.fresh_i64("tile_size")
    self.tile_params.append((tile_size, self.num_tiled_adverbs))
    self.num_tiled_adverbs += 1
    num_tiles = self.div(niters, tile_size, name="num_tiles")
    loop_bound = self.mul(num_tiles, tile_size, "loop_bound")

    i, i_after, merge = self.loop_counter("i")

    cond = self.lt(i, loop_bound)
    elt_t = expr.type.elt_type
    slice_t = array_type.make_slice_type(i.type, i_after.type, Int64)
    tile_bounds = syntax.Slice(i, i_after, syntax_helpers.one(Int64),
                               type=slice_t)
    nested_args = [args[0]] + [self.index_along_axis(arg, axis, tile_bounds)
                               for arg in args[1:]]
    print "nested_args:", nested_args
    for n, arg in enumerate(nested_args):
      print "nested_arg[" + str(n) + "].type:", arg.type
    print fn.type_env

    # TODO: Use shape inference to figure out how large of an array
    # I need to allocate here!
    array_result = self.alloc_array(elt_t, niters)
    self.blocks.push()
    self.assign(i_after, self.add(i, tile_size))
    output_idxs = syntax.Index(array_result, tile_bounds, type=fn.return_type)
    nested_call = syntax.Call(fn, nested_args, type=fn.return_type)
    self.assign(output_idxs, self.transform_expr(nested_call))

    body = self.blocks.pop()
    self.blocks += syntax.While(cond, body, merge)

    # Handle the straggler sub-tile
    cond = self.lt(loop_bound, niters)
    straggler_bounds = syntax.Slice(loop_bound, niters,
                                    syntax_helpers.one(Int64), type=slice_t)
    straggler_args = [args[0]] + \
                     [self.index_along_axis(arg, axis, straggler_bounds)
                      for arg in args[1:]]
    self.blocks.push()
    straggler_output = syntax.Index(array_result, straggler_bounds,
                                    type=fn.return_type)
    nested_call = syntax.Call(fn, straggler_args, type=fn.return_type)
    self.assign(straggler_output, self.transform_expr(nested_call))
    body = self.blocks.pop()
    self.blocks += syntax.If(cond, body, [], {})
    return array_result

  def post_apply(self, fn):
    tile_param_array = self.fresh_var(int32_array_t, "tile_params")
    fn.arg_names.append(tile_param_array.name)
    assignments = []
    for var, counter in self.tile_params:
      assignments.append(
          syntax.Assign(var,
                        self.index(tile_param_array, counter, temp=False)))
    fn.body = assignments + fn.body
    return fn
