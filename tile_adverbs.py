import adverb_helpers
import adverbs
import args
import array_type
import copy
import names
import syntax
import syntax_helpers
import type_inference

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
  elif isinstance(expr, (syntax.PrimCall,syntax.Call,syntax.Invoke)):
    return free_vars_list(expr.args)
  elif isinstance(expr, syntax.Index):
    return free_vars(expr.value).union(free_vars(expr.index))
  elif isinstance(expr, syntax.Tuple):
    return free_vars_list(expr.elts)
  else:
    assert isinstance(expr, syntax.Const), ("%s is not a Const" % expr)
    return set()

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

  def gen_unpack_tree(self, adverb_tree, exps, v_names, block):
    exps_left = {}
    for arg in v_names:
      exps_left[arg] = len(self.expansions[arg])

    def make_type_env(names):
      new_type_env = {}
      for name in names:
        new_type_env[name] = array_type.increase_rank(self.fn.type_env[name],
                                                      exps_left[name])
      return new_type_env

    def order_args(depth):
      cur_depth_args = []
      other_args = []
      for arg in v_names:
        arg_exps = self.expansions[arg]
        if depth in arg_exps:
          cur_depth_args.append(arg)
          exps_left[arg] -= 1
        else:
          other_args.append(arg)
      return (cur_depth_args, other_args)

    def gen_unpack_fn(depth_idx, arg_order):
      if depth_idx > len(exps):
        # Create type env for innermost fn
        inner_arg_types = [self.fn.type_env[name] for name in arg_order]
        inner_type_env = dict(*zip(arg_order, inner_arg_types))

        # For each stmt in body, add its lhs free vars to the type env
        return_t = Int32 # Dummy type
        for s in block:
          if isinstance(s, syntax.Assign):
            lhs_vars = free_vars(s.lhs)
            lhs_names = [var.name for var in lhs_vars]
            lhs_types = [self.fn.type_env[name] for name in lhs_names]
            for name, t in zip(lhs_names, lhs_types):
              inner_type_env[name] = t
          elif isinstance(s, syntax.Return):
            return_t = self.fn.type_env[s.rhs]
        inner_args = args.Args(position=arg_order)
        return syntax.TypedFn(name=names.fresh("expanded_assign"),
                              arg_names=inner_args,
                              body=block,
                              input_types=inner_arg_types,
                              return_type=return_t,
                              type_env=inner_type_env)
      else:
        depth = exps[depth_idx]
        new_type_env = make_type_env(arg_order)
        (cur_depth_args, other_args) = order_args(depth)
        new_arg_order = other_args + cur_depth_args
        nested_fn = gen_unpack_fn(depth_idx+1, new_arg_order)
        closure_args = [syntax.Var(name, type=new_type_env[name])
                        for name in other_args]
        closure = syntax.Closure(nested_fn.name, closure_args)
        direct_args = [syntax.Var(name, type=new_type_env[name])
                       for name in cur_depth_args]
        axis = 0 # When unpacking a non-adverb assignment, all axes are 0
        new_adverb = adverb_tree[depth_idx](closure, direct_args, axis)
        body = [syntax.Return(new_adverb)]
        outer_args = args.Args(position=direct_args)
        outer_arg_types = [new_type_env[name] for name in outer_args]
        return_t = type_inference.infer_map_type(closure.return_type,
                                                 outer_arg_types,
                                                 axis)
        return syntax.TypedFn(name=names.fresh("expanded_assign"),
                              arg_names=outer_args,
                              body=body,
                              input_types=outer_arg_types,
                              return_type=return_t,
                              type_env=new_type_env)

    (cur_depth_args, other_args) = order_args(exps[0])
    return (cur_depth_args, gen_unpack_fn(0, other_args + cur_depth_args))

  def get_exps(self, v_names):
    exps = list(set([self.expansions[name] for name in v_names]))
    exps.sort()
    return exps

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
      exps = self.get_exps(fv_names)
      map_tree = [adverbs.Map for _ in exps]
      inner_body = [stmt, syntax.Return(stmt.lhs)]
      cur_depth_args, unpack_fn = \
          self.gen_unpack_tree(map_tree, exps, fv_names, inner_body)
      new_rhs = syntax.Invoke(unpack_fn, cur_depth_args)
      return syntax.Assign(stmt.lhs, new_rhs)

  def transform_Return(self, stmt):
    if isinstance(stmt.rhs, adverbs.Adverb):
      new_rhs = self.transform_expr(stmt.rhs)
      return syntax.Return(stmt.lhs, new_rhs)

    return stmt

  def transform_Map(self, expr):
    # TODO: Have to handle naming collisions in the expansions dict
    depth = len(self.adverbs_visited)
    self.push_exp(adverbs.Map)
    for fn_arg, map_arg in zip(expr.fn.args, expr.args):
      self.expansions[fn_arg] = self.expansions[map_arg] + [depth]
    # Depends on inlining for this to work
    has_adverbs = False
    new_fn = syntax.TypedFn
    for stmt in expr.fn.body:
      if isinstance(stmt.rhs, adverbs.Adverb):
        has_adverbs = True
        break
    exps = self.get_exps(expr.fn.arg_names)
    if has_adverbs:
      new_body = self.transform_block(expr.fn.body)
      new_fn = self.gen_unpack_tree([], exps, expr.fn.arg_names, new_body)
    else:
      new_fn = self.gen_unpack_tree(self.adverbs_visited, exps,
                                    expr.fn.arg_names, expr.fn.body)
    axis = [len(self.expansions[arg]) + a - 1
            for arg, a in zip(expr.args, expr.axis)]
    self.pop_exp()
    return adverbs.TiledMap(new_fn, expr.args, axis)

  def transform_Reduce(self, expr):
    # TODO: Have to handle naming collisions in the expansions dict
    depth = len(self.adverbs_visited)
    self.push_exp(adverbs.Reduce)
    for fn_arg, map_arg in zip(expr.fn.args, expr.args):
      self.expansions[fn_arg] = self.expansions[map_arg] + [depth]
    # Depends on inlining for this to work
    has_adverbs = False
    new_fn = syntax.TypedFn
    for stmt in expr.fn.body:
      if isinstance(stmt.rhs, adverbs.Adverb):
        has_adverbs = True
        break
    exps = self.get_exps(expr.fn.arg_names)
    if has_adverbs:
      new_body = self.transform_block(expr.fn.body)
      new_fn = self.gen_unpack_tree([], exps, expr.fn.arg_names, new_body)
    else:
      new_fn = self.gen_unpack_tree(self.adverbs_visited, exps,
                                    expr.fn.arg_names, expr.fn.body)
    axis = [len(self.expansions[arg]) + a - 1
            for arg, a in zip(expr.args, expr.axis)]
    self.pop_exp()
    return adverbs.TiledReduce(new_fn, expr.args, axis)

class LowerTiledAdverbs(LowerAdverbs):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    self.tile_params = []
    self.num_tiled_adverbs = 0

  def transform_TiledMap(self, expr):
    fn, args, axis = self.adverb_prelude(expr)

    # TODO: Should make sure that all the shapes conform here,
    # but we don't yet have anything like assertions or error handling
    max_arg = adverb_helpers.max_rank_arg(args)
    niters = self.shape(max_arg, axis)

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
    nested_args = [self.index_along_axis(arg, axis, tile_bounds)
                   for arg in args]

    # TODO: Use shape inference to figure out how large of an array
    # I need to allocate here!
    array_result = self.alloc_array(elt_t, niters)
    self.blocks.push()
    self.assign(i_after, self.add(i, tile_size))
    call_result = self.invoke(fn, nested_args)
    output_idxs = syntax.Index(array_result, tile_bounds, type=call_result.type)
    self.assign(output_idxs, call_result)

    body = self.blocks.pop()
    self.blocks += syntax.While(cond, body, merge)

    # Handle the straggler sub-tile
    cond = self.lt(loop_bound, niters)
    straggler_bounds = syntax.Slice(loop_bound, niters,
                                    syntax_helpers.one(Int64), type=slice_t)
    straggler_args = [self.index_along_axis(arg, axis, straggler_bounds)
                      for arg in args]
    self.blocks.push()
    straggler_result = self.invoke(fn, straggler_args)
    straggler_output = syntax.Index(array_result, straggler_bounds,
                                    type=call_result.type)
    self.assign(straggler_output, straggler_result)
    body = self.blocks.pop()
    self.blocks += syntax.If(cond, body, [], {})
    return array_result

  def post_apply(self, fn):
    tile_param_array = self.fresh_var(int32_array_t, "tile_params")
    fn.args.arg_slots.append(tile_param_array.name)
    assignments = []
    for var, counter in self.tile_params:
      assignments.append(
          syntax.Assign(var,
                        self.index(tile_param_array, counter, temp=False)))
    fn.body = assignments + fn.body
    return fn
