import adverb_helpers
import adverbs
import array_type
import closure_type
import copy
import names
import syntax
import syntax_helpers
import tuple_type

from core_types import Int32, Int64
from lower_adverbs import LowerAdverbs
from transform import Transform

int64_array_t = array_type.make_array_type(Int64, 1)

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
    assert isinstance(expr, syntax.Const), "%s is not a Const" % expr
    return set()

class FindAdverbs(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    self.has_adverbs = False

  def transform_Map(self, expr):
    self.has_adverbs = True
    return expr

  def transform_Reduce(self, expr):
    self.has_adverbs = True
    return expr

  def transform_Scan(self, expr):
    self.has_adverbs = True
    return expr

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

  def get_num_expansions_at_depth(self, arg, depth):
    exps = self.get_expansions(arg)

    for i,v in enumerate(exps):
      if v >= depth:
        return len(exps) - i

    return 0

  def gen_unpack_tree(self, adverb_tree, depths, v_names, block, type_env):
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
            if isinstance(s.value, str):
              return_t = type_env[s.value.name]
            else:
              return_t = s.value.type

        # The innermost function always uses all the variables
        arg_types = [array_type.increase_rank(type_env[arg], 1)
                     for arg in v_names]
        fn = syntax.TypedFn(name=names.fresh("expanded_assign"),
                            arg_names=v_names,
                            body=block,
                            input_types=arg_types,
                            return_type=return_t,
                            type_env=inner_type_env)
        return (v_names, arg_types, [], [], fn)
      else:
        # Get the current depth
        depth = depths[depth_idx]

        # Order the arguments for the current depth, i.e. for the nested fn
        cur_arg_names, fixed_arg_names = order_args(depth)

        # Make a type env for this function based on the number of expansions
        # left for each arg
        new_type_env = {}
        for arg in cur_arg_names + fixed_arg_names:
          rank_increase = self.get_num_expansions_at_depth(arg, depth)
          new_type_env[arg] = \
              array_type.increase_rank(type_env[arg], rank_increase)

        cur_arg_types = []
        for arg in cur_arg_names:
          cur_arg_types.append(array_type.increase_rank(new_type_env[arg], 1))
        fixed_arg_types = [type_env[arg] for arg in fixed_arg_names]

        # Generate the nested fn and its fixed and normal args
        nested_arg_names, nested_arg_types, \
            nested_fixed_names, nested_fixed_types, nested_fn = \
            gen_unpack_fn(depth_idx+1)
        nested_args = [syntax.Var(name, type=t)
                       for name, t in zip(nested_arg_names, nested_arg_types)]
        nested_fixed_args = \
            [syntax.Var(name, type=t)
             for name, t in zip(nested_fixed_names, nested_fixed_types)]
        closure_t = closure_type.make_closure_type(nested_fn,
                                                   nested_fixed_types)
        nested_closure = syntax.Closure(nested_fn, nested_fixed_args,
                                        type=closure_t)

        # Make an adverb that wraps the nested fn
        axis = 0 # When unpacking a non-adverb assignment, all axes are 0
        return_t = array_type.increase_rank(nested_fn.return_type, 1)
        new_adverb = adverb_tree[depth_idx](nested_closure, nested_args, axis,
                                            type=return_t)

        # Add the adverb to the body of the current fn and return the fn
        fn = syntax.TypedFn(name=names.fresh("expanded_assign"),
                            arg_names=fixed_arg_names + cur_arg_names,
                            body=[syntax.Return(new_adverb)],
                            input_types=fixed_arg_types + cur_arg_types,
                            return_type=return_t,
                            type_env=new_type_env)
        return (cur_arg_names, cur_arg_types,
                fixed_arg_names, fixed_arg_types, fn)

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
    if isinstance(stmt.rhs, adverbs.Adverb):
      new_rhs = self.transform_expr(stmt.rhs)
      return syntax.Assign(stmt.lhs, new_rhs)
    elif len(self.adverbs_visited) > 0:
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
    else:
      # Do nothing if we're not inside a nesting of tiled adverbs
      return stmt

  def transform_Return(self, stmt):
    if isinstance(stmt.value, adverbs.Adverb):
      return syntax.Return(self.transform_expr(stmt.value))

    return stmt

  def tile_adverb(self, expr, adverb, tiledAdverb):
    # TODO: Have to handle naming collisions in the expansions dict
    depth = len(self.adverbs_visited)
    self.push_exp(adverb)
    for fn_arg, map_arg in zip(expr.fn.arg_names, expr.args):
      new_expansions = copy.deepcopy(self.get_expansions(map_arg.name))
      new_expansions.append(depth)
      self.expansions[fn_arg] = new_expansions
      print self.expansions

    new_fn = syntax.TypedFn
    arg_names = fixed_arg_names = []
    depths = self.get_depths_list(expr.fn.arg_names)
    find_adverbs = FindAdverbs(expr.fn)
    find_adverbs.apply(copy=False)

    if find_adverbs.has_adverbs:
      arg_names = expr.fn.arg_names
      input_types = []
      new_type_env = copy.copy(expr.fn.type_env)
      for arg, t in zip(arg_names, expr.fn.input_types):
        new_type = array_type.increase_rank(t, 1)
        input_types.append(new_type)
        new_type_env[arg] = new_type
      return_t = array_type.increase_rank(expr.fn.return_type, 1)
      new_fn = syntax.TypedFn(name=names.fresh("expanded_adverb_fn"),
                              arg_names=arg_names,
                              body=self.transform_block(expr.fn.body),
                              input_types=input_types,
                              return_type=return_t,
                              type_env=new_type_env)
    else:
      arg_names, _, fixed_arg_names, _, new_fn = \
          self.gen_unpack_tree(self.adverbs_visited, depths, expr.fn.arg_names,
                               expr.fn.body, expr.fn.type_env)

    #TODO: below is for when we have multiple axes
    #axis = [len(self.get_expansions(arg)) + a
    #        for arg, a in zip(expr.args, expr.axis)]
    arg_idxs = [expr.fn.arg_names.index(arg)
                for arg in fixed_arg_names + arg_names]
    args = [expr.args[idx] for idx in arg_idxs]
    for arg in args:
      rank_increase = len(self.get_expansions(arg.name))
      if depth in self.get_expansions(arg.name):
        rank_increase -= 1
      arg.type = array_type.increase_rank(arg.type, rank_increase)
    axis = len(self.get_expansions(expr.fn.arg_names[0])) + expr.axis - 1
    #axis = expr.axis
    self.pop_exp()
    return tiledAdverb(new_fn, args, axis, type=new_fn.return_type)

  def transform_Map(self, expr):
    return self.tile_adverb(expr, adverbs.Map, adverbs.TiledMap)

  def transform_Reduce(self, expr):
    return self.tile_adverb(expr, adverbs.Reduce, adverbs.TiledReduce)

  def transform_Scan(self, expr):
    return self.tile_adverb(expr, adverbs.Scan, adverbs.TiledScan)

  def post_apply(self, fn):
    #print fn
    return fn

class LowerTiledAdverbs(Transform):
  def __init__(self, fn, nesting_idx=-1, tile_param_array=None):
    Transform.__init__(self, fn)
    self.num_tiled_adverbs = 0
    self.nesting_idx = nesting_idx
    self.tiling = True
    if tile_param_array == None:
      self.tile_param_array = self.fresh_var(int64_array_t, "tile_params")
    else:
      self.tile_param_array = tile_param_array

  def transform_TypedFn(self, expr):
    nested_lower = LowerTiledAdverbs(expr, nesting_idx=self.nesting_idx,
                                     tile_param_array=self.tile_param_array)
    return nested_lower.apply()

  def transform_Map(self, expr):
    self.tiling = False
    return expr

  def transform_Reduce(self, expr):
    self.tiling = False
    return expr

  def transform_Scan(self, expr):
    self.tiling = False
    return expr

  def transform_TiledMap(self, expr):
    self.nesting_idx += 1
    fn = expr.fn # TODO: could be a Closure
    args = expr.args
    axis = syntax_helpers.unwrap_constant(expr.axis)

    # TODO: Should make sure that all the shapes conform here,
    # but we don't yet have anything like assertions or error handling
    max_arg = adverb_helpers.max_rank_arg(args)
    niters = self.shape(max_arg, axis)

    # Create the tile size variable and find the number of tiles
    tile_size = self.index(self.tile_param_array, self.nesting_idx)
    self.num_tiled_adverbs += 1
    num_tiles = self.div(niters, tile_size, name="num_tiles")
    loop_bound = self.mul(num_tiles, tile_size, "loop_bound")

    i, i_after, merge = self.loop_counter("i")

    cond = self.lt(i, loop_bound)
    elt_t = expr.type.elt_type
    slice_t = array_type.make_slice_type(i.type, i.type, Int64)

    # TODO: Use shape inference to figure out how large of an array
    # I need to allocate here!
    array_result = self.alloc_array(elt_t, self.shape(max_arg))

    self.blocks.push()
    self.assign(i_after, self.add(i, tile_size))
    tile_bounds = syntax.Slice(i, i_after, syntax_helpers.one(Int64),
                               type=slice_t)
    nested_args = [self.index_along_axis(arg, axis, tile_bounds)
                   for arg in args]
    output_idxs = self.index_along_axis(array_result, axis, tile_bounds)
    #syntax.Index(array_result, tile_bounds, type=fn.return_type)
    transformed_fn = self.transform_expr(fn)
    nested_has_tiles = \
        transformed_fn.arg_names[-1] == self.tile_param_array.name
    if nested_has_tiles:
      nested_args.append(self.tile_param_array)
    nested_call = syntax.Call(transformed_fn, nested_args, type=fn.return_type)
    self.assign(output_idxs, nested_call)
    body = self.blocks.pop()

    self.blocks += syntax.While(cond, body, merge)

    # Handle the straggler sub-tile
    cond = self.lt(loop_bound, niters)

    self.blocks.push()
    straggler_bounds = syntax.Slice(loop_bound, niters,
                                    syntax_helpers.one(Int64), type=slice_t)
    straggler_args = [self.index_along_axis(arg, axis, straggler_bounds)
                      for arg in args]
    straggler_output = self.index_along_axis(array_result, axis,
                                             straggler_bounds)
    #syntax.Index(array_result, straggler_bounds,
    #                                type=fn.return_type)
    nested_call = syntax.Call(transformed_fn, straggler_args,
                              type=fn.return_type)
    if nested_has_tiles:
      straggler_args.append(self.tile_param_array)
    self.assign(straggler_output, nested_call)
    body = self.blocks.pop()

    self.blocks += syntax.If(cond, body, [], {})
    return array_result

  def post_apply(self, fn):
    if self.tiling:
      fn.arg_names.append(self.tile_param_array.name)
      fn.input_types += (int64_array_t,)
      fn.type_env[self.tile_param_array.name] = int64_array_t
    #print fn
