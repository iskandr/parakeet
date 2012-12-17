import adverb_helpers
import array_type
import copy
import syntax
import syntax_helpers
import tuple_type

from core_types import Int32, Int64
from lower_adverbs import LowerAdverbs
from transform import Transform

int64_array_t = array_type.make_array_type(Int64, 1)

class LowerTiledAdverbs(Transform):
  def __init__(self, fn, nesting_idx=-1, tile_param_array=None):
    Transform.__init__(self, fn)
    self.num_tiled_adverbs = 0
    self.nesting_idx = nesting_idx
    self.tiling = False
    if tile_param_array == None:
      self.tile_param_array = self.fresh_var(int64_array_t, "tile_params")
    else:
      self.tile_param_array = tile_param_array

  def transform_TypedFn(self, expr):
    nested_lower = LowerTiledAdverbs(expr, nesting_idx=self.nesting_idx,
                                     tile_param_array=self.tile_param_array)
    return nested_lower.apply()

  def transform_Map(self, expr):
    #self.tiling = False
    return expr

  def transform_Reduce(self, expr):
    #self.tiling = False
    return expr

  def transform_Scan(self, expr):
    #self.tiling = False
    return expr

  def transform_TiledMap(self, expr):
    self.tiling = True
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

  def transform_TiledReduce(self, expr):
    self.tiling = True
    def alloc_maybe_array(isarray, t, shape):
      if isarray:
        return self.alloc_array(t, shape)
      else:
        return self.fresh_var(t)

    self.nesting_idx += 1
    fn = expr.combine # TODO: could be a Closure
    args = expr.args
    axis = syntax_helpers.unwrap_constant(expr.axis)
    combine = self.transform_expr(expr.combine)

    # TODO: Should make sure that all the shapes conform here,
    # but we don't yet have anything like assertions or error handling
    max_arg = adverb_helpers.max_rank_arg(args)
    niters = self.shape(max_arg, axis)

    # Create the tile size variable and find the number of tiles
    tile_size = self.index(self.tile_param_array, self.nesting_idx)
    self.num_tiled_adverbs += 1

    isarray = isinstance(expr.type, array_type.ArrayT)
    # TODO: Use shape inference so that I can preallocate an output array!
    #       For now, assuming that the reduction causes the reduced axis to
    #       disappear, leaving the others untouched.
    slice_t = array_type.make_slice_type(Int64, Int64, Int64)
    transformed_fn = self.transform_expr(fn)
    nested_has_tiles = \
        transformed_fn.arg_names[-1] == self.tile_param_array.name
    arg_shape = self.shape(max_arg)
    left_shape_slice = syntax.Slice(syntax_helpers.zero_i64, axis,
                                    syntax_helpers.one_i64, type=slice_t)
    left_type = tuple_type.make_tuple_type([Int64 for _ in range(axis)])
    left_shape = syntax.Index(arg_shape, left_shape_slice, type=left_type)
    right_shape_slice = syntax.Slice(axis, max_arg.type.rank,
                                     syntax_helpers.one_i64, type=slice_t)
    right_type = tuple_type.make_tuple_type([Int64 for _ in
                                             range(max_arg.type.rank - axis)])
    right_shape = syntax.Index(arg_shape, right_shape_slice, type=right_type)
    out_shape = self.concat_tuples(left_shape, right_shape)

    elt_t = array_type.elt_type(expr.type)
    output = alloc_maybe_array(isarray, elt_t, out_shape)

    init = syntax_helpers.const_scalar(expr.init)
    if init.type.rank < output.type.rank:
      print "Yo!! Gotta lift the init val dawg"

    self.blocks.push()
    initial_slice = syntax.Slice(syntax_helpers.zero_i64, tile_size,
                                 syntax_helpers.one(Int64), type=slice_t)
    initial_args = [init] + [self.index_along_axis(arg, axis, initial_slice)
                             for arg in args]
    if nested_has_tiles:
      initial_args.append(self.tile_param_array)
    initial_call = syntax.Call(transformed_fn, initial_args,
                               type=fn.return_type)

    rslt_before = alloc_maybe_array(isarray, elt_t, out_shape)
    loop_rslt = alloc_maybe_array(isarray, elt_t, out_shape)
    self.assign(rslt_before, initial_call)

    num_tiles = self.div(niters, tile_size, name="num_tiles")
    loop_bound = self.mul(num_tiles, tile_size, "loop_bound")
    i, i_after, merge = self.loop_counter("i", tile_size)
    loop_cond = self.lt(i, loop_bound)
    rslt_after = alloc_maybe_array(isarray, elt_t, out_shape)
    merge[loop_rslt.name] = (rslt_before, rslt_after)

    self.blocks.push()
    self.assign(i_after, self.add(i, tile_size))
    tile_bounds = syntax.Slice(i, i_after, syntax_helpers.one(Int64),
                               type=slice_t)
    nested_args = [loop_rslt] + [self.index_along_axis(arg, axis, tile_bounds)
                                 for arg in args]
    if nested_has_tiles:
      nested_args.append(self.tile_param_array)
    nested_call = syntax.Call(transformed_fn, nested_args, type=fn.return_type)
    self.assign(rslt_after, nested_call)
    loop_body = self.blocks.pop()
    self.blocks += syntax.While(loop_cond, loop_body, merge)

    straggler_cond = self.lt(loop_bound, niters)

    self.blocks.push()
    straggler_slice = syntax.Slice(loop_bound, niters, syntax_helpers.one_i64,
                                   type=slice_t)
    straggler_args = \
        [loop_rslt] + [self.index_along_axis(arg, axis, straggler_slice)
                       for arg in args]
    if nested_has_tiles:
      straggler_args.append(self.tile_param_array)
    straggler_call = syntax.Call(transformed_fn, straggler_args,
                                 type=fn.return_type)
    straggler_rslt = alloc_maybe_array(isarray, elt_t, out_shape)
    self.assign(straggler_rslt, straggler_call)
    straggler_body = self.blocks.pop()

    main_rslt = alloc_maybe_array(isarray, elt_t, out_shape)
    main_merge = {main_rslt.name : (straggler_rslt, loop_rslt)}
    self.blocks += syntax.If(straggler_cond, straggler_body, [], main_merge)
    main_body = self.blocks.pop()

    self.blocks.push()
    else_rslt = alloc_maybe_array(isarray, elt_t, out_shape)
    else_args = [init] + copy.copy(args)
    if nested_has_tiles:
      else_args.append(self.tile_param_array)
    else_call = syntax.Call(transformed_fn, else_args, type=fn.return_type)
    self.assign(else_rslt, else_call)
    else_body = self.blocks.pop()

    init_cond = self.lte(tile_size, niters)
    outer_merge = {output.name : (main_rslt, else_rslt)}
    self.blocks += syntax.If(init_cond, main_body, else_body, outer_merge)

    return output

  def post_apply(self, fn):
    if self.tiling:
      fn.arg_names.append(self.tile_param_array.name)
      fn.input_types += (int64_array_t,)
      fn.type_env[self.tile_param_array.name] = int64_array_t
      print fn
    return fn
