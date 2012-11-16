from core_types import Int32, Int64
import adverb_helpers, array_type, syntax, syntax_helpers, tuple_type
from lower_adverbs import LowerAdverbs
from transform import Transform

int32_array_t = array_type.make_array_type(Int32, 1)


class TileAdverbs(Transform):
  pass

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
    fn.args.argslots.append(tile_param_array.name)
    assignments = []
    for var, counter in self.tile_params:
      assignments.append(
          syntax.Assign(var,
                        self.index(tile_param_array, counter, temp=False)))
    fn.body = assignments + fn.body
