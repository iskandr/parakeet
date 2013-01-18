import adverbs
import array_type
import clone_function
import config
import syntax
import syntax_helpers
import tuple_type

from array_type import ArrayT
from core_types import Int64, ScalarT, NoneType
from inline import do_inline
from syntax_helpers import zero_i64, one_i64, slice_none
from transform import Transform

class LowerTiledAdverbs(Transform):
  default_reg_tile_size = 3
  default_last_reg_tile_size = 1

  def __init__(self,
               nesting_idx = -1,
               fixed_idx = -1,
               tile_sizes_param = None,
               fixed_tile_sizes = None,
               preallocate_output = False):
    Transform.__init__(self)
    self.nesting_idx = nesting_idx
    self.fixed_idx = fixed_idx
    self.tiling = False
    self.tile_sizes_param = tile_sizes_param
    self.fixed_tile_sizes = fixed_tile_sizes
    self.output_var = None
    self.preallocate_output = preallocate_output

    # For now, we'll assume that no closure variables have the same name.
    self.closure_vars = {}

  # TODO: Change this to be able to accept fixed tile sizes properly
  def pre_apply(self, fn, fixed_tile_sizes = None):
    if not self.tile_sizes_param:
      tile_type = \
          tuple_type.make_tuple_type([Int64] * fn.num_tiles)
      self.tile_sizes_param = self.fresh_var(tile_type, "tile_params")
    if self.fixed_tile_sizes is None:
      # TODO: Replace this with an estimate of size based on the number of
      #       floating point registers.
      self.fixed_tile_sizes = [self.default_reg_tile_size] * (fn.num_tiles - 1)
      self.fixed_tile_sizes.append(self.default_last_reg_tile_size)

    if self.preallocate_output:
      self.output_var = self.fresh_var(fn.return_type, "prealloc_output")
    return clone_function.CloneFunction().apply(fn)

  def post_apply(self, fn):
    # the arg names list seems to be shared among
    # multiple functions created by tiling,
    # so make a clean copy here
    # fn = clone_function.CloneFunction().apply(fn)
    fn.arg_names = tuple(fn.arg_names)

    if self.output_var:
      fn.arg_names += (self.output_var.name,)
      fn.input_types += (self.output_var.type,)

      fn.return_type = NoneType

    if self.tiling:
      fn.arg_names += (self.tile_sizes_param.name,)
      fn.input_types += (self.tile_sizes_param.type,)
      fn.type_env[self.tile_sizes_param.name] = self.tile_sizes_param.type
    return fn

  def transform_Return(self, stmt):
    value = self.transform_expr(stmt.value)
    if self.output_var:
      if not isinstance(stmt.value, adverbs.Tiled):
        self.comment("Copy instead of Return in %s" % self.fn.name)
        rank = self.output_var.type.rank
        slice_all = self.tuple([slice_none] * rank)
        self.setidx(self.output_var, slice_all, value)
      return None
    else:
      stmt.value = value
      return stmt

  def get_closure_arg(self, closure_elt):
    if isinstance(closure_elt, syntax.ClosureElt):
      if isinstance(closure_elt.closure, syntax.Closure):
        return closure_elt.closure.args[closure_elt.index]
      elif isinstance(closure_elt.closure, syntax.Var):
        closure = self.closure_vars[closure_elt.closure.name]
        return closure.args[closure_elt.index]
      else:
        assert False, "Unknown closure type for closure elt %s" % closure_elt
    elif isinstance(closure_elt, syntax.Var):
      return closure_elt
    else:
      assert False, "Unknown closure closure elt type %s" % closure_elt

  def transform_Assign(self, stmt):
    if isinstance(stmt.rhs, syntax.Closure):
      self.closure_vars[stmt.lhs.name] = stmt.rhs
    stmt.rhs = self.transform_expr(stmt.rhs)
    stmt.lhs = self.transform_lhs(stmt.lhs)
    return stmt

  def transform_TypedFn(self, expr, preallocate_output = False):
    nested_lower = LowerTiledAdverbs(nesting_idx = self.nesting_idx,
                                     fixed_idx = self.fixed_idx,
                                     tile_sizes_param = self.tile_sizes_param,
                                     fixed_tile_sizes = self.fixed_tile_sizes,
                                     preallocate_output = preallocate_output)
    return nested_lower.apply(expr)

  def transform_TiledMap(self, expr):
    args = expr.args
    axes = expr.axes

    # TODO: Should make sure that all the shapes conform here,
    # but we don't yet have anything like assertions or error handling
    niters = self.shape(expr.args[0],
                        syntax_helpers.unwrap_constant(axes[0]))

    # Create the tile size variable and find the number of tiles
    if expr.fixed_tile_size:
      self.fixed_idx += 1
      tile_size = syntax_helpers.const(self.fixed_tile_sizes[self.fixed_idx])
    else:
      self.tiling = True
      self.fn.has_tiles = True
      self.nesting_idx += 1
      tile_size = self.index(self.tile_sizes_param, self.nesting_idx,
                             temp = True, name = "tilesize")

    untiled_inner_fn = self.get_fn(expr.fn)
    if isinstance(untiled_inner_fn.return_type, ScalarT):
      tiled_inner_fn = self.transform_TypedFn(untiled_inner_fn)
    else:
      tiled_inner_fn = self.transform_TypedFn(untiled_inner_fn,
                                              preallocate_output = True)

    nested_has_tiles = tiled_inner_fn.has_tiles

    # Increase the nesting_idx by the number of tiles in the nested fn
    self.nesting_idx += tiled_inner_fn.num_tiles

    slice_t = array_type.make_slice_type(Int64, Int64, Int64)

    closure_args = [self.get_closure_arg(e)
                    for e in self.closure_elts(expr.fn)]

    if self.output_var and \
       not isinstance(untiled_inner_fn.return_type, ScalarT):
      array_result = self.output_var
    else:
      shape_args = closure_args + expr.args
      array_result = self._create_output_array(untiled_inner_fn, shape_args,
                                               [], "array_result")

    assert self.output_var is None or \
           self.output_var.type.__class__ is ArrayT, \
           "Invalid output var %s : %s" % \
           (self.output_var, self.output_var.type)

    def make_loop(start, stop, step, do_min = True):
      i = self.fresh_var(niters.type, "i")

      self.blocks.push()
      slice_stop = self.add(i, step, "slice_stop")
      slice_stop_min = self.min(slice_stop, niters, "slice_min") if do_min \
                       else slice_stop

      tile_bounds = syntax.Slice(i, slice_stop_min, one_i64, type = slice_t)
      nested_args = [self.index_along_axis(arg, axis, tile_bounds)
                     for arg, axis in zip(args, axes)]
      out_idx = self.fixed_idx if expr.fixed_tile_size else self.nesting_idx
      output_region = self.index_along_axis(array_result, out_idx, tile_bounds)
      nested_args.append(output_region)

      if nested_has_tiles:
        nested_args.append(self.tile_sizes_param)
      body = self.blocks.pop()
      do_inline(tiled_inner_fn,
                closure_args + nested_args,
                self.type_env,
                body,
                result_var = None)
      return syntax.ForLoop(i, start, stop, step, body, {})

    assert isinstance(tile_size, syntax.Expr)
    self.comment("TiledMap in %s" % self.fn.name)

    if expr.fixed_tile_size and \
       config.opt_reg_tiles_not_tile_size_dependent and \
       syntax_helpers.unwrap_constant(tile_size) > 1:
      num_tiles = self.div(niters, tile_size, "num_tiles")
      tile_stop = self.mul(num_tiles, tile_size, "tile_stop")
      loop1 = make_loop(zero_i64, tile_stop, tile_size, False)
      self.blocks.append(loop1)
      loop2_start = self.assign_temp(loop1.var, "loop2_start")
      self.blocks.append(make_loop(loop2_start, niters, one_i64, False))
    else:
      self.blocks.append(make_loop(zero_i64, niters, tile_size))
    return array_result

  def transform_expr(self, expr):
    if isinstance(expr, adverbs.Tiled):
      return Transform.transform_expr(self, expr)
    else:
      return expr

  def transform_TiledReduce(self, expr):
    args = expr.args
    axes = expr.axes

    # TODO: Should make sure that all the shapes conform here,
    # but we don't yet have anything like assertions or error handling.
    niters = self.shape(args[0], syntax_helpers.unwrap_constant(axes[0]))

    if expr.fixed_tile_size:
      self.fixed_idx += 1
      tile_size = syntax_helpers.const(self.fixed_tile_sizes[self.fixed_idx])
    else:
      self.tiling = True
      self.fn.has_tiles = True
      self.nesting_idx += 1
      tile_size = self.index(self.tile_sizes_param, self.nesting_idx,
                             temp = True, name = "tilesize")

    slice_t = array_type.make_slice_type(Int64, Int64, Int64)

    untiled_map_fn = self.get_fn(expr.fn)

    acc_type = untiled_map_fn.return_type
    acc_is_array = not isinstance(acc_type, ScalarT)

    tiled_map_fn = self.transform_TypedFn(untiled_map_fn)
    map_closure_args = [self.get_closure_arg(e)
                        for e in self.closure_elts(expr.fn)]

    untiled_combine = self.get_fn(expr.combine)
    combine_closure_args = []

    tiled_combine = self.transform_TypedFn(untiled_combine, acc_is_array)
    if self.output_var and acc_is_array:
      result = self.output_var
    else:
      shape_args = map_closure_args + args
      result = self._create_output_array(untiled_map_fn, shape_args,
                                         [], "loop_result")
    init = result
    rslt_t = result.type

    if not acc_is_array:
      result_before = self.fresh_var(rslt_t, "result_before")
      init = result_before

    # Lift the initial value and fill it.
    def init_unpack(i, cur):
      if i == 0:
        return syntax.Assign(cur, syntax_helpers.zero_f64)
      else:
        j = self.fresh_i64("j")
        start = zero_i64
        stop = self.shape(cur, 0)

        self.blocks.push()
        n = self.index_along_axis(cur, 0, j)
        self.blocks += init_unpack(i-1, n)
        body = self.blocks.pop()

        return syntax.ForLoop(j, start, stop, one_i64, body, {})
    num_exps = array_type.get_rank(init.type) - \
               array_type.get_rank(expr.init.type)

    # TODO: Get rid of this when safe to do so.
    if not expr.fixed_tile_size or True:
      self.comment("TiledReduce in %s: init_unpack" % self.fn.name)
      self.blocks += init_unpack(num_exps, init)

    # Loop over the remaining tiles.
    merge = {}

    if not acc_is_array:
      result_after = self.fresh_var(rslt_t, "result_after")
      merge[result.name] = (result_before, result_after)

    def make_loop(start, stop, step, do_min = True):
      i = self.fresh_var(niters.type, "i")
      self.blocks.push()
      slice_stop = self.add(i, step, "next_bound")
      slice_stop_min = self.min(slice_stop, stop) if do_min \
                       else slice_stop

      tile_bounds = syntax.Slice(i, slice_stop_min, one_i64, type = slice_t)
      nested_args = [self.index_along_axis(arg, axis, tile_bounds)
                     for arg, axis in zip(args, axes)]

      new_acc = self.fresh_var(tiled_map_fn.return_type, "new_acc")
      self.comment("TiledReduce in %s: map_fn " % self.fn.name)
      do_inline(tiled_map_fn,
                map_closure_args + nested_args,
                self.type_env,
                self.blocks.top(),
                result_var = new_acc)

      loop_body = self.blocks.pop()
      if acc_is_array:
        outidx = self.tuple([syntax_helpers.slice_none] * result.type.rank)
        result_slice = self.index(result, outidx, temp = False)
        self.comment("")
        do_inline(tiled_combine,
                  combine_closure_args + [result, new_acc, result_slice],
                  self.type_env,
                  loop_body,
                  result_var = None)
      else:
        do_inline(tiled_combine,
                  combine_closure_args + [result, new_acc],
                  self.type_env, loop_body,
                  result_var = result_after)
      return syntax.ForLoop(i, start, stop, step, loop_body, merge)

    assert isinstance(tile_size, syntax.Expr), "%s not an expr" % tile_size

    self.comment("TiledReduce in %s: combine" % self.fn.name)

    if expr.fixed_tile_size and \
       config.opt_reg_tiles_not_tile_size_dependent and \
       syntax_helpers.unwrap_constant(tile_size) > 1:
      num_tiles = self.div(niters, tile_size, "num_tiles")
      tile_stop = self.mul(num_tiles, tile_size, "tile_stop")
      loop1 = make_loop(zero_i64, tile_stop, tile_size, False)
      self.blocks.append(loop1)
      loop2_start = self.assign_temp(loop1.var, "loop2_start")
      self.blocks.append(make_loop(loop2_start, niters, one_i64, False))
    else:
      self.blocks.append(make_loop(zero_i64, niters, tile_size))

    return result
