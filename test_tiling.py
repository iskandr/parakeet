import adverbs
import array_type
import core_types
import numpy as np
import syntax
import syntax_helpers
import testing_helpers
import tile_adverbs

from function_registry import untyped_functions

id_fn = syntax.TypedFn(
  name = "id_fn",
  arg_names = ["x"],
  body = [syntax.Return(syntax_helpers.const(1))],
  return_type = core_types.Int32,
  type_env = {})

x_array = np.arange(100)
x_array_t = array_type.make_array_type(core_types.Int32, 1)

id_fn_2 = syntax.TypedFn(
  name = "id_fn_2",
  arg_names = ["x"],
  body = [syntax.Return(syntax.Var("x", type=core_types.Int32))],
  return_type = core_types.Int32,
  type_env = {"x":core_types.Int32})

map_fn = syntax.TypedFn(
  name = "map_fn",
  arg_names = ["X"],
  body = [syntax.Return(adverbs.Map(id_fn_2, ["X"], 0, type=x_array_t))],
  return_type = x_array_t,
  type_env = {"X":x_array_t})

def test_map_tiling():
  tiling_transform = tile_adverbs.TileAdverbs(map_fn)
  new_fn = tiling_transform.apply(copy=True)
  assert isinstance(new_fn, syntax.TypedFn)

def test_id_tiling():
  tiling_transform = tile_adverbs.TileAdverbs(id_fn_2)
  new_fn = tiling_transform.apply(copy=True)
  assert isinstance(new_fn, syntax.TypedFn)

def test_lowering():
  lower_tiling = tile_adverbs.LowerTiledAdverbs(id_fn)
  new_fn = lower_tiling.apply(copy=True)
  assert isinstance(new_fn, syntax.TypedFn)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
