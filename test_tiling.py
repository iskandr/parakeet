import adverbs
import array_type
import core_types
import lowering
import numpy as np
import syntax
import syntax_helpers
import testing_helpers
import tile_adverbs

from parakeet import each
from run_function import specialize_and_compile

id_fn = syntax.TypedFn(
  name = "id_fn",
  arg_names = ["x"],
  input_types = [core_types.Int32],
  body = [syntax.Return(syntax_helpers.const(1))],
  return_type = core_types.Int32,
  type_env = {})

x_array = np.arange(100)
x_array_t = array_type.make_array_type(core_types.Int32, 1)
x_2_array_t = array_type.make_array_type(core_types.Int32, 2)

id_fn_2 = syntax.TypedFn(
  name = "id_fn_2",
  arg_names = ["x"],
  input_types = [core_types.Int32],
  body = [syntax.Return(syntax.Var("x", type=core_types.Int32))],
  return_type = core_types.Int32,
  type_env = {"x":core_types.Int32})

map_fn = syntax.TypedFn(
  name = "map_fn",
  arg_names = ["X"],
  input_types = [x_array_t],
  body = [syntax.Return(adverbs.Map(id_fn_2, [syntax.Var("X", type=x_array_t)],
                                    0, type=x_array_t))],
  return_type = x_array_t,
  type_env = {"X":x_array_t})

map2_fn = syntax.TypedFn(
  name = "map2_fn",
  arg_names = ["X"],
  input_types = [x_2_array_t],
  body = [syntax.Return(adverbs.Map(map_fn, [syntax.Var("X", type=x_2_array_t)],
                                    0, type=x_2_array_t))],
  return_type = x_2_array_t,
  type_env = {"X":x_2_array_t})

def identity(x):
  return x

def map_id(X):
  return each(identity, X)

#def vm(x, y):
#  tmp = each(lambda x,y: x*y, x, y)
#  return reduce(lambda x,y: x+y, tmp)
#
#def test_vm_tiling():
#  _, typed, _, _ = specialize_and_compile(vm, [x_array, x_array])
#  print typed
#  tiling_transform = tile_adverbs.TileAdverbs()
#
#def test_map_tiling():
#  tiling_transform = tile_adverbs.TileAdverbs(map2_fn)
#  new_fn = tiling_transform.apply(copy=True)
#  print new_fn
#  assert isinstance(new_fn, syntax.TypedFn)
#
#def test_id_tiling():
#  tiling_transform = tile_adverbs.TileAdverbs(id_fn_2)
#  new_fn = tiling_transform.apply(copy=True)
#  assert isinstance(new_fn, syntax.TypedFn)

def test_lowering():
#  tiling_transform = tile_adverbs.TileAdverbs(map2_fn)
#  new_fn = tiling_transform.apply(copy=True)
#  print new_fn
#  lower_tiling = tile_adverbs.LowerTiledAdverbs(new_fn)
#  new_fn_2 = lower_tiling.apply(copy=True)
#  assert isinstance(new_fn_2, syntax.TypedFn)
#  print new_fn_2
  new_fn = lowering.lower(map2_fn, True)
  assert isinstance(new_fn, syntax.TypedFn)
  print new_fn

if __name__ == '__main__':
  testing_helpers.run_local_tests()
