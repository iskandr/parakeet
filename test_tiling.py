import adverbs
import array_type
import core_types
import interp
import llvm_backend
import lowering
import numpy as np
import parakeet
import run_function
import syntax
import syntax_helpers
import testing_helpers
import tile_adverbs
import transform

from parakeet import each
from run_function import specialize_and_compile

id_fn = syntax.TypedFn(
  name = "id_fn",
  arg_names = ["x"],
  input_types = [core_types.Int64],
  body = [syntax.Return(syntax_helpers.const(1))],
  return_type = core_types.Int64,
  type_env = {})

x_array = np.arange(100, dtype = np.int64)
x2_array = np.arange(100).reshape(10,10)
x_array_t = array_type.make_array_type(core_types.Int64, 1)
x_2_array_t = array_type.make_array_type(core_types.Int64, 2)

id_fn_2 = syntax.TypedFn(
  name = "id_fn_2",
  arg_names = ["x"],
  input_types = [core_types.Int64],
  body = [syntax.Return(syntax.Var("x", type=core_types.Int64))],
  return_type = core_types.Int64,
  type_env = {"x":core_types.Int64})

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

#def test_tiling():
#  rslt = parakeet.run(map_id, x_array)
#  print rslt

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
#  new_fn_3 = transform.apply_pipeline(new_fn_2, lowering.lowering_pipeline)
#  print new_fn_3
  new_fn = lowering.lower(map_fn, True)
  assert isinstance(new_fn, syntax.TypedFn)
  print new_fn
  llvm_fn, parakeet_fn, exec_engine = llvm_backend.compile_fn(new_fn)
  wrapper = run_function.CompiledFn(llvm_fn, parakeet_fn, exec_engine)
  rslt = wrapper([x2_array, np.array([3,3], dtype=np.int64)])
  print rslt

if __name__ == '__main__':
  testing_helpers.run_local_tests()
