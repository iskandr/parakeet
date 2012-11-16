from transform import apply_pipeline
from lower_adverbs import LowerAdverbs
from tile_adverbs import TileAdverbs, LowerTiledAdverbs
from lower_structs import LowerStructs
from lower_indexing import LowerIndexing
from simplify import Simplify

tiling_pipeline = [
  TileAdverbs, LowerTiledAdverbs
]

pipeline = [
  LowerAdverbs, LowerIndexing,
  Simplify,
  LowerStructs,
  Simplify,
]

_lowered_functions = {}
def lower(fundef, tile=False):
  key = (fundef.name, tile)
  if key in _lowered_functions:
    return _lowered_functions[key]
  else:
    if tile:
      fundef = apply_pipeline(fundef, tiling_pipeline, copy = True)
    # print "BEFORE LOWERING", fundef
    lowered_fn = apply_pipeline(fundef, pipeline, copy = True)
    # print "AFTER LOWERING", lowered_fn
    _lowered_functions[key] = lowered_fn
    _lowered_functions[(lowered_fn.name,tile)] = lowered_fn
    return lowered_fn