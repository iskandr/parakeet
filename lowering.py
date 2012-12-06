import syntax

from inline import Inliner
from lower_adverbs import LowerAdverbs
from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from simplify import Simplify
from tile_adverbs import TileAdverbs, LowerTiledAdverbs
from transform import apply_pipeline

tiling_pipeline = [
  TileAdverbs, LowerTiledAdverbs
]

no_tiling = [LowerAdverbs]

lowering_pipeline = [
  Simplify,
  Inliner,
  LowerIndexing,
  Simplify,
  LowerStructs,
  Simplify,
]

_lowered_functions = {}
def lower(fundef, tile=False):

  if isinstance(fundef, str):
    fundef = syntax.TypedFn.registry[fundef]

  key = (fundef, tile)
  if key in _lowered_functions:
    return _lowered_functions[key]
  else:
    if tile:
      lowered_fn = apply_pipeline(fundef, tiling_pipeline, copy = True)
    else:
      lowered_fn = apply_pipeline(fundef, no_tiling, copy = True)

    lowered_fn = apply_pipeline(lowered_fn, lowering_pipeline, copy = False)

    _lowered_functions[key] = lowered_fn
    _lowered_functions[(lowered_fn,tile)] = lowered_fn

    return lowered_fn
