import syntax

from fusion import Fusion
from inline import Inliner
from licm import LoopInvariantCodeMotion
from lower_adverbs import LowerAdverbs
from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from lower_tiled_adverbs import LowerTiledAdverbs
from simplify import Simplify
from tile_adverbs import TileAdverbs
from transform import apply_pipeline

tiling_pipeline = [
  TileAdverbs, LowerTiledAdverbs
]

lowering_pipeline = [
#  Simplify,
#  Fusion,
  LowerAdverbs,
#  Simplify,
  Inliner,
  LowerIndexing,
#  Simplify,
  LowerStructs,
#  Simplify,
#  LoopInvariantCodeMotion,
#  Simplify,
]

_lowered_functions = {}
def lower(fundef, tile=False):
  if isinstance(fundef, str):
    fundef = syntax.TypedFn.registry[fundef]

  key = (fundef, tile)
  if key in _lowered_functions:
    return _lowered_functions[key]
  else:
    lowered_fn = fundef
    if tile:
      lowered_fn = apply_pipeline(lowered_fn, tiling_pipeline, copy = True)
      lowered_fn = apply_pipeline(lowered_fn, lowering_pipeline, copy = False)
    else:
      lowered_fn = apply_pipeline(lowered_fn, lowering_pipeline, copy = True)

    _lowered_functions[key] = lowered_fn
    _lowered_functions[(lowered_fn,tile)] = lowered_fn

    return lowered_fn
