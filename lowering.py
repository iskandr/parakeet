import config
import syntax

from fusion import Fusion
from inline import Inliner
from licm import LoopInvariantCodeMotion
from lower_adverbs import LowerAdverbs
from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from lower_tiled_adverbs import LowerTiledAdverbs
from simplify import Simplify
from dead_code_elim import DCE 
from tile_adverbs import TileAdverbs
from transform import apply_pipeline
from clone_function import CloneFunction
# import config 

before_tiling = [
  CloneFunction
]

tiling_pipeline = [
  TileAdverbs, LowerTiledAdverbs
]

after_tiling = [
  Simplify,
  DCE, 
  Fusion,
  LowerAdverbs,
  Simplify,
  DCE, 
  Inliner,
  LowerIndexing,
  Simplify,
  DCE, 
  LowerStructs,
  Simplify,
  DCE, 
  # LoopInvariantCodeMotion,
  Simplify,
  DCE
]

_lowered_functions = {}
def lower(fundef, tile=False):
  if isinstance(fundef, str):
    fundef = syntax.TypedFn.registry[fundef]

  key = (fundef, tile)
  if key in _lowered_functions:
    return _lowered_functions[key]
  else:
    pipeline = before_tiling + (tiling_pipeline if tile else []) + after_tiling
    lowered_fn = apply_pipeline(fundef, pipeline)
    _lowered_functions[key] = lowered_fn
    _lowered_functions[(lowered_fn,tile)] = lowered_fn

    if config.print_lowered_function:
      print
      print "=== Lowered function ==="
      print 
      print repr(lowered_fn)
      print
    return lowered_fn
