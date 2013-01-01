import config
import syntax

from clone_function import CloneFunction
from copy_elimination import CopyElimination
from dead_code_elim import DCE
from fusion import Fusion
from licm import LoopInvariantCodeMotion
from lower_adverbs import lower_adverbs
from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from lower_tiled_adverbs import LowerTiledAdverbs
from mapify_allpairs import MapifyAllPairs
from simplify import Simplify
from tile_adverbs import TileAdverbs
from transform import apply_pipeline


_tiled_functions = {}
def tiling(old_fn):
  if old_fn.name in _tiled_functions:
    return _tiled_functions[old_fn.name]
  pipeline = [
    CloneFunction,
    MapifyAllPairs
  ]
  if config.opt_fusion:
    pipeline += [Fusion, Simplify, DCE]
  
  pipeline += [
    TileAdverbs, Simplify, DCE,
    LowerTiledAdverbs, Simplify, DCE
  ]
  if config.opt_fusion:
    pipeline += [Fusion, Simplify, DCE]
  new_fn = apply_pipeline(old_fn, pipeline)
  _tiled_functions[new_fn.name] = new_fn
  _tiled_functions[old_fn.name] = new_fn  
  return new_fn


_lowered_functions = {}
def lower(old_fn, tile=False):
  if isinstance(old_fn, str):
    old_fn = syntax.TypedFn.registry[old_fn]

  key = (old_fn.name, tile)
  if key in _lowered_functions:
    return _lowered_functions[key]
  else:
    
    if tile:
      old_fn = tiling(old_fn)
      num_tiles = old_fn.num_tiles
    else:
      num_tiles = 0
        
    loopy_fn = lower_adverbs(old_fn)

    final_pipeline = [CloneFunction]
    def add(*ts):
      final_pipeline.extend(ts + (Simplify, DCE))   
         
    if config.opt_copy_elimination:
      add(CopyElimination)

    add(LowerIndexing)
    add(LowerStructs)
    if config.opt_licm:
      add(LoopInvariantCodeMotion)

    lowered_fn = apply_pipeline(loopy_fn, final_pipeline)
    lowered_fn.num_tiles = num_tiles
    lowered_fn.has_tiles = (num_tiles > 0)

    _lowered_functions[key] = lowered_fn
    _lowered_functions[(lowered_fn.name,tile)] = lowered_fn

    if config.print_lowered_function:
      print
      print "=== Lowered function ==="
      print
      print repr(lowered_fn)
      print
    return lowered_fn
