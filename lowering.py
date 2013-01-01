import config
import syntax

from clone_function import CloneFunction
from copy_elimination import CopyElimination, PreallocAdverbOutput
from dead_code_elim import DCE
from fusion import Fusion
from inline import Inliner
from licm import LoopInvariantCodeMotion
from lower_adverbs import lower_adverbs
from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from lower_tiled_adverbs import LowerTiledAdverbs
from mapify_allpairs import MapifyAllPairs
from simplify import Simplify
from tile_adverbs import TileAdverbs
from transform import apply_pipeline

def build_pipeline(copy = False,
                   simplify = config.opt_cleanup_after_transforms,
                   inline = config.opt_inline,
                   fusion = config.opt_fusion,
                   licm = config.opt_licm):
  p = [CloneFunction] if copy else []

  if fusion:
    add(Fusion)

  if config.opt_copy_elimination:
    add(CopyElimination)
  return p

_lowered_functions = {}

def lower(fundef, tile=False):
  if isinstance(fundef, str):
    fundef = syntax.TypedFn.registry[fundef]

  key = (fundef, tile)
  if key in _lowered_functions:
    return _lowered_functions[key]
  else:
    fn = fundef
    num_tiles = 0
    if tile:
      prelim = [CloneFunction,
                MapifyAllPairs,
                Fusion, Simplify, DCE,
                TileAdverbs, Simplify, DCE,
                LowerTiledAdverbs, Simplify, DCE,
                Fusion, Simplify, DCE]
      fn = apply_pipeline(fundef, prelim)
      num_tiles = fn.num_tiles

    loopy_fn = lower_adverbs(fn)

    final_pipeline = []
    def add(*ts):
      final_pipeline.extend(ts)
      if config.opt_cleanup_after_transforms:
        final_pipeline.append(Simplify)
        final_pipeline.append(DCE)

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
    _lowered_functions[(lowered_fn,tile)] = lowered_fn

    if config.print_lowered_function:
      print
      print "=== Lowered function ==="
      print
      print repr(lowered_fn)
      print
    return lowered_fn
