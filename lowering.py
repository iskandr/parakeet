import config
import syntax

from clone_function import CloneFunction
from dead_code_elim import DCE
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

def build_pipeline(copy = True,
                   simplify = config.opt_simplify_when_lowering,
                   inline = config.opt_inline_when_lowering,
                   fusion = config.opt_fusion,
                   licm = config.opt_licm):
  p = [CloneFunction] if copy else []

  def add(t):
    p.append(t)
    if simplify:
      p.append(Simplify)
      p.append(DCE)

  if fusion:
    add(Fusion)

  add(LowerAdverbs)
  if inline:
    add(Inliner)

  add(LowerIndexing)
  add(LowerStructs)

  if licm:
    add(LoopInvariantCodeMotion)
  return p

_lowered_functions = {}
def lower(fundef, tile=False):
  if isinstance(fundef, str):
    fundef = syntax.TypedFn.registry[fundef]

  key = (fundef, tile)
  if key in _lowered_functions:
    return _lowered_functions[key]
  else:
    num_tiles = 0
    if tile:
      p = [CloneFunction,
           TileAdverbs, Simplify, DCE,
           LowerTiledAdverbs, Simplify, DCE]
      fundef = apply_pipeline(fundef, p)
      num_tiles = fundef.num_tiles
      print fundef

    pipeline = build_pipeline(copy=True)
    lowered_fn = apply_pipeline(fundef, pipeline)
    _lowered_functions[key] = lowered_fn
    _lowered_functions[(lowered_fn,tile)] = lowered_fn

    if tile:
      lowered_fn.num_tiles = num_tiles

    if config.print_lowered_function:
      print
      print "=== Lowered function ==="
      print
      print repr(lowered_fn)
      print
    return lowered_fn
