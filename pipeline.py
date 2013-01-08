import config

from copy_elimination import CopyElimination
from dead_code_elim import DCE
from fusion import Fusion
from inline import Inliner
from licm import LoopInvariantCodeMotion
from loop_fusion import LoopFusion
from lower_adverbs import LowerAdverbs
from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from lower_tiled_adverbs import LowerTiledAdverbs
from mapify_allpairs import MapifyAllPairs
from pipeline_phase import Phase
from simplify import Simplify
from tile_adverbs import TileAdverbs
from loop_unrolling import LoopUnrolling

fusion_opt = Phase(Fusion, config_param = 'opt_fusion')
inline_opt = Phase(Inliner, config_param = 'opt_inline')
high_level_optimizations = Phase([Simplify, inline_opt, fusion_opt],
                                 cleanup = [Simplify, DCE])

mapify = Phase(MapifyAllPairs, copy = False)
pre_tiling = Phase([mapify, fusion_opt], copy = True)
post_tiling = Phase([fusion_opt], copy = True)
tiling = Phase([pre_tiling, TileAdverbs, LowerTiledAdverbs],
               config_param = 'opt_tile',
               depends_on = high_level_optimizations,
               rename = True,
               memoize = False,
               cleanup = [Simplify, DCE])

copy_elim = Phase(CopyElimination, config_param = 'opt_copy_elimination')
licm = Phase(LoopInvariantCodeMotion, config_param = 'opt_licm', memoize = False)
loop_fusion = Phase(LoopFusion, config_param = 'opt_loop_fusion')
unroll = Phase(LoopUnrolling, config_param = 'opt_loop_unrolling')  
loopify = Phase([LowerAdverbs, inline_opt, licm, copy_elim, loop_fusion, licm],
                 depends_on = high_level_optimizations,  
                 cleanup = [Simplify, DCE],
                 copy = True)

def print_lowered(fn):
  if config.print_lowered_function:
    print
    print "=== Lowered function ==="
    print
    print repr(fn)
    print



lowering = Phase([LowerIndexing, unroll, licm,  LowerStructs, licm,  copy_elim],
                 depends_on = loopify,
                 copy = True,
                 run_after = print_lowered,
                 cleanup = [Simplify, DCE])

def lower_tiled(fn, ignore_config = True):
  """
  Tiling is awkward to stick into the transformation graph since it's
  non-idempotent and generates lots of previously unseen functions
  """
  
  tiled = tiling.apply(fn, ignore_config = ignore_config)
  loopy = loopify.apply(tiled, run_dependencies = False)
  lowered = lowering.apply(loopy, run_dependencies = False)
  return lowered
