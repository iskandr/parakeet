import config
import syntax_visitor

from copy_elimination import CopyElimination
from dead_code_elim import DCE
from fusion import Fusion
from inline import Inliner
from licm import LoopInvariantCodeMotion
from loop_fusion import LoopFusion
from loop_unrolling import LoopUnrolling
from lower_adverbs import LowerAdverbs
from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from lower_tiled_adverbs import LowerTiledAdverbs
from mapify_allpairs import MapifyAllPairs
from pipeline_phase import Phase
from range_propagation import RangePropagation
from redundant_load_elim import RedundantLoadElimination
from scalar_replacement import ScalarReplacement
from shape_elim import ShapeElimination
from simplify import Simplify
from tile_adverbs import TileAdverbs

class ContainsAdverbs(syntax_visitor.SyntaxVisitor):
  class Yes(Exception):
    pass
  def visit_Map(self, _):
    raise self.Yes()
  def visit_Reduce(self, _):
    raise self.Yes()
  def visit_Scan(self, _):
    raise self.Yes()
  def visit_AllPairs(self, _):
    raise self.Yes()

def contains_adverbs(fn):
  try:
    ContainsAdverbs().visit_fn(fn)
  except ContainsAdverbs.Yes:
    return True
  return False


fusion_opt = Phase(Fusion, config_param = 'opt_fusion', cleanup = [DCE],
                   memoize = False,
                   run_if = contains_adverbs)
inline_opt = Phase(Inliner, config_param = 'opt_inline', cleanup = [])
high_level_optimizations = Phase([Simplify, inline_opt, Simplify, DCE,
                                  fusion_opt, fusion_opt])
def print_loopy(fn):
  if config.print_loopy_function:
    print
    print "=== Loopy function ==="
    print
    print repr(fn)
    print

copy_elim = Phase(CopyElimination, config_param = 'opt_copy_elimination')
licm = Phase(LoopInvariantCodeMotion, config_param = 'opt_licm',
             memoize = False)

symbolic_range_propagation = Phase(RangePropagation,
                           config_param = 'opt_range_propagation',
                           memoize = False)
shape_elim = Phase(ShapeElimination,
                   config_param = 'opt_shape_elim')
loop_fusion = Phase(LoopFusion, config_param = 'opt_loop_fusion')
loopify = Phase([Simplify,
                 fusion_opt,
                 LowerAdverbs, inline_opt,
                 copy_elim,
                 licm,],
                depends_on = high_level_optimizations,
                cleanup = [Simplify, DCE],
                copy = True,
                run_if = contains_adverbs,
                post_apply = print_loopy)

mapify = Phase(MapifyAllPairs, copy = False)
pre_tiling = Phase([mapify, fusion_opt], copy = True)
post_tiling = Phase([fusion_opt, copy_elim], copy = True)
tiling = Phase([pre_tiling, TileAdverbs, LowerTiledAdverbs, post_tiling],
               config_param = 'opt_tile',
               depends_on = high_level_optimizations,
               rename = True,
               memoize = False,
               run_if = contains_adverbs,
               cleanup = [Simplify, DCE])

scalar_repl = Phase(ScalarReplacement, config_param = 'opt_scalar_replacement')
load_elim = Phase(RedundantLoadElimination,
                  config_param = 'opt_redundant_load_elimination')
unroll = Phase(LoopUnrolling, config_param = 'opt_loop_unrolling')

pre_lowering = Phase([symbolic_range_propagation,
                      shape_elim,
                      symbolic_range_propagation],
                     cleanup = [Simplify, DCE])
post_lowering = Phase([licm,
                       unroll,
                       licm,
                       Simplify,
                       load_elim,
                       scalar_repl,
                       ], cleanup = [Simplify, DCE])
lowering = Phase([pre_lowering,
                  LowerIndexing,
                  licm,
                  loop_fusion,
                  LowerStructs,
                  post_lowering,],
                 depends_on = loopify,
                 copy = True,
                 cleanup = [])

def lower_tiled(fn, ignore_config = True):
  """
  Tiling is awkward to stick into the transformation graph since it's
  non-idempotent and generates lots of previously unseen functions
  """

  tiled = tiling.apply(fn, ignore_config = ignore_config)
  loopy = loopify.apply(tiled, run_dependencies = False)
  lowered = lowering.apply(loopy, run_dependencies = False)
  return lowered