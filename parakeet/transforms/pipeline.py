from loopjit import Phase 

from .. analysis import contains_adverbs  
from fusion import Fusion
from lower_adverbs import LowerAdverbs
from mapify_allpairs import MapifyAllPairs
from shape_elim import ShapeElimination




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
# loop_fusion = Phase(LoopFusion, config_param = 'opt_loop_fusion')

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


