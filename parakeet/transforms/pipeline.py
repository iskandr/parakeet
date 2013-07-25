from .. import config 
from ..analysis import contains_adverbs, contains_calls 

from copy_elimination import CopyElimination
from dead_code_elim import DCE
from fusion import Fusion
from index_elimination import IndexElim
from indexify_adverbs import IndexifyAdverbs
from inline import Inliner
from licm import LoopInvariantCodeMotion
from loop_unrolling import LoopUnrolling
from lower_adverbs import LowerAdverbs
from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from imap_elim import IndexMapElimination

#from mapify_allpairs import MapifyAllPairs
#from maps_to_parfor import MapsToParFor
from parfor_to_nested_loops import ParForToNestedLoops
from phase import Phase
from redundant_load_elim import RedundantLoadElimination
from scalar_replacement import ScalarReplacement
from shape_elim import ShapeElimination
from shape_propagation import ShapePropagation
from simplify import Simplify
from value_range_propagation import RangePropagation
from parakeet.transforms import indexify_adverbs



####################################
#                                  #
#    HIGH LEVEL OPTIMIZATIONS      # 
#                                  #
####################################


fusion_opt = Phase(Fusion, config_param = 'opt_fusion', cleanup = [Simplify, DCE],
                   memoize = False,
                   run_if = contains_adverbs)
inline_opt = Phase(Inliner, 
                   config_param = 'opt_inline', 
                   cleanup = [], 
                   run_if = contains_calls)

copy_elim = Phase(CopyElimination, config_param = 'opt_copy_elimination')

licm = Phase(LoopInvariantCodeMotion, config_param = 'opt_licm',
             memoize = False)

indexify_adverbs = Phase([IndexifyAdverbs, 
                          inline_opt, Simplify, DCE, 
                          ShapePropagation, 
                          IndexMapElimination,
                          ], 
                         run_if = contains_adverbs) 
high_level_optimizations = Phase([
                                    Simplify, 
                                    inline_opt, Simplify, DCE,
                                    licm, Simplify, DCE, 
                                    fusion_opt, 
                                    fusion_opt, 
                                    copy_elim, Simplify, DCE, 
                                    indexify_adverbs, 
                                 ], 
                                 copy = True)

####################
#                  #
#     LOOPIFY      # 
#                  #
####################

def print_loopy(fn):
  if config.print_loopy_function:
    print
    print "=== Loopy function ==="
    print
    print repr(fn)
    print


symbolic_range_propagation = Phase(RangePropagation,
                           config_param = 'opt_range_propagation',
                           memoize = False)
shape_elim = Phase(ShapeElimination,
                   config_param = 'opt_shape_elim')
# loop_fusion = Phase(LoopFusion, config_param = 'opt_loop_fusion')


index_elim = Phase(IndexElim, config_param = 'opt_index_elimination')


lower_adverbs = Phase([LowerAdverbs], run_if = contains_adverbs)
loopify = Phase([
                   lower_adverbs,
                   ParForToNestedLoops, 
                   inline_opt, 
                   copy_elim,
                   licm,
                   symbolic_range_propagation,
                   shape_elim,
                   symbolic_range_propagation,
                   index_elim
                ],
                depends_on = high_level_optimizations,
                cleanup = [Simplify, DCE],
                copy = True,
                post_apply = print_loopy)


####################
#                  #
#     LOWERING     # 
#                  #
####################

scalar_repl = Phase(ScalarReplacement, config_param = 'opt_scalar_replacement')

# Currently incorrect in the presence of function calls
# TODO: Make this mark an slices that are call arguments as read & written to
load_elim = Phase(RedundantLoadElimination,
                  config_param = 'opt_redundant_load_elimination', 
                  run_if = lambda fn: not contains_calls(fn))
unroll = Phase(LoopUnrolling, config_param = 'opt_loop_unrolling')



post_lowering = Phase([
                         licm,
                         unroll,
                         licm,
                         Simplify,
                         load_elim,
                         scalar_repl,
                      ], 
                      cleanup = [Simplify, DCE])

lowering = Phase([
                    LowerIndexing,
                    licm,
                    LowerStructs,
                    post_lowering
                    
                 ],
                 depends_on = loopify,
                 copy = True,
                 cleanup = [])