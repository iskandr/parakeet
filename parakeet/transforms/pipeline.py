from .. import config 
from ..analysis import contains_adverbs, contains_calls, contains_loops
# from const_arg_specialization import ConstArgSpecialization 
from copy_elimination import CopyElimination
from dead_code_elim import DCE
from flattening import Flatten
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
from negative_index_elim import NegativeIndexElim
from offset_propagation import OffsetPropagation
from parfor_to_nested_loops import ParForToNestedLoops
from phase import Phase
from redundant_load_elim import RedundantLoadElimination
from scalar_replacement import ScalarReplacement
from shape_elim import ShapeElimination
from shape_propagation import ShapePropagation
from simplify import Simplify
from range_propagation import RangePropagation

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
                   run_if = contains_calls, 
                   memoize = False )

copy_elim = Phase(CopyElimination, 
                  config_param = 'opt_copy_elimination', 
                  memoize = True)

licm = Phase(LoopInvariantCodeMotion, 
             config_param = 'opt_licm',
             run_if = contains_loops, 
             memoize = True, 
             name = "LICM")

symbolic_range_propagation = Phase([RangePropagation, OffsetPropagation], 
                                    config_param = 'opt_range_propagation',
                                    name = "SymRangeProp", 
                                    memoize = True)
high_level_optimizations = Phase([
                                    Simplify, 
                                    inline_opt, 
                                    # ConstArgSpecialization, Simplify, DCE,
                                    symbolic_range_propagation,   
                                    licm,
                                    fusion_opt, 
                                    fusion_opt, 
                                    copy_elim,  
                                 ], 
                                 name = "HighLevelOpts", 
                                 copy = True, 
                                 cleanup = [Simplify, DCE])

indexify = Phase([IndexifyAdverbs, 
                  inline_opt, Simplify, DCE, 
                  ShapePropagation, 
                  IndexMapElimination,
                 ], 
                 run_if = contains_adverbs, 
                 copy = True, 
                 name = "Indexify", 
                 depends_on=high_level_optimizations) 

flatten = Phase([Flatten, inline_opt], name="Flatten", copy=True, depends_on=indexify)

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
                   shape_elim,
                   symbolic_range_propagation,
                   NegativeIndexElim, 
                   index_elim
                ],
                depends_on = flatten,
                cleanup = [Simplify, DCE],
                copy = True,
                name = "Loopify",
                post_apply = print_loopy)


####################
#                  #
#     LOWERING     # 
#                  #
####################

scalar_repl = Phase(ScalarReplacement, 
                    config_param = 'opt_scalar_replacement', 
                    run_if = contains_loops)

# Currently incorrect in the presence of function calls
# TODO: Make this mark an slices that are call arguments as read & written to
load_elim = Phase(RedundantLoadElimination,
                  config_param = 'opt_redundant_load_elimination', 
                  run_if = lambda fn: not contains_calls(fn))


unroll = Phase(LoopUnrolling, config_param = 'opt_loop_unrolling', 
               run_if = contains_loops)


lowering = Phase([
                    LowerIndexing,
                    licm,
                    LowerStructs,
                    licm, 
                    unroll, 
                    licm,
                    load_elim, 
                    scalar_repl, 
                    Simplify,
                    
                 ],
                 depends_on = loopify,
                 copy = True,
                 name = "Lowering", 
                 cleanup = [Simplify, DCE])