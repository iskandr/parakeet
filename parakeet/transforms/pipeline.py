from .. import config 
from ..analysis import (contains_adverbs, contains_calls, contains_loops, 
                        contains_structs, contains_array_operators)

from combine_nested_maps import CombineNestedMaps 
from copy_elimination import CopyElimination
from dead_code_elim import DCE

from flattening import Flatten
from fusion import Fusion
from imap_elim import IndexMapElimination
from index_elimination import IndexElim
from indexify_adverbs import IndexifyAdverbs

from inline import Inliner
from licm import LoopInvariantCodeMotion
from loop_unrolling import LoopUnrolling
from lower_adverbs import LowerAdverbs
from lower_array_operators import LowerArrayOperators
from lower_indexing import LowerIndexing
from lower_slices import LowerSlices
from lower_structs import LowerStructs
from negative_index_elim import NegativeIndexElim
from offset_propagation import OffsetPropagation
from parfor_to_nested_loops import ParForToNestedLoops
from phase import Phase
from range_propagation import RangePropagation
from redundant_load_elim import RedundantLoadElimination
from scalar_replacement import ScalarReplacement
from shape_elim import ShapeElimination
from simplify import Simplify
from simplify_array_operators import SimplifyArrayOperators
from specialize_fn_args import SpecializeFnArgs

####################################
#                                  #
#    HIGH LEVEL OPTIMIZATIONS      # 
#                                  #
####################################

normalize = Phase([Simplify], 
                  memoize = True, 
                  copy = False, 
                  cleanup = [], 
                  name = "Normalize")

fusion_opt = Phase(Fusion, 
                   config_param = 'opt_fusion',
                   memoize = False,
                   copy = False, 
                   run_if = contains_adverbs)

inline_opt = Phase(Inliner, 
                   config_param = 'opt_inline', 
                   cleanup = [], 
                   run_if = contains_calls, 
                   memoize = False, 
                   copy = False)



licm = Phase(LoopInvariantCodeMotion, 
             config_param = 'opt_licm',
             run_if = contains_loops, 
             memoize = False, 
             name = "LICM")


symbolic_range_propagation = Phase([RangePropagation, OffsetPropagation], 
                                    config_param = 'opt_range_propagation',
                                    name = "SymRangeProp",
                                    copy = False,  
                                    memoize = False, 
                                    cleanup = [])
 
shape_elim = Phase(ShapeElimination,
                   config_param = 'opt_shape_elim')





early_optimizations = Phase([
                               inline_opt,
                               symbolic_range_propagation,   
                               licm,
                             ],
                            name = "EarlyOpt",
                            copy = True, 
                            memoize = True, 
                            cleanup = [Simplify, DCE], 
                            depends_on = normalize, 
 
                            )

simplify_array_operators = Phase([SimplifyArrayOperators], 
                                 copy = False, 
                                 memoize = False, 
                                 run_if = contains_array_operators, 
                                 config_param = "opt_simplify_array_operators")

combine_nested_maps = Phase([CombineNestedMaps],
                            copy = False, 
                            memoize = False, 
                            run_if = contains_adverbs, 
                            config_param = "opt_combine_nested_maps")

arg_specialization = Phase([SpecializeFnArgs],
                           copy = False, 
                           memoize = False, 
                           config_param = "opt_specialize_fn_args")

adverb_optimizations = Phase([
                                fusion_opt,
                                simplify_array_operators, 
                                combine_nested_maps,
                                arg_specialization,
                                fusion_opt, 
                                arg_specialization,
                              ], 
                             run_if = contains_adverbs, 
                             depends_on = early_optimizations, 
                             copy = True, 
                             memoize = True, 
                             cleanup = [Simplify, DCE])

high_level_optimizations = Phase([ 
                                    LowerArrayOperators,
                                     
                                    NegativeIndexElim,
                                    shape_elim, 
                                    symbolic_range_propagation, 
                                    
                                 ], 
                                 depends_on = adverb_optimizations,
                                 name = "HighLevelOpts", 
                                 copy = True, 
                                 memoize = True, 
                                 cleanup = [Simplify, DCE])



copy_elim = Phase(CopyElimination, 
                  config_param = 'opt_copy_elimination', 
                  memoize = False)


def print_indexified(fn):
  if config.print_indexified_function:
    print
    print "=== Indexified function ==="
    print
    print repr(fn)
    print


indexify = Phase([
                    IndexifyAdverbs, Simplify, DCE, 
                 ],
                 name = "Indexify",  
                 run_if = contains_adverbs, 
                 depends_on= high_level_optimizations, 
                 copy = True,
                 memoize = True, 
                 post_apply = print_indexified) 

after_indexify = Phase([copy_elim, Simplify, DCE, 
                        LowerSlices, 
                        inline_opt, Simplify, DCE, 
                        IndexMapElimination], 
                       name = "AfterIndexify", 
                       depends_on = indexify, 
                       copy = True, 
                       memoize = True)


flatten = Phase([Flatten, inline_opt, Simplify, DCE ], name="Flatten", 
                depends_on=after_indexify,
                run_if = contains_structs,  
                copy=True, 
                memoize = True)

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




# loop_fusion = Phase(LoopFusion, config_param = 'opt_loop_fusion')


index_elim = Phase([NegativeIndexElim, IndexElim], config_param = 'opt_index_elimination')



lower_adverbs = Phase([LowerAdverbs, LowerSlices], run_if = contains_adverbs)


# Currently incorrect in the presence of function calls
# TODO: Make this mark an slices that are call arguments as read & written to
load_elim = Phase(RedundantLoadElimination,
                  config_param = 'opt_redundant_load_elimination', 
                  run_if = lambda fn: not contains_calls(fn))

loopify = Phase([lower_adverbs,
                 ParForToNestedLoops,
                 inline_opt,
                 LowerSlices, 
                 licm, 
                 shape_elim, 
                 symbolic_range_propagation,
                 load_elim,  
                 index_elim, 
                ],
                depends_on = after_indexify,
                cleanup = [Simplify, DCE],
                copy = True,
                memoize = True, 
                name = "Loopify",
                post_apply = print_loopy)


####################
#                  #
#     LOWERING     # 
#                  #
####################



lowering = Phase([
                    LowerIndexing,
                    licm,
                    LowerStructs,
                 ],
                 depends_on = loopify,
                 memoize = True, 
                 copy = True,
                 name = "Lowering", 
                 cleanup = [Simplify, DCE])

############################
#                          #
#  FINAL LOOP OPTIMIZATINS #
#                          #
############################

scalar_repl = Phase(ScalarReplacement, 
                    config_param = 'opt_scalar_replacement', 
                    run_if = contains_loops)



unroll = Phase([LoopUnrolling, licm], 
               config_param = 'opt_loop_unrolling', 
               run_if = contains_loops)



final_loop_optimizations = Phase([
                              licm, 
                              unroll, 
                              load_elim, 
                              scalar_repl, 
                              Simplify
                            ],
                           cleanup = [Simplify, DCE], 
                           copy = False, 
                           memoize = True,\
                           name = "FinalLoopOptimizations"
                           )