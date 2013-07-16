from .. import config, syntax 


from copy_elimination import CopyElimination
from dead_code_elim import DCE

from inline import Inliner
from licm import LoopInvariantCodeMotion
from loop_unrolling import LoopUnrolling

from lower_indexing import LowerIndexing
from lower_structs import LowerStructs
from pipeline_phase import Phase
from value_range_propagation import RangePropagation
from redundant_load_elim import RedundantLoadElimination
from scalar_replacement import ScalarReplacement
from simplify import Simplify
from index_elimination import IndexElim

inline_opt = Phase(Inliner, config_param = 'opt_inline', cleanup = [])
copy_elim = Phase(CopyElimination, config_param = 'opt_copy_elimination')
licm = Phase(LoopInvariantCodeMotion, config_param = 'opt_licm',
             memoize = False)

symbolic_range_propagation = Phase(RangePropagation,
                           config_param = 'opt_range_propagation',
                           memoize = False)

scalar_repl = Phase(ScalarReplacement, config_param = 'opt_scalar_replacement')
load_elim = Phase(RedundantLoadElimination,
                  config_param = 'opt_redundant_load_elimination')
unroll = Phase(LoopUnrolling, config_param = 'opt_loop_unrolling')

index_elim = Phase(IndexElim, config_param = 'opt_index_elimination')

pre_lowering = Phase([Simplify, 
                      inline_opt, 
                      copy_elim, 
                      licm, 
                      symbolic_range_propagation,
                      # shape_elim,
                      # symbolic_range_propagation,
                      index_elim ],
                     cleanup = [Simplify, DCE],
                     copy = True)
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
                  LowerStructs,
                  post_lowering,
                  LowerStructs,
                  Simplify],
                 copy = True,
                 cleanup = [])
