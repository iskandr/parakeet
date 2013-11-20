######################################
#          BACKEND SELECTION         #
######################################
#
#  'c': sequential, use gcc or clang to compile
#  'openmp': multi-threaded execution for array operations, requires gcc 4.4+
#  'llvm': deprecated
#  'interp': interpreter, will be dreadfully slow
#  'cuda': experimental GPU support
#

backend = 'openmp' 

######################################
#        PARAKEET OPTIMIZATIONS      #
######################################
  
    
opt_inline = True

opt_fusion = True
opt_combine_nested_maps = True

opt_specialize_fn_args = True 

# experimental!
opt_simplify_array_operators = False

opt_index_elimination = True
opt_range_propagation = True

opt_licm = True
opt_redundant_load_elimination = True
opt_stack_allocation = True

opt_shape_elim = True 

# replace 
#   a = alloc
#   ...
#   b[i:j] = a
#
#   with 
#     a = b[i:j]  
opt_copy_elimination = True

# recompile functions for distinct patterns of unit strides
stride_specialization = True 

# may dramatically increase compile time
opt_loop_unrolling = False

# suspiciously complex optimizations may introduce bugs 
# TODO: comb through carefully 
opt_scalar_replacement = False
    
# run verifier after each transformation 
opt_verify = True


#####################################
#            DEBUG OUTPUT           #
#####################################

# show untyped IR after it's translated from Python?
print_untyped_function = False

# show the higher level typed function after specialization?
print_specialized_function = False 

# show function after all data adverbs like Map/Reduce/Scan have been 
# lowered to use indexing explicitly into their inputs 
print_indexified_function = False

# print function after all adverbs have been turned to loops
print_loopy_function = False

# show lower level typed function before
# it gets translated to LLVM?
print_lowered_function = False

# before starting function specialization, print the fn name and input types 
print_before_specialization = False

# show the input function to each transformation?
print_functions_before_transforms =  []
                                        
# show the function produced by each transformation?
print_functions_after_transforms =   []

# show aliases and escape sets
print_escape_analysis = False

# how long did each transform take?
print_transform_timings = False

# print each transform's name when it runs
print_transform_names = False

# at exit, print the names of all specialized functions
print_specialized_function_names = False

# tell the backend to print whatever code it generates, 
# whether it's C, CUDA, or LLVM 
print_generated_code = False 

#####################################
#         DESPERATE MEASURES        #
#####################################

testing_find_broken_transform = False 

