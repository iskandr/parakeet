#####################################
#            DEBUG OUTPUT           #
#####################################

# show untyped IR after it's translated from Python?
print_untyped_function = False

# show the higher level typed function after specialization?
print_specialized_function = False

# show lower level typed function before
# it gets translated to LLVM?
print_lowered_function = True

# print function after all adverbs have been turned to loops
print_loopy_function = False

# show the input function to each transformation?
print_functions_before_transforms = False

# show the function produced by each transformation?
print_functions_after_transforms = False 

# show functions before tiling transformation?
print_functions_before_tiling = False

# show functions after tiling transformation?
print_tiled_adverbs = False

# show LLVM bytecode before optimization passes
print_unoptimized_llvm = False

# show LLVM bytecode after optimizations
print_optimized_llvm = False

# show aliases and escape sets
print_escape_analysis = False

# how long did each transform take?
print_transform_timings = False

# print each transform's name when it runs
print_transform_names = False

# at exit, print the names of all specialized functions
print_specialized_function_names = False

# show execution time on parallel backend?
print_parallel_exec_time = False

# print details about the ongoing tile size search
print_tile_search_info = False

# print generated assembly of compiled functions
print_x86 = False

######################################
#        PARAKEET OPTIMIZATIONS      #
######################################

opt_verify = True
opt_inline = True
opt_fusion = True
opt_licm = True
opt_copy_elimination = True
opt_stack_allocation = True
opt_range_propagation = True
opt_shape_elim = True
opt_scalar_replacement = True
opt_redundant_load_elimination = True

# may dramatically increase compile time
opt_loop_unrolling = True

# recompile functions for distinct patterns of unit strides
# in array arguments
stride_specialization = False

# Warning: loop fusion never fully implemented
opt_loop_fusion = False

######################################
#           LLVM OPTIONS             #
######################################

# run LLVM optimization passes
llvm_optimize = True

# number of times to run optimizations
llvm_num_passes = 4

# run verifier over generated LLVM code?
llvm_verify = False

######################################
#         RUNTIME OPTIONS            #
######################################

# Run the adverbs called from Python in parallel
call_from_python_in_parallel = False 

# Tile adverbs when they're run in parallel
opt_tile = True

# Add a level of tiling for registers
opt_reg_tile = True
opt_reg_tiles_not_tile_size_dependent = True

# Perform auto-tuning search for best tile parameters
opt_autotune_tile_sizes = True

use_cached_tile_sizes = True
