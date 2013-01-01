#####################################
#            DEBUG OUTPUT           #
#####################################

# show untyped IR after it's translated from Python?
print_untyped_function = False

# show the higher level typed function after
# specialization?
print_specialized_function = False

# show lower level typed function before
# it gets translated to LLVM?
print_lowered_function = False

# show the input function to each transformation?
print_functions_before_transforms = False# ['CopyElimination']

# show the function produced by each transformation?
print_functions_after_transforms = False#['CopyElimination']

# show functions after tiling transformation?
print_tiled_adverbs = False

# show LLVM bytecode before optimization passes
print_unoptimized_llvm = False

# show LLVM bytecode after optimizations
print_optimized_llvm = False

# how long did each transform take?
print_transform_timings = False

# at exit, print the names of all specialized functions
print_specialized_function_names = False

# show execution time on parallel backend?
print_parallel_exec_time = False

######################################
#        PARAKEET OPTIMIZATIONS      #
######################################
opt_inline = True
opt_fusion = True
opt_licm = True
opt_cleanup_after_transforms = True
opt_verify = True
opt_copy_elimination = False

######################################
#           LLVM OPTIONS             #
######################################

# run LLVM optimization passes
llvm_optimize = True

# number of times to run optimizations
llvm_num_passes = 4

# run verifier over generated LLVM code?
llvm_verify = True

######################################
#         RUNTIME OPTIONS            #
######################################

# Run the adverbs called from Python in parallel
call_from_python_in_parallel = False

# Tile adverbs when they're run in parallel
opt_tile = False
