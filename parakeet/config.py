
default_backend = 'c' #llvm 

######################################
#        PARAKEET OPTIMIZATIONS      #
######################################

opt_inline = True
opt_fusion = True
opt_licm = True
opt_stack_allocation = True
opt_index_elimination = True
opt_range_propagation = True
opt_shape_elim = True
opt_redundant_load_elimination = True  

# may dramatically increase compile time
opt_loop_unrolling = False

# suspiciously complex optimizations may introduce bugs 
# TODO: comb through carefully 
opt_scalar_replacement = False
opt_copy_elimination = False

# recompile functions for distinct patterns of unit strides
# in array arguments
stride_specialization = True

# run verifier after each transformation 
opt_verify = False

#####################################
#            DEBUG OUTPUT           #
#####################################

# show untyped IR after it's translated from Python?
print_untyped_function = False

# show the higher level typed function after specialization?
print_specialized_function = False 

# print function after all adverbs have been turned to loops
print_loopy_function = True

# show lower level typed function before
# it gets translated to LLVM?
print_lowered_function = False

# before starting function specialization, print the fn name and input types 
print_before_specialization = False

# show the input function to each transformation?
print_functions_before_transforms = ['Flatten', 'LowerSlices'] #['Fusion'] #['NegativeIndexElim'] #['Simplify']  

# show the function produced by each transformation?
print_functions_after_transforms = [ 'Flatten', 'LowerSlices']# ['Fusion'] #IndexifyAdverbs'] #['Fusion'] #['NegativeIndexElim'] # ['Simplify'] #['IndexifyAdverbs']  

# show aliases and escape sets
print_escape_analysis = False

# how long did each transform take?
print_transform_timings = False

# print each transform's name when it runs
print_transform_names = True 

# at exit, print the names of all specialized functions
print_specialized_function_names = False



