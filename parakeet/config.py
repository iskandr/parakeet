default_backend = 'c' #llvm


######################################
#        PARAKEET OPTIMIZATIONS      #
######################################
  
default_opt_level = 2 

def set_opt_level(n):
  assert 0 <= n <= 3, "Invalid optimization level %d" % n
  g = globals()

  if n > 0:
    g['opt_inline'] = True
    g['opt_fusion'] = True
    g['opt_stack_allocation'] = True
    g['opt_index_elimination'] = True
    g['opt_range_propagation'] = True 
    g['opt_shape_elim'] = True
  if n > 1:
    g['opt_licm'] = True  
    g['opt_redundant_load_elimination'] = True

  if n > 2: 
    g['opt_loop_unrolling'] = True

opt_inline = False
opt_fusion = False
opt_stack_allocation = False
opt_index_elimination = False
opt_range_propagation = False
opt_shape_elim = False
opt_redundant_load_elimination = False
opt_licm = False

# may dramatically increase compile time
opt_loop_unrolling = False

# suspiciously complex optimizations may introduce bugs 
# TODO: comb through carefully 
opt_scalar_replacement = False
opt_copy_elimination = False
    
# run verifier after each transformation 
opt_verify = True

# recompile functions for distinct patterns of unit strides
stride_specialization = True

set_opt_level(default_opt_level)

#####################################
#            DEBUG OUTPUT           #
#####################################

# show untyped IR after it's translated from Python?
print_untyped_function = False

# show the higher level typed function after specialization?
print_specialized_function = False

# print function after all adverbs have been turned to loops
print_loopy_function = False

# show lower level typed function before
# it gets translated to LLVM?
print_lowered_function = False

# before starting function specialization, print the fn name and input types 
print_before_specialization = False

# show the input function to each transformation?
print_functions_before_transforms = [] #['Flatten', 'LowerSlices', 'LowerAdverbs', 'IndexifyAdverbs']   

# show the function produced by each transformation?
print_functions_after_transforms =  [] #['Flatten', 'LowerSlices', 'LowerAdverbs', 'IndexifyAdverbs']  

# show aliases and escape sets
print_escape_analysis = False

# how long did each transform take?
print_transform_timings = False

# print each transform's name when it runs
print_transform_names = False

# at exit, print the names of all specialized functions
print_specialized_function_names = False

#####################################
#         DESPERATE MEASURES        #
#####################################

testing_find_broken_transform = False 

