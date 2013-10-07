default_backend = 'c' #llvm


######################################
#        PARAKEET OPTIMIZATIONS      #
######################################
  
default_opt_level = 2 


def set_opt_level(n):
  assert 0 <= n <= 3, "Invalid optimization level %d" % n
  g = globals()

  for name in ('opt_inline', 
               'opt_fusion', 
               'opt_index_elimination',
               'opt_range_propagation'):
    g[name] = n > 0
    
  for name in ('opt_licm', 
               'opt_redundant_load_elimination', 
               'opt_stack_allocation', 
               'opt_shape_elim', 
               'stride_specialization'):
    g[name] = n > 1 
    
  for name in ('opt_loop_unrolling',):
    g[name] = n > 2 
    
opt_inline = False
opt_fusion = False
opt_index_elimination = False
opt_range_propagation = False

opt_licm = False
opt_redundant_load_elimination = False
opt_stack_allocation = False
opt_shape_elim = False

# recompile functions for distinct patterns of unit strides
stride_specialization = False

# may dramatically increase compile time
opt_loop_unrolling = False

# suspiciously complex optimizations may introduce bugs 
# TODO: comb through carefully 
opt_scalar_replacement = False
opt_copy_elimination = False
    
# run verifier after each transformation 
opt_verify = True


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

