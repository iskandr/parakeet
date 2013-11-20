##########################
#  Performance Options   #
##########################
fast_math = True 
sse2 = True 
opt_level = '-O2'
# overload the default compiler path  
compiler_path = None

##########################
# Insert Debugging Code  #
##########################
debug = False
check_pyobj_types = True 

#########################
#  Verbose Printing     #
#########################
print_input_ir = False

print_line_numbers = False
 
print_function_source = False

print_commands = False
print_command_elapsed_time = False


# Generate a .c file or a .cpp? 
pure_c = True
delete_temp_files = True 

# if compiling C or OpenMP we can skip some of the craziness and 
# have distutils figure out the system config and compiler for us 
use_distutils = True

# show all the warnings? 
suppress_compiler_output = False 

# when compiling with NVCC, other headers get implicitly included 
# and cause warnings since Python redefines _POSIX_C_SOURCE  
# we can undefine it before including Python.h to get rid of those warnings 
undef_posix_c_source = True 