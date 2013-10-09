### 0.17 / October 9th, 2013 ###

- Added support for binding multiple variables in a for loop (i.e. "for (x,(y,z)) in enumerate(zip(ys,zs)):")
- More array constructors support 'dtype' argument 
- macros can now generate wrapper functions with keyword arguments 
- lib_helpers._get_type can pull element types out of a wider range of values 
- Added more unit tests from Python benchmarks 
- Added option to manually specify compiler in c_backend.config.compiler_path
- Slightly better support for negative indexing but negative step sizes are still mostly broken 
 
### 0.16.2 / October 1st, 2013 ###

- Moved version info into submodule so setup.py can run without full dependencies (thanks rjpower). 
- Fixed support for references to global arrays.
- Make C backend respect runtime changes to config flags. 
- Got rid of unncessary linking against libpython. 

### 0.16.1 / September 30th, 2013 ###

- Fixed bugs in C backend and a several optimizatons.
- Only print stderr if C backend fails to compile.  

### 0.16 / September 27th, 2013 ###

- Added flattening transformation which gets rid of all structures except scalars and pointers.
- Created new C backend which compiles with either gcc or clang.
- Disabled some of the more powerful optimizations (i.e. scalar replacement) which might introduce bugs. 
- Revamped the optimization pipeline to avoid running duplicate transformations and allow for better logging. 
- "Indexify" all adverbs by turning them into ParFor/IndexReduce/IndexScan. 

