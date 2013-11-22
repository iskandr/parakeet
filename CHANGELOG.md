### 0.23 / November 21st, 2013 ### 

- Generalized StrideSpecialization to ValueSpecialization 
- Significantly decreased overhead of calling into Parakeet (though still ~500x slower than a normal Python call)

### 0.22 / November 20th, 2013 ### 

- Changed NumPy calls to use newer API, cuts down on number of compile warnings
- If compilation fails, retry using distutils, may help some people on Windows

### 0.21 / November 19th, 2013 ### 

- Got rid of testing dependency on SciPy
- Deleted unused and unfinished optimizations
- Small misc. bugs 

### 0.20 / November 19th, 2013 ###

The last release added experimental CUDA support but the performance was terrible. This release includes lots of tweaks and optimizations necessary for getting beneficial speedups on the GPU. However, the default backend remains OpenMP since some program constructs will not work on the GPU and the nvcc compile times are unacceptably slow.

- Expanded and generalized fusion optimization
- Filled in missing methods from shape inference
- Using ShapeElimination on every function (repurposes the shape inference results as a symbolic execution optimization)
- Fixed lots of small bugs in other optimizations exposed by ShapeElimination
- Shaved off small amount of compile time by moving away from Node pseudo-ASTs to regular Python constructors
- Hackishly added int24 just as a sentinel for default values in reductions that need to cast up to int32 from bool, int8, int16.
- Eliminate redundant & constant array operator arguments with SpecializeFnArgs

### 0.19 / November 4th, 2013 ###

- Added experimental CUDA backend (use by passing _backend='cuda' to functions wrapped by @jit)

### 0.18 / October 30th, 2013 ###

- Added OpenMP backend (runs most map-like computations across multiple threads)
- Stack-allocate representations for all structured types in C
- Disabled Flattening -- tricky transform needs careful audit
- Debugged and enabled CopyElimination
- Fixed negative step in slices 
- Added RLock around AST translation to play nice with Python threads (thanks Russell Power)
- Fixed link argument order for building on cygwin in Windows (thanks Yves-Rémi Van Eycke)

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

