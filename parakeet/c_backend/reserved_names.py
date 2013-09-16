keywords = set([
  "auto", 
  "break",
  "case",
  "char",
  "const",
  "continue", 
  "default",
  "do",
  "double",
  "else",
  "enum",
  "extern", 
  "float",
  "for",
  "goto",
  "if", 
  "int",
  "long",
  "register",
  "return"
  "short"
  "signed"
  "sizeof"
  "static"
  "struct",
  "switch"
  "typedef", 
  "union", 
  "unsigned",
  "void", 
  "volatile",
  "while"])

macro_names = set(["assert"])
util_names = set(["printf", "malloc", "free", ])
math_names = set([
  "acos",
  "acosf", 
  "acosl",
  
  "asin",
  "asinf",
  "asinl",
  
  "atan",
  "atanf",
  "atanl",
  
  "atan2",
  "atan2f",
  "atan2l",
  
  "atanh", 
  "atanhf",
  "atanhl",
  
  "atof",
  "ceil"
  
  "cos",
  "cosf",
  "cosl",
  
  "cosh",
  "coshf",
  "coshl",
  
  "exp",
  "expf",
  "expl",
  
  "fabs",
  "floor",
  "frexp",
  "ldexp",
  "log",
  "log10", 
  "div", 
  
  "sin", 
  "sinf",
  "sinl",
  
  "sinh",
  "sinhf",
  "sinhl",
  
  "tan", 
  "tanf",
  "tanl",
  
  "tanh",
  "tanhf",
  "tanh2"             
])
all_reserved_names = keywords.union(macro_names).union(util_names).union(math_names)
 
def is_reserved(name):
  return name in all_reserved_names or name.startswith("Py") 