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
util_names = set(["printf", "malloc", "free"])
math_names = set([
  "acos",
  "asin",
  "atan",
  "atan2",
  "atof",
  "ceil"
  "cos",
  "cosh",
  "exp",
  "fabs",
  "floor",
  "frexp",
  "ldexp",
  "log",
  "log10"                
])
all_reserved_names = keywords.union(macro_names).union(util_names).union(math_names)
 
def is_reserved(name):
  return name in all_reserved_names or name.startswith("Py") 