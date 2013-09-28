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

import math 

base_math_names = [name for name in dir(math) if not name.startswith("__")]

float32_math_names = [name+'f' for name in base_math_names]
long_math_names = [name+"l" for name in base_math_names]
extra_math_names = [
  "atof", "cbrt", "cbrtf", "cbrtl",
  "div", 
  "exp2", "exp2f", "exp2l", 
  "j0", "j1", "jn",
  "log2", "log2f", "log2l", 
  "logb", "logbf", "logbl", 
  "lrintf", "lrintl", 
  "lround", "lroundf", "lroundl", 
  "remainder", "remainderf", "remainderl", 
  "remquo", "remquof", "remquol", 
  "rint", "rintf", "rintl", 
  "round", "roundf", "roundl",
  "scalb", "scalbln", "scalblnf", "scalblnl", "scalbn", "scalbnf", "scalbnl",  
  "signgam", 
  "tgamma", "tgammaf", "tgammal"  
  "y0", "y1", "yn",              
]
math_names = set(base_math_names+float32_math_names+long_math_names+extra_math_names)
all_reserved_names = keywords.union(macro_names).union(util_names).union(math_names)
 
def is_reserved(name):
  return name in all_reserved_names or name.startswith("Py") 