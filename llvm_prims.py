

import prims
import llvm.core as llc    

signed_int_comparisons = {
  prims.equal : llc.ICMP_EQ, 
  prims.not_equal : llc.ICMP_NE, 
  prims.greater : llc.ICMP_SGT, 
  prims.greater_equal : llc.ICMP_SGE, 
  prims.less : llc.ICMP_SLT, 
  prims.less_equal : llc.ICMP_SLE
}

unsigned_int_comparisons = {
  prims.equal : llc.ICMP_EQ, 
  prims.not_equal : llc.ICMP_NE, 
  prims.greater : llc.ICMP_UGT, 
  prims.greater_equal : llc.ICMP_UGE, 
  prims.less : llc.ICMP_ULT, 
  prims.less_equal : llc.ICMP_ULE
}  

float_comparisons = { 
  prims.equal : llc.FCMP_OEQ, 
  prims.not_equal : llc.FCMP_ONE, 
  prims.greater : llc.FCMP_OGT, 
  prims.greater_equal : llc.FCMP_OGE, 
  prims.less : llc.FCMP_OLT, 
  prims.less_equal : llc.FCMP_OLE                     
}


signed_binops = { 
  prims.add : 'add',  
  prims.subtract : 'sub', 
  prims.multiply : 'mul', 
  prims.divide : 'sdiv', 
  prims.mod : 'srem'
}

unsigned_binops = { 
  prims.add : 'add',  
  prims.subtract : 'sub', 
  prims.multiply : 'mul', 
  prims.divide : 'udiv', 
  prims.mod : 'urem', 
}

float_binops = { 
  prims.add : 'fadd',  
  prims.subtract : 'fsub', 
  prims.multiply : 'fmul', 
  prims.divide : 'fdiv', 
  prims.mod : 'frem', 
}

# Note: there is no division instruction between booleans
# so b1 / b2 should be translated to int(b1) / int(b2) 
bool_binops = { 
  prims.add : 'or_',
  prims.multiply : 'and_', 
  prims.subtract : 'xor'
}
