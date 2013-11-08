from .. ndtypes import ArrayT, PtrT 
from .. syntax import (Var, Alloc, ArrayView, Array, Struct, AllocArray, 
                       Map, IndexMap, OuterMap, Scan, IndexScan, 
                       ConstArray, ConstArrayLike
                       )
from syntax_visitor import SyntaxVisitor
 
# from syntax import Adverb   

array_alloc_classes = (AllocArray, Array, 
                       Map, IndexMap, OuterMap, 
                       Scan, IndexScan, 
                       ConstArray, ConstArrayLike)

class FindLocalArrays(SyntaxVisitor):
  def __init__(self):
    # hash table mapping from variable names to
    # statements allocating space
    self.local_allocs = {}

    # hash table mapping from variable names to
    # places where we create array views containing
    # locally allocated data pointers
    self.local_arrays = {}
    
    # associate each local array 
    # with its data allocation 
    self.array_to_alloc = {}
  
  def visit_merge(self, merge):
    """
    If both sides of a flow merge are a locally created 
    array originating on the same statement, then 
    also mark the new phi-bound variable as also being 
    that local array. 
    
    Similarly, if both sides are the same local pointer 
    allocation, then propagate that information to the 
    new variable being created by a phi-node
    """
    for (new_name, (left, right)) in merge.iteritems():
      if left.__class__ is Var and right.__class__ is Var:
        left_name = left.name 
        right_name = right.name 
        if left.type.__class__ is ArrayT and \
           left_name in self.local_arrays and \
           right_name in self.local_arrays:
          left_stmt = self.local_arrays[left_name]
          right_stmt = self.local_arrays[right_name]
          if left_stmt == right_stmt:
            self.local_arrays[new_name] = left_stmt 
            if left.name in self.array_to_alloc_name:
              self.array_to_alloc[new_name] = \
                  self.array_to_alloc[left.name]
        elif left.type.__class__ is PtrT and \
             left_name in self.local_allocs and \
             right_name in self.local_allocs:
          left_stmt = self.local_allocs[left_name]
          right_stmt = self.local_allocs[right_name]
          if left_stmt == right_stmt:
            self.local_allocs[new_name] = left_stmt
  
  def visit_Assign(self, stmt):
    if stmt.lhs.__class__ is Var:
      lhs_name = stmt.lhs.name 
      rhs_class = stmt.rhs.__class__
      if rhs_class is Alloc:
        self.local_allocs[lhs_name] = stmt
      elif rhs_class is ArrayView and \
          stmt.rhs.data.__class__ is Var:
        data_name = stmt.rhs.data.name 
        if data_name in self.local_allocs:
          self.local_arrays[lhs_name] = stmt
          self.array_to_alloc[lhs_name] = data_name  
      elif rhs_class is Struct and \
          stmt.rhs.type.__class__ is ArrayT and \
          stmt.rhs.args[0].__class__ is Var:
        data_name = stmt.rhs.args[0].name
        if data_name in self.local_allocs:
          self.local_arrays[lhs_name] = stmt
          self.array_to_alloc[lhs_name] = data_name 
      elif rhs_class in array_alloc_classes:
        self.local_arrays[stmt.lhs.name] = stmt
