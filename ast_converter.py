
import ast 

###############################################################################
#  Converter
###############################################################################

class InvalidFunction(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


class ASTConverter(ast.NodeVisitor):
  def __init__(self, global_refs):
    self.seen_functions = set([])
    self.arg_names = arg_names
    self.global_variables = set()
    self.global_refs = global_refs

  def visit_Module(self, node):
    """
     when we parse Python source, we get a single function definition 
     wrapped by a Module node
    """ 
    assert len(node.body) == 1
    body = node.body[0]
    print body 
    ptr = self.visit_function_def(node.body[0])
    print "ptr", ptr
    return ptr 
  
  # TODO: don't ignore function decorators!  
  def visit_FunctionDef(self, node):
    # we don't support variable argument functions
    assert node.args.vararg is None
    # we don't support dictionaries of keyword arguments  
    assert node.args.kwarg is None

    return self.visit_stmt_sequence(node.body)
    
  
  def visit_stmt_sequence(self, stmts, src_info=None):
    stmt_nodes = [self.visit_stmt(stmt) for stmt in stmts]
    print stmt_nodes
    return mk_block(stmt_nodes, src_info)

  def visit_stmt(self, node, src_info = None):
    #print "visit_stmt", node
    src_info = self.build_src_info(node)
    srcAddr = src_addr(src_info)
    if isinstance(node, ast.If):
      test = self.visit_expr(node.test)
      if_true = self.visit_stmt_sequence(node.body, src_info)
      if_false = self.visit_stmt_sequence(node.orelse, src_info)
      return LibPar.mk_if(test, if_true, if_false, srcAddr)
    elif isinstance(node, ast.Assign):
      assert len(node.targets) == 1
      return self.visit_assign(node.targets[0], node.value, src_info)
    elif isinstance(node, ast.Return):
      return self.visit_return(node, src_info)
      
    elif isinstance(node, ast.While):
      # infrequently used final iteration, not supported for now
      assert node.orelse == [] or node.orelse is None
      block = self.visit_stmt_sequence(node.body)
      test = self.visit_expr(node.test)
      return LibPar.mk_whileloop(test, block, srcAddr)
    elif isinstance(node, ast.Expr):
      return self.visit_expr(node.value)
    else:
      raise RuntimeError("Unsupported statement" + str(node))

  def visit_assign(self, lhs, rhs, src_info = None):
    """
    On the left-hand-side of an assignment we allow either a variable
    or a tuple of variables. Anything else should signal an error.
    The right-hand-side is expected to be an expression (and not a statement).
    """  
    def mk_lhs_var(node):
      return build_var(node.id, self.build_src_info(node))
    if isinstance(lhs, ast.Name):
      vars = [mk_lhs_var(lhs)]
    elif isinstance(lhs, ast.Tuple):
      assert all([isinstance(elt, ast.Name) for elt in lhs.elts])
      vars = [mk_lhs_var(elt) for elt in lhs.elts]
    else:
      raise RuntimeError("Unsupported LHS")
    rhs = self.visit_expr(rhs)
    return LibPar.mk_assign(list_to_ctypes_array(vars), len(vars), rhs, src_info)
       
  def visit_return(self, node, src_info = None):
    if node.value is None:
      values = []
    elif isinstance(node.value, ast.Tuple): 
      values = [self.visit_expr(v) for v in node.value.elts]
    else:
      values = [self.visit_expr(node.value)]
    return mk_return(values, src_info)


  def get_function_ref(self, node):
    if isinstance(node, ast.Name):
      funName = node.id
      if funName in self.global_refs:
        funRef = self.global_refs[funName]
      elif funName in __builtins__:
        funRef = __builtins__[funName]
      else:
        raise InvalidFunction(funName)
    elif isinstance(node, ast.Attribute):
      #For function calls like mod1.mod2.mod4.fun()
      moduleList = []
      nextNode = node
      funName = node.attr
      #Get a list of the chain of modules
      while isinstance(nextNode, ast.Attribute):
        moduleList = [nextNode.attr] + moduleList
        nextNode = nextNode.value
      currModule = self.global_refs[nextNode.id]

      for m in moduleList:
        try:
          currModule = currModule.__dict__[m]
        except AttributeError:
          if currModule in AutoTranslate:
            currModule = currModule.__getattribute__(m)
          elif m in NumpyArrayMethods and m == moduleList[-1]:
            currModule = currModule.__getattribute__(m)
          elif m in NumpyArrayMethods:
            raise InvalidFunction(moduleList.join('.'))
          else:
            raise InvalidFunction("Invalid object %s" % currModule)
      funRef = currModule
    else:
      raise InvalidFunction("Call.func shouldn't be " + name)
    
    if not hasattr(funRef, '__call__'):
      return None
    
    if funRef in AutoTranslate:
      funRef = AutoTranslate[funRef]
    
    if not hasattr(funRef,'__self__') or not funRef.__self__:
      self.seen_functions.add(funRef)
    return funRef

  def register_if_function(self, node):
    #Alex: Right now this will crash if a local variable method is an argument,
    #because the local method won't be registered or translated
    try:
      parts = flatten_var_attrs(node)
      if parts[0] in self.arg_names and len(parts) > 1:
        raise ParakeetUnsupported(\
          "[Parakeet] %s is not a valid function argument" % node_name)
      return self.get_function_ref(node)
    except RuntimeError:
      return None
  
  def python_fn_to_parakeet(self, python_fn, src_info = None):
    if python_fn in AutoTranslate:
      python_fn = AutoTranslate[python_fn]
    if python_fn in ParakeetOperators:
      return ast_prim(ParakeetOperators[python_fn])
    else:
      return build_var(global_fn_name(python_fn), src_info)
  

  def get_slice_children(self, node): 
    if isinstance(node, ast.Index):
      if isinstance(node.value, ast.Tuple):
        return [self.visit_expr(v) for v in node.value.elts]
      else:
        return [self.visit_expr(node.value)]
    elif isinstance(node, ast.Slice):
      def helper(child):
        if child is None: 
          return mk_none()
        else:
          return self.visit_expr(child)
      return map(helper, [node.lower, node.upper, node.step])
    else:
      raise ParakeetUnsupported("Slice of type " + str(type(node)) + " not supported") 
      
  def visit_expr(self, node):
    src_info = self.build_src_info(node)

    if isinstance(node, ast.Name):
      return build_var(node.id, src_info)
    elif isinstance(node, ast.BinOp):
      left = self.visit_expr(node.left)
      right = self.visit_expr(node.right)
      op_name = name_of_ast_node(node.op)
      return build_prim_call(op_name, [left, right], src_info)
    elif isinstance(node, ast.BoolOp):
      args = [self.visit_expr(v) for  v in node.values]
      return build_prim_call(name_of_ast_node(node.op), args, src_info)
    elif isinstance(node, ast.UnaryOp):
      arg = self.visit_expr(node.operand)
      return build_prim_call(name_of_ast_node(node.op), [arg], src_info)
    elif isinstance(node, ast.Compare):
      assert len(node.ops) == 1
      
      #Not sure when there are multiple ops or multiple comparators?
      assert len(node.comparators) == 1
      op_name = name_of_ast_node(node.ops[0])
      left = self.visit_expr(node.left)
      right = self.visit_expr(node.comparators[0])

      return build_prim_call(op_name, [left, right], src_info)
    elif isinstance(node, ast.Subscript):
      op_name = name_of_ast_node(node.slice) 
      # only Slice and Index supported 
      assert op_name != "Ellipsis" and op_name != "ExtSlice" 
      arr = self.visit_expr(node.value)
      indexArgs = self.get_slice_children(node.slice)
      args = [arr] + indexArgs
      return build_prim_call(op_name, args, src_info)
    elif isinstance(node, ast.Index):
      raise RuntimeError("[Parakeet] Unexpected index node in AST")
    elif isinstance(node, ast.Num):
      return build_num(node.n, src_info)
    elif isinstance(node, ast.Call):
      # neither var-args or dictionaries of keyword args are
      # supported by Parakeet since they make program 
      # analysis way harder
      assert node.starargs is None
      assert node.kwargs is None 
      fn = node.func
      args = node.args
      kwds = node.keywords 
      return self.visit_call(fn, args, kwds, src_info)
    elif isinstance(node, ast.Tuple):
       elts = [self.visit_expr(elt) for elt in node.elts]
       return mk_tuple(elts, src_info)
    else:
      raise RuntimeError("[Parakeet] AST node %s not supported " % type(node).__name__)
      return None

  def visit_call(self, fn, args, kwds, src_info = None):
    fn_name_parts = flatten_var_attrs(fn)
    assert len(fn_name_parts) > 0 
    base_name = fn_name_parts[0]
    # TODO: should be a check whether it's a local, not an arg 
    if base_name in self.arg_names:
      
      #Is the function an argument?
      #If so, we must be calling a numpy method on an array 
      assert len(fn_name_parts) == 1
      method_name = fn_name_parts[1]
      parakeet_obj =  build_var(base_name, src_info)
      self.visit_method_call(method_name, parakeet_obj, src_info)
    else:
      python_fn = self.get_function_ref(fn)
      if python_fn in Adverbs:
        assert len(args) > 1
        return self.visit_adverb(python_fn, args[0], args[1:],  kwds)
      else:
        return self.visit_simple_call(python_fn, args, kwds, src_info)
  
  def visit_method_call(self, method_name, parakeet_obj, src_info = None):
    if method_name in NumpyArrayMethods:
      prim_name = NumpyArrayMethods[method_name]
      return build_prim_call(prim_name, [parakeet_obj],  src_info)
    else:
      raise ParakeetUnsupported("Can't call method %s" % method_name)
    
  def visit_adverb(self, adverb, fn, args, kwds):
    python_fn = self.get_function_ref(fn)
    parakeet_fn = self.python_fn_to_parakeet(python_fn)
    parakeet_args = [parakeet_fn]
    for arg in args:
      parakeet_args.append(self.visit_expr(arg))

    parakeet_keywords = {}
    for pair in kwds:
      parakeet_keywords[pair.arg] = self.visit_expr(pair.value)
    assert adverb in ParakeetOperators 
    parakeet_adverb = ast_prim(ParakeetOperators[adverb])
    return build_call(parakeet_adverb, parakeet_args, parakeet_keywords) 
    
  def visit_simple_call(self, python_fn, args, kwds, src_info=None):
    """
    A simple call is neither a ufunc method nor an adverb
    """
    # have to handle arrays differently since they can contain
    # lists that otherwise are illegal in Parakeet programs
    if python_fn == np.array:
      # keywords on arrays not yet supported
      assert len(kwds) == 0
      assert len(args) == 1
      return self.visit_array_elts(args[0])
    else:
      parakeet_fn = self.python_fn_to_parakeet(python_fn)
      parakeet_args = [self.visit_expr(arg) for arg in args]
      parakeet_keywords = {}
      for pair in kwds:
        parakeet_keywords[pair.arg] = self.visit_expr(pair.value)
      return build_call(parakeet_fn, parakeet_args, parakeet_keywords) 
      
    
  def visit_array_elts(self, node):
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
      elts = [self.visit_expr(elt) for elt in node.elts]
      return mk_array(elts, self.build_src_info(node)) 
    else:
      raise ParakeetUnsupported('Array must have literal arguments') 

  def build_src_info(self, node):
    #Temporary to fix seg faults:
    return None 
    try:
      file_name = c_char_p(self.file_name)
      line = c_int(self.line_offset + node.lineno)
      col = c_int(node.col_offset)
      return _source_info_t(_c_source_info_t(file_name, line, col))
    except AttributeError:
      return _source_info_t(None)

 
