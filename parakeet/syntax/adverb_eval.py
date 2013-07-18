class AdverbEvalHelpers(object):
  """
  Describe the behavior of adverbs in terms of lower-level value and iteration
  constructs.

  To get something other than an unfathomably slow interpreter, override all the
  methods of BaseSemantics and make them work for some other domain (such as
  types, shapes, or compiled expressions)
  """
  
  


  def eval_reduce(self, map_fn, combine, init, values, axis):
    zero = self.int(0)
    one = self.int(1)
    
    if axis is  None or self.is_none(axis):
      assert len(values) == 1
      values = [self.ravel(values[0])]
      axis = 0
    
    niters, delayed_elts = self.map_prelude(map_fn, values, axis)
    first_acc_value = self.call(map_fn, [elt(zero) for elt in delayed_elts])
    if init is None or self.is_none(init):
      init = first_acc_value
    else:
      init = self.call(combine, [init, first_acc_value])
    def loop_body(acc, idx):
      elt = self.call(map_fn, [elt(idx) for elt in delayed_elts])
      new_acc_value = self.call(combine, [acc.get(), elt])
      acc.update(new_acc_value)
    return self.accumulate_loop(one, niters, loop_body, init)

  def eval_scan(self, map_fn, combine, emit, init, values, axis):

    niters, delayed_elts = self.map_prelude(map_fn, values, axis)
    def delayed_map_result(idx):
      return self.call(map_fn, self.force_list(delayed_elts, idx))
    init = self.acc_prelude(init, combine, delayed_map_result)
    output = self.create_output_array(emit, [init], niters)
    self.setidx(output, self.int(0), self.call(emit, [init]))
    def loop_body(acc, idx):
      output_indices = self.build_slice_indices(self.rank(output), 0, idx)
      new_acc_value = self.call(combine, [acc.get(), delayed_map_result(idx)])
      acc.update(new_acc_value)
      output_value = self.call(emit, [new_acc_value])
      self.setidx(output, output_indices, output_value)
    self.accumulate_loop(self.int(1), niters, loop_body, init)
    return output

  def eval_allpairs(self, fn, x, y, axis):
    if axis is None: 
      x = self.ravel(x)
      y = self.ravel(y)
      axis = 0
    nx = self.size_along_axis(x, axis)
    ny = self.size_along_axis(y, axis)
    outer_shape = self.tuple( [nx, ny] )
    zero = self.int(0)
    first_x = self.slice_along_axis(x, axis, zero)
    first_y = self.slice_along_axis(y, axis, zero)
    output =  self.create_output_array(fn, [first_x, first_y], outer_shape)
    def outer_loop_body(i):
      xi = self.slice_along_axis(x, axis, i)
      def inner_loop_body(j):
        yj = self.slice_along_axis(y, axis, j)
        out_idx = self.tuple([i,j])
        self.setidx(output, out_idx, self.call(fn, [xi, yj]))
      self.loop(zero, ny, inner_loop_body)
    self.loop(zero, nx, outer_loop_body)
    return output
  
  def eval_index_map(self, fn, shape, output = None): 
    dims = self.tuple_elts(shape)
    if len(dims) == 1:
      shape = dims[0]
      
    if output is None:
      output = self.create_output_array(fn, [shape], shape)

    n_loops = len(dims)
    def build_loops(index_vars = ()):
      n_indices = len(index_vars)
      if n_indices == n_loops:
        if n_indices > 1:
          idx_tuple = self.tuple(index_vars)
        else:
          idx_tuple = index_vars[0]
        elt_result =  self.call(fn, (idx_tuple,))
        self.setidx(output, index_vars, elt_result)
      else:
        def loop_body(idx):
          build_loops(index_vars + (idx,))
        self.loop(self.int(0), dims[n_indices], loop_body)
    build_loops()
    return output 
    

    
    