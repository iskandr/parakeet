
class AdverbSemantics(object):
  """
  Describe the behavior of adverbs in terms of
  lower-level value and iteration constructs.

  To get something other than an unfathomably slow
  interpreter, override all the methods of BaseSemantics
  and make them work for some other domain (such as types,
  shapes, or compiled expressions)
  """
  def invoke_delayed(self, fn, args, idx):
    curr_args = [x(idx) for x in args]
    return self.invoke(fn, curr_args)

  def build_slice_indices(self, rank, axis, idx):
    indices = []
    for i in xrange(rank):
      if i == axis:
        indices.append(idx)
      else:
        s = self.slice_value(self.none, self.none, self.int(1))
        indices.append(s)
    return self.tuple(indices)

  def slice_along_axis(self, arr, axis, idx):
    r = self.rank(arr)
    index_tuple = self.build_slice_indices(r, axis, idx)
    return self.index(arr, index_tuple)

  def delayed_elt(self, x, axis):
    return lambda idx: self.slice_along_axis(x, axis, idx)

  def sizes_along_axis(self, xs, axis):
    axis_sizes = [self.size_along_axis(x, axis)
                  for x in xs
                  if self.rank(x) >= axis]
    assert len(axis_sizes) > 0
    # all arrays should agree in their dimensions along the
    # axis we're iterating over
    self.check_equal_sizes(axis_sizes)
    return axis_sizes 

  def map_prelude(self, map_fn, xs, axis):
    if not isinstance(xs, (list, tuple)):
      xs = [xs]
    axis_sizes = self.sizes_along_axis(xs, axis)
    elts = [self.delayed_elt(x, axis) for x in xs]
    def delayed_map_result(idx):
      return self.invoke_delayed(map_fn, elts, idx)
    return axis_sizes[0], delayed_map_result

  def acc_prelude(self, init, combine, delayed_map_result):
    if init is None:
      init = delayed_map_result(self.int(0))
    else:
      # combine the provided initializer with
      # transformed first value of the data
      # in case we need to coerce up
      init = self.invoke(combine, [init, delayed_map_result(self.int(0))])
    return init, self.int(1)
  
  def create_result(self, first_elt, outer_shape):
    if not self.is_tuple(outer_shape):
      outer_shape = self.tuple([outer_shape])
       
    inner_shape = self.shape(first_elt)
    result_shape = self.concat_tuples(outer_shape, inner_shape)    
    result = self.alloc_array(self.elt_type(first_elt), result_shape)

    return result   
  
  def eval_map(self, f,  values, axis):
    niters, delayed_map_result = self.map_prelude(f, values, axis)
    zero = self.int(0)
    first_output = delayed_map_result(zero)
    result = self.create_result(first_output, niters)
    self.setidx(result, self.int(0), first_output)
        
    def loop_body(idx):
      output_indices = self.build_slice_indices(self.rank(result), axis, idx)
      self.setidx(result, output_indices, delayed_map_result(idx))
    self.loop(self.int(1), niters, loop_body)
    return result 

  def eval_reduce(self, map_fn, combine, init, values, axis):
    niters, delayed_map_result = self.map_prelude(map_fn, values, axis)
    init, start_idx = self.acc_prelude(init, combine, delayed_map_result)
    def loop_body(acc, idx):
      new_acc_value = self.invoke(combine, [acc.get(), delayed_map_result(idx)])
      acc.update(new_acc_value)   
    return self.accumulate_loop(start_idx, niters, loop_body, init)
    
  def eval_scan(self, map_fn, combine, emit, init, values, axis):
    niters, delayed_map_result = self.map_prelude(map_fn, values, axis)
    init, start_idx = self.acc_prelude(init, combine, delayed_map_result)
    first_output = self.invoke(emit, [init])
    result = self.create_result(first_output, niters)
    self.setidx(result, self.int(0), first_output)
    
    def loop_body(acc, idx):
      output_indices = self.build_slice_indices(self.rank(result), axis, idx)
      new_acc_value = self.invoke(combine, [acc.get(), delayed_map_result(idx)])
      acc.update(new_acc_value)      
      output_value = self.invoke(emit, [new_acc_value])
      self.setidx(result, output_indices, output_value)
    self.accumulate_loop(start_idx, niters, loop_body, init)
    return result 
  
  def eval_allpairs(self, fn, x, y, axis):
    nx = self.size_along_axis(x, axis)
    ny = self.size_along_axis(y, axis)
    outer_shape = self.tuple( [nx, ny] )
    zero = self.int(0)
    first_x = self.slice_along_axis(x, axis, zero)
    first_y = self.slice_along_axis(y, axis, zero)

    first_output = self.invoke(fn, [first_x, first_y])
    result = self.create_result(first_output, outer_shape)
    def outer_loop_body(i):
      xi = self.slice_along_axis(x, axis, i)
      def inner_loop_body(j):
        yj = self.slice_along_axis(y, axis, j)
        out_idx = self.tuple([i,j])
        self.setidx(result, out_idx, self.invoke(fn, [xi, yj]))
      self.loop(zero, ny, inner_loop_body)
    self.loop(zero, nx, outer_loop_body)
    return result 
    
    