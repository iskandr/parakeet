class AdverbSemantics(object):
  """
  Describe the behavior of adverbs in terms of lower-level value and iteration
  constructs.

  To get something other than an unfathomably slow interpreter, override all the
  methods of BaseSemantics and make them work for some other domain (such as
  types, shapes, or compiled expressions)
  """

  def build_slice_indices(self, rank, axis, idx):
    if rank == 1:
      assert axis == 0
      return idx

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
    if r > axis:
      index_tuple = self.build_slice_indices(r, axis, idx)
      return self.index(arr, index_tuple)
    else:
      return arr

  def delayed_elt(self, x, axis):
    return lambda idx: self.slice_along_axis(x, axis, idx)

  def delay_list(self, xs, axis):
    return [self.delayed_elt(x, axis) for x in xs]

  def force_list(self, delayed_elts, idx):
    return [e(idx) for e in delayed_elts]

  def sizes_along_axis(self, xs, axis):
    axis_sizes = [self.size_along_axis(x, axis)
                  for x in xs
                  if self.rank(x) > axis]

    assert len(axis_sizes) > 0
    # all arrays should agree in their dimensions along the
    # axis we're iterating over
    self.check_equal_sizes(axis_sizes)
    return axis_sizes

  def map_prelude(self, map_fn, xs, axis):
    axis_sizes = self.sizes_along_axis(xs, axis)
    return axis_sizes[0], self.delay_list(xs, axis)

  def acc_prelude(self, init, combine, delayed_map_result):
    zero = self.int(0)
    if init is None or self.is_none(init):
      return delayed_map_result(zero)
    else:
      # combine the provided initializer with
      # transformed first value of the data
      # in case we need to coerce up
      return self.invoke(combine, [init, delayed_map_result(zero)])

  def create_result(self, elt_type, inner_shape, outer_shape):
    if not self.is_tuple(outer_shape):
      outer_shape = self.tuple([outer_shape])
    result_shape = self.concat_tuples(outer_shape, inner_shape)
    result = self.alloc_array(elt_type, result_shape)
    return result

  def create_output_array(self, fn, inputs, extra_dims):
    if hasattr(self, "_create_output_array"):
      try:
        return self._create_output_array(fn, inputs, extra_dims)
      except:
        print "Failed to allocate output array using shape inference"
        pass
    inner_result = self.invoke(fn, inputs)
    inner_shape = self.shape(inner_result)
    elt_t = self.elt_type(inner_result)
    return self.create_result(elt_t, inner_shape, extra_dims)

  def output_slice(self, output, axis, idx):
    r = self.rank(output)
    if r > 1:
      output_indices = self.build_slice_indices(r, axis, idx)
    elif r == 1:
      output_idx = self.slice_value(idx, self.none, self.int(1))
      output_indices = self.tuple([output_idx])
    else:
      output_idx = self.slice_value(self.none, self.none, self.none)
      output_indices = self.tuple([output_idx])
    return self.index(output, output_indices)

  def eval_map(self, f, values, axis, output = None):
    niters, delayed_elts = self.map_prelude(f, values, axis)
    zero = self.int(0)
    first_elts = self.force_list(delayed_elts, zero)
    if output is None:
      output = self.create_output_array(f, first_elts, niters)
    def loop_body(idx):
      output_indices = self.build_slice_indices(self.rank(output), 0, idx)
      elt_result = self.invoke(f, [elt(idx) for elt in delayed_elts])
      self.setidx(output, output_indices, elt_result)
    self.loop(zero, niters, loop_body)
    return output

  def eval_reduce(self, map_fn, combine, init, values, axis):
    niters, delayed_elts = self.map_prelude(map_fn, values, axis)
    zero = self.int(0)
    one = self.int(1)
    first_acc_value = self.invoke(map_fn, [elt(zero) for elt in delayed_elts])
    if init is None or self.is_none(init):
      init = first_acc_value
    else:
      init = self.invoke(combine, [init, first_acc_value])
    def loop_body(acc, idx):
      elt = self.invoke(map_fn, [elt(idx) for elt in delayed_elts])
      new_acc_value = self.invoke(combine, [acc.get(), elt])
      acc.update(new_acc_value)
    return self.accumulate_loop(one, niters, loop_body, init)

  def eval_scan(self, map_fn, combine, emit, init, values, axis):
    niters, delayed_elts = self.map_prelude(map_fn, values, axis)
    def delayed_map_result(idx):
      return self.invoke(map_fn, self.force_list(delayed_elts, idx))
    init = self.acc_prelude(init, combine, delayed_map_result)
    output = self.create_output_array(emit, [init], niters)
    self.setidx(output, self.int(0), self.invoke(emit, [init]))
    def loop_body(acc, idx):
      output_indices = self.build_slice_indices(self.rank(output), 0, idx)
      new_acc_value = self.invoke(combine, [acc.get(), delayed_map_result(idx)])
      acc.update(new_acc_value)
      output_value = self.invoke(emit, [new_acc_value])
      self.setidx(output, output_indices, output_value)
    self.accumulate_loop(self.int(1), niters, loop_body, init)
    return output

  def eval_allpairs(self, fn, x, y, axis):
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
        self.setidx(output, out_idx, self.invoke(fn, [xi, yj]))
      self.loop(zero, ny, inner_loop_body)
    self.loop(zero, nx, outer_loop_body)
    return output
