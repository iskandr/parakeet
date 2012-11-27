
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

  def map_prelude(self, map_fn, xs, axis):
    if not isinstance(xs, (list, tuple)):
      xs = [xs]
    axis_sizes = [self.size_along_axis(x, axis)
                  for x in xs
                  if self.rank(x) >= axis]
    assert len(axis_sizes) > 0

    # all arrays should agree in their dimensions along the
    # axis we're iterating over
    self.check_equal_sizes(axis_sizes)
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
    return self.accumulator(init), self.int(1)
    
  def eval_map(self, f,  values, axis):
    niters, delayed_map_result = self.map_prelude(f, values, axis)
    first_output = delayed_map_result(self.int(0))
    result = self.repeat_array(niters, first_output)
    
    def loop_body(idx):
      output_indices = self.build_slice_indices(self.rank(result), axis, idx)
      self.setidx(result, output_indices, delayed_map_result(idx))
    self.loop(0, niters, loop_body)
    return result 

  def eval_reduce(self, map_fn, combine, init, values, axis):
    niters, delayed_map_result = self.map_prelude(map_fn, values, axis)
    acc, start_idx = self.acc_prelude(init, combine, delayed_map_result)
    def loop_body(idx):
      old_acc_value = self.get_acc(acc)
      new_acc_value = self.invoke(combine, [old_acc_value, delayed_map_result(idx)]) 
      self.set_acc(acc, new_acc_value)
    self.loop(start_idx, niters, loop_body)
    return self.get_acc(acc)
  
  def eval_scan(self, map_fn, combine, emit, init, values, axis):
    niters, delayed_map_result = self.map_prelude(map_fn, values, axis)
    acc, start_idx = self.acc_prelude(init, combine, delayed_map_result)
    emitted_elt_repr = lambda idx: self.invoke(emit, [self.get_acc(acc)])
    first_output = emitted_elt_repr(self.int(0))
    result = self.repeat_array(niters, first_output)
    
    def loop_body(idx):
      output_indices = self.build_slice_indices(self.rank(result), axis, idx)
      self.set_acc(acc,
        self.invoke(combine, [self.get_acc(acc), delayed_map_result(idx)]))
      self.setidx(result, output_indices, emitted_elt_repr(idx))
    self.loop(start_idx, niters, loop_body)
    return result
