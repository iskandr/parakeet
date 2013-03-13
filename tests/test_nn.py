# TODO: change this from Theano into Parakeet

class LogisticRegression(object):
  def __init__(self, input, n_in, n_out):
      self.W = theano.shared(value=N.zeros((n_in, n_out), dtype=theano.config.floatX), name='W')
      self.b = theano.shared(value=N.zeros((n_out,), dtype=theano.config.floatX), name='b')
      self.p_y_given_x = T.nnet.softmax(theano.dot(input, self.W) + self.b)
      self.y_pred = T.argmax(self.p_y_given_x, axis=1)
      self.params = [self.W, self.b]

  def negative_log_likelihood(self, y):
    return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
  
  def errors(self, y):
    return T.mean(T.neq(self.y_pred, y))

class HiddenLayer(object):
  def __init__(self, rng, input, n_in, n_out, activation=T.tanh, W=None, b=None):
    self.input = input
    if W is None:
      W_values = rand_array(rng, 6.0, (n_in, n_out), theano.config.floatX)
      W = theano.shared(value=W_values, name='W', borrow=True)
    
    if activation == T.nnet.sigmoid: W_values *= 4
    if b is None:
      b_values = N.zeros((n_out,), dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)

    self.W = W
    self.b = b
    self.output = activation(theano.dot(input, self.W) + self.b)
    self.params = [self.W, self.b]

def l1(w): return abs(w).sum()
def l2(w): return (w ** 2).sum()

class MLP(object):
  def __init__(self, rng, input, n_in, n_hidden, n_out):
    self.hidden = HiddenLayer(rng, input, n_in, n_hidden, T.tanh)
    self.logreg = LogisticRegression(input=self.hidden.output, n_in=n_hidden, n_out=n_out)
    layers = [self.hidden, self.logreg]
    self.L1 = sum([l1(layer.W) for layer in layers])
    self.L2 = sum([l2(layer.W) for layer in layers])

    self.negative_log_likelihood = self.logreg.negative_log_likelihood
    self.errors = self.logreg.errors
    self.params = self.hidden.params + self.logreg.params

