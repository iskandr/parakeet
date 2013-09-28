import parakeet as par
from parakeet import testing_helpers, jit 
import numpy as np


class Network(object):
  def __init__(self):
    self.layers = []
    
    self.start_epoch()
    
  def start_epoch(self):
    self.sse = 0
    self.n_updates = 0
     
  def mse(self):
    return self.sse / self.n_updates 
   
  def __iadd__(self, layer):
    self.layers.append(layer)
    return self
    
  def predict(self, x):
    for layer in self.layers:
      x = layer.fprop(x)
    return x
  
  def bprop(self, err):
    for layer in reversed(self.layers[1:]):
      err = layer.bprop(err)
  
  def update(self, x, y):
    y_pred = self.predict(x)
    err = y_pred - y
    self.sse += err**2 
    self.n_updates += 1
    self.bprop(err)


@jit
def sigmoid(x):
  return 1.0 / (1.0 + par.exp(-x))

@jit
def d_sigmoid(x):
  s = sigmoid(x)
  return s * (1-s) 

@jit
def dot(x,y):
  return sum(x*y)

@jit
def fprop_linear(x, W, b):
  def dot_add(w_row, b_elt):
    return sum(w_row * x) + b_elt 
  return par.each(dot_add, W, b)
 
@jit 
def fprop_logistic(x, W, b):
  return sigmoid(fprop_linear(x,W,b))

@jit
def bprop_logistic(x, W, b, err):
  return W*0.001

class LogisticLayer(object):
  def __init__(self, n_in, n_out, learning_rate = 0.001):
    self.W = np.random.randn( n_out, n_in)
    self.bias = np.random.randn(n_out)
    self.last_input = None
    self.last_output = None 
    self.learning_rate = learning_rate
    
  def __str__(self):
    return "LogisticLayer(in = %d, out = %d)" % (self.W.shape[0], self.W.shape[1])
  
  def fprop(self, x):
    self.last_input = x
    self.last_output = fprop_logistic(x, self.W, self.bias)

    return self.last_output
  
  def bprop(self, err):
    weighted_err = 0  
    return weighted_err 
    

@jit 
def tanh_fprop(x, w, b):
  return par.tanh(sum(x*w) + b)

def tanh_bprop(x, w, b, err):
  return 0

class TanhLayer(object):
  def __init__(self, n, w = None, bias = None):
    if w is None:
      w = np.random.randn(n)
    if bias is None:
      bias = np.random.randn()
    self.w = w
    self.bias = bias
    self.last_x = None
    
  def fprop(self, data):
    self.last_data = data
    self.last_output = tanh_fprop(data, self.w, self.bias)
    return self.last_output
     
  def bprop(self, err):
    self.last_delta = tanh_bprop(self.last_data, self.w, self.bias, err)
    

n_in = 1000 
n_hidden = 50
n_out = 1 

mlp = Network()
mlp += LogisticLayer(n_in, n_hidden)
mlp += LogisticLayer(n_hidden, n_out)

def test_mlp():
  x = np.random.randn(1000)
  y = mlp.predict(x)
  assert len(y) == n_out
  assert np.all(y >= 0)
  assert np.all(y <= 1)
  


if __name__ == '__main__':
  testing_helpers.run_local_tests()
