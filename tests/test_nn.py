import parakeet 
import testing_helpers

import numpy as np 

def init_layer(input_size, n_elts):
  return np.random.randn(n_elts, input_size)

def create_mlp(n_inputs, n_hidden, n_outputs):
  W1 = init_layer(n_inputs, n_hidden)
  W2 = init_layer(n_hidden, n_outputs)
  return (W1, W2)

def dot(x,y):
  sum(x*y)

def fprop_elt(x,w):
  return dot(x,w)

def fprop_layer(layer, x):
  return [fprop_elt(x,w) for w in layer]

def fprop(network, x):
  for layer in network:
    x = parakeet.run(fprop_layer, layer,x)
  return x

def fprop_python(network, x): 
  for layer in network:
    x = fprop_layer(layer,x)
  return x

def test_nn():
  network = create_mlp(1000, 50, 1)
  x = np.random.randn(1000)
  assert testing_helpers.eq(fprop(network, x), fprop_python(network, x))
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()
