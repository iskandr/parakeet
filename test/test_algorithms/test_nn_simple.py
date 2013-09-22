from parakeet import jit, testing_helpers 
import numpy as np 

def init_layer(input_size, n_elts):
  return np.random.randn(n_elts, input_size)

def create_mlp(n_inputs, n_hidden, n_outputs):
  W1 = init_layer(n_inputs, n_hidden)
  W2 = init_layer(n_hidden, n_outputs)
  return (W1, W2)

def dot(x,y):
  return sum(x*y)

def fprop_elt(x,w):
  return np.tanh(dot(x,w))

def fprop_layer(layer, x):
  return [fprop_elt(x,w) for w in layer]

fprop_layer_parakeet = jit(fprop_layer)

def fprop(network, x):
  for layer in network:
    x = fprop_layer_parakeet(layer, x)
  return x

def fprop_python(network, x): 
  for layer in network:
    x = np.array(fprop_layer(layer,x))
  return x

n = 1000
d = 500
Xpos = np.random.randn(n,d)+1.0
Xneg = np.random.randn(n,d)-1.0
X = np.vstack([Xpos, Xneg])
Y = np.vstack([np.ones(n), np.ones(n)*-1])

network = create_mlp(d,50,1)

def test_fprop():
  parakeet_result = fprop(network, X[0])
  print "Parakeet result", parakeet_result
  python_result = fprop_python(network, X[0])
  print "Python result", python_result
  assert testing_helpers.eq(parakeet_result, python_result)
    
if __name__ == '__main__':
  testing_helpers.run_local_tests()
