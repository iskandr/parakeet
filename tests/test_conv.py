import numpy as np
import parakeet
import testing_helpers

def mk_weights(radius, sigma=1):
    window_size = 2*radius+1
    weights = np.zeros( (window_size, window_size) )
    for i in xrange(-radius, radius):
        for j in xrange(-radius, radius):
            weights[i+radius,j+radius] = np.exp(-(i**2 + j**2)/sigma**2)
    return weights

def blur(x, weights):
    flat_weights = weights.ravel()
    def f(window):
        return parakeet.dot(window, flat_weights)
    return parakeet.conv(f, x, shape=weights.shape)

def test_conv():
    m,n = 200,300
    x = np.random.rand(m,n)
    radius = 2
    weights = mk_weights(radius)
    y = blur(x,weights)
    print y.shape
    assert y.shape == (m-radius*2, n-radius*2)

if __name__ == '__main__':
    testing_helpers.run_local_tests()
