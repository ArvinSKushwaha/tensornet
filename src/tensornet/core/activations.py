import cupy as np
from .layers import Layer
import numba as nb


def relu(x):
    return np.maximum(x, 0)


def d_relu(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x

def expit(x):
    return 1/(np.exp(-x)+1)

@nb.jit(parallel=True, forceobj=True)
def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    assert x.ndim == 2
    classes = x.shape[1]
    x = x - np.mean(x, keepdims=True)
    x = x/np.std(x, keepdims=True)
    x = np.exp(x)
    x = x / np.sum(x, -1, keepdims=True)
    
    return x

@nb.jit(parallel=True, forceobj=True)
def softmax_derivative(x):
    assert x.ndim == 2
    classes = x.shape[1]
    x = x - np.mean(x, keepdims=True)
    x = x/np.std(x, keepdims=True)
    out = np.zeros((x.shape[0], classes, classes))
    for i in range(classes):
        for j in range(classes):
            out[:, i, j] = x[:, i] * (1 - x[:, i]) if (i == j) else -x[:, i]*x[:, j]
    return out

activations = {
    "linear": [
        lambda x: x,
        lambda x: 1
    ],
    "sigmoid": [
        expit,
        lambda x: expit(x)*(1-expit(x))
    ],
    "tanh": [
        np.tanh,
        lambda x: 1 - np.tanh(x)**2
    ],
    "relu": [
        relu,
        d_relu
    ],
    "softmax": [
        stablesoftmax,
        softmax_derivative   
    ]
}


class Activation(Layer):
    def __init__(self, fname="linear", function=None, d_function=None):
        super().__init__()
        if(function and d_function):
            self.func = function
            self.d_func = d_function
        else:
            self.func, self.d_func = activations[fname]

    def forward(self, x):
        return self.func(x)

    def diff(self, x):
        return self.d_func(x) 

    def summary(self):
        pass

class LeakyReLU(Activation):
    def __init__(self, alpha=0):
        self.alpha = alpha
        def leaky(x):
            x[x < 0] *= alpha
            return x
        
        def d_leaky(x):
            x[x < 0] = -alpha
            x[x > 0] = 1
            return x
        super().__init__(function=leaky, d_function=d_leaky)