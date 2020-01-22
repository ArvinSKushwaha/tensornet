import numpy as np
from scipy.special import expit
from .layer import Layer

def relu(x):
    x[x<0] = 0
    return x

def d_relu(x):
    x[x<0] = 0
    x[x>0] = 1
    return x

activations = {
    "linear": [
        lambda x: x,
        lambda x: 1
    ],
    "sigmoid": [
        lambda x: expit(x),
        lambda x: expit(x)*(1-expit(x))
    ],
    "relu": [
        relu,
        d_relu
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