import numpy as np

class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        """Takes in an input "x" and outputs the layer's result"""
        pass

    def backprop(self, x):
        """Backpropagation will return gradients for each of the weights"""
        pass

    def summary(self):
        """Prints info about the layer"""
        pass

    def __call__(self, x):
        return self.forward(x)
