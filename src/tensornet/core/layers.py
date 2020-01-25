import cupy as np

class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        """Takes in an input "x" and outputs the layer's result"""
        pass

    def backprop(self, x, grads):
        """Backpropagation will return gradients for each of the weights"""
        pass

    def summary(self):
        """Prints info about the layer"""
        pass

    def __call__(self, x):
        return self.forward(x)

class Dense(Layer):
    def __init__(self, output_dims, input_dims):
        super().__init__()
        self.output_size = output_dims
        self.input_size = input_dims
        self.weights = np.random.randn(input_dims, output_dims) * np.sqrt(2/(input_dims))
        self.bias = np.random.randn(output_dims)
    
    def forward(self, x):
        return (x @ self.weights) + self.bias

    def summary(self):
        pass

    def __call__(self, x):
        return self.forward(x)