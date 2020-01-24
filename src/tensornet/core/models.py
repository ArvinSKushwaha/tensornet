import cupy as np
from . import Activation, Dense, Layer

class Model:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def foward_(self, x):
        outputs = []
        for layer in self.layers:
            outputs.append(x)
            x = layer(x)
        return outputs, x
    
    def backprop(self, x_train, y_train, loss, d_loss, lr=1e-3):
        outputs, out = self.foward_(x_train)
        derivative = d_loss(out, y_train)
        for index in range(len(self.layers)-1, -1, -1):
            if(isinstance(self.layers[index], (Activation,))):
                derivative = derivative * self.layers[index].diff(outputs[index]) * lr
            if(isinstance(self.layers[index], (Dense,))):
                # print(derivative.shape, inputs.shape, layer.bias.shape)
                dEdw = (outputs[index].T @ derivative) * lr
                dEdb = (np.ones((outputs[index].shape[0],)) @ derivative) * lr
                # print(dEdb.shape)
                self.layers[index].weights -= dEdw
                self.layers[index].bias -= dEdb
                derivative = derivative @ self.layers[index].weights.T * lr
                # print("Change:", np.average(dEdw))
    
    def __call__(self, x):
        return self.forward(x)