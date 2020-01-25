import cupy as np
from . import Activation, Dense, Layer

class Model:
    def __init__(self, layers):
        self.layers = layers
        self.prev_updates = [None] * len(layers)
    
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
    
    def compiled(self, loss, d_loss):
        self.loss = loss
        self.d_loss = d_loss

    def train_batch(self, x_train, y_train, loss=None, d_loss=None, lr=1e-3, cross_entropy=False):
        if(not cross_entropy):
            assert (loss and d_loss) or (self.loss and self.d_loss)
            if(not (loss and d_loss)):
                loss = self.loss
                d_loss = self.d_loss
            outputs, out = self.foward_(x_train)
            derivative = d_loss(out, y_train) * lr
            for index in range(len(self.layers)-1, -1, -1):
                if(isinstance(self.layers[index], (Activation,))):
                    derivative = derivative * self.layers[index].diff(outputs[index])
                if(isinstance(self.layers[index], (Dense,))):
                    # print(derivative.shape, inputs.shape, layer.bias.shape)
                    if(self.prev_updates[index]):
                        dEdw = (outputs[index].T @ derivative) * lr + 0.9 * self.prev_updates[index][0]
                        dEdb = (np.ones((outputs[index].shape[0],)) @ derivative) * lr + 0.9 * self.prev_updates[index][1]
                    else:
                        dEdw = (outputs[index].T @ derivative) * lr
                        dEdb = (np.ones((outputs[index].shape[0],)) @ derivative) * lr
                    self.prev_updates[index] = [dEdw, dEdb]
                    self.layers[index].weights -= dEdw
                    self.layers[index].bias -= dEdb
                    derivative = derivative @ self.layers[index].weights.T * lr
                    # print("Change:", np.average(dEdw))
        else:
            outputs, out = self.foward_(x_train)
            assert out.ndim == 2
            classes = out.shape[1]
            derivative = (out - y_train) * lr
            for index in range(len(self.layers)-2, -1, -1):
                if(isinstance(self.layers[index], (Activation,))):
                    derivative = derivative * self.layers[index].diff(outputs[index])
                if(isinstance(self.layers[index], (Dense,))):
                    # print(derivative.shape, inputs.shape, layer.bias.shape)
                    if(self.prev_updates[index]):
                        dEdw = (outputs[index].T @ derivative) * lr + 0.9 * self.prev_updates[index][0]
                        dEdb = (np.ones((outputs[index].shape[0],)) @ derivative) * lr + 0.9 * self.prev_updates[index][1]
                    else:
                        dEdw = (outputs[index].T @ derivative) * lr
                        dEdb = (np.ones((outputs[index].shape[0],)) @ derivative) * lr
                    self.prev_updates[index] = [dEdw, dEdb]
                    self.layers[index].weights -= dEdw
                    self.layers[index].bias -= dEdb
                    derivative = derivative @ self.layers[index].weights.T * lr
                    # print("Change:", np.average(dEdw))
    def fit(self, x_train, y_train, epochs=1, batch_size=64):
        pass
        
    def __call__(self, x):
        return self.forward(x)