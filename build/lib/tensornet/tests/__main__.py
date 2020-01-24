from .. import *
import numpy as np

d = Model([
    Dense(128, 784),
    Activation("relu"),
    Dense(64, 128),
    Activation("relu"),
    Dense(10, 64),
    Activation('tanh')
])

x_train, y_train = (np.random.rand(10000, 784), 2*np.random.rand(10000, 10) - 1)

def loss(y_pred, y_real):
    return (y_pred - y_real)**2

def d_loss(y_pred, y_real):
    return 2 * (y_pred - y_real)

d.backprop(x_train, y_train, loss, d_loss)