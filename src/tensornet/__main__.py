from . import *
import cupy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

d = Model([
    Dense(512, 3072),
    LeakyReLU(0.4),
    Dense(343, 512),
    LeakyReLU(0.4),
    Dense(100, 343),
    LeakyReLU(0.4),
    Dense(10, 100),
    Activation("sigmoid"),
])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.array(x_train).reshape((50000, 3072))
x_test = np.array(x_test).reshape((10000, 3072))

y_train = np.array(to_categorical(y_train))
y_test = np.array(to_categorical(y_test))

def loss(y_pred, y_real):
    return ((y_pred - y_real)**2)/2

def d_loss(y_pred, y_real):
    return (y_pred - y_real)

BATCH_SIZE = 1024

for epoch in range(1000):
    print("="*10, "EPOCH", epoch, "="*10)
    test_indices = np.random.randint(x_test.shape[0], size=BATCH_SIZE)
    test_pred, test_real = d(x_test[test_indices]), y_test[test_indices]
    print("Loss:", np.average(loss(test_pred, test_real)))
    print("Accuracy:", np.count_nonzero(np.argmax(test_pred, -1) == np.argmax(test_real, -1))/BATCH_SIZE * 100, "%")
    for i in range(10):
        train_indices = np.random.randint(x_train.shape[0], size=BATCH_SIZE)
        d.backprop(x_train[train_indices], y_train[train_indices], loss, d_loss)