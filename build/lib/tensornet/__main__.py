from . import *
import cupy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

d = Model([
    Dense(512, 784),
    LeakyReLU(0.3),]\
    + [Dense(512, 512), LeakyReLU(0.3)] * 2\
    + [Dense(343, 512),
    LeakyReLU(0.3),
    Dense(128, 343),
    LeakyReLU(0.3),
    Dense(100, 128),
    LeakyReLU(0.3),
    Dense(10, 100),
    Activation("softmax"),
])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.array(x_train).reshape((60000, 784))
x_test = np.array(x_test).reshape((10000, 784))

y_train = np.array(to_categorical(y_train))
y_test = np.array(to_categorical(y_test))

TEST_BATCH_SIZE = 1000
TRAIN_BATCH_SIZE = 1000

lr = 1e-3

for epoch in range(100000):
    print("="*10, "EPOCH", epoch, "="*10)
    test_indices = np.random.randint(x_test.shape[0], size=TEST_BATCH_SIZE)
    test_pred, test_real = d(x_test[test_indices]), y_test[test_indices]
    print("Loss:", np.average(-np.log(test_pred[:, np.argmax(test_real, -1)])))
    print("Accuracy:", np.sum(np.argmax(test_pred, -1) == np.argmax(test_real, -1))/(test_pred.shape[0]) * 100, "%")
    for i in range(100):
        train_indices = np.random.randint(x_train.shape[0], size=TRAIN_BATCH_SIZE)
        d.train_batch(x_train[train_indices], y_train[train_indices], lr=lr, cross_entropy=True)
        lr *= 0.999