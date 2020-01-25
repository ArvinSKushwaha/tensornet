import cupy as np

def mse(y_pred, y_real):
    return 0.5 * (y_pred - y_real)**2

def d_mse(y_pred, y_real):
    return y_pred - y_real